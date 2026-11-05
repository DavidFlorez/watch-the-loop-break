"""One-command ephemeral Google Colab GPU runner (G4 default) with live streaming & auto-stop."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpu.common import utc_now_iso
from gpu.gcp_run import (
    _is_safe_result_relpath,
    create_upload_bundle,
    merge_downloaded_outputs,
)

DEFAULT_COLAB_GPU = "G4"
VALID_COLAB_GPUS = ("G4", "L4", "A100", "T4", "H100")
DEFAULT_SESSION_NAME = "wsp-g4"
DEFAULT_EXEC_TIMEOUT = 10800  # 3 hours (overrides `colab exec` default 30.0s timeout)

_SESSION_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$")
_SAFE_ARCHIVE_NAME_RE = re.compile(r"^wsp_out(?:_summary|_budget_\d+)\.tar\.gz$")


def validate_session_name(name: str) -> str:
    """Validate a Colab CLI session name against a strict allow-list regex."""
    cleaned = (name or "").strip()
    if not cleaned or not _SESSION_NAME_RE.match(cleaned):
        raise ValueError(
            f"Invalid Colab session name '{name}'. Must match {_SESSION_NAME_RE.pattern}"
        )
    return cleaned


def validate_colab_gpu(gpu: str) -> str:
    """Validate and normalize a Colab GPU variant ('G4', 'L4', 'A100', 'T4', 'H100')."""
    cleaned = (gpu or "").strip().upper()
    if cleaned not in VALID_COLAB_GPUS:
        raise ValueError(
            f"Unsupported Colab GPU '{gpu}'; expected one of {VALID_COLAB_GPUS}"
        )
    return cleaned


def parse_gpu_candidates(gpu: str, fallback_gpus: str = "") -> List[str]:
    """Return deduplicated ordered list of Colab GPU variants to attempt."""
    ordered = [validate_colab_gpu(gpu)]
    for item in (fallback_gpus or "").split(","):
        tok = item.strip()
        if not tok:
            continue
        norm = validate_colab_gpu(tok)
        if norm not in ordered:
            ordered.append(norm)
    return ordered


def resolve_ordered_budgets(
    budgets: Sequence[int],
    *,
    include_4k: bool = False,
    include_32k: bool = False,
) -> List[int]:
    """Return deduplicated ordered token budgets honoring `--include-4k` and `--include-32k`."""
    ordered: List[int] = []
    if include_4k:
        ordered.append(4096)
    for b in budgets:
        val = int(b)
        if val <= 0:
            raise ValueError(f"Budget must be positive, got {val}")
        if val not in ordered:
            ordered.append(val)
    if include_32k and 32768 not in ordered:
        ordered.append(32768)
    if not ordered:
        raise ValueError("At least one token budget is required.")
    return ordered


def build_colab_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return environment with unbuffered output and `certifi` CA bundle for `colab` CLI."""
    env = dict(base_env if base_env is not None else os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        import certifi

        ca_path = certifi.where()
        if ca_path and os.path.isfile(ca_path):
            env.setdefault("SSL_CERT_FILE", ca_path)
            env.setdefault("REQUESTS_CA_BUNDLE", ca_path)
    except Exception:
        pass
    return env


def _run_colab(
    args: Sequence[str],
    *,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = 180,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Execute `colab --auth=oauth2 ...` with stdin closed and TLS certs configured."""
    cmd = ["colab", "--auth=oauth2", *[str(a) for a in args]]
    proc = subprocess.run(
        cmd,
        env=build_colab_env(env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        err_detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"colab command failed ({proc.returncode}): {' '.join(cmd)}\n{err_detail}"
        )
    return proc


def build_colab_shell_script(
    *,
    budgets: Sequence[int],
    alphas: Sequence[float],
    n_problems: int,
    samples: int,
    seed: int,
    dtype: str,
    gpu_mem: float,
    enforce_eager: bool,
    smoke_only: bool,
    include_4k: bool = False,
    include_32k: bool = False,
    skip_smoke: bool = False,
    out_root: str = "assets",
    seed_jsonl: Optional[str] = None,
    derive_budgets: Sequence[int] = (),
    cached_forced_root: Optional[str] = None,
    chunk_size: int = 50,
    progress_interval_s: float = 300.0,
    skip_aggregate: bool = False,
) -> str:
    """Build remote bash script executed inside `/content/wsp/repo` using isolated `uv` virtualenv."""
    ordered_budgets = resolve_ordered_budgets(
        budgets, include_4k=include_4k, include_32k=include_32k
    )
    budget_tokens = " ".join(str(int(b)) for b in ordered_budgets)
    alpha_tokens = " ".join(shlex.quote(f"{float(a):g}") for a in alphas)
    eager_flag = "--enforce-eager" if enforce_eager else ""
    out_q = shlex.quote(str(out_root))
    seed_flag = f"--seed-jsonl {shlex.quote(str(seed_jsonl))}" if seed_jsonl else ""
    cached_forced_flag = (
        f"--cached-forced-root {shlex.quote(str(cached_forced_root))}" if cached_forced_root else ""
    )

    derive_step = ""
    if derive_budgets:
        der_str = ",".join(str(int(b)) for b in sorted({int(b) for b in derive_budgets}))
        derive_step = f"""
    echo "=== [WSP Colab] Deriving nested budgets ({der_str}) + Pass-2 forced answers for alpha=${{ALPHA}} ==="
    "$VENV_PY" -u gpu/derive_budgets.py \\
      --src {out_q} \\
      --out-root {out_q} \\
      --budgets {shlex.quote(der_str)} \\
      --alphas "$ALPHA" \\
      --run-forced \\
      --ckpt-dir /content/wsp/ckpt \\
      --dtype {shlex.quote(dtype)} \\
      --gpu-mem {float(gpu_mem)} \\
      --seed {int(seed)} \\
      {cached_forced_flag} \\
      {eager_flag}
"""

    agg_step = ""
    if not skip_aggregate:
        agg_step = f"""
echo "=== [WSP Colab] Aggregating cross-budget results ==="
"$VENV_PY" -u gpu/aggregate.py --out-root {out_q} --notes NOTES.md
"""

    smoke_section = ""
    if smoke_only or not skip_smoke:
        smoke_section = f"""
echo "=== [WSP Colab] Running smoke test on N=8 MATH-500 problems ==="
"$VENV_PY" -u gpu/smoke_test.py \\
  --n 8 \\
  --max-tokens 1024 \\
  --seed {int(seed)} \\
  --out-root {out_q} \\
  --gpu-mem {float(gpu_mem)} \\
  {eager_flag}
"""

    sweep_section = ""
    if not smoke_only:
        sweep_section = f"""
BUDGETS=({budget_tokens})
ALPHAS=({alpha_tokens})
for BUDGET in "${{BUDGETS[@]}}"; do
  for ALPHA in "${{ALPHAS[@]}}"; do
    echo "=== [WSP Colab] Running budget=${{BUDGET}} alpha=${{ALPHA}} out_root={out_root} ==="
    rm -rf /content/wsp/ckpt
    mkdir -p /content/wsp/ckpt
    "$VENV_PY" -u gpu/make_ckpt.py --alpha "$ALPHA" --out /content/wsp/ckpt
    "$VENV_PY" -u gpu/vllm_sweep.py \\
      --alpha "$ALPHA" \\
      --budget "$BUDGET" \\
      --n {int(n_problems)} \\
      --samples {int(samples)} \\
      --seed {int(seed)} \\
      --dtype {shlex.quote(dtype)} \\
      --gpu-mem {float(gpu_mem)} \\
      --out-root {out_q} \\
      --ckpt-dir /content/wsp/ckpt \\
      --chunk-size {int(chunk_size)} \\
      --progress-interval {float(progress_interval_s)} \\
      {seed_flag} \\
      {eager_flag}
{derive_step}
    rm -rf /content/wsp/ckpt
  done
done
{agg_step}
"""

    return f"""#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export HF_HOME=/content/wsp/hf_cache
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

cleanup_scratch() {{
  rm -rf /content/wsp/ckpt
}}
trap cleanup_scratch EXIT

cd /content/wsp/repo

echo "=== [WSP Colab] GPU & Driver Telemetry ==="
nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total --format=csv,noheader
nvidia-smi

if ! command -v uv >/dev/null 2>&1; then
  echo "=== [WSP Colab] Installing uv ==="
  python3 -m pip install -q uv || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

VENV_DIR="/content/wsp/venv"
VENV_PY="$VENV_DIR/bin/python"
export PATH="$VENV_DIR/bin:/usr/local/cuda/bin:$PATH"
if [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
fi
if [ ! -x "$VENV_PY" ]; then
  uv venv "$VENV_DIR" --python 3.12 || uv venv "$VENV_DIR" --python 3.11 || uv venv "$VENV_DIR" --python 3.10
fi

echo "=== [WSP Colab] Installing vLLM + CUDA runtime inside isolated uv venv ==="
if ! uv pip install --python "$VENV_PY" -r requirements-gpu.txt --torch-backend=auto; then
  uv pip install --python "$VENV_PY" -r requirements-gpu.txt
fi

if ! command -v nvcc >/dev/null 2>&1; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
fi

"$VENV_PY" -c "import shutil, torch, vllm, transformers, math_verify; print('[WSP Colab] Verified vllm=' + vllm.__version__ + ', torch=' + torch.__version__ + ', transformers=' + transformers.__version__ + ', cuda=' + str(torch.cuda.is_available()) + ', ninja=' + str(shutil.which('ninja')) + ', nvcc=' + str(shutil.which('nvcc')))"
{smoke_section}{sweep_section}
echo "=== [WSP Colab] Remote bash script finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
"""


_REMOTE_SUPERVISOR_SOURCE = r'''from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

WSP_ROOT = Path("/content/wsp")
REPO_DIR = WSP_ROOT / "repo"
RC_PATH = WSP_ROOT / "runner.rc"
RC_TMP_PATH = WSP_ROOT / "runner.rc.tmp"

FORBIDDEN_EXTS = (".safetensors", ".pt", ".bin", ".ckpt")
PROTECTED_N50_RE = re.compile(
    r"^assets/(?:budget_(?:4096|8192|16384|32768)/gpu-rtx-pro-6000-bfloat16/|sweep_results\.csv$)"
)


def _is_packageable_file(path: Path, repo_dir: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    name = path.name.lower()
    if name.startswith(".write.lock") or name.startswith(".log.lock") or name.endswith(".log"):
        return False
    if name.endswith(FORBIDDEN_EXTS):
        return False
    rel = path.relative_to(repo_dir).as_posix()
    if PROTECTED_N50_RE.match(rel):
        return False
    return True


def _package_results(repo_dir: Path) -> list[str]:
    import tarfile

    created: list[str] = []
    if not repo_dir.is_dir():
        return created

    summary_tar = Path("/content/wsp_out_summary.tar.gz")
    summary_files: list[tuple[Path, str]] = []
    by_budget: dict[str, list[tuple[Path, str]]] = {}

    assets_dir = repo_dir / "assets"
    if assets_dir.is_dir():
        for item in sorted(assets_dir.rglob("*")):
            if not _is_packageable_file(item, repo_dir):
                continue
            rel = item.relative_to(repo_dir).as_posix()
            parts = Path(rel).parts
            budget_suffix = None
            for part in parts:
                if part.startswith("budget_") and part.replace("budget_", "").isdigit():
                    budget_suffix = part.replace("budget_", "")
                    break

            if budget_suffix is not None:
                by_budget.setdefault(budget_suffix, []).append((item, rel))
                if not rel.endswith(".jsonl.gz"):
                    summary_files.append((item, rel))
            else:
                summary_files.append((item, rel))

    for b_suffix in sorted(by_budget.keys(), key=int):
        b_files = by_budget[b_suffix]
        b_tar = Path(f"/content/wsp_out_budget_{b_suffix}.tar.gz")
        with tarfile.open(b_tar, "w:gz") as tf:
            for src_p, arc_rel in b_files:
                tf.add(src_p, arcname=arc_rel)
        created.append(b_tar.name)

    if summary_files:
        with tarfile.open(summary_tar, "w:gz") as tf:
            for src_p, arc_rel in summary_files:
                tf.add(src_p, arcname=arc_rel)
        created.insert(0, summary_tar.name)

    return created


def main() -> None:
    runner_sh = REPO_DIR / "run_remote.sh"
    ret_code = 1
    archives: list[str] = []
    try:
        proc = subprocess.Popen(
            ["bash", str(runner_sh)],
            cwd=str(REPO_DIR),
            stdin=subprocess.DEVNULL,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        ret_code = proc.wait()
    except Exception as exc:
        sys.stdout.write(f"[WSP Colab] Supervisor caught exception: {exc}\n")
        sys.stdout.flush()
    finally:
        shutil.rmtree(WSP_ROOT / "ckpt", ignore_errors=True)
        shutil.rmtree(WSP_ROOT / "hf_cache", ignore_errors=True)
        archives = _package_results(REPO_DIR)
        sys.stdout.write(f"[WSP Colab] Packaged result archives: {archives}\n")
        sys.stdout.flush()
        RC_TMP_PATH.write_text(
            json.dumps({"returncode": int(ret_code), "archives": archives}),
            encoding="utf-8",
        )
        os.replace(RC_TMP_PATH, RC_PATH)


if __name__ == "__main__":
    main()
'''


_REMOTE_TAILER_HELPERS = r'''def _stream_log_slice(start_offset: int, slice_timeout_s: int) -> None:
    log_path = WSP_ROOT / "runner.log"
    rc_path = WSP_ROOT / "runner.rc"
    pos = max(0, int(start_offset))
    slice_start = time.monotonic()
    last_emit = slice_start

    while True:
        if log_path.is_file():
            with open(log_path, "rb") as fh:
                fh.seek(pos)
                chunk = fh.read()
                if chunk:
                    pos += len(chunk)
                    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
                    last_emit = time.monotonic()

        if rc_path.is_file():
            # Drain any trailing bytes written right before runner.rc appeared
            if log_path.is_file():
                with open(log_path, "rb") as fh:
                    fh.seek(pos)
                    tail = fh.read()
                    if tail:
                        pos += len(tail)
                        sys.stdout.write(tail.decode("utf-8", errors="replace"))
                        sys.stdout.flush()
            try:
                rc_info = json.loads(rc_path.read_text(encoding="utf-8"))
                rc_val = int(rc_info.get("returncode", 1))
            except Exception:
                rc_val = 1
            sys.stdout.write(f"\n[WSP_SLICE_STATE]=DONE offset={pos} rc={rc_val}\n")
            sys.stdout.flush()
            return

        now = time.monotonic()
        if now - last_emit >= 20.0:
            sys.stdout.write(f"[WSP Colab] heartbeat: remote sweep running (offset={pos})\n")
            sys.stdout.flush()
            last_emit = now

        if now - slice_start >= float(slice_timeout_s):
            sys.stdout.write(f"\n[WSP_SLICE_STATE]=CONTINUE offset={pos} rc=-1\n")
            sys.stdout.flush()
            return

        time.sleep(0.25)
'''


def build_colab_remote_driver(slice_timeout_s: int = 300) -> str:
    """Return initial remote Python driver run via `colab exec -f`.

    Unpacks `/content/wsp_bundle.tar.gz`, starts a detached supervisor running
    `/content/wsp/repo/run_remote.sh` into `/content/wsp/runner.log` (and packaging
    safe results on completion), and streams the first WebSocket slice.
    """
    return (
        "from __future__ import annotations\n\n"
        "import json\n"
        "import os\n"
        "import shutil\n"
        "import subprocess\n"
        "import sys\n"
        "import tarfile\n"
        "import time\n"
        "from pathlib import Path\n\n"
        'BUNDLE_PATH = Path("/content/wsp_bundle.tar.gz")\n'
        'WSP_ROOT = Path("/content/wsp")\n'
        'REPO_DIR = WSP_ROOT / "repo"\n'
        f"SUPERVISOR_CODE = {_REMOTE_SUPERVISOR_SOURCE!r}\n\n"
        + _REMOTE_TAILER_HELPERS
        + f"""

def _safe_extract_tar(src_tar: Path, dst_dir: Path) -> None:
    dst_resolved = dst_dir.resolve()
    dst_resolved.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src_tar, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.name or member.name.startswith("/") or "\\\\" in member.name:
                raise ValueError(f"Unsafe tar member path: {{member.name!r}}")
            target = (dst_resolved / member.name).resolve()
            if target != dst_resolved and not str(target).startswith(str(dst_resolved) + os.sep):
                raise ValueError(f"Tar path traversal blocked: {{member.name!r}}")
            if member.issym() or member.islnk():
                raise ValueError(f"Refusing symlink/hardlink in tar: {{member.name!r}}")
        tf.extractall(dst_resolved)


def main() -> None:
    for old in Path("/content").glob("wsp_out*.tar.gz"):
        old.unlink(missing_ok=True)
    for old_meta in ("runner.log", "runner.rc", "runner.rc.tmp", "supervisor.py"):
        (WSP_ROOT / old_meta).unlink(missing_ok=True)
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR, ignore_errors=True)
    shutil.rmtree(WSP_ROOT / "ckpt", ignore_errors=True)

    if not BUNDLE_PATH.is_file():
        raise FileNotFoundError(f"Upload bundle not found at {{BUNDLE_PATH}}")

    _safe_extract_tar(BUNDLE_PATH, WSP_ROOT)
    BUNDLE_PATH.unlink(missing_ok=True)

    runner_sh = REPO_DIR / "run_remote.sh"
    if not runner_sh.is_file():
        raise FileNotFoundError(f"Expected {{runner_sh}} after unpacking bundle")

    sup_path = WSP_ROOT / "supervisor.py"
    sup_path.write_text(SUPERVISOR_CODE, encoding="utf-8")
    log_fh = open(WSP_ROOT / "runner.log", "w", encoding="utf-8", buffering=1)
    try:
        subprocess.Popen(
            [sys.executable, "-u", str(sup_path)],
            cwd=str(WSP_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_fh.close()

    _stream_log_slice(start_offset=0, slice_timeout_s={int(slice_timeout_s)})


if __name__ == "__main__":
    main()
"""
    )


def build_colab_poll_driver(start_offset: int, slice_timeout_s: int = 300) -> str:
    """Return follow-up remote driver that tails `/content/wsp/runner.log` from `start_offset`."""
    return (
        "from __future__ import annotations\n\n"
        "import json\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n\n"
        'WSP_ROOT = Path("/content/wsp")\n\n'
        + _REMOTE_TAILER_HELPERS
        + f"""

if __name__ == "__main__":
    _stream_log_slice(start_offset={max(0, int(start_offset))}, slice_timeout_s={int(slice_timeout_s)})
"""
    )



def extract_tar_safely(archive_path: Path, staging_dir: Path) -> None:
    """Extract only safe relative members from `archive_path` into `staging_dir`."""
    staging_resolved = staging_dir.resolve()
    staging_resolved.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            rel = member.name.lstrip("./")
            if not _is_safe_result_relpath(rel):
                continue
            target = (staging_resolved / rel).resolve()
            if not str(target).startswith(str(staging_resolved) + os.sep):
                raise ValueError(f"Unsafe tar extraction path: {member.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            src_fh = tf.extractfile(member)
            if src_fh is None:
                continue
            with src_fh, open(target, "wb") as dst_fh:
                shutil.copyfileobj(src_fh, dst_fh)


def list_active_sessions(env: Optional[Dict[str, str]] = None) -> List[str]:
    """Return raw lines from `colab --auth=oauth2 sessions`."""
    res = _run_colab(["sessions"], env=env, timeout=30, check=True)
    lines: List[str] = []
    for raw in (res.stdout or "").splitlines():
        cleaned = raw.strip()
        if not cleaned or cleaned.startswith("[colab] No active sessions"):
            continue
        lines.append(cleaned)
    return lines


def session_exists(session_name: str, session_lines: Sequence[str]) -> bool:
    """Return True if `session_name` appears as the named prefix of any active session line."""
    for line in session_lines:
        if line.startswith("[?]"):
            continue
        first_col = line.split("|", 1)[0].strip()
        if first_col == session_name:
            return True
    return False


def stop_session_safe(
    session_name: str,
    *,
    env: Optional[Dict[str, str]] = None,
) -> None:
    """Best-effort immediate termination of `session_name` so compute units are not burned."""
    print(f"[Colab] Stopping session '{session_name}' ...", flush=True)
    res = _run_colab(
        ["stop", "--session", session_name],
        env=env,
        timeout=90,
        check=False,
    )
    if res.returncode == 0:
        print(f"[Colab] Session '{session_name}' stopped.", flush=True)
    else:
        detail = (res.stderr or res.stdout or "").strip()
        print(
            f"[Colab] Warning: `colab stop -s {session_name}` exited {res.returncode}: {detail}",
            flush=True,
        )


def _collect_extra_bundle_files(
    repo_root: Path,
    *,
    seed_jsonl: Optional[str] = None,
    cached_forced_root: Optional[str] = None,
) -> List[str]:
    extras: List[str] = []
    if seed_jsonl:
        rel = Path(seed_jsonl).as_posix().lstrip("./")
        if (repo_root / rel).is_file():
            extras.append(rel)
    if cached_forced_root:
        c_root = repo_root / cached_forced_root
        if c_root.is_dir():
            for p in sorted(c_root.glob("budget_*/gpu-*/*")):
                if p.is_file() and p.name in ("sweep_per_problem.jsonl.gz", "meta.json"):
                    rel = p.relative_to(repo_root).as_posix()
                    if rel not in extras:
                        extras.append(rel)
    return extras


_SLICE_STATE_RE = re.compile(
    r"\[WSP_SLICE_STATE\]=(DONE|CONTINUE)\s+offset=(\d+)\s+rc=(-?\d+)\s*$"
)


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    """Terminate (and if needed SIGKILL) a child process and its process group."""
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None

    if proc.poll() is None:
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=3.0)
        except Exception:
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=3.0)
            except Exception:
                pass


def _run_single_exec_slice(
    *,
    session_name: str,
    script_path: Path,
    env: Dict[str, str],
    slice_exec_timeout_s: int,
    line_inactivity_s: float = 85.0,
) -> tuple[Optional[str], int, int, bool]:
    """Run one `colab exec` slice with an inactivity watchdog against hung WebSockets."""
    exec_cmd = [
        "colab",
        "--auth=oauth2",
        "exec",
        "--session",
        session_name,
        "--file",
        str(script_path),
        "--timeout",
        str(slice_exec_timeout_s),
    ]
    exec_proc = subprocess.Popen(
        exec_cmd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert exec_proc.stdout is not None

    line_q: queue.Queue[Optional[str]] = queue.Queue()

    def _reader() -> None:
        assert exec_proc.stdout is not None
        try:
            for raw in exec_proc.stdout:
                line_q.put(raw)
        finally:
            line_q.put(None)

    reader_t = threading.Thread(target=_reader, daemon=True)
    reader_t.start()

    slice_state: Optional[str] = None
    offset = -1
    rc = -1
    stalled = False

    while True:
        try:
            item = line_q.get(timeout=line_inactivity_s)
        except queue.Empty:
            stalled = True
            _terminate_process_group(exec_proc)
            return slice_state, offset, rc, True

        if item is None:
            break

        clean = item.rstrip("\n")
        match = _SLICE_STATE_RE.search(clean.strip())
        if match:
            slice_state = match.group(1)
            offset = int(match.group(2))
            rc = int(match.group(3))
            prefix = clean[: match.start()].rstrip()
            if prefix:
                print(prefix, flush=True)
        else:
            print(clean, flush=True)

    proc_rc = exec_proc.wait()
    if slice_state == "DONE":
        return slice_state, offset, rc, False
    if slice_state == "CONTINUE":
        return slice_state, offset, -1, False
    return None, offset, proc_rc, stalled


def _execute_colab_detached(
    *,
    safe_session: str,
    tmp_path: Path,
    initial_driver_path: Path,
    colab_env: Dict[str, str],
    total_timeout_s: int,
    slice_timeout_s: int = 300,
) -> tuple[int, bool]:
    """Execute remote driver via resilient 5-minute WebSocket slices with reconnect."""
    deadline = time.monotonic() + float(total_timeout_s)
    stream_offset = 0
    slice_idx = 0
    consecutive_disconnects = 0

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Colab remote execution exceeded overall timeout of {total_timeout_s}s "
                f"(last log offset={stream_offset})."
            )

        this_slice_s = min(int(slice_timeout_s), max(30, int(remaining)))
        this_exec_timeout_s = this_slice_s + 180

        if slice_idx == 0:
            script_path = initial_driver_path
        else:
            script_path = tmp_path / f"colab_poll_{slice_idx}.py"
            script_path.write_text(
                build_colab_poll_driver(
                    start_offset=stream_offset, slice_timeout_s=this_slice_s
                ),
                encoding="utf-8",
            )

        slice_state, new_offset, rc, stalled = _run_single_exec_slice(
            session_name=safe_session,
            script_path=script_path,
            env=colab_env,
            slice_exec_timeout_s=this_exec_timeout_s,
        )
        if new_offset >= 0:
            stream_offset = max(stream_offset, new_offset)

        if slice_state == "DONE":
            return int(rc), (int(rc) == 0)

        if slice_state == "CONTINUE":
            consecutive_disconnects = 0
            slice_idx += 1
            continue

        # WebSocket dropped or stalled before slice completion: check whether VM is still alive
        active_lines = list_active_sessions(env=colab_env)
        if not session_exists(safe_session, active_lines):
            raise RuntimeError(
                f"Colab session '{safe_session}' was terminated or preempted on server "
                f"(last streamed log offset={stream_offset})."
            )

        consecutive_disconnects += 1
        if consecutive_disconnects > 12:
            raise RuntimeError(
                f"Colab session '{safe_session}' lost WebSocket stream {consecutive_disconnects} "
                f"consecutive times (last streamed log offset={stream_offset})."
            )

        reason = "stalled" if stalled else "closed prematurely"
        print(
            f"[Colab] WebSocket {reason} while VM '{safe_session}' is still active; "
            f"reconnecting at log offset {stream_offset} (attempt {consecutive_disconnects}/12) ...",
            flush=True,
        )
        time.sleep(3.0)
        slice_idx = max(1, slice_idx + 1)


def run_on_colab(
    *,
    gpu: str = DEFAULT_COLAB_GPU,
    fallback_gpus: str = "",
    high_mem: bool = False,
    session_name: str = DEFAULT_SESSION_NAME,
    reuse_session: bool = False,
    budgets: Sequence[int] = (8192, 16384),
    include_4k: bool = False,
    include_32k: bool = False,
    skip_smoke: bool = False,
    alphas: Sequence[float] = (0.0, 1.0, 1.25),
    n_problems: int = 50,
    samples: int = 1,
    seed: int = 0,
    dtype: str = "auto",
    gpu_mem: float = 0.85,
    enforce_eager: bool = False,
    smoke_only: bool = False,
    out_root: str = "assets",
    seed_jsonl: Optional[str] = None,
    derive_budgets: Sequence[int] = (),
    cached_forced_root: Optional[str] = None,
    chunk_size: int = 50,
    progress_interval_s: float = 300.0,
    skip_aggregate: bool = False,
    exec_timeout: int = DEFAULT_EXEC_TIMEOUT,
    keep_session: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Provision a Colab GPU session (G4 default), run the vLLM sweep, download results, and stop session."""
    safe_session = validate_session_name(session_name)
    gpu_candidates = parse_gpu_candidates(gpu, fallback_gpus=fallback_gpus)
    ordered_budgets = resolve_ordered_budgets(
        budgets, include_4k=include_4k, include_32k=include_32k
    )
    timeout_s = max(60, int(exec_timeout))

    shell_script = build_colab_shell_script(
        budgets=ordered_budgets,
        alphas=alphas,
        n_problems=n_problems,
        samples=samples,
        seed=seed,
        dtype=dtype,
        gpu_mem=gpu_mem,
        enforce_eager=enforce_eager,
        smoke_only=smoke_only,
        skip_smoke=skip_smoke,
        out_root=out_root,
        seed_jsonl=seed_jsonl,
        derive_budgets=derive_budgets,
        cached_forced_root=cached_forced_root,
        chunk_size=chunk_size,
        progress_interval_s=progress_interval_s,
        skip_aggregate=skip_aggregate,
    )
    driver_py = build_colab_remote_driver()
    extra_files = _collect_extra_bundle_files(
        REPO_ROOT,
        seed_jsonl=seed_jsonl,
        cached_forced_root=cached_forced_root,
    )

    if dry_run:
        plan = {
            "dry_run": True,
            "session_name": safe_session,
            "gpu_candidates": gpu_candidates,
            "high_mem": bool(high_mem),
            "budgets": ordered_budgets,
            "derive_budgets": [int(b) for b in derive_budgets],
            "out_root": str(out_root),
            "seed_jsonl": seed_jsonl,
            "cached_forced_root": cached_forced_root,
            "extra_bundle_files": extra_files,
            "alphas": [float(a) for a in alphas],
            "exec_timeout_s": timeout_s,
            "keep_session": bool(keep_session),
            "colab_new_cmd": [
                "colab",
                "--auth=oauth2",
                "new",
                "--session",
                safe_session,
                "--gpu",
                gpu_candidates[0],
                *(["--high-mem"] if high_mem else []),
            ],
            "shell_script": shell_script,
            "driver_py": driver_py,
        }
        print(
            json.dumps(
                {k: v for k, v in plan.items() if k not in ("shell_script", "driver_py")},
                indent=2,
            ),
            flush=True,
        )
        return plan

    colab_env = build_colab_env()
    active_lines = list_active_sessions(env=colab_env)
    already_active = session_exists(safe_session, active_lines)
    if already_active and not reuse_session:
        raise RuntimeError(
            f"Colab session '{safe_session}' already exists. Pass `--reuse-session` or `--session <other>`."
        )

    active_gpu: Optional[str] = None
    created_here = False
    copied_files: List[str] = []

    try:
        if already_active:
            print(f"[Colab] Reusing existing session '{safe_session}'.", flush=True)
        else:
            last_err = ""
            for cand_gpu in gpu_candidates:
                cmd = ["new", "--session", safe_session, "--gpu", cand_gpu]
                if high_mem:
                    cmd.append("--high-mem")
                print(
                    f"[Colab] Provisioning session '{safe_session}' with --gpu {cand_gpu} ...",
                    flush=True,
                )
                res = _run_colab(cmd, env=colab_env, timeout=180, check=False)
                if res.returncode == 0:
                    active_gpu = cand_gpu
                    created_here = True
                    print(
                        f"[Colab] Session '{safe_session}' READY (gpu={cand_gpu}).",
                        flush=True,
                    )
                    break
                last_err = (res.stderr or res.stdout or "").strip()
                print(
                    f"[Colab] Failed to allocate --gpu {cand_gpu} (exit {res.returncode}): {last_err}",
                    flush=True,
                )

            if not created_here:
                raise RuntimeError(
                    f"Could not allocate Colab session across {gpu_candidates}. Last error:\n{last_err}"
                )

        with tempfile.TemporaryDirectory(prefix="wsp_colab_stage_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            bundle_path = tmp_path / "wsp_bundle.tar.gz"
            driver_path = tmp_path / "colab_driver.py"
            create_upload_bundle(
                REPO_ROOT,
                runner_script=shell_script,
                bundle_path=bundle_path,
                extra_repo_files=extra_files,
            )
            driver_path.write_text(driver_py, encoding="utf-8")

            print(
                f"[Colab] Uploading bundle ({bundle_path.stat().st_size / 1024:.1f} KB) to /content/wsp_bundle.tar.gz ...",
                flush=True,
            )
            _run_colab(
                [
                    "upload",
                    "--session",
                    safe_session,
                    str(bundle_path),
                    "/content/wsp_bundle.tar.gz",
                ],
                env=colab_env,
                timeout=180,
                check=True,
            )

            print(
                f"[Colab] Executing remote setup & vLLM sweep (timeout={timeout_s}s, sliced WebSocket streaming) ...",
                flush=True,
            )
            exec_ret, saw_clean_sentinel = _execute_colab_detached(
                safe_session=safe_session,
                tmp_path=tmp_path,
                initial_driver_path=driver_path,
                colab_env=colab_env,
                total_timeout_s=timeout_s,
            )

            # Discover & download packaged archives even if a later step failed partway through
            ls_res = _run_colab(
                ["ls", "--session", safe_session, "content"],
                env=colab_env,
                timeout=60,
                check=False,
            )
            remote_archives: List[str] = []
            if ls_res.returncode == 0:
                for raw_name in (ls_res.stdout or "").splitlines():
                    cleaned = raw_name.strip().rstrip("/")
                    if _SAFE_ARCHIVE_NAME_RE.match(cleaned) and cleaned not in remote_archives:
                        remote_archives.append(cleaned)

            if not remote_archives:
                all_expected_b = sorted({*ordered_budgets, *(int(b) for b in derive_budgets)})
                remote_archives = [
                    "wsp_out_summary.tar.gz",
                    *(f"wsp_out_budget_{b}.tar.gz" for b in all_expected_b),
                ]

            staging_dir = tmp_path / "staging"
            archives_dir = tmp_path / "archives"
            archives_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"[Colab] Downloading {len(remote_archives)} archive(s): {remote_archives} ...",
                flush=True,
            )
            failed_archives: List[str] = []
            for arch_name in remote_archives:
                if not _SAFE_ARCHIVE_NAME_RE.match(arch_name):
                    continue
                local_arch = archives_dir / arch_name
                dl_ok = False
                dl_err = ""
                for attempt in range(1, 4):
                    dl_res = _run_colab(
                        [
                            "download",
                            "--session",
                            safe_session,
                            f"/content/{arch_name}",
                            str(local_arch),
                        ],
                        env=colab_env,
                        timeout=600,
                        check=False,
                    )
                    if dl_res.returncode == 0 and local_arch.is_file():
                        extract_tar_safely(local_arch, staging_dir)
                        dl_ok = True
                        break
                    dl_err = (dl_res.stderr or dl_res.stdout or "").strip()
                    print(
                        f"[Colab] Retry {attempt}/3 downloading {arch_name} ({dl_err})",
                        flush=True,
                    )
                if not dl_ok:
                    failed_archives.append(f"{arch_name}: {dl_err}")

            copied_files = merge_downloaded_outputs(staging_dir, REPO_ROOT)
            print(
                f"[Colab] Merged {len(copied_files)} file(s) safely into local repo.",
                flush=True,
            )
            if failed_archives:
                print(
                    f"[Colab] Warning: some remote archives failed to download: {failed_archives}",
                    flush=True,
                )

            if exec_ret != 0 or not saw_clean_sentinel:
                raise RuntimeError(
                    f"Remote Colab sweep did not complete cleanly (exec_ret={exec_ret}, "
                    f"sentinel={saw_clean_sentinel}, salvaged={len(copied_files)} file(s))."
                )
            if failed_archives:
                raise RuntimeError(f"Failed to download archive(s): {failed_archives}")

        # Re-run local aggregation when writing directly to default `assets`
        if not skip_aggregate and str(out_root) == "assets":
            agg_py = REPO_ROOT / ".venv" / "bin" / "python"
            py_exec = str(agg_py) if agg_py.is_file() else sys.executable
            if Path(py_exec).is_file() and copied_files:
                subprocess.run(
                    [
                        py_exec,
                        "-u",
                        str(REPO_ROOT / "gpu" / "aggregate.py"),
                        "--out-root",
                        "assets",
                        "--notes",
                        "NOTES.md",
                    ],
                    cwd=str(REPO_ROOT),
                    check=False,
                )

        return {
            "completed_at_utc": utc_now_iso(),
            "session_name": safe_session,
            "gpu": active_gpu or gpu_candidates[0],
            "budgets": ordered_budgets,
            "derive_budgets": [int(b) for b in derive_budgets],
            "out_root": str(out_root),
            "alphas": [float(a) for a in alphas],
            "synced_files": copied_files,
        }
    finally:
        if created_here and not keep_session:
            stop_session_safe(safe_session, env=colab_env)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the watch-the-loop-break vLLM sweep on Google Colab (G4 default) via `colab` CLI."
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=DEFAULT_COLAB_GPU,
        help=f"Colab GPU accelerator variant: {', '.join(VALID_COLAB_GPUS)} (default: {DEFAULT_COLAB_GPU}).",
    )
    parser.add_argument(
        "--fallback-gpus",
        type=str,
        default="",
        help="Optional comma-separated fallback GPUs if primary allocation fails (e.g. 'L4,A100,T4').",
    )
    parser.add_argument(
        "--high-mem",
        action="store_true",
        help="Pass `--high-mem` to `colab new` (applicable to T4/A100; G4 has a single 96GB shape).",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=DEFAULT_SESSION_NAME,
        help=f"Colab session name (default: {DEFAULT_SESSION_NAME}).",
    )
    parser.add_argument(
        "--reuse-session",
        action="store_true",
        help="Reuse an existing session with `--session` instead of creating a new one.",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="8192,16384",
        help="Comma-separated Pass-1 token budgets (default: 8192,16384).",
    )
    parser.add_argument(
        "--include-4k",
        action="store_true",
        help="Prepend 4096 to the budget list for direct MPS vs GPU comparison.",
    )
    parser.add_argument(
        "--include-32k",
        action="store_true",
        help="Append 32768 to the budget list for full 32k paper comparison.",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.0,1.0,1.25",
        help="Comma-separated blend coefficients alpha (default: 0.0,1.0,1.25).",
    )
    parser.add_argument("--n", type=int, default=50, help="Number of MATH-500 problems (default: 50).")
    parser.add_argument("--samples", type=int, default=1, help="Samples per problem (default: 1).")
    parser.add_argument("--seed", type=int, default=0, help="Dataset & sampling seed (default: 0).")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
        help="Model dtype (default: auto -> bfloat16 on G4/L4/A100, smoke-tested on T4).",
    )
    parser.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization (default: 0.85).")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile in vLLM.")
    parser.add_argument("--smoke-only", action="store_true", help="Run only the 8-problem smoke test.")
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the 8-problem smoke test when dtype has already been verified on this GPU.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="assets",
        help="Target output root directory under repo (default: assets).",
    )
    parser.add_argument(
        "--seed-jsonl",
        type=str,
        default=None,
        help="Optional repo-relative sweep_per_problem.jsonl.gz to upload and seed cached rows.",
    )
    parser.add_argument(
        "--derive-budgets",
        type=str,
        default="",
        help="Optional comma-separated budgets (e.g. '4096,8192,16384,32768') to derive + force from 32k.",
    )
    parser.add_argument(
        "--cached-forced-root",
        type=str,
        default=None,
        help="Optional repo-relative assets root (e.g. 'assets') to upload & reuse cached Pass-2 forced continuations.",
    )
    parser.add_argument("--chunk-size", type=int, default=50, help="Problems per checkpoint chunk (default: 50).")
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=300.0,
        help="Seconds between 5-min progress/ETA logs during generation (default: 300).",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip running gpu/aggregate.py on NOTES.md during this sweep.",
    )
    parser.add_argument(
        "--exec-timeout",
        type=int,
        default=DEFAULT_EXEC_TIMEOUT,
        help=f"Timeout in seconds passed to `colab exec --timeout` (default: {DEFAULT_EXEC_TIMEOUT}).",
    )
    parser.add_argument(
        "--keep-session",
        action="store_true",
        help="Keep the Colab session running after completion instead of stopping it in `finally:`.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution plan without provisioning a Colab session.",
    )
    args = parser.parse_args(argv)

    budgets_list = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    alphas_list = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    derive_list = [int(x.strip()) for x in (args.derive_budgets or "").split(",") if x.strip()]
    if not budgets_list:
        raise ValueError("At least one budget is required.")
    if not alphas_list:
        raise ValueError("At least one alpha is required.")

    summary = run_on_colab(
        gpu=args.gpu,
        fallback_gpus=args.fallback_gpus,
        high_mem=args.high_mem,
        session_name=args.session,
        reuse_session=args.reuse_session,
        budgets=budgets_list,
        include_4k=args.include_4k,
        include_32k=args.include_32k,
        skip_smoke=args.skip_smoke,
        alphas=alphas_list,
        n_problems=args.n,
        samples=args.samples,
        seed=args.seed,
        dtype=args.dtype,
        gpu_mem=args.gpu_mem,
        enforce_eager=args.enforce_eager,
        smoke_only=args.smoke_only,
        out_root=args.out_root,
        seed_jsonl=args.seed_jsonl,
        derive_budgets=derive_list,
        cached_forced_root=args.cached_forced_root,
        chunk_size=args.chunk_size,
        progress_interval_s=args.progress_interval,
        skip_aggregate=args.skip_aggregate,
        exec_timeout=args.exec_timeout,
        keep_session=args.keep_session,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(
            f"[Colab] Done! session={summary['session_name']} ({summary['gpu']}) | "
            f"Synced {len(summary['synced_files'])} file(s).",
            flush=True,
        )


if __name__ == "__main__":
    main()
