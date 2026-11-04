"""One-command ephemeral Google Compute Engine GPU runner with L4->T4 fallback & auto-teardown."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpu.common import utc_now_iso

METRICS_ENV_TAG = "datacloud.antigravity"
DEFAULT_MAX_MINUTES = 90
MAX_ALLOWED_MINUTES = 240
DEFAULT_BOOT_DISK_GB = 60
DEFAULT_IMAGE_PROJECT = "deeplearning-platform-release"
DEFAULT_IMAGE_FAMILY = "common-cu124-ubuntu-2204-py310"

CANDIDATE_ZONES: Tuple[str, ...] = (
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-east1-b",
    "us-east1-c",
    "us-east4-a",
    "us-west1-a",
    "us-west1-b",
    "europe-west4-a",
)

GPU_PROFILES: Dict[str, Dict[str, Any]] = {
    "l4": {
        "gpu_tag": "l4",
        "accelerator_type": "nvidia-l4",
        "machine_type": "g2-standard-4",
        "vram_gb": 24,
        "compute_cap": "8.9",
        "native_bf16": True,
        "spot_quota_metric": "PREEMPTIBLE_NVIDIA_L4_GPUS",
        "standard_quota_metric": "NVIDIA_L4_GPUS",
        "spot_usd_per_hr": 0.22,
        "standard_usd_per_hr": 0.71,
    },
    "t4": {
        "gpu_tag": "t4",
        "accelerator_type": "nvidia-tesla-t4",
        "machine_type": "n1-standard-4",
        "vram_gb": 16,
        "compute_cap": "7.5",
        "native_bf16": False,
        "spot_quota_metric": "PREEMPTIBLE_NVIDIA_T4_GPUS",
        "standard_quota_metric": "NVIDIA_T4_GPUS",
        "spot_usd_per_hr": 0.11,
        "standard_usd_per_hr": 0.35,
    },
}

_GCP_PROJECT_RE = re.compile(r"^[a-z][a-z0-9-]{4,29}$")
_GCP_ZONE_RE = re.compile(r"^[a-z]+-[a-z0-9-]+[0-9]-[a-z]$")
_GCP_NAME_RE = re.compile(r"^[a-z]([-a-z0-9]{0,61}[a-z0-9])?$")

_UPLOAD_EXACT_FILES: Tuple[str, ...] = (
    "weights.py",
    "metrics.py",
    "sweep.py",
    "app.py",
    "requirements-gpu.txt",
    "README.md",
    "NOTES.md",
    ".gitignore",
)


def validate_gcp_identifier(value: str, pattern: re.Pattern[str], label: str) -> str:
    """Validate a GCP resource identifier against a strict allow-list regex."""
    cleaned = (value or "").strip()
    if not cleaned or not pattern.match(cleaned):
        raise ValueError(f"Invalid GCP {label} '{value}'. Must match {pattern.pattern}")
    return cleaned


def region_from_zone(zone: str) -> str:
    """Extract GCP region name (e.g. 'us-central1') from zone (e.g. 'us-central1-a')."""
    z = validate_gcp_identifier(zone, _GCP_ZONE_RE, "zone")
    return z.rsplit("-", 1)[0]


def build_gcloud_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return environment with non-interactive prompts disabled and mandatory GCP attribution."""
    env = dict(base_env if base_env is not None else os.environ)
    env["CLOUDSDK_CORE_DISABLE_PROMPTS"] = "1"
    existing = (env.get("CLOUDSDK_METRICS_ENVIRONMENT") or "").strip()
    tokens = existing.split() if existing else []
    if METRICS_ENV_TAG not in tokens:
        tokens.append(METRICS_ENV_TAG)
    env["CLOUDSDK_METRICS_ENVIRONMENT"] = " ".join(tokens)
    return env


def _run_gcloud(
    args: Sequence[str],
    *,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = 120,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Execute a `gcloud` CLI command with attribution env and stdin closed."""
    cmd = ["gcloud", *[str(a) for a in args]]
    proc = subprocess.run(
        cmd,
        env=build_gcloud_env(env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        err_detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"gcloud command failed ({proc.returncode}): {' '.join(cmd)}\n{err_detail}")
    return proc


def check_region_gpu_quota(
    quotas: Sequence[Dict[str, Any]],
    gpu_key: str,
    *,
    is_spot: bool = True,
) -> Optional[float]:
    """Return available count (`limit - usage`) for `gpu_key` in region quotas, or `None` if unlisted."""
    if gpu_key not in GPU_PROFILES:
        raise ValueError(f"Unknown gpu_key '{gpu_key}'; expected one of {tuple(GPU_PROFILES)}")
    prof = GPU_PROFILES[gpu_key]
    quota_map: Dict[str, float] = {}
    for q in quotas:
        metric = str(q.get("metric") or "").strip()
        if not metric:
            continue
        try:
            avail = float(q.get("limit", 0.0)) - float(q.get("usage", 0.0))
        except (TypeError, ValueError):
            avail = 0.0
        quota_map[metric] = avail

    if is_spot and prof["spot_quota_metric"] in quota_map:
        return quota_map[prof["spot_quota_metric"]]
    if prof["standard_quota_metric"] in quota_map:
        return quota_map[prof["standard_quota_metric"]]
    return None


def check_global_gpu_quota(quotas: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Return available global `GPUS_ALL_REGIONS` quota (`limit - usage`), or `None` if unlisted."""
    for q in quotas:
        if str(q.get("metric") or "").strip() == "GPUS_ALL_REGIONS":
            try:
                return float(q.get("limit", 0.0)) - float(q.get("usage", 0.0))
            except (TypeError, ValueError):
                return 0.0
    return None


def resolve_project_and_account(explicit_project: Optional[str] = None) -> Tuple[str, str]:
    """Verify active `gcloud` authentication and resolve an accessible GCP project ID."""
    acct_res = _run_gcloud(
        ["auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
        timeout=30,
        check=False,
    )
    account = (acct_res.stdout or "").strip().splitlines()[0].strip() if acct_res.stdout.strip() else ""
    if not account:
        raise RuntimeError(
            "No active `gcloud` account found. Run `gcloud auth login` first and re-run."
        )

    # Probe accessible projects for active account
    proj_res = _run_gcloud(
        ["projects", "list", "--format=value(projectId)", "--limit=25"],
        timeout=30,
        check=False,
    )
    if proj_res.returncode != 0 and "invalid_grant" in (proj_res.stderr or ""):
        raise RuntimeError(
            f"Active gcloud account '{account}' has an expired token (`invalid_grant`).\n"
            f"Run: `gcloud auth login {account}` (or `gcloud config set account <ACCOUNT>`) and retry."
        )
    accessible_projects = [line.strip() for line in (proj_res.stdout or "").splitlines() if line.strip()]

    candidate = (
        explicit_project
        or os.environ.get("GCP_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or ""
    ).strip()

    if not candidate:
        cfg_res = _run_gcloud(["config", "get-value", "project"], timeout=15, check=False)
        cfg_proj = (cfg_res.stdout or "").strip()
        if cfg_proj and cfg_proj in accessible_projects:
            candidate = cfg_proj
        else:
            avail_str = ", ".join(accessible_projects) if accessible_projects else "<none>"
            raise RuntimeError(
                f"Configured gcloud project '{cfg_proj or '<unset>'}' is not accessible for account '{account}'.\n"
                f"Accessible projects on this machine: {avail_str}\n"
                "Pass `--project <PROJECT_ID>` (with billing + Compute Engine enabled) or create one:\n"
                "  gcloud projects create <NEW_PROJECT_ID>\n"
                "  gcloud beta billing projects link <NEW_PROJECT_ID> --billing-account=<BILLING_ID>\n"
                "  python gpu/gcp_run.py --project <NEW_PROJECT_ID>"
            )

    validated = validate_gcp_identifier(candidate, _GCP_PROJECT_RE, "project ID")
    return validated, account


def ensure_compute_api_enabled(project: str, dry_run: bool = False) -> None:
    """Verify `compute.googleapis.com` is enabled on `project` (or enable it automatically)."""
    if dry_run:
        return
    check_res = _run_gcloud(
        [
            "services",
            "list",
            "--enabled",
            f"--project={project}",
            "--filter=config.name:compute.googleapis.com",
            "--format=value(config.name)",
        ],
        timeout=30,
        check=False,
    )
    if check_res.returncode == 0 and "compute.googleapis.com" in (check_res.stdout or ""):
        return

    print(f"[GCP] Enabling compute.googleapis.com on project '{project}' ...", flush=True)
    enable_res = _run_gcloud(
        ["services", "enable", "compute.googleapis.com", f"--project={project}"],
        timeout=120,
        check=False,
    )
    if enable_res.returncode != 0:
        err = (enable_res.stderr or enable_res.stdout or "").strip()
        raise RuntimeError(
            f"Failed to enable Compute Engine API (`compute.googleapis.com`) on project '{project}'.\n"
            f"Ensure an active GCP Billing Account is linked to '{project}'.\n"
            f"Link billing: https://console.cloud.google.com/billing/linkedaccount?project={project}\n"
            f"Details: {err}"
        )


def resolve_latest_dlvm_image() -> Tuple[str, str]:
    """Return `(image_project, image_family)` for Google Deep Learning VM Ubuntu CUDA image."""
    res = _run_gcloud(
        [
            "compute",
            "images",
            "list",
            f"--project={DEFAULT_IMAGE_PROJECT}",
            "--no-standard-images",
            "--filter=family ~ '^common-cu12' AND name ~ 'ubuntu' AND status = 'READY'",
            "--sort-by=~creationTimestamp",
            "--limit=1",
            "--format=value(family)",
        ],
        timeout=30,
        check=False,
    )
    fam = (res.stdout or "").strip().splitlines()[0].strip() if res.stdout.strip() else ""
    return DEFAULT_IMAGE_PROJECT, (fam or DEFAULT_IMAGE_FAMILY)


def select_gpu_candidates(
    *,
    gpu_pref: str = "auto",
    zone_override: Optional[str] = None,
    provisioning: str = "spot",
    project: Optional[str] = None,
    inspect_live_quota: bool = True,
) -> List[Dict[str, Any]]:
    """Return ordered list of `(gpu_key, zone, provisioning, profile, avail_quota)` placement candidates."""
    gpu_pref_norm = (gpu_pref or "auto").strip().lower()
    if gpu_pref_norm not in ("auto", "l4", "t4"):
        raise ValueError(f"Unsupported --gpu '{gpu_pref}'; expected auto, l4, or t4.")

    prov_norm = (provisioning or "spot").strip().lower()
    if prov_norm not in ("auto", "spot", "standard"):
        raise ValueError(f"Unsupported --provisioning '{provisioning}'; expected auto, spot, or standard.")

    gpu_order = ["l4", "t4"] if gpu_pref_norm == "auto" else [gpu_pref_norm]
    zones = [validate_gcp_identifier(zone_override, _GCP_ZONE_RE, "zone")] if zone_override else list(CANDIDATE_ZONES)
    prov_order = ["spot", "standard"] if prov_norm == "auto" else [prov_norm]

    region_cache: Dict[str, List[Dict[str, Any]]] = {}
    global_avail: Optional[float] = None

    if inspect_live_quota and project:
        proj_info = _run_gcloud(
            ["compute", "project-info", "describe", f"--project={project}", "--format=json(quotas)"],
            timeout=30,
            check=False,
        )
        if proj_info.returncode == 0 and proj_info.stdout.strip():
            try:
                p_quotas = json.loads(proj_info.stdout).get("quotas", [])
                global_avail = check_global_gpu_quota(p_quotas)
            except Exception:
                global_avail = None

        if global_avail is not None and global_avail < 1.0:
            raise RuntimeError(
                f"Project '{project}' has 0 available global GPU quota (`GPUS_ALL_REGIONS`).\n"
                f"Request a free quota increase (to >= 1) in GCP Console:\n"
                f"  https://console.cloud.google.com/iam-admin/quotas?project={project}&metric=GPUs"
            )

        unique_regions = sorted({region_from_zone(z) for z in zones})
        for reg in unique_regions:
            r_res = _run_gcloud(
                ["compute", "regions", "describe", reg, f"--project={project}", "--format=json(quotas)"],
                timeout=30,
                check=False,
            )
            if r_res.returncode == 0 and r_res.stdout.strip():
                try:
                    region_cache[reg] = json.loads(r_res.stdout).get("quotas", [])
                except Exception:
                    region_cache[reg] = []

    candidates: List[Dict[str, Any]] = []
    quota_skipped: List[str] = []
    for gpu_key in gpu_order:
        prof = GPU_PROFILES[gpu_key]
        for prov_mode in prov_order:
            is_spot = prov_mode == "spot"
            for z in zones:
                reg = region_from_zone(z)
                avail = None
                if inspect_live_quota and reg in region_cache:
                    avail = check_region_gpu_quota(region_cache[reg], gpu_key, is_spot=is_spot)
                    if avail is not None and avail < 1.0:
                        quota_skipped.append(f"{z}:{gpu_key}({prov_mode})={avail:g}")
                        continue
                candidates.append(
                    {
                        "gpu_key": gpu_key,
                        "zone": z,
                        "region": reg,
                        "provisioning": prov_mode,
                        "profile": prof,
                        "available_quota": avail,
                    }
                )

    if not candidates and quota_skipped:
        raise RuntimeError(
            f"No GPU quota >= 1 available in project '{project}' across candidate zones.\n"
            f"Checked quotas: {', '.join(quota_skipped[:8])}\n"
            f"Request GPU quota (NVIDIA_L4_GPUS or NVIDIA_T4_GPUS >= 1) in GCP Console:\n"
            f"  https://console.cloud.google.com/iam-admin/quotas?project={project}&metric=NVIDIA"
        )
    return candidates


def build_create_instance_cmd(
    *,
    vm_name: str,
    project: str,
    zone: str,
    gpu_key: str,
    provisioning: str = "spot",
    max_minutes: int = DEFAULT_MAX_MINUTES,
    boot_disk_gb: int = DEFAULT_BOOT_DISK_GB,
    image_project: str = DEFAULT_IMAGE_PROJECT,
    image_family: str = DEFAULT_IMAGE_FAMILY,
) -> List[str]:
    """Construct `gcloud compute instances create` CLI arguments with hard auto-delete timer."""
    safe_vm = validate_gcp_identifier(vm_name, _GCP_NAME_RE, "VM name")
    safe_proj = validate_gcp_identifier(project, _GCP_PROJECT_RE, "project ID")
    safe_zone = validate_gcp_identifier(zone, _GCP_ZONE_RE, "zone")
    if gpu_key not in GPU_PROFILES:
        raise ValueError(f"Unsupported gpu_key '{gpu_key}'")
    if provisioning not in ("spot", "standard"):
        raise ValueError(f"Unsupported provisioning '{provisioning}'")
    max_min = int(max_minutes)
    if max_min < 5 or max_min > MAX_ALLOWED_MINUTES:
        raise ValueError(f"--max-minutes must be between 5 and {MAX_ALLOWED_MINUTES}, got {max_min}")

    prof = GPU_PROFILES[gpu_key]
    cmd = [
        "compute",
        "instances",
        "create",
        safe_vm,
        f"--project={safe_proj}",
        f"--zone={safe_zone}",
        f"--machine-type={prof['machine_type']}",
        f"--accelerator=type={prof['accelerator_type']},count=1",
        "--maintenance-policy=TERMINATE",
        f"--boot-disk-size={int(boot_disk_gb)}GB",
        "--boot-disk-type=pd-balanced",
        f"--image-project={image_project}",
        f"--image-family={image_family}",
        "--metadata=install-nvidia-driver=True",
        f"--max-run-duration={max_min * 60}s",
        "--instance-termination-action=DELETE",
        "--labels=app=watch-the-loop-break,workload=vllm-sweep,managed-by=gcp-run",
    ]
    if provisioning == "spot":
        cmd.append("--provisioning-model=SPOT")
    return cmd


def build_remote_runner_script(
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
) -> str:
    """Build remote bash script executed inside `/opt/wsp/repo` using isolated `uv` virtualenv."""
    ordered_budgets = list(budgets)
    if include_4k and 4096 not in ordered_budgets:
        ordered_budgets = [4096, *ordered_budgets]

    budget_tokens = " ".join(str(int(b)) for b in ordered_budgets)
    alpha_tokens = " ".join(shlex.quote(f"{float(a):g}") for a in alphas)
    eager_flag = "--enforce-eager" if enforce_eager else ""

    sweep_section = ""
    if not smoke_only:
        sweep_section = f"""
BUDGETS=({budget_tokens})
ALPHAS=({alpha_tokens})
for BUDGET in "${{BUDGETS[@]}}"; do
  for ALPHA in "${{ALPHAS[@]}}"; do
    echo "=== [WSP GPU] Running budget=${{BUDGET}} alpha=${{ALPHA}} ==="
    rm -rf /tmp/wsp_ckpt
    mkdir -p /tmp/wsp_ckpt
    "$VENV_PY" -u gpu/make_ckpt.py --alpha "$ALPHA" --out /tmp/wsp_ckpt
    "$VENV_PY" -u gpu/vllm_sweep.py \\
      --alpha "$ALPHA" \\
      --budget "$BUDGET" \\
      --n {int(n_problems)} \\
      --samples {int(samples)} \\
      --seed {int(seed)} \\
      --dtype {shlex.quote(dtype)} \\
      --gpu-mem {float(gpu_mem)} \\
      --out-root assets \\
      --ckpt-dir /tmp/wsp_ckpt \\
      {eager_flag}
    rm -rf /tmp/wsp_ckpt
  done
done

echo "=== [WSP GPU] Aggregating cross-budget results ==="
"$VENV_PY" -u gpu/aggregate.py --out-root assets --notes NOTES.md
"""

    return f"""#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

cleanup_scratch() {{
  rm -rf /tmp/wsp_ckpt
}}
trap cleanup_scratch EXIT

cd /opt/wsp/repo

echo "=== [WSP GPU] Waiting for NVIDIA driver readiness ==="
for i in $(seq 1 45); do
  if nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader; then
    break
  fi
  echo "Waiting for nvidia-smi ($i/45)..."
  sleep 4
done
nvidia-smi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

VENV_DIR="/opt/wsp/venv"
VENV_PY="$VENV_DIR/bin/python"
if [ ! -x "$VENV_PY" ]; then
  uv venv "$VENV_DIR" --python 3.11 || uv venv "$VENV_DIR" --python 3.10
fi

echo "=== [WSP GPU] Installing vLLM + CUDA runtime inside isolated uv venv ==="
if ! uv pip install --python "$VENV_PY" -r requirements-gpu.txt --torch-backend=auto; then
  uv pip install --python "$VENV_PY" -r requirements-gpu.txt
fi

echo "=== [WSP GPU] Running smoke test on N=8 MATH-500 problems ==="
"$VENV_PY" -u gpu/smoke_test.py \\
  --n 8 \\
  --max-tokens 1024 \\
  --seed {int(seed)} \\
  --out-root assets \\
  --gpu-mem {float(gpu_mem)} \\
  {eager_flag}
{sweep_section}
echo "=== [WSP GPU] Packaging GPU artifacts ==="
rm -rf /tmp/wsp_out
mkdir -p /tmp/wsp_out/assets
if [ -f NOTES.md ]; then
  cp NOTES.md /tmp/wsp_out/NOTES.md
fi
if [ -f assets/budget_curve_gpu.png ]; then
  cp assets/budget_curve_gpu.png /tmp/wsp_out/assets/budget_curve_gpu.png
fi
for sf in assets/gpu_smoke_*.json; do
  if [ -f "$sf" ]; then
    cp "$sf" /tmp/wsp_out/assets/
  fi
done
for bd in assets/budget_*; do
  if [ -d "$bd" ]; then
    cp -R "$bd" /tmp/wsp_out/assets/
  fi
done
find /tmp/wsp_out -type f -name '*.safetensors' -delete
echo "=== [WSP GPU] Completed successfully at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
"""


def _is_safe_result_relpath(rel_path: str) -> bool:
    """Validate relative artifact path copied back from remote VM."""
    if not rel_path or rel_path.startswith("/") or "\\" in rel_path:
        return False
    p = Path(rel_path)
    if p.is_absolute() or ".." in p.parts:
        return False
    lower = p.name.lower()
    if lower.endswith((".safetensors", ".pt", ".bin", ".ckpt")):
        return False
    if rel_path == "NOTES.md":
        return True
    if p.parts[0] == "assets":
        return True
    return False


def create_upload_bundle(repo_root: Path, runner_script: str, bundle_path: Path) -> None:
    """Package local code + scripts (no `.safetensors`, `.venv`, or `.git`) into a tar.gz bundle."""
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "w:gz") as tf:
        for fname in _UPLOAD_EXACT_FILES:
            fpath = repo_root / fname
            if fpath.is_file():
                tf.add(fpath, arcname=f"repo/{fname}")

        gpu_dir = repo_root / "gpu"
        for pyfile in sorted(gpu_dir.glob("*.py")):
            if pyfile.is_file():
                tf.add(pyfile, arcname=f"repo/gpu/{pyfile.name}")

        mps_csv = repo_root / "assets" / "sweep_results.csv"
        if mps_csv.is_file():
            tf.add(mps_csv, arcname="repo/assets/sweep_results.csv")

        script_bytes = runner_script.encode("utf-8")
        info = tarfile.TarInfo(name="repo/run_remote.sh")
        info.size = len(script_bytes)
        info.mode = 0o755
        tf.addfile(info, fileobj=_BytesIO(script_bytes))


class _BytesIO:
    """Minimal read-only binary stream wrapper for `tarfile.TarFile.addfile`."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk = self._data[self._pos :]
            self._pos = len(self._data)
            return chunk
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk


def merge_downloaded_outputs(staging_dir: Path, repo_root: Path) -> List[str]:
    """Safely merge downloaded GPU results from `staging_dir` into `repo_root` without touching MPS assets."""
    repo_resolved = repo_root.resolve()
    copied: List[str] = []
    if not staging_dir.is_dir():
        return copied

    for src in sorted(staging_dir.rglob("*")):
        if not src.is_file() or src.is_symlink():
            continue
        rel = src.relative_to(staging_dir).as_posix()
        if not _is_safe_result_relpath(rel):
            continue
        dst = (repo_resolved / rel).resolve()
        if not str(dst).startswith(str(repo_resolved) + os.sep):
            raise ValueError(f"Unsafe target path outside repository: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    return copied


def _wait_for_ssh(vm_name: str, project: str, zone: str, timeout_s: int = 210) -> None:
    """Wait until SSH on the newly created GCE VM accepts commands."""
    deadline = time.monotonic() + timeout_s
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        res = _run_gcloud(
            [
                "compute",
                "ssh",
                vm_name,
                f"--project={project}",
                f"--zone={zone}",
                "--command=echo ready",
                "--",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
            ],
            timeout=25,
            check=False,
        )
        if res.returncode == 0 and "ready" in (res.stdout or ""):
            return
        print(f"[GCP] Waiting for VM SSH readiness (attempt {attempt}) ...", flush=True)
        time.sleep(6)
    raise TimeoutError(f"Timed out after {timeout_s}s waiting for SSH on {vm_name} ({zone}).")


def delete_instance_safe(vm_name: str, project: str, zone: str) -> None:
    """Best-effort immediate deletion of the ephemeral GCE VM."""
    print(f"[GCP] Deleting ephemeral VM '{vm_name}' in {zone} ...", flush=True)
    res = _run_gcloud(
        [
            "compute",
            "instances",
            "delete",
            vm_name,
            f"--project={project}",
            f"--zone={zone}",
            "--quiet",
        ],
        timeout=180,
        check=False,
    )
    if res.returncode == 0:
        print(f"[GCP] VM '{vm_name}' deleted.", flush=True)
    else:
        err = (res.stderr or "").strip()
        if "was not found" in err:
            print(f"[GCP] VM '{vm_name}' already removed.", flush=True)
        else:
            print(f"[GCP] Warning: delete returned exit {res.returncode}: {err}", flush=True)


def run_on_gcp(
    *,
    project: Optional[str] = None,
    gpu: str = "auto",
    zone: Optional[str] = None,
    provisioning: str = "spot",
    budgets: Sequence[int] = (8192, 16384),
    alphas: Sequence[float] = (0.0, 1.0, 1.25),
    n_problems: int = 50,
    samples: int = 1,
    seed: int = 0,
    dtype: str = "auto",
    gpu_mem: float = 0.85,
    enforce_eager: bool = False,
    smoke_only: bool = False,
    include_32k: bool = False,
    include_4k: bool = False,
    max_minutes: int = DEFAULT_MAX_MINUTES,
    boot_disk_gb: int = DEFAULT_BOOT_DISK_GB,
    keep_vm: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Provision an ephemeral GCE GPU VM, run the vLLM sweep, sync results locally, and delete the VM."""
    effective_budgets = list(budgets)
    if include_32k and 32768 not in effective_budgets:
        effective_budgets.append(32768)

    runner_script = build_remote_runner_script(
        budgets=effective_budgets,
        alphas=alphas,
        n_problems=n_problems,
        samples=samples,
        seed=seed,
        dtype=dtype,
        gpu_mem=gpu_mem,
        enforce_eager=enforce_eager,
        smoke_only=smoke_only,
        include_4k=include_4k,
    )

    if dry_run:
        safe_proj = validate_gcp_identifier(project or "example-gcp-project", _GCP_PROJECT_RE, "project ID")
        candidates = select_gpu_candidates(
            gpu_pref=gpu,
            zone_override=zone,
            provisioning=provisioning,
            project=None,
            inspect_live_quota=False,
        )
        top = candidates[0]
        vm_name = f"wsp-sweep-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        create_cmd = build_create_instance_cmd(
            vm_name=vm_name,
            project=safe_proj,
            zone=top["zone"],
            gpu_key=top["gpu_key"],
            provisioning=top["provisioning"],
            max_minutes=max_minutes,
            boot_disk_gb=boot_disk_gb,
        )
        plan = {
            "dry_run": True,
            "project": safe_proj,
            "selected_candidate": {k: v for k, v in top.items() if k != "profile"},
            "machine_type": top["profile"]["machine_type"],
            "accelerator": f"1x {top['profile']['accelerator_type']} ({top['profile']['vram_gb']}GB)",
            "budgets": effective_budgets,
            "alphas": [float(a) for a in alphas],
            "max_minutes_auto_delete": int(max_minutes),
            "gcloud_create_cmd": ["gcloud", *create_cmd],
            "runner_script": runner_script,
        }
        print(json.dumps({k: v for k, v in plan.items() if k != "runner_script"}, indent=2))
        return plan

    resolved_project, account = resolve_project_and_account(explicit_project=project)
    print(f"[GCP] Account: {account} | Project: {resolved_project}", flush=True)
    ensure_compute_api_enabled(resolved_project, dry_run=False)

    candidates = select_gpu_candidates(
        gpu_pref=gpu,
        zone_override=zone,
        provisioning=provisioning,
        project=resolved_project,
        inspect_live_quota=True,
    )
    img_project, img_family = resolve_latest_dlvm_image()
    print(f"[GCP] VM Image: {img_project} / {img_family}", flush=True)

    ts_slug = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    vm_name = f"wsp-sweep-{ts_slug}"

    created_zone: Optional[str] = None
    created_gpu_key: Optional[str] = None
    created_prov: Optional[str] = None
    last_err = ""

    for cand in candidates:
        c_gpu = cand["gpu_key"]
        c_zone = cand["zone"]
        c_prov = cand["provisioning"]
        prof = cand["profile"]
        rate = prof["spot_usd_per_hr"] if c_prov == "spot" else prof["standard_usd_per_hr"]
        print(
            f"[GCP] Attempting to provision {prof['machine_type']} + 1x {prof['accelerator_type']} "
            f"({c_prov.upper()}, ~${rate:.2f}/hr, auto-delete <= {max_minutes}m) in {c_zone} ...",
            flush=True,
        )
        create_cmd = build_create_instance_cmd(
            vm_name=vm_name,
            project=resolved_project,
            zone=c_zone,
            gpu_key=c_gpu,
            provisioning=c_prov,
            max_minutes=max_minutes,
            boot_disk_gb=boot_disk_gb,
            image_project=img_project,
            image_family=img_family,
        )
        c_res = _run_gcloud(create_cmd, timeout=180, check=False)
        if c_res.returncode == 0:
            created_zone = c_zone
            created_gpu_key = c_gpu
            created_prov = c_prov
            print(f"[GCP] Provisioned '{vm_name}' in {c_zone} ({c_gpu.upper()}, {c_prov}).", flush=True)
            break

        last_err = (c_res.stderr or c_res.stdout or "").strip()
        if "ZONE_RESOURCE_POOL_EXHAUSTED" in last_err or "does not have enough resources" in last_err:
            print(f"[GCP] Stockout in {c_zone} ({c_gpu}/{c_prov}); trying next candidate...", flush=True)
            continue
        if "QUOTA_EXCEEDED" in last_err:
            print(f"[GCP] Quota exceeded in {c_zone} ({c_gpu}/{c_prov}); trying next candidate...", flush=True)
            continue
        raise RuntimeError(f"Failed to create GCE VM in {c_zone}:\n{last_err}")

    if not created_zone:
        raise RuntimeError(
            f"Could not provision a GPU VM across {len(candidates)} candidate(s). Last error:\n{last_err}"
        )

    copied_files: List[str] = []
    try:
        _wait_for_ssh(vm_name=vm_name, project=resolved_project, zone=created_zone)

        with tempfile.TemporaryDirectory(prefix="wsp_gcp_stage_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            bundle_path = tmp_path / "wsp_bundle.tar.gz"
            create_upload_bundle(REPO_ROOT, runner_script=runner_script, bundle_path=bundle_path)

            print("[GCP] Uploading workspace code to VM ...", flush=True)
            _run_gcloud(
                [
                    "compute",
                    "scp",
                    str(bundle_path),
                    f"{vm_name}:~/wsp_bundle.tar.gz",
                    f"--project={resolved_project}",
                    f"--zone={created_zone}",
                ],
                timeout=120,
                check=True,
            )

            print("[GCP] Starting remote setup & vLLM sweep (streaming live output) ...", flush=True)
            remote_exec_cmd = (
                "sudo mkdir -p /opt/wsp && sudo chown -R $(whoami):$(whoami) /opt/wsp && "
                "tar -xzf ~/wsp_bundle.tar.gz -C /opt/wsp && rm -f ~/wsp_bundle.tar.gz && "
                "bash /opt/wsp/repo/run_remote.sh"
            )
            ssh_proc = subprocess.Popen(
                [
                    "gcloud",
                    "compute",
                    "ssh",
                    vm_name,
                    f"--project={resolved_project}",
                    f"--zone={created_zone}",
                    f"--command={remote_exec_cmd}",
                ],
                env=build_gcloud_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert ssh_proc.stdout is not None
            for line in ssh_proc.stdout:
                print(line.rstrip("\n"), flush=True)
            ssh_ret = ssh_proc.wait()

            # Even if a later budget timed out or failed, salvage any completed assets before deleting VM
            download_dir = tmp_path / "downloaded"
            download_dir.mkdir(parents=True, exist_ok=True)
            print("[GCP] Downloading completed results from VM ...", flush=True)
            _run_gcloud(
                [
                    "compute",
                    "ssh",
                    vm_name,
                    f"--project={resolved_project}",
                    f"--zone={created_zone}",
                    "--command=bash -lc 'mkdir -p /tmp/wsp_out/assets && cp -R /opt/wsp/repo/assets/budget_* /tmp/wsp_out/assets/ 2>/dev/null || true; cp /opt/wsp/repo/assets/gpu_smoke_*.json /opt/wsp/repo/assets/budget_curve_gpu.png /tmp/wsp_out/assets/ 2>/dev/null || true; find /tmp/wsp_out -type f -name \"*.safetensors\" -delete'",
                ],
                timeout=60,
                check=False,
            )
            scp_out = _run_gcloud(
                [
                    "compute",
                    "scp",
                    "--recurse",
                    f"{vm_name}:/tmp/wsp_out",
                    str(download_dir),
                    f"--project={resolved_project}",
                    f"--zone={created_zone}",
                ],
                timeout=180,
                check=False,
            )
            wsp_out_dir = download_dir / "wsp_out"
            if scp_out.returncode == 0 and wsp_out_dir.is_dir():
                copied_files = merge_downloaded_outputs(wsp_out_dir, REPO_ROOT)
                print(f"[GCP] Synced {len(copied_files)} file(s) into local assets/.", flush=True)

            if ssh_ret != 0:
                raise RuntimeError(f"Remote execution failed with exit code {ssh_ret}.")

        # Re-run local aggregation so 4k MPS baseline + new GCP results are combined in NOTES.md & chart
        agg_py = REPO_ROOT / ".venv" / "bin" / "python"
        py_exec = str(agg_py) if agg_py.is_file() else sys.executable
        if Path(py_exec).is_file():
            subprocess.run(
                [py_exec, "-u", str(REPO_ROOT / "gpu" / "aggregate.py"), "--out-root", "assets", "--notes", "NOTES.md"],
                cwd=str(REPO_ROOT),
                check=False,
            )

        return {
            "completed_at_utc": utc_now_iso(),
            "project": resolved_project,
            "account": account,
            "vm_name": vm_name,
            "zone": created_zone,
            "gpu_key": created_gpu_key,
            "provisioning": created_prov,
            "synced_files": copied_files,
        }
    finally:
        if not keep_vm and created_zone:
            delete_instance_safe(vm_name=vm_name, project=resolved_project, zone=created_zone)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the watch-the-loop-break vLLM sweep on an ephemeral Google Cloud GPU VM with auto-teardown."
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Google Cloud Project ID (defaults to GCP_PROJECT env var or active accessible gcloud project).",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="auto",
        choices=("auto", "l4", "t4"),
        help="GPU accelerator preference: auto (L4 native bf16 first, T4 fallback), l4, or t4.",
    )
    parser.add_argument(
        "--zone",
        type=str,
        default=None,
        help="Optional explicit GCE zone (e.g. us-central1-a). If omitted, auto-selects a zone with GPU quota.",
    )
    parser.add_argument(
        "--provisioning",
        type=str,
        default="spot",
        choices=("spot", "standard", "auto"),
        help="VM provisioning model (default: spot for ~65%% cost savings; or 'auto' / 'standard').",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="8192,16384",
        help="Comma-separated Pass-1 token budgets to run on GPU (default: 8192,16384).",
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
        help="Model dtype on VM (default: auto -> bfloat16 on L4, smoke-tested fp16/fp32 on T4).",
    )
    parser.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization (default: 0.85).")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile in vLLM.")
    parser.add_argument("--smoke-only", action="store_true", help="Run only the 8-problem smoke test on the VM.")
    parser.add_argument("--include-32k", action="store_true", help="Also run budget=32768 after 8192 & 16384.")
    parser.add_argument("--include-4k", action="store_true", help="Also run budget=4096 on GPU for direct MPS vs GPU comparison.")
    parser.add_argument(
        "--max-minutes",
        type=int,
        default=DEFAULT_MAX_MINUTES,
        help=f"Hard GCP auto-delete wall-clock cap in minutes (5..{MAX_ALLOWED_MINUTES}, default: {DEFAULT_MAX_MINUTES}).",
    )
    parser.add_argument(
        "--boot-disk-gb",
        type=int,
        default=DEFAULT_BOOT_DISK_GB,
        help=f"VM boot disk size in GB (default: {DEFAULT_BOOT_DISK_GB}).",
    )
    parser.add_argument(
        "--keep-vm",
        action="store_true",
        help="Keep VM after completion instead of deleting immediately (not recommended; max-run-duration still applies).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print provisioning plan and remote commands without creating any cloud resources.",
    )
    args = parser.parse_args(argv)

    budgets_list = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    alphas_list = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    if not budgets_list:
        raise ValueError("At least one budget is required.")
    if not alphas_list:
        raise ValueError("At least one alpha is required.")

    summary = run_on_gcp(
        project=args.project,
        gpu=args.gpu,
        zone=args.zone,
        provisioning=args.provisioning,
        budgets=budgets_list,
        alphas=alphas_list,
        n_problems=args.n,
        samples=args.samples,
        seed=args.seed,
        dtype=args.dtype,
        gpu_mem=args.gpu_mem,
        enforce_eager=args.enforce_eager,
        smoke_only=args.smoke_only,
        include_32k=args.include_32k,
        include_4k=args.include_4k,
        max_minutes=args.max_minutes,
        boot_disk_gb=args.boot_disk_gb,
        keep_vm=args.keep_vm,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(
            f"[GCP] Done! VM={summary['vm_name']} ({summary['zone']}) | "
            f"Synced {len(summary['synced_files'])} file(s).",
            flush=True,
        )


if __name__ == "__main__":
    main()
