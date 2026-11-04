"""Shared GPU environment detection, scratch management, locking, and vLLM engine utilities."""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import gc
import gzip
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
SHARD_FILENAME = "model.safetensors-00001-of-00001.safetensors"
STOP_TOKEN_IDS: Tuple[int, int] = (248046, 248044)
SAMPLING_DEFAULTS: Dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.5,
}
FORCE_ANSWER_SUFFIX = (
    "\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n"
    "</think>\n\n"
)
FORCE_BOX_OPEN_SUFFIX = "\n\nTherefore, the final answer is \\boxed{"
FORCED_MAX_TOKENS = 512

# Official 32k MATH-500 reference points from Takehi et al. (2026) / lm_understand-logs
PAPER_32K_REFERENCE: Dict[str, Dict[str, Any]] = {
    "base": {
        "alpha": 0.0,
        "budget": 32768,
        "accuracy_pct": 7.9,
        "hit_cap_pct": 93.0,
        "label": "Paper 32k Base (α=0.0)",
    },
    "edit": {
        "alpha": 1.0,
        "budget": 32768,
        "accuracy_pct": 48.1,
        "hit_cap_pct": 33.05,
        "label": "Paper 32k Edited p50c (α=1.0)",
    },
}

_NON_ASCII_RUN_RE = re.compile(r"[^\x00-\x7F\s]{8,}")
_NON_ASCII_SEQ_RE = re.compile(r"[^\x00-\x7F]{12,}")
_REPEATED_NAN_WORD_RE = re.compile(r"(?:\bnan\b[\s,;]*){6,}", re.IGNORECASE)


def utc_now_iso() -> str:
    """Return current UTC ISO-8601 timestamp string."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_today_str() -> str:
    """Return current UTC YYYY-MM-DD date string."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").exists()


def is_colab() -> bool:
    if Path("/content").exists():
        return True
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def get_scratch_root() -> Path:
    """Return a writable scratch directory outside the git repo for temporary checkpoints."""
    repo_resolved = REPO_ROOT.resolve()
    candidates: List[Path] = []
    if is_kaggle():
        candidates.append(Path("/kaggle/tmp"))
    if is_colab():
        candidates.append(Path("/content/tmp"))
    candidates.append(Path(tempfile.gettempdir()))

    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            resolved = cand.resolve()
            # Ensure we never place raw checkpoints inside the repository tree
            if resolved == repo_resolved or repo_resolved in resolved.parents:
                continue
            probe = resolved / f".probe_{os.getpid()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return resolved
        except Exception:
            continue

    fallback = Path(tempfile.gettempdir()).resolve()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def make_scratch_ckpt_dir(alpha: float, prefix: str = "wsp_ckpt") -> Path:
    """Create a unique scratch checkpoint directory outside the repository."""
    root = get_scratch_root()
    safe_alpha = f"{alpha:g}".replace("-", "m").replace(".", "p")
    ckpt_dir = root / f"{prefix}_a{safe_alpha}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def cleanup_ckpt_dir(ckpt_dir: Optional[Union[str, Path]]) -> None:
    """Remove a scratch checkpoint directory immediately to free disk space."""
    if ckpt_dir is None:
        return
    path = Path(ckpt_dir).resolve()
    if not path.exists():
        return
    repo_resolved = REPO_ROOT.resolve()
    if path == repo_resolved:
        raise ValueError("Refusing to delete repository root!")
    shutil.rmtree(path, ignore_errors=True)


def normalize_gpu_tag(gpu_name: str) -> str:
    """Normalize an NVIDIA GPU name into a clean lowercase filesystem tag (e.g., 'Tesla T4' -> 't4')."""
    raw = (gpu_name or "").strip().lower()
    if not raw or raw in ("unknown", "none", "cpu"):
        return "cpu"

    for known in ("b200", "b100", "h200", "h100", "a100", "a10g", "a10", "l40s", "l40", "l4", "v100", "p100", "t4"):
        if re.search(rf"\b{known}\b", raw) or known in raw:
            return known

    if re.search(r"\brtx\s+pro\s+6000\b", raw):
        return "rtx-pro-6000"

    rtx_match = re.search(r"rtx\s*[-_]?\s*(\d{4}(?:\s*ti|\s*super)?)", raw)
    if rtx_match:
        return "rtx" + re.sub(r"\s+", "", rtx_match.group(1))

    cleaned = re.sub(r"\b(nvidia|geforce|tesla|quadro|pcie|sxm\d*|max-q|\d+gb)\b", " ", raw)
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned).strip("-")
    return cleaned or "gpu"


def query_nvidia_smi() -> Dict[str, Any]:
    """Query `nvidia-smi` for GPU name(s), driver version, CUDA version, and compute capability."""
    info: Dict[str, Any] = {
        "available": False,
        "driver_version": None,
        "cuda_version": None,
        "gpu_count": 0,
        "gpu_name": None,
        "compute_capability": None,
        "memory_total_mib": None,
        "gpus": [],
    }
    try:
        csv_res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,compute_cap,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if csv_res.returncode != 0 or not csv_res.stdout.strip():
            return info

        gpus: List[Dict[str, Any]] = []
        for line in csv_res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                idx = len(gpus)
            try:
                cc_val: Optional[float] = float(parts[3])
            except ValueError:
                cc_val = None
            try:
                mem_mib: Optional[int] = int(float(parts[4]))
            except ValueError:
                mem_mib = None
            gpus.append(
                {
                    "index": idx,
                    "name": parts[1],
                    "driver_version": parts[2],
                    "compute_capability": cc_val,
                    "memory_total_mib": mem_mib,
                    "tag": normalize_gpu_tag(parts[1]),
                }
            )

        if not gpus:
            return info

        # Determine active GPU index if CUDA_VISIBLE_DEVICES is set
        active_idx = 0
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd and cvd != "-1":
            first_tok = cvd.split(",")[0].strip()
            if first_tok.isdigit():
                req_idx = int(first_tok)
                if 0 <= req_idx < len(gpus):
                    active_idx = req_idx

        active_gpu = gpus[active_idx]
        info["available"] = True
        info["driver_version"] = active_gpu["driver_version"]
        info["gpu_count"] = len(gpus)
        info["gpu_name"] = active_gpu["name"]
        info["compute_capability"] = active_gpu["compute_capability"]
        info["memory_total_mib"] = active_gpu["memory_total_mib"]
        info["gpus"] = gpus

        # Parse CUDA version from standard `nvidia-smi` banner
        banner_res = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if banner_res.returncode == 0 and banner_res.stdout:
            m = re.search(r"CUDA Version:\s*([0-9.]+)", banner_res.stdout)
            if m:
                info["cuda_version"] = m.group(1)
    except Exception:
        pass
    return info


def supports_native_bfloat16(smi_info: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when the active NVIDIA GPU has compute capability >= 8.0 (e.g., Ampere A100 or Ada L4)."""
    info = smi_info if smi_info is not None else query_nvidia_smi()
    cc = info.get("compute_capability")
    if cc is None and info.get("gpus"):
        cc = info["gpus"][0].get("compute_capability")
    try:
        return cc is not None and float(cc) >= 8.0
    except (TypeError, ValueError):
        return False


def detect_gpu_info(gpu_tag_override: Optional[str] = None) -> Dict[str, Any]:
    """Detect GPU metadata and return normalized tag (e.g. 't4', 'l4')."""
    smi = query_nvidia_smi()
    gpu_name = smi.get("gpu_name")
    if not gpu_name:
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None

    tag = normalize_gpu_tag(gpu_tag_override) if gpu_tag_override else normalize_gpu_tag(gpu_name or "t4")
    return {
        "gpu_tag": tag,
        "gpu_name": gpu_name or tag.upper(),
        "compute_capability": smi.get("compute_capability"),
        "native_bfloat16": supports_native_bfloat16(smi),
        "nvidia_smi": smi,
    }


def check_min_compute_capability(min_cc: float = 7.5, require_gpu: bool = True) -> Dict[str, Any]:
    """Verify at least one NVIDIA GPU is present with compute capability >= `min_cc` (e.g. 7.5 for T4)."""
    smi = query_nvidia_smi()
    if not smi["available"] or not smi["gpus"]:
        if require_gpu:
            raise RuntimeError(
                "No NVIDIA GPU detected via nvidia-smi. Enable a GPU runtime (e.g., Kaggle T4 x2, Colab T4, or GCE L4/T4)."
            )
        return smi

    for gpu in smi["gpus"]:
        cc = gpu.get("compute_capability")
        if cc is not None and cc < min_cc:
            raise RuntimeError(
                f"GPU {gpu['index']} ({gpu['name']}) has compute capability {cc:.1f} < {min_cc:.1f}. "
                "vLLM requires Turing/Volta+ GPUs such as NVIDIA T4 (CC 7.5); older GPUs (e.g., P100 CC 6.0) are unsupported."
            )
    return smi


def get_pip_freeze_filtered() -> List[str]:
    """Return frozen package lines matching `vllm|torch|transformers|triton` (with uv/importlib fallback)."""
    pattern = re.compile(r"vllm|torch|transformers|triton", re.IGNORECASE)
    for cmd in (
        [sys.executable, "-m", "pip", "freeze"],
        ["uv", "pip", "freeze", "--python", sys.executable],
    ):
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                matched = [line.strip() for line in res.stdout.splitlines() if pattern.search(line)]
                if matched:
                    return matched
        except Exception:
            continue

    try:
        from importlib import metadata as importlib_metadata

        dists = []
        for dist in importlib_metadata.distributions():
            name = (dist.metadata.get("Name") or "").strip()
            ver = (dist.version or "").strip()
            if name and ver and pattern.search(name):
                dists.append(f"{name}=={ver}")
        return sorted(set(dists), key=str.lower)
    except Exception:
        return []


def collect_environment_metadata(gpu_tag_override: Optional[str] = None) -> Dict[str, Any]:
    """Collect Python, pip package versions (`vllm|torch|transformers|triton`), and `nvidia-smi` telemetry."""
    gpu_info = detect_gpu_info(gpu_tag_override=gpu_tag_override)
    pkg_versions: Dict[str, Optional[str]] = {}
    for pkg in ("vllm", "torch", "transformers", "triton", "datasets", "math_verify", "safetensors"):
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, "__version__", "installed")
        except Exception:
            pkg_versions[pkg] = None

    return {
        "python_version": platform.python_version(),
        "python_full": sys.version,
        "platform": platform.platform(),
        "runtime": "kaggle" if is_kaggle() else ("colab" if is_colab() else "local"),
        "gpu_tag": gpu_info["gpu_tag"],
        "gpu_name": gpu_info["gpu_name"],
        "compute_capability": gpu_info["compute_capability"],
        "native_bfloat16": gpu_info["native_bfloat16"],
        "nvidia_smi": gpu_info["nvidia_smi"],
        "pip_freeze_filtered": get_pip_freeze_filtered(),
        "package_versions": pkg_versions,
    }


def detect_nan_junk(text: str, token_ids: Optional[Sequence[int]] = None) -> Optional[str]:
    """Return a failure reason if `text` / `token_ids` shows fp16 NaN garbage, else `None`."""
    if token_ids is not None and len(token_ids) == 0:
        return "empty token_ids (0 tokens generated)"
    if not text or not text.strip():
        return "empty or whitespace-only text"
    if "\ufffd" in text:
        count = text.count("\ufffd")
        return f"contains U+FFFD replacement character ({count} occurrences)"
    m_run = _NON_ASCII_RUN_RE.search(text)
    if m_run:
        return f"long run of non-ASCII symbols ({len(m_run.group(0))} chars: {m_run.group(0)[:24]!r})"
    m_seq = _NON_ASCII_SEQ_RE.search(text)
    if m_seq:
        return f"long sequence of non-ASCII codepoints ({len(m_seq.group(0))} chars)"
    m_nan = _REPEATED_NAN_WORD_RE.search(text)
    if m_nan:
        return f"repeated 'nan' token output ({m_nan.group(0)[:32]!r})"
    return None


@contextlib.contextmanager
def file_lock(lock_path: Union[str, Path]) -> Iterator[None]:
    """Acquire an exclusive `fcntl.flock` lock around concurrent multi-GPU writes."""
    lpath = Path(lock_path)
    lpath.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lpath), os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def atomic_write_text(path: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
    """Atomically write text to `path` via a same-directory temp file and `os.replace`."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp"
    try:
        with open(tmp_path, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_gzip_jsonl(path: Union[str, Path], records: Iterable[Dict[str, Any]]) -> None:
    """Atomically write an iterable of dicts as gzipped JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp"
    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8") as gz:
            for rec in records:
                gz.write(json.dumps(rec, ensure_ascii=False) + "\n")
            gz.flush()
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def make_tokens_prompt(token_ids: Sequence[int]) -> Any:
    """Construct a vLLM `TokensPrompt` (with dict fallback for lightweight mocks)."""
    ids_list = [int(t) for t in token_ids]
    try:
        from vllm.inputs import TokensPrompt

        return TokensPrompt(prompt_token_ids=ids_list)
    except Exception:
        return {"prompt_token_ids": ids_list}


def create_vllm_engine(
    ckpt_dir: Union[str, Path],
    dtype: str = "float16",
    max_model_len: int = 10240,
    gpu_memory_utilization: float = 0.85,
    seed: int = 0,
    enforce_eager: bool = False,
    llm_cls: Optional[Any] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Instantiate a vLLM `LLM` engine with logged compatibility fallbacks.

    1. Tries `language_model_only=True`. If rejected by that vLLM version (`TypeError`),
       falls back to `limit_mm_per_prompt={"image": 0, "video": 0}` and logs the fallback.
    2. If `torch.compile` / Inductor fails while `enforce_eager=False`, retries with
       `enforce_eager=True` and logs the fallback.
    """
    bin_dir = str(Path(sys.executable).resolve().parent)
    path_parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    if Path("/usr/local/cuda/bin").is_dir() and "/usr/local/cuda/bin" not in path_parts:
        path_parts.insert(0, "/usr/local/cuda/bin")
    if bin_dir and bin_dir not in path_parts:
        path_parts.insert(0, bin_dir)
    os.environ["PATH"] = os.pathsep.join(path_parts)
    if not os.environ.get("CUDA_HOME") and Path("/usr/local/cuda").is_dir():
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
    if not os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") and shutil.which("nvcc") is None:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    if llm_cls is None:
        from vllm import LLM as _VllmLLM

        llm_cls = _VllmLLM

    try:
        import vllm as _vllm_mod

        vllm_version = getattr(_vllm_mod, "__version__", "unknown")
    except Exception:
        vllm_version = "unknown"

    fallbacks_triggered: List[str] = []
    engine_kwargs: Dict[str, Any] = {
        "model": str(ckpt_dir),
        "dtype": str(dtype),
        "language_model_only": True,
        "max_model_len": int(max_model_len),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "seed": int(seed),
        "enforce_eager": bool(enforce_eager),
        "mamba_ssm_cache_dtype": "float32",
    }

    def _try_build(kwargs_in: Dict[str, Any]) -> Any:
        try:
            return llm_cls(**kwargs_in)
        except TypeError as exc:
            err_msg = str(exc)
            if "language_model_only" in err_msg:
                msg = (
                    "vLLM rejected `language_model_only=True`; falling back to "
                    "`limit_mm_per_prompt={'image': 0, 'video': 0}`"
                )
                print(f"[vLLM Fallback] {msg} ({exc})", flush=True)
                fallbacks_triggered.append(msg)
                kwargs_in.pop("language_model_only", None)
                kwargs_in["limit_mm_per_prompt"] = {"image": 0, "video": 0}
                try:
                    return llm_cls(**kwargs_in)
                except TypeError as exc2:
                    if "limit_mm_per_prompt" in str(exc2):
                        msg2 = "vLLM also rejected `limit_mm_per_prompt`; omitting multimodal kwargs"
                        print(f"[vLLM Fallback] {msg2} ({exc2})", flush=True)
                        fallbacks_triggered.append(msg2)
                        kwargs_in.pop("limit_mm_per_prompt", None)
                        return llm_cls(**kwargs_in)
                    raise
            raise

    try:
        llm = _try_build(engine_kwargs)
    except Exception as exc:
        if not engine_kwargs["enforce_eager"]:
            msg = f"vLLM initialization failed with enforce_eager=False ({exc}); retrying with enforce_eager=True"
            print(f"[vLLM Fallback] {msg}", flush=True)
            fallbacks_triggered.append(msg)
            engine_kwargs["enforce_eager"] = True
            llm = _try_build(engine_kwargs)
        else:
            raise

    meta = {
        "vllm_version": vllm_version,
        "model": str(ckpt_dir),
        "dtype": str(dtype),
        "max_model_len": int(max_model_len),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "seed": int(seed),
        "enforce_eager": bool(engine_kwargs["enforce_eager"]),
        "mamba_ssm_cache_dtype": "float32",
        "language_model_only": not any("language_model_only" in fb for fb in fallbacks_triggered),
        "limit_mm_per_prompt": (
            {"image": 0, "video": 0}
            if any("limit_mm_per_prompt" in fb for fb in fallbacks_triggered)
            and not any("omitting multimodal kwargs" in fb for fb in fallbacks_triggered)
            else None
        ),
        "fallbacks_triggered": fallbacks_triggered,
    }
    return llm, meta


def release_vllm_engine(llm: Any) -> None:
    """Tear down a vLLM engine and reclaim GPU memory."""
    if llm is not None:
        try:
            eng = getattr(llm, "llm_engine", None)
            executor = getattr(eng, "model_executor", None)
            if executor is not None and hasattr(executor, "shutdown"):
                executor.shutdown()
        except Exception:
            pass
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
