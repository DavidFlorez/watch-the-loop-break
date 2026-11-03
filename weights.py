"""Weight verification, hardware auto-detection, and weight-space interpolation `blend(alpha)`.

Reproduces and extends Experiment 8 of "Sparks of Weight-Space Programming"
(https://rikiyat.github.io/blog/weight-space-programming/) on:
- Base model: `Qwen/Qwen3.5-0.8B`
- Edited model: `RikiyaT/Qwen3.5-0.8B-weight-space-programming`
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B"
EDIT_MODEL_ID = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"

TARGET_TENSORS: Tuple[str, str] = (
    "model.language_model.layers.15.mlp.down_proj.weight",
    "model.language_model.layers.19.mlp.down_proj.weight",
)

EXPECTED_SHA256 = {
    "model.language_model.layers.15.mlp.down_proj.weight": {
        "base": "ef60d2c687bab19464dca9c8bfd7777e46ad226df0c9e25643b0bab04b5a1cca",
        "edit": "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53",
    },
    "model.language_model.layers.19.mlp.down_proj.weight": {
        "base": "638eedeb701b94aa861f4438e615c07ec6c60ee703db1bfee2de577bb6498c3e",
        "edit": "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2",
    },
}


def detect_hardware(preferred: Optional[str] = None) -> Tuple[str, torch.dtype, str]:
    """Auto-detect hardware backend (CUDA -> MPS -> CPU) and appropriate dtype.

    Returns:
        (device_str, torch_dtype, human_readable_description)
    """
    if preferred in ("cuda", "mps", "cpu"):
        device = preferred
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        desc = f"CUDA ({name}, {vram_gb:.1f} GB VRAM, dtype={dtype})"
    elif device == "mps":
        dtype = torch.bfloat16
        desc = f"Apple Silicon MPS (dtype={dtype})"
    else:
        dtype = torch.float32
        desc = f"CPU (dtype={dtype})"

    return device, dtype, desc


def resolve_safetensors_files(repo_id: str) -> Dict[str, str]:
    """Return mapping of `{tensor_name: local_safetensors_filepath}` for a HF repo."""
    try:
        idx_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
        with open(idx_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map: Dict[str, str] = index_data["weight_map"]
        shard_files = sorted(set(weight_map.values()))
        local_shards = {
            shard: hf_hub_download(repo_id=repo_id, filename=shard)
            for shard in shard_files
        }
        return {tensor_name: local_shards[shard] for tensor_name, shard in weight_map.items()}
    except Exception:
        single_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        with safe_open(single_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
        return {k: single_path for k in keys}


def bf16_sha256(tensor: torch.Tensor) -> str:
    """Compute the SHA-256 hex digest of raw bfloat16 bytes of a tensor."""
    t_bf16 = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    raw_bytes = t_bf16.view(torch.uint16).numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def verify_weight_diff(
    base_repo: str = BASE_MODEL_ID,
    edit_repo: str = EDIT_MODEL_ID,
) -> Dict[str, Any]:
    """Compare all tensors between `Qwen/Qwen3.5-0.8B` and `RikiyaT/Qwen3.5-0.8B-weight-space-programming`.

    Confirms that ONLY `model.language_model.layers.15.mlp.down_proj.weight` and
    `model.language_model.layers.19.mlp.down_proj.weight` differ, verifies their
    SHA-256 digests, and characterizes the rank-1 64-column structure of the edit.
    """
    base_map = resolve_safetensors_files(base_repo)
    edit_map = resolve_safetensors_files(edit_repo)

    base_keys = set(base_map.keys())
    edit_keys = set(edit_map.keys())
    if base_keys != edit_keys:
        only_base = sorted(base_keys - edit_keys)
        only_edit = sorted(edit_keys - base_keys)
        raise AssertionError(f"Tensor key mismatch! Only in base: {only_base}, only in edit: {only_edit}")

    differing_tensors = []
    total_params = 0
    total_changed_bf16_entries = 0
    tensor_reports: Dict[str, Any] = {}

    # Group keys by (base_file, edit_file) so each shard is opened once
    open_base: Dict[str, Any] = {}
    open_edit: Dict[str, Any] = {}
    try:
        for path in set(base_map.values()):
            open_base[path] = safe_open(path, framework="pt", device="cpu")
        for path in set(edit_map.values()):
            open_edit[path] = safe_open(path, framework="pt", device="cpu")

        for key in sorted(base_keys):
            t_base = open_base[base_map[key]].get_tensor(key)
            t_edit = open_edit[edit_map[key]].get_tensor(key)
            assert t_base.shape == t_edit.shape, f"Shape mismatch on {key}: {t_base.shape} vs {t_edit.shape}"
            total_params += t_base.numel()

            # Exact bit-level comparison in uint16/bfloat16 view
            b_bits = t_base.to(torch.bfloat16).view(torch.uint16)
            e_bits = t_edit.to(torch.bfloat16).view(torch.uint16)
            diff_mask = b_bits != e_bits
            n_changed = int(diff_mask.sum().item())

            if n_changed > 0:
                differing_tensors.append(key)
                total_changed_bf16_entries += n_changed

                w_base_f32 = t_base.to(torch.float32)
                w_edit_f32 = t_edit.to(torch.float32)
                delta = w_edit_f32 - w_base_f32

                col_norms = torch.linalg.norm(delta, dim=0)
                changed_col_mask = (diff_mask.any(dim=0))
                changed_col_indices = torch.where(changed_col_mask)[0].tolist()
                n_changed_cols = len(changed_col_indices)
                n_changed_rows = int(diff_mask.any(dim=1).sum().item())

                # SVD on the active columns to verify rank-1 directional projection
                delta_active = delta[:, changed_col_mask]
                u, s, vh = torch.linalg.svd(delta_active, full_matrices=False)
                top_sv = float(s[0].item())
                second_sv = float(s[1].item()) if s.numel() > 1 else 0.0
                rank1_energy_frac = float((s[0] ** 2 / (s**2).sum()).item())

                base_hash = bf16_sha256(t_base)
                edit_hash = bf16_sha256(t_edit)
                expected = EXPECTED_SHA256.get(key, {})

                rel_frob = float((torch.linalg.norm(delta) / torch.linalg.norm(w_base_f32)).item())
                max_abs_diff = float(delta.abs().max().item())

                tensor_reports[key] = {
                    "shape": list(t_base.shape),
                    "dtype": str(t_base.dtype).replace("torch.", ""),
                    "numel": int(t_base.numel()),
                    "changed_bf16_entries": n_changed,
                    "touched_columns_entries": n_changed_cols * int(t_base.shape[0]),
                    "changed_columns": n_changed_cols,
                    "changed_rows": n_changed_rows,
                    "changed_column_indices": changed_col_indices,
                    "relative_frobenius_change": rel_frob,
                    "max_abs_diff": max_abs_diff,
                    "svd_top5": [float(x) for x in s[:5].tolist()],
                    "sv1_to_sv2_ratio": top_sv / max(second_sv, 1e-12),
                    "rank1_energy_fraction": rank1_energy_frac,
                    "base_sha256": base_hash,
                    "edit_sha256": edit_hash,
                    "base_sha256_matches_manifest": base_hash == expected.get("base"),
                    "edit_sha256_matches_manifest": edit_hash == expected.get("edit"),
                    "col_norms": col_norms.numpy(),
                    "u_principal": u[:, 0].numpy(),
                }
    finally:
        # safe_open contexts clean up on GC
        open_base.clear()
        open_edit.clear()

    assert tuple(differing_tensors) == TARGET_TENSORS, (
        f"Expected only {TARGET_TENSORS} to differ, but found: {differing_tensors}"
    )

    # Check cosine similarity between Layer 15 and Layer 19 repetition directions
    u15 = tensor_reports[TARGET_TENSORS[0]]["u_principal"]
    u19 = tensor_reports[TARGET_TENSORS[1]]["u_principal"]
    u15_u19_cos = float(np.dot(u15, u19) / (np.linalg.norm(u15) * np.linalg.norm(u19)))

    summary = {
        "base_repo": base_repo,
        "edit_repo": edit_repo,
        "total_tensors_checked": len(base_keys),
        "total_parameters": total_params,
        "differing_tensor_names": differing_tensors,
        "only_target_tensors_differ": tuple(differing_tensors) == TARGET_TENSORS,
        "total_touched_entries": sum(
            tensor_reports[k]["touched_columns_entries"] for k in differing_tensors
        ),
        "total_changed_bf16_entries": total_changed_bf16_entries,
        "fraction_of_model_touched": sum(
            tensor_reports[k]["touched_columns_entries"] for k in differing_tensors
        )
        / total_params,
        "u15_u19_cosine_similarity": u15_u19_cos,
        "tensors": {
            k: {
                kk: vv
                for kk, vv in v.items()
                if kk not in ("col_norms", "u_principal")
            }
            for k, v in tensor_reports.items()
        },
        "_raw": tensor_reports,
    }
    return summary


def load_target_tensors(
    base_repo: str = BASE_MODEL_ID,
    edit_repo: str = EDIT_MODEL_ID,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Load only the two target MLP `down_proj.weight` tensors from base and edited checkpoints."""
    base_map = resolve_safetensors_files(base_repo)
    edit_map = resolve_safetensors_files(edit_repo)
    base_tensors: Dict[str, torch.Tensor] = {}
    edit_tensors: Dict[str, torch.Tensor] = {}

    for name in TARGET_TENSORS:
        with safe_open(base_map[name], framework="pt", device="cpu") as fb:
            base_tensors[name] = fb.get_tensor(name).clone()
        with safe_open(edit_map[name], framework="pt", device="cpu") as fe:
            edit_tensors[name] = fe.get_tensor(name).clone()

    return base_tensors, edit_tensors


# Module-level cache for standalone `blend(alpha)` usage
_CACHED_BASE_TENSORS: Dict[str, torch.Tensor] = {}
_CACHED_EDIT_TENSORS: Dict[str, torch.Tensor] = {}
_ACTIVE_MODEL: Optional[Any] = None


def register_active_model(
    model: Any,
    base_tensors: Optional[Dict[str, torch.Tensor]] = None,
    edit_tensors: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Register a loaded Hugging Face model so `blend(alpha)` can update it in-place."""
    global _ACTIVE_MODEL, _CACHED_BASE_TENSORS, _CACHED_EDIT_TENSORS
    _ACTIVE_MODEL = model
    if base_tensors is not None:
        _CACHED_BASE_TENSORS = base_tensors
    if edit_tensors is not None:
        _CACHED_EDIT_TENSORS = edit_tensors


def _resolve_parameter_by_name(model: torch.nn.Module, name: str) -> torch.nn.Parameter:
    """Locate parameter `name` inside `model`, handling both multimodal and text-only prefixes."""
    params = dict(model.named_parameters())
    if name in params:
        return params[name]
    # Fallback if loaded via text-only AutoModelForCausalLM where `model.language_model.` is `model.`
    alt_name = name.replace("model.language_model.", "model.")
    if alt_name in params:
        return params[alt_name]
    raise KeyError(f"Could not find tensor '{name}' (or '{alt_name}') in model parameters.")


@torch.no_grad()
def blend(
    alpha: float,
    model: Optional[torch.nn.Module] = None,
    base_tensors: Optional[Dict[str, torch.Tensor]] = None,
    edit_tensors: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Set `layers.15.mlp.down_proj.weight` and `layers.19.mlp.down_proj.weight` to:

        W(alpha) = W_base + alpha * (W_edit - W_base)

    Arithmetic is performed in float32 and cast back to the target parameter dtype
    (bfloat16 on CUDA/MPS or float32 on CPU). If `model` (or a registered active model)
    is provided, its parameters are updated in-place and returned.

    Args:
        alpha: Interpolation coefficient (0.0 = base model, 1.0 = edited model, >1.0 = extrapolated).
        model: Optional PyTorch model whose parameters will be updated in-place.
        base_tensors: Optional pre-loaded dict of base target tensors.
        edit_tensors: Optional pre-loaded dict of edited target tensors.

    Returns:
        Dictionary mapping the two target tensor names to their blended tensors.
    """
    global _CACHED_BASE_TENSORS, _CACHED_EDIT_TENSORS
    if base_tensors is None or edit_tensors is None:
        if not _CACHED_BASE_TENSORS or not _CACHED_EDIT_TENSORS:
            _CACHED_BASE_TENSORS, _CACHED_EDIT_TENSORS = load_target_tensors()
        base_tensors = _CACHED_BASE_TENSORS
        edit_tensors = _CACHED_EDIT_TENSORS

    target_model = model if model is not None else _ACTIVE_MODEL
    blended_out: Dict[str, torch.Tensor] = {}
    alpha_f = float(alpha)

    for name in TARGET_TENSORS:
        w_base = base_tensors[name].to(torch.float32)
        w_edit = edit_tensors[name].to(torch.float32)
        w_blend_f32 = w_base + alpha_f * (w_edit - w_base)
        # Convert through bfloat16 first so alpha=0.0 and alpha=1.0 match the exact bfloat16 checkpoints
        w_blend_bf16 = w_blend_f32.to(torch.bfloat16)

        if target_model is not None:
            param = _resolve_parameter_by_name(target_model, name)
            param.copy_(w_blend_bf16.to(device=param.device, dtype=param.dtype))
            blended_out[name] = param.detach()
        else:
            blended_out[name] = w_blend_bf16

    return blended_out


def save_weight_diff_chart(summary: Dict[str, Any], out_path: str = "assets/weight_diff_analysis.png") -> str:
    """Save a diagnostic chart of the weight-space edit in Layers 15 and 19."""
    import matplotlib.pyplot as plt

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    raw = summary["_raw"]
    t15 = raw[TARGET_TENSORS[0]]
    t19 = raw[TARGET_TENSORS[1]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: Layer 15 column L2 norms
    ax = axes[0]
    cols = np.arange(len(t15["col_norms"]))
    norms15 = t15["col_norms"]
    ax.bar(cols, norms15, width=6.0, color="#2563eb", alpha=0.85, label="Edited columns (64 / 3584)")
    ax.set_title("Layer 15 MLP down_proj: ||ΔW[:, j]||₂", fontsize=11, fontweight="bold")
    ax.set_xlabel("MLP Intermediate Column Index j (0..3583)")
    ax.set_ylabel("L2 Norm of Column Change (bf16)")
    ax.set_xlim(0, 3584)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.text(
        0.04,
        0.92,
        f"Changed cols: {t15['changed_columns']}/3584\n"
        f"Changed entries: {t15['changed_bf16_entries']:,}\n"
        f"||ΔW||_F / ||W||_F = {t15['relative_frobenius_change']*100:.2f}%",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eff6ff", "edgecolor": "#bfdbfe"},
    )

    # Panel 2: Layer 19 column L2 norms
    ax = axes[1]
    norms19 = t19["col_norms"]
    ax.bar(cols, norms19, width=6.0, color="#7c3aed", alpha=0.85, label="Edited columns (64 / 3584)")
    ax.set_title("Layer 19 MLP down_proj: ||ΔW[:, j]||₂", fontsize=11, fontweight="bold")
    ax.set_xlabel("MLP Intermediate Column Index j (0..3583)")
    ax.set_ylabel("L2 Norm of Column Change (bf16)")
    ax.set_xlim(0, 3584)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.text(
        0.04,
        0.92,
        f"Changed cols: {t19['changed_columns']}/3584\n"
        f"Changed entries: {t19['changed_bf16_entries']:,}\n"
        f"||ΔW||_F / ||W||_F = {t19['relative_frobenius_change']*100:.2f}%",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f5f3ff", "edgecolor": "#ddd6fe"},
    )

    # Panel 3: Singular value spectrum of ΔW (proving rank-1 projection)
    ax = axes[2]
    idx = np.arange(1, 6)
    sv15 = np.array(t15["svd_top5"])
    sv19 = np.array(t19["svd_top5"])
    width = 0.35
    ax.bar(idx - width / 2, sv15, width=width, color="#2563eb", label="Layer 15 ΔW")
    ax.bar(idx + width / 2, sv19, width=width, color="#7c3aed", label="Layer 19 ΔW")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"σ_{i}" for i in idx])
    ax.set_title("Singular Values of ΔW (Rank-1 Loop Direction)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Singular Value σ_i (log scale)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.text(
        0.04,
        0.58,
        f"L15 rank-1 energy: {t15['rank1_energy_fraction']*100:.4f}%\n"
        f"L19 rank-1 energy: {t19['rank1_energy_fraction']*100:.4f}%\n"
        f"cos(u₁₅, u₁₉) = {summary['u15_u19_cosine_similarity']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )

    fig.suptitle(
        "Weight-Space Programming Diff: Qwen/Qwen3.5-0.8B vs RikiyaT/Qwen3.5-0.8B-weight-space-programming",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    """CLI entrypoint to verify weight differences, test `blend(alpha)`, and save report + chart."""
    device, dtype, hw_desc = detect_hardware()
    print(f"[Hardware] {hw_desc}")
    print(f"[Verify] Comparing {BASE_MODEL_ID} vs {EDIT_MODEL_ID} ...")
    summary = verify_weight_diff()

    # Verify blend(0.0) and blend(1.0) reproduce exact SHA-256 digests
    base_t, edit_t = load_target_tensors()
    b0 = blend(0.0, base_tensors=base_t, edit_tensors=edit_t)
    b1 = blend(1.0, base_tensors=base_t, edit_tensors=edit_t)
    for name in TARGET_TENSORS:
        assert bf16_sha256(b0[name]) == EXPECTED_SHA256[name]["base"], f"blend(0.0) hash mismatch on {name}"
        assert bf16_sha256(b1[name]) == EXPECTED_SHA256[name]["edit"], f"blend(1.0) hash mismatch on {name}"

    chart_path = save_weight_diff_chart(summary)
    report_out = {k: v for k, v in summary.items() if k != "_raw"}
    report_out["blend_0_and_1_sha256_verified"] = True
    report_out["hardware_detected"] = {"device": device, "dtype": str(dtype), "description": hw_desc}

    os.makedirs("assets", exist_ok=True)
    with open("assets/weight_verification.json", "w", encoding="utf-8") as f:
        json.dump(report_out, f, indent=2)

    print(json.dumps(report_out, indent=2))
    print(f"[Chart] Saved weight diff visualization to {chart_path}")


if __name__ == "__main__":
    main()
