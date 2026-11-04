"""Build a blended Qwen3.5-0.8B safetensors checkpoint directory in scratch for vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from gpu.common import BASE_MODEL_REVISION, SHARD_FILENAME
from weights import (
    BASE_MODEL_ID,
    EDIT_MODEL_ID,
    EXPECTED_SHA256,
    TARGET_TENSORS,
    bf16_sha256,
    blend,
    load_target_tensors,
)


def _sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_checkpoint(
    alpha: float,
    out_dir: Union[str, Path],
    base_repo: str = BASE_MODEL_ID,
    edit_repo: str = EDIT_MODEL_ID,
    revision: str = BASE_MODEL_REVISION,
    symlink_non_weights: bool = False,
) -> Dict[str, Any]:
    """Create a complete Hugging Face checkpoint at `out_dir` with `W(alpha)` on the two target tensors.

    Downloads `base_repo` at `revision`, copies/symlinks every non-shard file (keeping `config.json`
    unchanged with `Qwen3_5ForConditionalGeneration`), replaces ONLY `TARGET_TENSORS` in
    `model.safetensors-00001-of-00001.safetensors` with `W_base + alpha * (W_edit - W_base)` computed in
    `float32` and cast to `bfloat16`, and saves with `safetensors.torch.save_file(..., metadata={"format": "pt"})`.
    """
    alpha_f = float(alpha)
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    snapshot_dir = Path(
        snapshot_download(
            repo_id=base_repo,
            revision=revision,
        )
    ).resolve()

    src_shard = snapshot_dir / SHARD_FILENAME
    if not src_shard.is_file():
        raise FileNotFoundError(f"Expected weight shard not found in snapshot: {src_shard}")

    # 1. Copy or symlink every file except `model.safetensors-00001-of-00001.safetensors`
    copied_files: list[str] = []
    for src_item in sorted(snapshot_dir.iterdir()):
        rel_name = src_item.name
        if rel_name == SHARD_FILENAME or rel_name.startswith("."):
            continue
        dst_item = out_path / rel_name
        if dst_item.exists() or dst_item.is_symlink():
            if dst_item.is_dir() and not dst_item.is_symlink():
                shutil.rmtree(dst_item)
            else:
                dst_item.unlink()

        if src_item.is_dir():
            shutil.copytree(src_item, dst_item)
        elif symlink_non_weights and rel_name != "config.json":
            os.symlink(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)
        copied_files.append(rel_name)

    # Verify config.json is completely unchanged and preserves Qwen3_5ForConditionalGeneration
    src_cfg_path = snapshot_dir / "config.json"
    dst_cfg_path = out_path / "config.json"
    if not dst_cfg_path.is_file():
        raise FileNotFoundError(f"config.json missing from checkpoint output: {dst_cfg_path}")
    src_cfg_hash = _sha256_file(src_cfg_path)
    dst_cfg_hash = _sha256_file(dst_cfg_path)
    assert src_cfg_hash == dst_cfg_hash, "config.json digest mismatch between snapshot and output!"
    with open(dst_cfg_path, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    architectures = cfg_data.get("architectures", [])
    assert architectures == ["Qwen3_5ForConditionalGeneration"], (
        f"Expected architectures=['Qwen3_5ForConditionalGeneration'], got {architectures}"
    )

    # 2. Load the single weight shard and replace ONLY `TARGET_TENSORS` with W(alpha) in bfloat16
    state_dict = load_file(str(src_shard), device="cpu")
    orig_keys = set(state_dict.keys())
    for t_name in TARGET_TENSORS:
        if t_name not in orig_keys:
            raise KeyError(f"Target tensor '{t_name}' missing from shard {src_shard}")

    base_tensors, edit_tensors = load_target_tensors(base_repo=base_repo, edit_repo=edit_repo)
    blended = blend(alpha_f, base_tensors=base_tensors, edit_tensors=edit_tensors)

    target_hashes: Dict[str, str] = {}
    for t_name in TARGET_TENSORS:
        w_bf16 = blended[t_name].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        t_hash = bf16_sha256(w_bf16)
        target_hashes[t_name] = t_hash

        if abs(alpha_f - 0.0) < 1e-12:
            expected_base = EXPECTED_SHA256[t_name]["base"]
            assert t_hash == expected_base, (
                f"alpha=0.0 SHA-256 mismatch on {t_name}: got {t_hash}, expected {expected_base}"
            )
        elif abs(alpha_f - 1.0) < 1e-12:
            expected_edit = EXPECTED_SHA256[t_name]["edit"]
            assert t_hash == expected_edit, (
                f"alpha=1.0 SHA-256 mismatch on {t_name}: got {t_hash}, expected {expected_edit}"
            )

        state_dict[t_name] = w_bf16

    assert set(state_dict.keys()) == orig_keys, "Shard tensor key set mutated unexpectedly!"

    # 3. Save safetensors shard atomically with format="pt" metadata
    dst_shard = out_path / SHARD_FILENAME
    tmp_shard = out_path / f".{SHARD_FILENAME}.{os.getpid()}.tmp"
    try:
        save_file(state_dict, str(tmp_shard), metadata={"format": "pt"})
        os.replace(tmp_shard, dst_shard)
    finally:
        if tmp_shard.exists():
            tmp_shard.unlink(missing_ok=True)

    return {
        "alpha": alpha_f,
        "out_dir": str(out_path),
        "base_repo": base_repo,
        "edit_repo": edit_repo,
        "revision": revision,
        "config_sha256": dst_cfg_hash,
        "architectures": architectures,
        "total_tensors": len(state_dict),
        "target_tensors_sha256": target_hashes,
        "non_shard_files_count": len(copied_files),
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a blended Qwen3.5-0.8B safetensors checkpoint in scratch for vLLM."
    )
    parser.add_argument("--alpha", type=float, required=True, help="Interpolation coefficient alpha.")
    parser.add_argument("--out", type=str, required=True, help="Scratch output directory for the checkpoint.")
    parser.add_argument("--base-repo", type=str, default=BASE_MODEL_ID, help="Base Hugging Face repo ID.")
    parser.add_argument("--edit-repo", type=str, default=EDIT_MODEL_ID, help="Edited Hugging Face repo ID.")
    parser.add_argument(
        "--revision",
        type=str,
        default=BASE_MODEL_REVISION,
        help="Pinned git revision of base model.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink large non-weight assets instead of copying (config.json is always copied & verified).",
    )
    args = parser.parse_args(argv)

    summary = build_checkpoint(
        alpha=args.alpha,
        out_dir=args.out,
        base_repo=args.base_repo,
        edit_repo=args.edit_repo,
        revision=args.revision,
        symlink_non_weights=args.symlink,
    )
    print(f"[make_ckpt] Built checkpoint for alpha={args.alpha:g} -> {summary['out_dir']}", flush=True)
    for name, h in summary["target_tensors_sha256"].items():
        print(f"  - {name}: sha256={h[:16]}...", flush=True)


if __name__ == "__main__":
    main()
