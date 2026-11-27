#!/usr/bin/env python3
"""Convert Google's official `google/embeddinggemma-2` text encoder into the
single-function (seq_len=512), INT8-quantized Core ML package used by
`CoreMLEmbeddingModel.swift`.

Provenance (both Apache-2.0):
  - Weights: https://huggingface.co/google/embeddinggemma-2
             pinned revision: f9972ac22e696a105c482202646979e33e4b5a1e
  - ANE text-tower rewrite (`egcm`):
             https://github.com/Delavalom/embeddinggemma-coreml
             pinned commit: a53192ec293ac9138cda9d0c84e7d8267bb60bf1

Requirements (Apple Silicon Mac with Xcode command-line tools + `uv`, e.g. `brew install uv`):
  git clone https://github.com/Delavalom/embeddinggemma-coreml.git /tmp/egcm
  cd /tmp/egcm && git checkout a53192ec293ac9138cda9d0c84e7d8267bb60bf1
  uv sync --group convert

Usage (~1-2 minutes on Apple Silicon):
  uv run --project /tmp/egcm --no-sync python scripts/convert-embedding-model.py \
      --egcm-dir /tmp/egcm \
      --out-dir ./embeddinggemma-2-coreml-int8 \
      --release-dir ./release-assets

Outputs:
  1) `<out-dir>/text/` — on-device directory layout expected by CoreMLEmbeddingModel:
       - tokenizer.json
       - embeddings.int8             (262144 x 512 symmetric per-row INT8 token table)
       - embeddings.int8.scale.f16   (262144 Float16 per-row scales)
       - EmbeddingGemma2Text.mlmodelc/
  2) `<release-dir>/` (optional) — flattened GitHub Release assets + SHA256SUMS.txt
     matching `CoreMLEmbeddingModel.textManifest`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HF_REPO = "google/embeddinggemma-2"
HF_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
HF_FILES = ("config.json", "tokenizer.json", "model.safetensors")
SEQ_LEN = 512

# Exact mapping from on-device relative path -> flat GitHub Release asset name
RELEASE_FILES = [
    ("config.json", "config.json"),
    ("text/EmbeddingGemma2Text.mlmodelc/analytics/coremldata.bin", "analytics-coremldata.bin"),
    ("text/EmbeddingGemma2Text.mlmodelc/coremldata.bin", "coremldata.bin"),
    ("text/EmbeddingGemma2Text.mlmodelc/metadata.json", "metadata.json"),
    ("text/EmbeddingGemma2Text.mlmodelc/model.mil", "model.mil"),
    ("text/EmbeddingGemma2Text.mlmodelc/weights/weight.bin", "weight.bin"),
    ("text/embeddings.int8", "embeddings.int8"),
    ("text/embeddings.int8.scale.f16", "embeddings.int8.scale.f16"),
    ("text/tokenizer.json", "tokenizer.json"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_google_weights(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for fname in HF_FILES:
        target = dest / fname
        if target.exists() and target.stat().st_size > 0:
            print(f"  using cached {target}", flush=True)
            continue
        url = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{fname}"
        print(f"  downloading {url} ...", flush=True)
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(target)


def make_verification_sequences() -> list[tuple[str, "np.ndarray"]]:
    """24 deterministic token sequences spanning short queries and long passages."""
    import numpy as np

    rng = np.random.default_rng(20261125)
    lengths = [8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224,
               256, 280, 310, 340, 370, 400, 420, 450, 475, 490, 505, 512]
    cases = []
    for idx, length in enumerate(lengths):
        mid = rng.integers(500, 50_000, size=length - 2, dtype=np.int32)
        ids = np.concatenate([[2], mid, [1]]).astype(np.int32)  # <bos> ... <eos>
        cases.append((f"seq_{idx:02d}_L{length}", ids))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--egcm-dir", type=Path, required=True,
                    help="Path to checkout of https://github.com/Delavalom/embeddinggemma-coreml")
    ap.add_argument("--weights-dir", type=Path, default=Path("./google-eg2-raw"),
                    help="Directory for downloaded google/embeddinggemma-2 weights")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory for compiled Core ML model + INT8 token table")
    ap.add_argument("--release-dir", type=Path, default=None,
                    help="Optional directory to write flattened release assets + SHA256SUMS.txt")
    ap.add_argument("--skip-verify", action="store_true",
                    help="Skip ANE cosine-similarity parity verification against FP32 PyTorch")
    args = ap.parse_args()

    sys.path.insert(0, str(args.egcm_dir.resolve()))

    import coremltools as ct
    import coremltools.optimize.coreml as cto
    import numpy as np
    import torch
    from egcm.config import load as load_cfg
    from egcm.model.embedder import _RopeOwner, l2_normalize
    from egcm.model.masks import key_mask
    from egcm.model.text import TextTower
    from egcm.weights import TABLE_KEY, build_text_tower, load_hf

    class TextEmbedder512(_RopeOwner):
        """Exact I/O contract for CoreMLEmbeddingModel.swift:
          inputs_embeds:  fp16 [1, 512, 512] (pre-scaled by sqrt(512))
          attention_mask: fp16 [1, 512]      (1.0 valid, 0.0 padding)
          -> embedding:   fp16 [1, 768]      (mean-pooled + L2-normalized)
        """

        def __init__(self, tower: TextTower, seq_len: int = SEQ_LEN):
            super().__init__()
            self.tower = tower
            self.seq_len = seq_len
            self._register_rope(tower, seq_len)

        def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            emb = inputs_embeds.transpose(1, 2).unsqueeze(2)  # (1, 512, 1, 512)
            tokens = self.tower(emb, key_mask(attention_mask), self._rope())  # (1, 768, 1, 512)
            valid = attention_mask.view(1, 1, 1, -1)
            weights = valid / valid.sum(-1, keepdim=True)
            pooled = (tokens * weights).sum(-1, keepdim=True)
            return l2_normalize(pooled.reshape(1, -1))

    t0 = time.monotonic()
    print(f"1) Ensuring official Google weights ({HF_REPO} @ {HF_REVISION[:12]})...", flush=True)
    download_google_weights(args.weights_dir)

    tcfg = load_cfg(args.weights_dir)[0]
    sd = load_hf(args.weights_dir)
    tower = build_text_tower(tcfg, sd)
    raw_table = sd[TABLE_KEY].float()  # (262144, 512)
    embed_scale = math.sqrt(tcfg.hidden)
    print(f"   Loaded TextTower (layers={tcfg.layers}, hidden={tcfg.hidden}, out={tcfg.out_dim}) "
          f"and token table {tuple(raw_table.shape)} in {time.monotonic() - t0:.1f}s", flush=True)

    print("2) Quantizing token embedding table symmetrically per row (INT8 + Float16 scales)...", flush=True)
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    text_dir = args.out_dir / "text"
    text_dir.mkdir(parents=True)

    t_np = raw_table.numpy()
    row_max = np.max(np.abs(t_np), axis=1)
    scales_f16 = np.where(row_max > 0, row_max / 127.0, 1.0).astype(np.float16)
    q_i8 = np.clip(np.rint(t_np / scales_f16.astype(np.float32)[:, None]), -127, 127).astype(np.int8)

    q_i8.tofile(text_dir / "embeddings.int8")
    scales_f16.tofile(text_dir / "embeddings.int8.scale.f16")
    shutil.copy2(args.weights_dir / "tokenizer.json", text_dir / "tokenizer.json")
    shutil.copy2(args.weights_dir / "config.json", args.out_dir / "config.json")

    model_pt = TextEmbedder512(tower, SEQ_LEN).eval()

    print("3) Tracing, converting to Core ML FP16 -> INT8, and compiling .mlmodelc...", flush=True)
    t_conv = time.monotonic()
    dummy_emb = torch.randn(1, SEQ_LEN, tcfg.hidden, dtype=torch.float32)
    dummy_mask = torch.ones(1, SEQ_LEN, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model_pt, (dummy_emb, dummy_mask))

    ml_fp16 = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="inputs_embeds", shape=(1, SEQ_LEN, tcfg.hidden), dtype=np.float16),
            ct.TensorType(name="attention_mask", shape=(1, SEQ_LEN), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="embedding", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    qcfg = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_channel",
            weight_threshold=2048,
        )
    )
    ml_int8 = cto.linear_quantize_weights(ml_fp16, config=qcfg)

    pkg_path = args.out_dir / "EmbeddingGemma2Text.mlpackage"
    ml_int8.save(str(pkg_path))
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(pkg_path), str(text_dir)],
        check=True,
    )
    shutil.rmtree(pkg_path)
    print(f"   Converted, quantized & compiled in {time.monotonic() - t_conv:.1f}s", flush=True)

    if not args.skip_verify:
        print("4) Verifying ANE output cosine similarity against Google FP32 PyTorch...", flush=True)
        cases = make_verification_sequences()
        mlmodelc_path = text_dir / "EmbeddingGemma2Text.mlmodelc"
        compiled = ct.models.CompiledMLModel(str(mlmodelc_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        cosines = []
        with torch.no_grad():
            for _, ids in cases:
                length = len(ids)
                emb_fp32 = torch.zeros(1, SEQ_LEN, tcfg.hidden, dtype=torch.float32)
                mask_fp32 = torch.zeros(1, SEQ_LEN, dtype=torch.float32)
                emb_fp32[0, :length, :] = raw_table[torch.from_numpy(ids)] * embed_scale
                mask_fp32[0, :length] = 1.0
                v_ref = model_pt(emb_fp32, mask_fp32).numpy().ravel().astype(np.float64)

                embeds_f16 = np.zeros((1, SEQ_LEN, tcfg.hidden), dtype=np.float16)
                mask_f16 = np.zeros((1, SEQ_LEN), dtype=np.float16)
                rows = q_i8[ids].astype(np.float32) * (scales_f16[ids].astype(np.float32)[:, None] * embed_scale)
                embeds_f16[0, :length, :] = rows.astype(np.float16)
                mask_f16[0, :length] = np.float16(1.0)
                v_ane = compiled.predict({"inputs_embeds": embeds_f16, "attention_mask": mask_f16})["embedding"].ravel().astype(np.float64)

                cos = float(np.dot(v_ane, v_ref) / (np.linalg.norm(v_ane) * np.linalg.norm(v_ref) + 1e-12))
                cosines.append(cos)
        print(f"   Parity over 24 sequences: mean_cosine={np.mean(cosines):.6f}, min_cosine={np.min(cosines):.6f}", flush=True)

    if args.release_dir is not None:
        print(f"5) Packaging flattened release assets into {args.release_dir}...", flush=True)
        if args.release_dir.exists():
            shutil.rmtree(args.release_dir)
        args.release_dir.mkdir(parents=True)
        sums = []
        manifest_entries = []
        for rel_src, flat_name in RELEASE_FILES:
            src = args.out_dir / rel_src
            dst = args.release_dir / flat_name
            shutil.copy2(src, dst)
            digest = sha256_file(dst)
            size = dst.stat().st_size
            sums.append(f"{digest}  {flat_name}")
            manifest_entries.append((rel_src, flat_name, size, digest))
        (args.release_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
        print("\nManifest entries for CoreMLEmbeddingModel.textManifest:")
        for rel_src, flat_name, size, digest in manifest_entries:
            print(
                f'        ManifestFile(\n'
                f'            relativePath: "{rel_src}",\n'
                f'            remotePath: "{flat_name}",\n'
                f'            sizeBytes: {size:_},\n'
                f'            sha256: "{digest}",\n'
                f'        ),'
            )

    print(f"\nDone in {time.monotonic() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
