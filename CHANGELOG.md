# Changelog

All notable changes to **ComfyUI-GGUF-qwen35** (the IKBlue fork) are recorded
here. This file covers the fork only; upstream ComfyUI-GGUF history lives at
[city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The baseline is upstream **2.0.0** (GitHub main), taken as-is.

## [Unreleased]

Nothing yet.

## [2.1.0] — 2026-09-22

First release of the fork. Adds Qwen3.5 text-encoder support, an optional mmproj
vision tower, and the V3 node schema.

### Added

- **Qwen3.5 (`qwen35`) text-encoder support** (`qwen35_support.py`, new).
  ComfyUI previously raised
  `ValueError: Unexpected text model architecture type in GGUF file: 'qwen35'`.
  The conversion remaps the llama.cpp layout onto ComfyUI's native qwen35
  checkpoint:
  - RMSNorm weights are un-centred in the GGUF, so `1.0` is subtracted
  - `ssm_a` holds `A`, so `A_log = log(-A)`
  - per-head SSM tensors (`A_log`, `dt_bias`, `in_proj_a`, `in_proj_b`) use an
    *evens-then-odds* channel grouping over the 32 value heads
  - `conv1d` uses 64 groups of 128 reordered `0..32, 34,36,..,62, 33,35,..,63`
  - llama.cpp key names → ComfyUI qwen35 module names
  - every tensor is dequantised on return, because a `GGMLTensor` reports the
    dequantised shape while keeping quantised storage

- **Optional mmproj vision tower** (`vision_from_mmproj`).
  qwen35 text-encoder GGUFs are frequently language-only, while ComfyUI's Qwen35
  unconditionally builds a vision tower and calls it for image inputs, so those
  files failed on image-edit workflows with
  `mat1 and mat2 shapes cannot be multiplied (3520x1152 and 3456x1152)`.
  The fork merges the matching llama.cpp `mmproj` GGUF. Handles the ggml traps
  that `tensor.data` has reversed axes and that BF16 surfaces as `uint8`, so
  mmproj linear weights arrive `(in, out)` and are transposed.

- **V3 node schema** (`nodes_v3.py`, new). All six nodes are re-expressed with
  `io.ComfyNode` / `io.Schema` / `comfy_entrypoint`. `node_id` and display names
  match the V1 ids, so existing workflows keep resolving.
  `UnetLoaderGGUFAdvanced` gains real typed inputs for `dequant_dtype`,
  `patch_dtype` and `patch_on_device` (V1 read them as undeclared keyword
  arguments, so they never appeared in the UI).

- `NOTICE`, `CHANGELOG.md`, and an expanded `README.md` documenting the
  conversion, the vision workflow and the verification status.

### Changed

- `loader.gguf_clip_loader()` accepts an optional `vision_path` and refuses to
  merge a vision file that shares any key with the text encoder.
- `CLIPLoaderGGUF` gained an optional `vision_name` input; it defaults to `none`,
  so text-only workflows are unaffected.
- `__init__.py` now exposes `comfy_entrypoint` and sets
  `NODE_CLASS_MAPPINGS = None` on the V3 path, falling back to the V1 mappings
  when `comfy_api.latest` is unavailable.
- Node category changed from upstream's `bootleg` to **`IKBlue`**.
- `pyproject.toml` repointed at this fork while keeping upstream credited.

### Fixed

- NotImplementedError message typo: `"Mixig scaled FP8"` → `"Mixing scaled FP8"`.

### Notes

- Verified against the Qwen-Image-2.1 **PE-T2I (9B)** weights only; the channel
  permutation tables assume 32 value heads, so 2B/4B/27B need their own checks.
- The fused `attn_qkv` of the 8 full-attention layers is intentionally **not**
  split (see the limitations section of the README) — the int8 reference could
  not pin the packing order to zero error, and guessing would degrade output
  silently.

## Upstream baseline

| Version | Note |
|---|---|
| 2.0.0 | upstream GitHub `main`, the starting point of this fork |

[Unreleased]: https://github.com/IKBlue/ComfyUI-GGUF-qwen35/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/IKBlue/ComfyUI-GGUF-qwen35/releases/tag/v2.1.0
