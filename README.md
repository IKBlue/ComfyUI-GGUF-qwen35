# ComfyUI-GGUF-qwen35

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

GGUF quantization support for ComfyUI, **with Qwen3.5 (`qwen35`) text-encoder
support and an optional mmproj vision tower**, migrated to the **V3 node schema**.

This is a fork of [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
by **IKBlue**. Upstream provides all GGUF loading, the GGML custom operations,
the dequantisation kernels and every non-qwen35 model — that work is City96's and
is reused here unchanged apart from the additions listed below. See
[NOTICE](NOTICE) for the full attribution, and [CHANGELOG.md](CHANGELOG.md) for
what this fork changed and when.

---

## Contents

- [What this fork adds](#what-this-fork-adds)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Load a diffusion model](#load-a-diffusion-model)
  - [Load a text encoder](#load-a-text-encoder)
  - [Qwen3.5 prompt enhancers and the vision tower](#qwen35-prompt-enhancers-and-the-vision-tower)
- [The qwen35 conversion, step by step](#the-qwen35-conversion-step-by-step)
- [Verification status](#verification-status)
- [Known limitations](#known-limitations)
- [Project layout](#project-layout)
- [Development](#development)
- [Credits and licence](#credits-and-licence)

---

## What this fork adds

Three things, on top of upstream:

1. **Qwen3.5 (`qwen35`) text-encoder support.** ComfyUI rejects these GGUFs with
   `ValueError: Unexpected text model architecture type in GGUF file: 'qwen35'`
   because the llama.cpp layout differs from ComfyUI's native qwen35 checkpoint.
   `qwen35_support.py` performs the remap.
2. **Optional mmproj vision tower.** Many qwen35 text-encoder GGUFs are exported
   language-only; ComfyUI's `Qwen35` always builds a vision tower and calls it for
   image inputs, so those files break on image-edit workflows. Supply the matching
   `mmproj` GGUF and the fork merges it in.
3. **V3 node schema.** All six nodes are available via `comfy_api.latest`
   (`io.ComfyNode` / `io.Schema` / `comfy_entrypoint`), with a V1 fallback for
   older ComfyUI builds.

| File | Origin | Change |
|---|---|---|
| `qwen35_support.py` | **new** (IKBlue) | qwen35 text-encoder remap + mmproj vision-tower converter |
| `nodes_v3.py` | **new** (IKBlue) | all six nodes in the V3 schema |
| `gguf_patcher.py` | City96 code, **extracted** (IKBlue) | `GGUFModelPatcher` + `*_gguf` folder registration, moved out of `nodes.py` so V1 and V3 share one copy |
| `__init__.py` | IKBlue | exports `comfy_entrypoint`, falls back to V1 |
| `loader.py` | City96 | +qwen35 branch, +optional `vision_path` |
| `nodes.py` | City96 | loading moved to free functions (shared with V3); +optional `vision_name` |
| `README.md`, `NOTICE`, `CHANGELOG.md`, `pyproject.toml` | IKBlue | fork metadata |

Everything else (`ops.py`, `dequant.py`, `tools/`) is upstream, unchanged.

---

## Requirements

- A recent ComfyUI. The **V3 schema** needs `comfy_api.latest`; without it the
  pack automatically falls back to the V1 nodes, so old builds keep working.
- `gguf>=0.13.0` (plus optional `sentencepiece`, `protobuf` for tokenizer
  recreation) — see `requirements.txt`.

## Installation

Clone into ComfyUI's custom nodes directory:

```bash
git clone https://github.com/IKBlue/ComfyUI-GGUF-qwen35 ComfyUI/custom_nodes/ComfyUI-GGUF
```

For a standalone ComfyUI release, from the folder containing `run_nvidia_gpu.bat`:

```bash
git clone https://github.com/IKBlue/ComfyUI-GGUF-qwen35 ComfyUI/custom_nodes/ComfyUI-GGUF
.\python_embeded\python.exe -s -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-GGUF\requirements.txt
```

The nodes appear under the **`IKBlue`** category (upstream uses `bootleg`, so the
two packs can coexist without mixing).

---

## Usage

The node ids are identical to upstream, so existing workflows keep working after
swapping the folder.

### Load a diffusion model

| Node | Output | Notes |
|---|---|---|
| `Unet Loader (GGUF)` | `MODEL` | pick a `.gguf` from `models/diffusion_models` or `models/unet` |
| `Unet Loader (GGUF/Advanced)` | `MODEL` | adds `dequant_dtype`, `patch_dtype`, `patch_on_device` |

Replace the stock *Load Diffusion Model* node with the GGUF loader. For full
precision output keep `dequant_dtype`/`patch_dtype` at `default`.

### Load a text encoder

| Node | Output |
|---|---|
| `CLIPLoader (GGUF)` | `CLIP` |
| `DualCLIPLoader (GGUF)` | `CLIP` |
| `TripleCLIPLoader (GGUF)` | `CLIP` |
| `QuadrupleCLIPLoader (GGUF)` | `CLIP` |

`type` accepts the same values as the built-in `CLIPLoader`
(`stable_diffusion`, `sd3`, `flux`, `qwen_image`, …). Picking the wrong one is the
most common cause of a shape mismatch at load time.

### Qwen3.5 prompt enhancers and the vision tower

If the text-encoder GGUF was exported **without** a vision tower, loading
succeeds but any image input fails with:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (3520x1152 and 3456x1152)
```

`1152` is the vision tower width — the 9B language tower is 4096 — so that error
means the vision path was reached with no vision weights present.

Fix: download the matching **mmproj** GGUF (`type = mmproj`,
`clip.has_vision_encoder = true`) and select it in `vision_name`:

```
CLIPLoader (GGUF)
  clip_name   : Qwen-Image-2.1-PE-T2I.Q5_K_S.gguf
  type        : qwen_image
  vision_name : mmproj-Qwen3.5-9B-BF16.gguf      <- optional, new
```

`vision_name` defaults to `none` and listing it is optional, so text-only
workflows are unaffected. Any `.gguf` whose filename contains `mmproj` or
`vision` shows up in the list — put the file in `models/text_encoders` or
`models/clip` so ComfyUI scans it.

---

## The qwen35 conversion, step by step

Derived and verified tensor-by-tensor against the official reference weights
[`Qwen/Qwen-Image-2.1-PE-T2I`](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I)
(architecture `qwen3_5`, `qwen35_9b`: 32 layers, hidden 4096, intermediate
12288, 16 attention heads / 4 KV heads, head_dim 256, 24 linear-attention +
8 full-attention layers at interval 4).

`convert_qwen35()` performs five transformations:

1. **RMSNorm centring.** qwen35 GGUF stores norm weights un-centred, while
   ComfyUI's `RMSNorm` uses `add=1` — so `1.0` is subtracted. Applies to
   `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm` and the final
   `norm`.
2. **`ssm_a` → `A_log`.** The GGUF holds `A` (negative); ComfyUI stores `A_log`,
   so `A_log = log(-A)`.
3. **Value-head channel order.** The per-head SSM tensors — `A_log`, `dt_bias`,
   `in_proj_a`, `in_proj_b` — are grouped *evens then odds* over the 32 value
   heads. A single permutation is correct for every layer.
4. **`conv1d` channel groups.** 8192 channels stored as 64 groups of 128,
   reordered `0..32, 34,36,..,62, 33,35,..,63`.
5. **Key renaming.** llama.cpp names → ComfyUI qwen35 module names, e.g.
   `blk.N.attn_qkv.weight` → `model.language_model.layers.N.linear_attn.in_proj_qkv.weight`.

Every tensor is dequantised before being returned. This matters: a `GGMLTensor`
reports the dequantised shape while keeping quantised storage, so
`load_state_dict` would otherwise read the wrong inner dimension.

`vision_from_mmproj()` maps the mmproj onto ComfyUI's `model.visual.*`:

| mmproj | ComfyUI |
|---|---|
| `v.blk.N.ln1` / `ln2` | `model.visual.blocks.N.norm1` / `norm2` |
| `v.blk.N.attn_qkv` | `model.visual.blocks.N.attn.qkv` |
| `v.blk.N.attn_out` | `model.visual.blocks.N.attn.proj` |
| `v.blk.N.ffn_up` / `ffn_down` | `model.visual.blocks.N.mlp.linear_fc1` / `linear_fc2` |
| `v.patch_embd` + `.1` | `patch_embed.proj` (two temporal slices → Conv3d) |
| `v.position_embd` | `pos_embed` |
| `v.post_ln`, `mm.0`, `mm.2` | `merger.norm`, `merger.linear_fc1`, `merger.linear_fc2` |

Two ggml layout traps are handled explicitly:

- `tensor.shape` is the *logical* shape while `tensor.data` has its axes
  **reversed**, stored in that tensor's own dtype (BF16 appears as `uint8` with
  the last dimension doubled).
- Consequently mmproj linear weights are `(in, out)` and need a transpose —
  unlike the language tower, which already stores them as `(out, in)`.

---

## Verification status

| Check | Result |
|---|---|
| Layer-type layout vs `comfy.text_encoders.qwen35._qwen35_layer_types()` | identical |
| Converted tensor keys present in the reference key set | all, 0 unexpected |
| Tensor shapes vs the official checkpoint | all match |
| Norm offset, `A_log` transform, channel permutations (un-quantised F32 tensors) | exact, 0 error, all 24 linear-attention layers |
| Language GGUF + mmproj tensor count | 427 + 333 = **760**, = the official checkpoint, 0 key overlap |
| `Qwen35VisionModel.load_state_dict` | 0 missing / 0 unexpected / 0 shape mismatch |
| Synthetic vision forward pass | OK, returns the 4096-d language width |
| `comfy.sd.load_text_encoder_state_dicts` | `Qwen35TEModel_` |
| Real `CLIPLoaderGGUF.execute(..., vision_name=...)` (V3, no stubs) | `NodeOutput` → `Qwen35TEModel_` |
| V3 migration: entrypoint, all six schemas, id/display-name parity with V1 | pass |

## Known limitations

- **Not yet numerically compared against the reference model.** The mapping is
  verified structurally and exactly on the un-quantised tensors, but a
  token-by-token comparison of the GGUF text path against the int8 reference has
  not been run (it needs both models resident). If prompt-enhancement output
  looks wrong, the first suspect is item below.
- **The fused `attn_qkv` of the 8 full-attention layers is not split.** The GGUF
  packs q/k/v/gate into one 8192×4096 tensor while ComfyUI wants separate
  `q_proj` / `k_proj` / `v_proj`. It is currently mapped through as-is. The
  reference is int8-quantised there, so the packing order could not be derived
  with zero error; guessing would degrade output silently, so it was left alone.
  This does not affect the vision tower.
- **Verified against the 9B PE-T2I weights only — and now enforced.** The
  channel tables assume 32 value heads, so `convert_qwen35()` checks the GGUF's
  value-head count and **refuses** a mismatch instead of silently corrupting the
  weights (the 4B weights have 16). Likewise `vision_from_mmproj()` rejects an
  mmproj whose vision geometry is not the validated 9B one (hidden 1152, patch
  16). Both guards raise with an explanatory message; supporting another size
  means deriving and validating its own tables.
- **Language-only GGUFs need an mmproj** for any image input, as described above.

---

## Project layout

```
ComfyUI-GGUF-qwen35/
├── __init__.py            # V3 entry point (+ V1 fallback)
├── nodes_v3.py            # V3 schema nodes          (IKBlue, new)
├── nodes.py               # V1 schema nodes + shared loading helpers
├── gguf_patcher.py        # GGUFModelPatcher + *_gguf folder registration
├── qwen35_support.py      # qwen35 + mmproj conversion (IKBlue, new)
├── loader.py              # GGUF readers and key maps  (upstream + qwen35)
├── ops.py                 # GGML custom operations     (upstream)
├── dequant.py             # dequantisation kernels     (upstream)
├── tools/                 # conversion / quantisation helpers (upstream)
├── NOTICE                 # attribution
├── CHANGELOG.md           # fork changes
├── LICENSE                # Apache-2.0 (upstream, unchanged)
└── pyproject.toml         # package + ComfyUI registry metadata
```

Files must stay in this flat layout: `loader.py` consumes `qwen35_support.py` via
a relative import and `ops.py` imports `comfy.ops`, so moving them breaks the
pack.

### Why the loading logic is in free functions

A method cannot be shared between the two schemas, because V1 binds `self` while
V3 binds `cls`. So the shared work lives at module level in `nodes.py` and both
schemas call it directly:

| Helper | Purpose |
|---|---|
| `_filename_list()` / `_vision_filename_list()` | the CLIP file lists |
| `_load_clip_state_dicts(paths, vision_path)` | read each path; merge the mmproj tower |
| `_load_text_encoder(paths, clip_type, clip_data)` | build the `CLIP` with the GGML ops |
| `_clip_type(name)` | map a `type` widget value onto `comfy.sd.CLIPType` |

`GGUFModelPatcher` and the `*_gguf` folder registration live in
`gguf_patcher.py` for the same reason — that file also performs the registration
at import, so the folder keys exist no matter which schema path loads.

## Development

Everything runs on CPU and needs no GPU to check:

```bash
python -c "import ast,glob; [ast.parse(open(p,encoding='utf-8').read(),p) for p in glob.glob('*.py')]"

# expand the V3 entrypoint and validate every schema
python - <<'PY'
import asyncio
from nodes_v3 import comfy_entrypoint
ext = asyncio.run(comfy_entrypoint())
for cls in asyncio.run(ext.get_node_list()):
    s = cls.GET_SCHEMA(); s.validate()
    print(s.node_id, s.category, [i.id for i in s.inputs])
PY
```

Conversion helpers (model → GGUF, quantization with a patched llama.cpp) live in
[`tools/README.md`](tools/README.md) and are upstream's.

## Credits and licence

- **Upstream project:** [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
  — City96, Apache-2.0. All GGUF reading, GGML ops, dequantisation and non-qwen35
  model support.
- **qwen35 support, mmproj vision tower, V3 migration, docs:** IKBlue, Apache-2.0.
- **Reference weights:** [Qwen/Qwen-Image-2.1-PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I)
  — Qwen team.
- **mmproj vision tower:** [lmstudio-community/Qwen3.5-9B-GGUF](https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF).
- **V3 schema reference:** [ComfyUI V3 migration guide](https://docs.comfy.org/custom-nodes/v3_migration).

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE). Modified files
carry a notice stating that they were changed, as required by Apache-2.0 §4(b).
