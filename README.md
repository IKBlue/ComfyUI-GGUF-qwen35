# ComfyUI-GGUF-qwen35

> **This is a fork of [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).**
> It adds **Qwen3.5 (`qwen35`) text-encoder GGUF support**, which the upstream
> project does not currently handle. All GGUF reading, dequantisation, GGML
> custom ops and everything unrelated to qwen35 are **City96's original work**.
> Licensed Apache-2.0, same as upstream. See [NOTICE](NOTICE) for the full
> modification list.

## What this fixes

Loading a Qwen3.5 text encoder with `CLIPLoader (GGUF)` fails with:

```
ValueError: Unexpected text model architecture type in GGUF file: 'qwen35'
```

`qwen35` is a hybrid architecture (24 linear-attention + 8 full-attention layers
at 32 layers total), and its GGUF tensor layout does not match ComfyUI's native
qwen35 checkpoint. This fork adds the missing conversion.

## Changes vs upstream

Only two files differ:

| File | Change |
|---|---|
| `qwen35_support.py` | **new** — remaps the llama.cpp qwen35 layout onto ComfyUI's qwen35 checkpoint layout |
| `loader.py` | 6 lines — `"qwen35"` added to `TXT_ARCH_LIST`, plus one branch in `gguf_clip_loader()` |

The conversion performs five things:

1. **RMSNorm centring** — qwen35 GGUF stores norm weights un-centred, ComfyUI's
   `RMSNorm` uses `add=1`, so `1.0` is subtracted.
2. **`ssm_a` → `A_log`** — the GGUF holds `A` (negative), ComfyUI stores
   `A_log`, so `A_log = log(-A)`.
3. **Value-head channel order** — the per-head SSM tensors (`A_log`, `dt_bias`,
   `in_proj_a`, `in_proj_b`) are grouped *evens then odds* over the 32 value
   heads. One single permutation is correct for every layer.
4. **`conv1d` channel groups** — 8192 channels stored as 64 groups of 128,
   reordered `0..32, 34,36,..,62, 33,35,..,63`.
5. **Key renaming** — llama.cpp names (`blk.N.attn_qkv.weight`, …) to ComfyUI
   qwen35 module names (`model.language_model.layers.N.linear_attn.*`, …).

Tensors are dequantised before being returned. This matters: a `GGMLTensor`
reports the dequantised shape while keeping quantised storage, so
`load_state_dict` would otherwise read the wrong inner dimension.

## Verification

The mapping was derived and checked tensor-by-tensor against the **official
reference weights**, [`Qwen/Qwen-Image-2.1-PE-T2I`](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I)
(architecture `qwen3_5`, `qwen35_9b`: 32 layers, hidden 4096, intermediate
12288, 16 attention heads / 4 KV heads, head_dim 256):

* layer-type layout (24 linear-attention / 8 full-attention, `full_attention_interval=4`)
  matches `comfy.text_encoders.qwen35._qwen35_layer_types()` exactly;
* every converted tensor key exists in the reference key set (0 unexpected keys);
* all shapes match;
* the norm offset, the `A_log` transform and the channel permutations reproduce
  the reference **exactly (0 error)** on the un-quantised (`F32`) tensors of all
  24 linear-attention layers;
* the converted state dict loads through
  `comfy.sd.load_text_encoder_state_dicts()` as `Qwen35TEModel_`.

**Scope note:** this was verified against the Qwen-Image-2.1 PE-T2I (9B) weights
only. Other qwen35 sizes (2B / 4B / 27B) may need separate validation — the
`HEAD_PERM` / `CONV_GROUP_ORDER` tables assume 32 value heads.

## Install

Drop-in replacement for the upstream node pack:

```
git clone https://github.com/IKBlue/ComfyUI-GGUF-qwen35 ComfyUI/custom_nodes/ComfyUI-GGUF
```

or copy `qwen35_support.py` and the `loader.py` change into an existing
`ComfyUI-GGUF` checkout.

## Credit

* Upstream project: **[city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** — City96, Apache-2.0
* qwen35 support: **IKBlue**, Apache-2.0
* Reference weights: **[Qwen/Qwen-Image-2.1-PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I)** — Qwen team

---

*The original upstream README follows, unchanged.*

---

# ComfyUI-GGUF
GGUF Quantization support for native ComfyUI models

This is currently very much WIP. These custom nodes provide support for model files stored in the GGUF format popularized by [llama.cpp](https://github.com/ggerganov/llama.cpp).

While quantization wasn't feasible for regular UNET models (conv2d), transformer/DiT models such as flux seem less affected by quantization. This allows running it in much lower bits per weight variable bitrate quants on low-end GPUs. For further VRAM savings, a node to load a quantized version of the T5 text encoder is also included.

![Comfy_Flux1_dev_Q4_0_GGUF_1024](https://github.com/user-attachments/assets/70d16d97-c522-4ef4-9435-633f128644c8)

Note: The "Force/Set CLIP Device" is **NOT** part of this node pack. Do not install it if you only have one GPU. Do not set it to cuda:0 then complain about OOM errors if you do not undestand what it is for. There is not need to copy the workflow above, just use your own workflow and replace the stock "Load Diffusion Model" with the "Unet Loader (GGUF)" node.

## Installation

> [!IMPORTANT]  
> Make sure your ComfyUI is on a recent-enough version to support custom ops when loading the UNET-only.

To install the custom node normally, git clone this repository into your custom nodes folder (`ComfyUI/custom_nodes`) and install the only dependency for inference (`pip install --upgrade gguf`)

```
git clone https://github.com/city96/ComfyUI-GGUF
```

To install the custom node on a standalone ComfyUI release, open a CMD inside the "ComfyUI_windows_portable" folder (where your `run_nvidia_gpu.bat` file is) and use the following commands:

```
git clone https://github.com/city96/ComfyUI-GGUF ComfyUI/custom_nodes/ComfyUI-GGUF
.\python_embeded\python.exe -s -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-GGUF\requirements.txt
```

On MacOS sequoia, torch 2.4.1 seems to be required, as 2.6.X nightly versions cause a "M1 buffer is not large enough" error. See [this issue](https://github.com/city96/ComfyUI-GGUF/issues/107) for more information/workarounds.

## Models

GGUF quantizations of many models can be found on [Comfy-Org's HuggingFace](https://huggingface.co/Comfy-Org) and [city96's HuggingFace](https://huggingface.co/city96). For a full list of supported models, check the [README in the tools folder](tools/README.md).

Note that quantized models are not currently supported when using a `Stable-Diffusion` checkpoint with an embedded VAE. Use a standalone VAE instead. You can find the original full precision models on the [Comfy-Org HuggingFace](https://huggingface.co/Comfy-Org) repository.
