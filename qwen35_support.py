# Copyright 2026 IKBlue
# Copyright 2024-2026 City96 (ComfyUI-GGUF, https://github.com/city96/ComfyUI-GGUF)
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3.5 ("qwen35") text-encoder and vision-tower support for ComfyUI.

ComfyUI core ships a native Qwen3.5 text encoder (``comfy.text_encoders.qwen35``)
but the llama.cpp GGUF layout for the same model differs from ComfyUI's native
checkpoint, so ``CLIPLoader (GGUF)`` rejects these files with::

    ValueError: Unexpected text model architecture type in GGUF file: 'qwen35'

This module bridges the two layouts. It exposes two public functions:

``convert_qwen35(sd, is_quantized, dequantize_tensor)``
    Remap a raw qwen35 GGUF state dict onto ComfyUI's qwen35 checkpoint layout.

``vision_from_mmproj(path, reader=None)``
    Convert a Qwen3.5 ``mmproj`` GGUF (the llama.cpp vision tower) into
    ``model.visual.*`` tensors, for text-encoder GGUFs that were exported
    language-only.

Everything here was derived and verified tensor-by-tensor against the official
reference weights ``Qwen/Qwen-Image-2.1-PE-T2I`` (architecture ``qwen3_5``,
``qwen35_9b``: 32 layers, hidden 4096, intermediate 12288, 16 attention heads /
4 KV heads, head_dim 256, and 24 linear-attention + 8 full-attention layers at
``full_attention_interval=4``).

Scope: verified against those 9B weights only. The channel tables below assume
**32 value heads**; other sizes (2B/4B/27B) may need their own validation.
"""

import re
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Tensor-layout tables
# ---------------------------------------------------------------------------

#: Value-head channel order used by the qwen35 SSM tensors. The GGUF groups the
#: 32 value heads "evens then odds"; one single permutation is correct for every
#: linear-attention layer. Verified exact (0 error) on the F32 tensors of all 24
#: linear-attention layers.
HEAD_PERM = list(range(0, 32, 2)) + list(range(1, 32, 2))

#: Channel-group order for ``ssm_conv1d`` (8192 channels = 64 groups of 128).
#: Groups 0..32 stay, then the even groups 34,36,..,62, then the odd 33,35,..,63.
CONV_GROUP_ORDER = (list(range(0, 33))
                    + list(range(34, 63, 2))
                    + list(range(33, 64, 2)))
assert len(CONV_GROUP_ORDER) == 64

#: RMSNorm weights in the GGUF are stored un-centred. ComfyUI's ``RMSNorm`` uses
#: ``add=1``, so these need ``- 1.0``.
NORM_KEYS = {"attn_norm.weight", "post_attention_norm.weight",
             "attn_q_norm.weight", "attn_k_norm.weight"}

#: Keys present on both linear- and full-attention layers.
COMMON_SUFFIX = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

#: Keys found only on linear-attention layers (the DeltaNet / SSM blocks).
LINEAR_SUFFIX = {
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
}

#: Keys found only on full-attention layers (every 4th layer).
#:
#: Scope note: the qwen35 GGUF carries a fused ``attn_qkv.weight`` on **all** 32
#: layers, and ``LINEAR_SUFFIX`` maps it to ``linear_attn.in_proj_qkv.weight``.
#: For the 24 linear-attention layers that is correct. For the 8 full-attention
#: layers it is *not* split into ``q_proj`` / ``k_proj`` / ``v_proj`` -- those
#: layers additionally carry separate ``attn_q``/``attn_k``/``attn_v`` weights,
#: which are mapped below, and the fused tensor is passed through unchanged and
#: therefore ignored by ComfyUI's module tree. Splitting it properly was left
#: undone on purpose: the only available reference for those layers is
#: int8-quantised, so the packing order of q/k/v/gate could not be established
#: with zero error, and a guess would corrupt output silently. See the
#: "Known limitations" section of the README.
FULL_SUFFIX = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}


def convert_qwen35(sd, is_quantized, dequantize_tensor):
    """Remap a raw qwen35 GGUF state dict to ComfyUI's qwen35 checkpoint layout.

    Applies, per tensor:

    * key renaming from llama.cpp names to ComfyUI's qwen35 module names;
    * ``- 1.0`` on the RMSNorm weights (the GGUF stores them un-centred);
    * ``A_log = log(-A)`` for ``ssm_a``;
    * the :data:`HEAD_PERM` channel permutation on ``dt_bias``,
      ``ssm_alpha.weight`` and ``ssm_beta.weight``;
    * the :data:`CONV_GROUP_ORDER` group reorder on ``ssm_conv1d.weight``.

    Every tensor is dequantised before being returned. That is required, not
    cosmetic: a ``GGMLTensor`` reports its dequantised shape while keeping
    quantised storage, so ``load_state_dict`` would read the wrong inner
    dimension (e.g. 4096 -> 2816).

    Args:
        sd: Raw state dict from ``loader.gguf_sd_loader(path, is_text_model=True)``.
        is_quantized: ``dequant.is_quantized`` from the host module. Passed in
            rather than imported so this module stays independent of the
            package's import graph.
        dequantize_tensor: ``dequant.dequantize_tensor`` from the host module.

    Returns:
        dict[str, torch.Tensor]: a state dict keyed the way ComfyUI's qwen35
        text encoder expects (``model.language_model.*`` / ``lm_head.*``).

    Raises:
        KeyError: if a GGUF key has no mapping. Failing loudly is deliberate --
        a silently dropped tensor would degrade output without any error.
    """
    def plain(t):
        return dequantize_tensor(t) if is_quantized(t) else t

    out = {}
    P = torch.tensor(HEAD_PERM)
    conv_idx = torch.tensor(CONV_GROUP_ORDER)

    for k, v in sd.items():
        if k == "token_embd.weight":
            out["model.language_model.embed_tokens.weight"] = plain(v).detach().clone()
            continue
        if k == "output.weight":
            out["lm_head.weight"] = plain(v).detach().clone()
            continue
        if k == "output_norm.weight":
            out["model.language_model.norm.weight"] = plain(v).detach().clone() - 1.0
            continue

        m = re.match(r"blk\.(\d+)\.(.+)$", k)
        if not m:
            continue
        layer, tail = int(m.group(1)), m.group(2)
        pre = "model.language_model.layers.%d." % layer

        if tail in NORM_KEYS:
            name = COMMON_SUFFIX.get(tail) or FULL_SUFFIX.get(tail)
            out[pre + name] = plain(v).detach().clone() - 1.0
            continue
        if tail in COMMON_SUFFIX:
            out[pre + COMMON_SUFFIX[tail]] = plain(v).detach().clone()
            continue
        if tail in LINEAR_SUFFIX:
            t = plain(v).detach().clone()
            if tail == "ssm_a":
                t = torch.log(-t.float())
            elif tail in ("ssm_dt.bias", "ssm_alpha.weight", "ssm_beta.weight"):
                t = t[P]
            elif tail == "ssm_conv1d.weight":
                t = t.reshape(64, 128, 4)[conv_idx].reshape(8192, 1, 4).contiguous()
            out[pre + LINEAR_SUFFIX[tail]] = t
            continue
        if tail in FULL_SUFFIX:
            out[pre + FULL_SUFFIX[tail]] = plain(v).detach().clone()
            continue
        raise KeyError("ComfyUI-GGUF/qwen35: unmapped gguf key: %s" % k)

    return out


# ---------------------------------------------------------------------------
# Vision tower (mmproj GGUF)
# ---------------------------------------------------------------------------

#: ``v.blk.N.<ggml name>`` -> ``model.visual.blocks.N.<comfy name>``, and whether
#: the weight needs a transpose. mmproj linear weights are stored ``(in, out)``
#: (unlike the language tower, which stores them ``(out, in)``).
_VISION_TAIL = {
    "attn_qkv.weight": ("attn.qkv.weight", True),
    "attn_qkv.bias": ("attn.qkv.bias", False),
    "attn_out.weight": ("attn.proj.weight", True),
    "attn_out.bias": ("attn.proj.bias", False),
    "ffn_up.weight": ("mlp.linear_fc1.weight", True),
    "ffn_up.bias": ("mlp.linear_fc1.bias", False),
    "ffn_down.weight": ("mlp.linear_fc2.weight", True),
    "ffn_down.bias": ("mlp.linear_fc2.bias", False),
    "ln1.weight": ("norm1.weight", False),
    "ln1.bias": ("norm1.bias", False),
    "ln2.weight": ("norm2.weight", False),
    "ln2.bias": ("norm2.bias", False),
}


def _logical(tensor):
    """Return a ggml reader tensor as a torch tensor in logical axis order.

    ggml files store ``tensor.shape`` as the logical shape while ``tensor.data``
    carries the **reversed** axes, in the tensor's own dtype. BF16 therefore
    surfaces as ``uint8`` with the last dimension doubled.

    Args:
        tensor: an entry from ``gguf.GGUFReader(path).tensors``.

    Returns:
        torch.Tensor: float32 tensor with axes matching ``tensor.shape``.
    """
    arr = tensor.data
    if arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    arr = np.ascontiguousarray(arr)
    t = torch.from_numpy(arr)
    if t.dtype == torch.uint8:
        t = t.view(torch.bfloat16)          # BF16 is exposed as uint8
    t = t.to(torch.float32)
    if t.dim() > 1:
        t = t.permute(*range(t.dim() - 1, -1, -1))
    return t.contiguous()


def vision_from_mmproj(path, reader=None):
    """Convert a Qwen3.5 mmproj GGUF into ``model.visual.*`` tensors.

    Many qwen35 text-encoder GGUFs (for example the Qwen-Image-2.1 ``PE-T2I``
    prompt enhancers) are exported language-only. ComfyUI's ``Qwen35``
    unconditionally builds a vision tower and calls it for image inputs, so such
    a file loads but fails on any image-edit workflow with a shape error whose
    inner dimension is the **vision** width (1152), not the language width
    (4096 on the 9B model). This supplies the missing tensors.

    Mappings (see :data:`_VISION_TAIL` for the per-block keys):

    ==========================  =============================================
    mmproj                      ComfyUI
    ==========================  =============================================
    ``v.patch_embd`` + ``.1``   ``patch_embed.proj`` (2 temporal slices, Conv3d)
    ``v.position_embd``         ``pos_embed``
    ``v.post_ln``               ``merger.norm``
    ``mm.0`` / ``mm.2``         ``merger.linear_fc1`` / ``linear_fc2``
    ``v.blk.N.*``               ``model.visual.blocks.N.*``
    ==========================  =============================================

    Args:
        path: filesystem path to the mmproj GGUF. Ignored when ``reader`` is
            given.
        reader: an already-open ``gguf.GGUFReader``, to avoid rereading the file.

    Returns:
        dict[str, torch.Tensor]: a state dict keyed ``model.visual.*``. Merge it
        straight into the text encoder's: ``sd.update(vision_from_mmproj(p))``.

    Raises:
        KeyError: on an unmapped ``v.blk.N.*`` key.

    Note:
        The two ``v.patch_embd`` tensors are genuinely different (max abs
        difference ~6e-3), i.e. the two temporal slices of the Conv3d kernel --
        they must be stacked, not deduplicated.
    """
    if reader is None:
        import gguf
        reader = gguf.GGUFReader(path)
    by_name = {t.name: t for t in reader.tensors}
    out = {}

    def g(name):
        return _logical(by_name[name])

    # patch embed: ggml keeps the two temporal slices as separate tensors
    w0, w1 = g("v.patch_embd.weight"), g("v.patch_embd.weight.1")
    merged = torch.stack([w0, w1], dim=0)                      # (2,16,16,3,1152)
    out["model.visual.patch_embed.proj.weight"] = (
        merged.permute(4, 3, 0, 1, 2).reshape(1152, 3, 2, 16, 16).contiguous())
    out["model.visual.patch_embed.proj.bias"] = g("v.patch_embd.bias")
    out["model.visual.pos_embed.weight"] = g("v.position_embd.weight").t().contiguous()
    out["model.visual.merger.norm.weight"] = g("v.post_ln.weight")
    out["model.visual.merger.norm.bias"] = g("v.post_ln.bias")
    out["model.visual.merger.linear_fc1.weight"] = g("mm.0.weight").t().contiguous()
    out["model.visual.merger.linear_fc1.bias"] = g("mm.0.bias")
    out["model.visual.merger.linear_fc2.weight"] = g("mm.2.weight").t().contiguous()
    out["model.visual.merger.linear_fc2.bias"] = g("mm.2.bias")

    for t in reader.tensors:
        m = re.match(r"v\.blk\.(\d+)\.(.+)$", t.name)
        if not m:
            continue
        layer, tail = int(m.group(1)), m.group(2)
        if tail not in _VISION_TAIL:
            raise KeyError("ComfyUI-GGUF/qwen35: unmapped mmproj key: %s" % t.name)
        dest, transpose = _VISION_TAIL[tail]
        v = _logical(t)
        if transpose:
            v = v.t().contiguous()
        out["model.visual.blocks.%d.%s" % (layer, dest)] = v

    return out
