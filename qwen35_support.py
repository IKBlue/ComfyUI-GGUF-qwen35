# Qwen3.5 ("qwen35") text-encoder support for ComfyUI-GGUF.
#
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
#
# ---------------------------------------------------------------------------
# The qwen35 GGUF layout differs from ComfyUI's native qwen35 checkpoint in
# several ways. These rules were derived and verified tensor-by-tensor against
# the official reference weights (Qwen/Qwen-Image-2.1-PE-T2I, arch "qwen3_5",
# qwen35_9b: 32 layers, hidden 4096, inter 12288, 16/4 heads, head_dim 256):
#
#   1. RMSNorm weights are stored un-centred (comfy's RMSNorm uses add=1)
#      -> subtract 1.0.  Verified exact on q_norm/k_norm/attn_norm/
#         post_attention_norm/output_norm.
#   2. ``ssm_a`` holds A (negative); comfy stores A_log  ->  A_log = log(-A).
#      Verified exact on all 24 linear-attention layers.
#   3. The per-head SSM tensors (ssm_a -> A_log, ssm_dt.bias -> dt_bias,
#      ssm_alpha.weight -> in_proj_a.weight, ssm_beta.weight -> in_proj_b.weight)
#      are grouped "evens then odds" over the 32 value heads.  One single
#      permutation is correct for every layer.  Verified exact (0 error) on the
#      F32 tensors of all 24 linear-attention layers.
#   4. ``ssm_conv1d.weight`` (8192 channels) is stored as 64 groups of 128
#      reordered as: 0..32, then 34,36,..,62, then 33,35,..,63.  Verified exact
#      on every layer.
#   5. llama.cpp tensor names -> ComfyUI qwen35 module names (see the maps below).
#
# NOTE: verified against this model only.  Other qwen35 variants (2B/4B/27B) may
# need separate validation.
#
# Tensors are dequantised before being returned: a GGMLTensor reports the
# dequantised shape while keeping quantised storage, which makes
# ``load_state_dict`` read the wrong inner dimension (e.g. 4096 -> 2816).
#
# ---------------------------------------------------------------------------
# Optional vision tower support
# -----------------------------
# qwen35 text-encoder GGUFs are often exported language-only (e.g. the
# "PE-T2I" prompt enhancers).  ComfyUI's Qwen35 always builds its vision tower
# and calls it for image inputs, so such a file cannot handle image-edit
# workflows.  The missing vision weights are available from the matching mmproj
# GGUF (llama.cpp multimodal projector), e.g.
# lmstudio-community/Qwen3.5-9B-GGUF/mmproj-Qwen3.5-9B-BF16.gguf.
#
# ``vision_from_mmproj`` converts one of those into ``model.visual.*``:
#   * v.blk.N.ln1/ln2      -> model.visual.blocks.N.norm1/norm2
#   * v.blk.N.attn_qkv     -> model.visual.blocks.N.attn.qkv
#   * v.blk.N.attn_out     -> model.visual.blocks.N.attn.proj
#   * v.blk.N.ffn_up/down  -> model.visual.blocks.N.mlp.linear_fc1/fc2
#   * v.patch_embd + .1    -> patch_embed.proj   (two temporal slices -> Conv3d)
#   * v.position_embd      -> pos_embed
#   * v.post_ln / mm.0/2   -> merger.norm / merger.linear_fc1/fc2
#
# Layout gotcha: in a ggml file ``tensor.shape`` is the logical shape while
# ``tensor.data`` has its axes REVERSED, stored in the tensor's dtype (BF16
# shows up as uint8 with the last dim doubled).  Linear weights in the mmproj
# are therefore ``(in, out)`` and need a transpose.  Verified: all 333 tensors
# load into Qwen35VisionModel with 0 missing / 0 unexpected / 0 shape
# mismatches, and a synthetic forward returns the 4096-d language width.

import re
import numpy as np
import torch

# value-head channel order used by the qwen35 SSM tensors ("evens then odds")
HEAD_PERM = list(range(0, 32, 2)) + list(range(1, 32, 2))
# conv1d channel groups (64 groups x 128 channels)
CONV_GROUP_ORDER = (list(range(0, 33))
                    + list(range(34, 63, 2))
                    + list(range(33, 64, 2)))
assert len(CONV_GROUP_ORDER) == 64

NORM_KEYS = {"attn_norm.weight", "post_attention_norm.weight",
             "attn_q_norm.weight", "attn_k_norm.weight"}

COMMON_SUFFIX = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

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

    ``sd``                raw state dict from ``gguf_sd_loader``
    ``is_quantized``      ``dequant.is_quantized`` from the host module
    ``dequantize_tensor`` ``dequant.dequantize_tensor`` from the host module
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
# vision tower (mmproj GGUF)
# ---------------------------------------------------------------------------

# v.blk.N.<ggml name> -> model.visual.blocks.N.<comfy name>, and whether the
# weight needs a transpose (mmproj stores linear weights as (in, out))
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
    """raw ggml tensor -> torch tensor in logical (torch) axis order"""
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

    Returns a state dict that can simply be merged into the language tower's
    (``out.update(vision_from_mmproj(mmproj_path))``).
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
