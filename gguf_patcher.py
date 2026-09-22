# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Modified 2026 by IKBlue: extracted from nodes.py so the V1 and V3 node
# modules share one definition instead of keeping two verbatim copies.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared GGUF plumbing used by both the V1 and V3 node modules.

The node definitions in :mod:`nodes` need two pieces of infrastructure that have
nothing to do with the schema itself, so they live here instead.

:class:`GGUFModelPatcher`
    A ``ModelPatcher`` that understands GGML-quantised weights: it keeps
    quantised tensors on the offload device and attaches LoRA patches to them
    instead of materialising a float copy, and it releases mmap'd modules after
    the first load so Windows can drop the file mapping.

:func:`update_folder_names_and_paths`
    Registers the ``*_gguf`` pseudo folder keys so the loaders can offer
    ``.gguf`` files from the corresponding ComfyUI model directories.

Attribution: both definitions come from City96's ComfyUI-GGUF
(https://github.com/city96/ComfyUI-GGUF), Apache-2.0, and are unchanged apart
from being moved here.
"""

import torch
import logging
import collections

import comfy.lora
import comfy.float
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import folder_paths

from .ops import move_patch_to_device
from .dequant import is_quantized, is_torch_compatible


def update_folder_names_and_paths(key, targets=[]):
    """Register a ``.gguf``-only pseudo folder key onto ComfyUI's folder map.

    ``key`` (e.g. ``"unet_gguf"``) is given the same underlying paths as the
    first entry of ``targets`` that already exists (e.g. ``"diffusion_models"``),
    but restricted to the ``.gguf`` extension. If ``key`` already had paths of
    its own that differ from the resolved target, a warning is logged and the
    target's paths win.

    Callers must still perform the registration themselves at import time; this
    helper only mutates ``folder_paths.folder_names_and_paths``.
    """
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")


# Register the .gguf-only pseudo keys here rather than in the node modules, so
# they exist no matter which schema path is loaded (V3 never imports nodes.py's
# registration calls). The loaders read these keys in their schemas.
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])


class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    """``ModelPatcher`` variant that keeps quantised GGML weights quantised.

    Key behaviours beyond the base class:

    ``patch_weight_to_device``
        For a GGML weight (a ``GGMLTensor``) the patch is attached to the
        quantised tensor rather than dequantised into a float copy, which is
        what makes LoRA on a quantised checkpoint fit in VRAM at all.

    ``pin_weight_to_device`` / ``load``
        ``load`` records every named module and, on the first low-VRAM load,
        moves offloaded modules to the load device and back so nothing stays
        linked to the mmap; ``pin_weight_to_device`` drops the corresponding
        bookkeeping entry.

    ``clone``
        Preserves ``patch_on_device`` / ``mmap_released`` and forces a size
        recalculation when upgrading a plain ``ModelPatcher`` in place.
    """

    patch_on_device = False

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
            # TODO: do we ever have legitimate duplicate patches? (i.e. patch on top of patched weight)
            out_weight.patches = [(patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                )

            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = comfy.lora.calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        # TODO: Find another way to not unload after patches
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)


    def pin_weight_to_device(self, key):
        op_key = key.rsplit('.', 1)[0]
        if not self.mmap_released and op_key in self.named_modules_to_munmap:
            # TODO: possible to OOM, find better way to detach
            self.named_modules_to_munmap[op_key].to(self.load_device).to(self.offload_device)
            del self.named_modules_to_munmap[op_key]
        super().pin_weight_to_device(key)

    mmap_released = False
    named_modules_to_munmap = {}

    def load(self, *args, force_patch_weights=False, **kwargs):
        if not self.mmap_released:
            self.named_modules_to_munmap = dict(self.model.named_modules())

        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.named_modules_to_munmap.items():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked and self.load_device != self.offload_device:
                logging.info(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    # TODO: possible to OOM, find better way to detach
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True
            self.named_modules_to_munmap = {}

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        # GGUF specific clone values below
        n.patch_on_device = getattr(self, "patch_on_device", False)
        n.mmap_released = getattr(self, "mmap_released", False)
        if src_cls != GGUFModelPatcher:
            n.size = 0 # force recalc
        return n
