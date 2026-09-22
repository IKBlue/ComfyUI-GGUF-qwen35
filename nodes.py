# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Modified 2026 by IKBlue:
#   * GGUFModelPatcher and the *_gguf folder registration moved to
#     gguf_patcher.py so nodes.py and nodes_v3.py share one definition
#   * the CLIP state-dict loader and text-encoder builder are now free
#     functions (module level) so the V1 and V3 node classes can both call
#     them -- a method cannot be shared, because V1 binds `self` and V3
#     binds `cls`
#   * optional "vision_name" input on CLIPLoaderGGUF (mmproj vision tower)
#   * node category changed from "bootleg" to "IKBlue"
#
# The V1 node classes below remain the ones ComfyUI registers when
# comfy_api.latest is unavailable; otherwise nodes_v3.py is used.

"""V1 (legacy schema) node classes for the GGUF loaders.

Kept alongside :mod:`nodes_v3` so ComfyUI builds that predate the V3 schema
keep working. The shared loading logic lives in free functions:

``_filename_list()``
    Every ``.gguf``/safetensors text encoder ComfyUI can see.
``_vision_filename_list()``
    The subset that looks like an mmproj / vision tower.
``_load_clip_state_dicts(paths, vision_path)``
    Read each path into a state dict; merges the mmproj vision tower when a
    single GGUF is loaded.
``_load_text_encoder(clip_paths, clip_type, clip_data)``
    Build the ``CLIP`` object with the GGML custom ops and wrap its patcher.

See :mod:`gguf_patcher` for the model patcher and the folder registration.
"""
import torch
import logging
import inspect

import nodes
import comfy.sd
import comfy.utils
import comfy.model_management
import folder_paths

from .ops import GGMLOps
from .loader import gguf_sd_loader, gguf_clip_loader
from .gguf_patcher import GGUFModelPatcher, update_folder_names_and_paths

# Add a custom keys for files ending in .gguf
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])


# ---------------------------------------------------------------------------
# shared loading helpers (free functions: usable from V1 methods and V3
# classmethods alike, since neither `self` nor `cls` is involved)
# ---------------------------------------------------------------------------

def _filename_list():
    """All text-encoder files the GGUF CLIP loaders may offer."""
    files = []
    files += folder_paths.get_filename_list("clip")
    files += folder_paths.get_filename_list("clip_gguf")
    return sorted(files)


def _vision_filename_list():
    """The mmproj / vision-tower GGUF files among :func:`_filename_list`."""
    return sorted(f for f in _filename_list()
                  if "mmproj" in f.lower() or "vision" in f.lower())


def _load_clip_state_dicts(ckpt_paths, vision_path=None):
    """Read each path into a state dict, merging the mmproj vision tower.

    ``vision_path`` only applies when a single GGUF is being loaded -- a vision
    tower belongs to one text encoder, so it is ignored for the multi-encoder
    loaders.

    Non-GGUF files are read with ``load_torch_file``; mixing scaled FP8 with
    GGUF is rejected because only one set of custom ops can be active.
    """
    clip_data = []
    for p in ckpt_paths:
        if p.endswith(".gguf"):
            sd = gguf_clip_loader(p, vision_path=(vision_path if len(ckpt_paths) == 1 else None))
        else:
            sd = comfy.utils.load_torch_file(p, safe_load=True)
            if "scaled_fp8" in sd: # NOTE: Scaled FP8 would require different custom ops, but only one can be active
                raise NotImplementedError(f"Mixing scaled FP8 with GGUF is not supported! Use regular CLIP loader or switch model(s)\n({p})")
        clip_data.append(sd)
    return clip_data


def _load_text_encoder(clip_paths, clip_type, clip_data):
    """Build the text-encoder ``CLIP`` for already-loaded state dicts."""
    clip = comfy.sd.load_text_encoder_state_dicts(
        clip_type = clip_type,
        state_dicts = clip_data,
        model_options = {
            "custom_operations": GGMLOps,
            "initial_device": comfy.model_management.text_encoder_offload_device()
        },
        embedding_directory = folder_paths.get_folder_paths("embeddings"),
    )
    clip.patcher = GGUFModelPatcher.clone(clip.patcher)
    return clip


def _clip_type(name):
    """Map a ``type`` widget value onto ``comfy.sd.CLIPType``."""
    return getattr(comfy.sd.CLIPType, name.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)


class UnetLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "IKBlue"
    TITLE = "Unet Loader (GGUF)"

    def load_unet(self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None):
        ops = GGMLOps()

        if dequant_dtype in ("default", None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ["target"]:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

        if patch_dtype in ("default", None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ["target"]:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)

        # init model
        unet_path = folder_paths.get_full_path("unet", unet_name)
        sd, extra = gguf_sd_loader(unet_path)

        kwargs = {}
        valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
        if "metadata" in valid_params:
            kwargs["metadata"] = extra.get("metadata", {})

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}, **kwargs,
        )
        if model is None:
            logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
            raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        return (model,)

class UnetLoaderGGUFAdvanced(UnetLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
            }
        }
    TITLE = "Unet Loader (GGUF/Advanced)"

class CLIPLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.CLIPLoader.INPUT_TYPES()
        return {
            "required": {
                "clip_name": (s.get_filename_list(),),
                "type": base["required"]["type"],
            },
            "optional": {
                # Qwen3.5 only: supply the matching mmproj GGUF when the text
                # encoder GGUF was exported language-only (no vision tower).
                "vision_name": (["none"] + s.get_vision_filename_list(),),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "IKBlue"
    TITLE = "CLIPLoader (GGUF)"

    @classmethod
    def get_filename_list(s):
        return _filename_list()

    @classmethod
    def get_vision_filename_list(s):
        "mmproj / vision-tower GGUF files that can be merged into a text encoder"
        return _vision_filename_list()

    def load_clip(self, clip_name, type="stable_diffusion", vision_name="none"):
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip_type = _clip_type(type)
        vision_path = None
        if vision_name not in (None, "", "none"):
            vision_path = (folder_paths.get_full_path("clip", vision_name)
                           or folder_paths.get_full_path("clip_gguf", vision_name))
        return (_load_text_encoder([clip_path], clip_type,
                                   _load_clip_state_dicts([clip_path], vision_path=vision_path)),)

class DualCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.DualCLIPLoader.INPUT_TYPES()
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "type": base["required"]["type"],
            }
        }

    TITLE = "DualCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, type):
        clip_paths = (folder_paths.get_full_path("clip", clip_name1),
                      folder_paths.get_full_path("clip", clip_name2))
        return (_load_text_encoder(clip_paths, _clip_type(type),
                                   _load_clip_state_dicts(clip_paths)),)

class TripleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "clip_name3": file_options,
            }
        }

    TITLE = "TripleCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, type="sd3"):
        clip_paths = (folder_paths.get_full_path("clip", clip_name1),
                      folder_paths.get_full_path("clip", clip_name2),
                      folder_paths.get_full_path("clip", clip_name3))
        return (_load_text_encoder(clip_paths, _clip_type(type),
                                   _load_clip_state_dicts(clip_paths)),)

class QuadrupleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        return {
            "required": {
            "clip_name1": file_options,
            "clip_name2": file_options,
            "clip_name3": file_options,
            "clip_name4": file_options,
        }
    }

    TITLE = "QuadrupleCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4, type="stable_diffusion"):
        clip_paths = (folder_paths.get_full_path("clip", clip_name1),
                      folder_paths.get_full_path("clip", clip_name2),
                      folder_paths.get_full_path("clip", clip_name3),
                      folder_paths.get_full_path("clip", clip_name4))
        return (_load_text_encoder(clip_paths, _clip_type(type),
                                   _load_clip_state_dicts(clip_paths)),)

NODE_CLASS_MAPPINGS = {
    "UnetLoaderGGUF": UnetLoaderGGUF,
    "CLIPLoaderGGUF": CLIPLoaderGGUF,
    "DualCLIPLoaderGGUF": DualCLIPLoaderGGUF,
    "TripleCLIPLoaderGGUF": TripleCLIPLoaderGGUF,
    "QuadrupleCLIPLoaderGGUF": QuadrupleCLIPLoaderGGUF,
    "UnetLoaderGGUFAdvanced": UnetLoaderGGUFAdvanced,
}
