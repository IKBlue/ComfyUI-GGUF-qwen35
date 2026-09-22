# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Modified 2026 by IKBlue: migrated to the ComfyUI V3 node schema
# (comfy_api.latest: io.Schema / io.ComfyNode / comfy_entrypoint).
#
# See https://docs.comfy.org/custom-nodes/v3_migration

"""ComfyUI V3 (``comfy_api.latest``) definitions for the GGUF loader nodes.

This is the only node module in the pack: there is no V1 fallback, so importing
this file requires a ComfyUI build that provides ``comfy_api.latest``.

Six nodes, each an :class:`io.ComfyNode` with a ``define_schema`` classmethod and
a classmethod ``execute`` returning :class:`io.NodeOutput`:

===========================  ==========  ==========================================
node_id                      output      notes
===========================  ==========  ==========================================
``UnetLoaderGGUF``           ``MODEL``   diffusion model from ``models/unet`` or
                                         ``models/diffusion_models``
``UnetLoaderGGUFAdvanced``   ``MODEL``   adds ``dequant_dtype``, ``patch_dtype``,
                                         ``patch_on_device``
``CLIPLoaderGGUF``           ``CLIP``    adds an optional ``vision_name`` mmproj
``DualCLIPLoaderGGUF``       ``CLIP``    two text encoders
``TripleCLIPLoaderGGUF``     ``CLIP``    three text encoders
``QuadrupleCLIPLoaderGGUF``  ``CLIP``    four text encoders
===========================  ==========  ==========================================

Registration
------------
ComfyUI's custom-node loader checks ``NODE_CLASS_MAPPINGS`` **first** and returns
immediately, so the ``comfy_entrypoint`` branch is only reached when that
attribute is ``None`` or absent. :mod:`__init__` therefore sets
``NODE_CLASS_MAPPINGS = None``; there is no legacy mapping to fall back to.

Shared plumbing that has nothing to do with the node schema (the GGUF model
patcher and the ``*_gguf`` folder registration) lives in :mod:`gguf_patcher`;
the qwen35 / mmproj conversion lives in :mod:`qwen35_support`.

Attribution: the node definitions are derived from the V1 node classes in
upstream ComfyUI-GGUF by City96 (https://github.com/city96/ComfyUI-GGUF),
Apache-2.0.
"""
import torch
import logging
import inspect

import nodes
import comfy.sd
import comfy.utils
import comfy.model_management
import folder_paths

from comfy_api.latest import ComfyExtension, io

from .ops import GGMLOps
from .loader import gguf_sd_loader, gguf_clip_loader
from .gguf_patcher import GGUFModelPatcher


# ---------------------------------------------------------------------------
# shared loading helpers
#
# These are plain functions rather than node methods so that all six nodes (and
# the dual/triple/quadruple wrappers in particular) can call the same code.
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


def _clip_type_options():
    """The same ``type`` list the core CLIPLoader exposes."""
    return nodes.CLIPLoader.INPUT_TYPES()["required"]["type"][0]


def _clip_type(name):
    """Map a ``type`` widget value onto ``comfy.sd.CLIPType``."""
    return getattr(comfy.sd.CLIPType, name.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)


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


class UnetLoaderGGUF(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UnetLoaderGGUF",
            display_name="Unet Loader (GGUF)",
            category="IKBlue",
            description="Load a diffusion model in GGUF format.",
            inputs=[
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("unet_gguf")),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None) -> io.NodeOutput:
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
        return io.NodeOutput(model)


class UnetLoaderGGUFAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UnetLoaderGGUFAdvanced",
            display_name="Unet Loader (GGUF/Advanced)",
            category="IKBlue",
            description="Load a diffusion model in GGUF format with dequantisation options.",
            inputs=[
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("unet_gguf")),
                io.Combo.Input("dequant_dtype",
                               options=["default", "target", "float32", "float16", "bfloat16"],
                               default="default"),
                io.Combo.Input("patch_dtype",
                               options=["default", "target", "float32", "float16", "bfloat16"],
                               default="default"),
                io.Boolean.Input("patch_on_device", default=False),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, unet_name, dequant_dtype="default", patch_dtype="default", patch_on_device=False) -> io.NodeOutput:
        return UnetLoaderGGUF.execute(
            unet_name,
            dequant_dtype=dequant_dtype,
            patch_dtype=patch_dtype,
            patch_on_device=patch_on_device,
        )


class CLIPLoaderGGUF(io.ComfyNode):
    @classmethod
    def get_filename_list(cls):
        files = []
        files += folder_paths.get_filename_list("clip")
        files += folder_paths.get_filename_list("clip_gguf")
        return sorted(files)

    @classmethod
    def get_vision_filename_list(cls):
        "mmproj / vision-tower GGUF files that can be merged into a text encoder"
        return sorted(f for f in cls.get_filename_list()
                      if "mmproj" in f.lower() or "vision" in f.lower())

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPLoaderGGUF",
            display_name="CLIPLoader (GGUF)",
            category="IKBlue",
            description="Load a text encoder in GGUF format.",
            inputs=[
                io.Combo.Input("clip_name", options=cls.get_filename_list()),
                io.Combo.Input("type", options=_clip_type_options(), default="stable_diffusion"),
                # Qwen3.5 only: supply the matching mmproj GGUF when the text
                # encoder GGUF was exported language-only (no vision tower).
                io.Combo.Input("vision_name",
                               options=["none"] + cls.get_vision_filename_list(),
                               default="none",
                               optional=True,
                               tooltip="Qwen3.5 only: mmproj GGUF supplying the vision tower "
                                       "for text encoders exported without one."),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def get_filename_list(cls):
        return _filename_list()

    @classmethod
    def get_vision_filename_list(cls):
        return _vision_filename_list()

    @classmethod
    def execute(cls, clip_name, type="stable_diffusion", vision_name="none") -> io.NodeOutput:
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip_type = _clip_type(type)
        vision_path = None
        if vision_name not in (None, "", "none"):
            vision_path = (folder_paths.get_full_path("clip", vision_name)
                           or folder_paths.get_full_path("clip_gguf", vision_name))
        return io.NodeOutput(_load_text_encoder(
            [clip_path], clip_type,
            _load_clip_state_dicts([clip_path], vision_path=vision_path)))


class DualCLIPLoaderGGUF(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        file_options = _filename_list()
        return io.Schema(
            node_id="DualCLIPLoaderGGUF",
            display_name="DualCLIPLoader (GGUF)",
            category="IKBlue",
            description="Load two text encoders in GGUF format.",
            inputs=[
                io.Combo.Input("clip_name1", options=file_options),
                io.Combo.Input("clip_name2", options=file_options),
                io.Combo.Input("type", options=_clip_type_options(), default="stable_diffusion"),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip_name1, clip_name2, type="stable_diffusion") -> io.NodeOutput:
        clip_paths = (folder_paths.get_full_path("clip", clip_name1),
                      folder_paths.get_full_path("clip", clip_name2))
        clip_type = _clip_type(type)
        return io.NodeOutput(_load_text_encoder(
            clip_paths, clip_type, _load_clip_state_dicts(clip_paths)))


class TripleCLIPLoaderGGUF(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        file_options = _filename_list()
        return io.Schema(
            node_id="TripleCLIPLoaderGGUF",
            display_name="TripleCLIPLoader (GGUF)",
            category="IKBlue",
            description="Load three text encoders in GGUF format.",
            inputs=[
                io.Combo.Input("clip_name1", options=file_options),
                io.Combo.Input("clip_name2", options=file_options),
                io.Combo.Input("clip_name3", options=file_options),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip_name1, clip_name2, clip_name3, type="sd3") -> io.NodeOutput:
        clip_paths = (folder_paths.get_full_path("clip", clip_name1),
                      folder_paths.get_full_path("clip", clip_name2),
                      folder_paths.get_full_path("clip", clip_name3))
        clip_type = _clip_type(type)
        return io.NodeOutput(_load_text_encoder(
            clip_paths, clip_type, _load_clip_state_dicts(clip_paths)))


class QuadrupleCLIPLoaderGGUF(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        file_options = _filename_list()
        return io.Schema(
            node_id="QuadrupleCLIPLoaderGGUF",
            display_name="QuadrupleCLIPLoader (GGUF)",
            category="IKBlue",
            description="Load four text encoders in GGUF format.",
            inputs=[
                io.Combo.Input("clip_name1", options=file_options),
                io.Combo.Input("clip_name2", options=file_options),
                io.Combo.Input("clip_name3", options=file_options),
                io.Combo.Input("clip_name4", options=file_options),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip_name1, clip_name2, clip_name3, clip_name4,
                type="stable_diffusion") -> io.NodeOutput:
        clip_paths = (folder_paths.get_full_path("clip", clip_name1),
                      folder_paths.get_full_path("clip", clip_name2),
                      folder_paths.get_full_path("clip", clip_name3),
                      folder_paths.get_full_path("clip", clip_name4))
        clip_type = _clip_type(type)
        return io.NodeOutput(_load_text_encoder(
            clip_paths, clip_type, _load_clip_state_dicts(clip_paths)))


V3_NODES = [
    UnetLoaderGGUF,
    UnetLoaderGGUFAdvanced,
    CLIPLoaderGGUF,
    DualCLIPLoaderGGUF,
    TripleCLIPLoaderGGUF,
    QuadrupleCLIPLoaderGGUF,
]


class GGUFCustomNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return V3_NODES


async def comfy_entrypoint() -> ComfyExtension:
    return GGUFCustomNodesExtension()
