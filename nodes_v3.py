# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Modified 2026 by IKBlue: migrated to the ComfyUI V3 node schema
# (comfy_api.latest: io.Schema / io.ComfyNode / comfy_entrypoint).
#
# See https://docs.comfy.org/custom-nodes/v3_migration

"""ComfyUI V3 (``comfy_api.latest``) definitions for the GGUF loader nodes.

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

Every ``node_id`` and ``display_name`` matches the corresponding V1 class in
:mod:`nodes`, so existing workflows keep resolving after the migration.

Import order and registration
-----------------------------
ComfyUI's custom-node loader checks ``NODE_CLASS_MAPPINGS`` **first** and returns
immediately, so the ``comfy_entrypoint`` branch is only reached when that
attribute is ``None`` or absent. :mod:`__init__` therefore sets
``NODE_CLASS_MAPPINGS = None`` when this module imports cleanly, and only falls
back to the V1 mappings when ``comfy_api.latest`` is unavailable. Exposing both
would silently keep running V1.

The V1 classes stay in :mod:`nodes` (which this module imports for its shared
helpers and for the core type lists) because custom-node discovery only loads
top-level entries of ``custom_nodes/`` -- ``nodes.py`` as a submodule is never
registered on its own, so there is no double registration.

Attribution: derived from the V1 node classes in upstream ComfyUI-GGUF by City96
(https://github.com/city96/ComfyUI-GGUF), Apache-2.0.
"""
import torch
import logging
import inspect
import collections

import nodes
import comfy.sd
import comfy.lora
import comfy.float
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import folder_paths

from comfy_api.latest import ComfyExtension, io

from .ops import GGMLOps, move_patch_to_device
from .loader import gguf_sd_loader, gguf_clip_loader
from .dequant import is_quantized, is_torch_compatible


def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")

# Add a custom keys for files ending in .gguf
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])


class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
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


def _clip_type_options():
    """The same list the core CLIPLoader exposes."""
    return nodes.CLIPLoader.INPUT_TYPES()["required"]["type"][0]


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
    def load_data(cls, ckpt_paths, vision_path=None):
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

    @classmethod
    def load_patcher(cls, clip_paths, clip_type, clip_data):
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

    @classmethod
    def execute(cls, clip_name, type="stable_diffusion", vision_name="none") -> io.NodeOutput:
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        vision_path = None
        if vision_name not in (None, "", "none"):
            vision_path = (folder_paths.get_full_path("clip", vision_name)
                           or folder_paths.get_full_path("clip_gguf", vision_name))
        return io.NodeOutput(cls.load_patcher([clip_path], clip_type,
                                              cls.load_data([clip_path], vision_path=vision_path)))


class DualCLIPLoaderGGUF(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        file_options = CLIPLoaderGGUF.get_filename_list()
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
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return io.NodeOutput(CLIPLoaderGGUF.load_patcher(clip_paths, clip_type,
                                                         CLIPLoaderGGUF.load_data(clip_paths)))


class TripleCLIPLoaderGGUF(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        file_options = CLIPLoaderGGUF.get_filename_list()
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
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return io.NodeOutput(CLIPLoaderGGUF.load_patcher(clip_paths, clip_type,
                                                         CLIPLoaderGGUF.load_data(clip_paths)))


class QuadrupleCLIPLoaderGGUF(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        file_options = CLIPLoaderGGUF.get_filename_list()
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
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return io.NodeOutput(CLIPLoaderGGUF.load_patcher(clip_paths, clip_type,
                                                         CLIPLoaderGGUF.load_data(clip_paths)))


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
