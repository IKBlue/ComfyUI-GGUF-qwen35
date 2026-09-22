# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Modified 2026 by IKBlue: exports the V3 entry point only.
#
# SPDX-License-Identifier: Apache-2.0

"""ComfyUI-GGUF-qwen35 -- GGUF loaders in the ComfyUI V3 node schema.

ComfyUI's custom-node loader checks ``NODE_CLASS_MAPPINGS`` **first** and returns
immediately; the ``comfy_entrypoint`` branch is an ``elif`` and is therefore only
reached when that attribute is ``None`` or absent. Since this pack is V3-only,
``NODE_CLASS_MAPPINGS`` is set to ``None`` here rather than to a mapping -- having
both would silently keep running the V1 classes.

Requires a ComfyUI build providing ``comfy_api.latest``. There is deliberately no
V1 fallback: on an older build the import below fails and ComfyUI reports the
pack as unloadable, rather than quietly registering nodes from a second, V1
implementation that would have to be maintained in parallel.
"""

from .nodes_v3 import V3_NODES, comfy_entrypoint

# Must be None (not a mapping) so the loader takes the comfy_entrypoint branch.
NODE_CLASS_MAPPINGS = None
NODE_DISPLAY_NAME_MAPPINGS = None

__all__ = [
    "comfy_entrypoint",
    "V3_NODES",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
