# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Modified 2026 by IKBlue: exports the V3 entry point only.
#
# SPDX-License-Identifier: Apache-2.0

"""ComfyUI-GGUF-qwen35 -- GGUF loaders in the ComfyUI V3 node schema.

ComfyUI's custom-node loader checks ``NODE_CLASS_MAPPINGS`` **first** and returns
immediately; the ``comfy_entrypoint`` branch is an ``elif`` and is therefore only
reached when that attribute is ``None`` or absent. This pack is V3-only, so the
name is left **undefined** here -- see the note below for why it must not be set
to ``None`` either.

Requires a ComfyUI build providing ``comfy_api.latest``. There is deliberately no
V1 fallback: on an older build the import below fails and ComfyUI reports the
pack as unloadable, rather than quietly registering nodes from a second, V1
implementation that would have to be maintained in parallel.
"""

from .nodes import V3_NODES, comfy_entrypoint

# NOTE: do NOT define NODE_CLASS_MAPPINGS here -- not even as None.
#
# ComfyUI's loader (nodes.py) tests `getattr(module, "NODE_CLASS_MAPPINGS", None)
# is not None` and only falls through to comfy_entrypoint when that is false, so
# leaving the name undefined is what selects the V3 branch; setting it to None
# works there too.
#
# But ComfyUI-HotReloadHack does
#     getattr(module, "NODE_CLASS_MAPPINGS", {}).keys()
# where the {} default only applies when the attribute is *absent*. An explicit
# None raises `'NoneType' object has no attribute 'keys'`, aborting its reload
# and leaving its class bookkeeping out of sync -- so the name must be omitted
# rather than set to None.

__all__ = [
    "comfy_entrypoint",
    "V3_NODES",
]
