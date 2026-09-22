# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Modified 2026 by IKBlue: V3 entry point with a V1 fallback.
#
# ComfyUI's custom-node loader (nodes.py) checks these in order and returns
# after the FIRST match:
#
#   1. NODE_CLASS_MAPPINGS is not None  -> register V1 nodes, stop
#   2. comfy_entrypoint exists          -> register V3 nodes (elif)
#
# so the two paths are mutually exclusive.  We therefore expose V3 whenever the
# versioned Comfy API is importable, and fall back to the legacy V1 classes
# otherwise.  Setting NODE_CLASS_MAPPINGS to None (rather than omitting it) is
# what lets the "elif comfy_entrypoint" branch run on newer ComfyUI builds.
try:
    import comfy.utils
except ImportError:
    pass
else:
    __all__ = []

    try:
        from .nodes_v3 import V3_NODES, comfy_entrypoint
    except ImportError as e:
        # Older ComfyUI without comfy_api.latest -> legacy V1 schema only.
        import logging
        logging.info(f"ComfyUI-GGUF: V3 schema unavailable ({e}); using the V1 nodes.")
        from .nodes import NODE_CLASS_MAPPINGS
        NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
        __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
    else:
        # V3 path: NODE_CLASS_MAPPINGS must be None so the loader takes the
        # comfy_entrypoint branch instead.
        NODE_CLASS_MAPPINGS = None
        NODE_DISPLAY_NAME_MAPPINGS = None
        __all__ = ["comfy_entrypoint", "V3_NODES", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
