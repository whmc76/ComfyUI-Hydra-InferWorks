if __package__:
    from . import nodes as _nodes

    NODE_CLASS_MAPPINGS = _nodes.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = _nodes.NODE_DISPLAY_NAME_MAPPINGS
    CAPABILITY_STATUS = _nodes.CAPABILITY_STATUS

    __all__ = [
        "NODE_CLASS_MAPPINGS",
        "NODE_DISPLAY_NAME_MAPPINGS",
        "CAPABILITY_STATUS",
    ]

    if hasattr(_nodes, "comfy_entrypoint"):
        comfy_entrypoint = _nodes.comfy_entrypoint
        __all__.append("comfy_entrypoint")
else:
    # Pytest may collect a custom-node repository root as the top-level
    # ``__init__`` module. ComfyUI always loads this file as a package.
    __all__ = []

