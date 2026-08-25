from __future__ import annotations

from importlib import import_module
import warnings


CAPABILITY_MODULES = (
    ("tts", "tts_nodes"),
    ("asr", "asr_nodes"),
    ("heygem", "heygem_nodes"),
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
CAPABILITY_STATUS = {}
_LOADED_CAPABILITY_MODULES = {}


def _load_capability(capability: str, module_name: str) -> None:
    try:
        module = import_module(f".{module_name}", package=__package__)
        classes = dict(getattr(module, "NODE_CLASS_MAPPINGS", {}))
        displays = dict(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))
        duplicate_nodes = sorted(set(classes).intersection(NODE_CLASS_MAPPINGS))
        if duplicate_nodes:
            raise RuntimeError(
                f"hydra_inferworks_duplicate_node_types:{','.join(duplicate_nodes)}"
            )
        NODE_CLASS_MAPPINGS.update(classes)
        NODE_DISPLAY_NAME_MAPPINGS.update(displays)
        _LOADED_CAPABILITY_MODULES[capability] = module
        CAPABILITY_STATUS[capability] = {
            "available": True,
            "module": module_name,
            "node_types": sorted(classes),
            "error": None,
        }
    except Exception as error:
        CAPABILITY_STATUS[capability] = {
            "available": False,
            "module": module_name,
            "node_types": [],
            "error": f"{type(error).__name__}: {error}",
        }
        warnings.warn(
            f"Hydra InferWorks capability '{capability}' is unavailable: {error}",
            RuntimeWarning,
            stacklevel=2,
        )


for _capability, _module_name in CAPABILITY_MODULES:
    _load_capability(_capability, _module_name)


def require_capability(capability: str):
    status = CAPABILITY_STATUS.get(capability)
    if not status or status["available"] is not True:
        detail = status["error"] if status else "unknown capability"
        raise RuntimeError(f"hydra_inferworks_capability_unavailable:{capability}:{detail}")
    return _LOADED_CAPABILITY_MODULES[capability]


_heygem_module = _LOADED_CAPABILITY_MODULES.get("heygem")
if _heygem_module is not None and hasattr(_heygem_module, "comfy_entrypoint"):
    comfy_entrypoint = _heygem_module.comfy_entrypoint

