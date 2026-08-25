import sys
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


_ROOT = Path(__file__).resolve().parents[1]
_MODELS = tempfile.TemporaryDirectory(prefix="hydra-inferworks-tts-models-")
_TEMP = tempfile.TemporaryDirectory(prefix="hydra-inferworks-tts-temp-")
_folder_paths = sys.modules.setdefault("folder_paths", SimpleNamespace())
_folder_paths.models_dir = _MODELS.name
_folder_paths.get_temp_directory = lambda: _TEMP.name
_comfy = sys.modules.setdefault("comfy", ModuleType("comfy"))
_model_management = ModuleType("comfy.model_management")
_model_management.soft_empty_cache = lambda: None
_comfy.model_management = _model_management
sys.modules.setdefault("comfy.model_management", _model_management)

_SPEC = spec_from_file_location("hydra_inferworks_tts_contract", _ROOT / "tts_nodes.py")
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _FakeWorker:
    last_command = None

    def __init__(self, command):
        type(self).last_command = dict(command)
        self.info = {
            "device": "cuda:0",
            "precision": "fp32",
            "quantization": "none",
            "optimization_profile": command["optimization_profile"],
            "acceleration": {
                "gpt_flash_attention": False,
                "torch_compile": True,
                "bigvgan_cuda_kernel": True,
            },
            "runtime_profile": {"profile_key": command["optimization_profile"]},
        }

    def close(self):
        return None


class TtsLoaderCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_dir = Path(_MODELS.name) / _MODULE.DEFAULT_MODEL_DIRECTORY
        for relative in (*_MODULE.MAIN_MODEL_FILES, *_MODULE.AUXILIARY_MODEL_FILES):
            target = model_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()

    def test_missing_profile_preserves_v1_custom_inputs(self):
        with patch.object(_MODULE, "_ensure_runtime"), patch.object(_MODULE, "TopTTSWorker", _FakeWorker):
            model, _ = _MODULE.TopTTS25ModelLoader().load(
                _MODULE.DEFAULT_MODEL_DIRECTORY,
                "auto",
                "fp32",
                False,
                True,
                True,
            )
        command = _FakeWorker.last_command
        self.assertEqual(command["optimization_profile"], "legacy_custom")
        self.assertFalse(command["use_bf16"])
        self.assertTrue(command["use_cuda_kernel"])
        self.assertTrue(command["use_torch_compile"])
        self.assertFalse(command["use_gpt_acceleration"])
        self.assertEqual(model.optimization_profile, "legacy_custom")

    def test_new_graph_ui_default_remains_quality_bf16(self):
        optional = _MODULE.TopTTS25ModelLoader.INPUT_TYPES()["optional"]
        self.assertEqual(optional["optimization_profile"][1]["default"], "quality_bf16")


if __name__ == "__main__":
    unittest.main()
