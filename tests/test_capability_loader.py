import importlib.util
import json
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class CapabilityLoaderTests(unittest.TestCase):
    def test_one_missing_capability_does_not_remove_other_nodes(self):
        modules = {
            ".tts_nodes": SimpleNamespace(
                NODE_CLASS_MAPPINGS={"TopTTS25Synthesize": object},
                NODE_DISPLAY_NAME_MAPPINGS={"TopTTS25Synthesize": "TTS"},
            ),
            ".asr_nodes": SimpleNamespace(
                NODE_CLASS_MAPPINGS={"HydraTranscriptReceipt": object},
                NODE_DISPLAY_NAME_MAPPINGS={"HydraTranscriptReceipt": "ASR"},
            ),
        }

        def fake_import(name, package=None):
            if name == ".heygem_nodes":
                raise ImportError("optional HeyGem dependency unavailable")
            return modules[name]

        spec = importlib.util.spec_from_file_location(
            "hydra_inferworks_test.nodes",
            ROOT / "nodes.py",
        )
        module = importlib.util.module_from_spec(spec)
        with patch("importlib.import_module", side_effect=fake_import), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec.loader.exec_module(module)

        self.assertIn("TopTTS25Synthesize", module.NODE_CLASS_MAPPINGS)
        self.assertIn("HydraTranscriptReceipt", module.NODE_CLASS_MAPPINGS)
        self.assertTrue(module.CAPABILITY_STATUS["tts"]["available"])
        self.assertTrue(module.CAPABILITY_STATUS["asr"]["available"])
        self.assertFalse(module.CAPABILITY_STATUS["heygem"]["available"])

    def test_manifest_preserves_all_public_class_types(self):
        manifest = json.loads((ROOT / "node-manifest.v1.json").read_text(encoding="utf-8"))
        preserved = set(manifest["compatibility"]["preserved_class_types"])
        self.assertEqual(
            preserved,
            {
                "TopTTS25ModelLoader",
                "TopTTS25EmotionVector",
                "TopTTS25Synthesize",
                "TopTTS25UnloadModel",
                "HydraQwen3LongAsrTranscribe",
                "HydraQwen3ForcedAlign",
                "HydraTranscriptReceipt",
                "HydraHeyGemLongformAvatar",
            },
        )


if __name__ == "__main__":
    unittest.main()

