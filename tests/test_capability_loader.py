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
                "HydraQwen3ASRModelLoader",
                "HydraQwen3LongAsrTranscribe",
                "HydraQwen3ForcedAlign",
                "HydraTranscriptReceipt",
                "HydraHeyGemLongformAvatar",
            },
        )

    def test_inference_profiles_keep_unverified_quantization_out_of_production(self):
        profiles = json.loads((ROOT / "inference-profiles.v1.json").read_text(encoding="utf-8"))
        self.assertEqual(profiles["tts"]["production_default"], "quality_bf16")
        self.assertEqual(profiles["asr"]["production_default"], "transformers_bf16_sdpa")
        self.assertFalse(profiles["truth_rules"]["bf16_is_quantization"])
        self.assertEqual(profiles["asr"]["quantized_candidates"]["status"], "not_admitted")

    def test_asr_production_attestations_pin_exact_official_files(self):
        manifest = json.loads((ROOT / "asr-model-attestations.v1.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["contract_version"], "hydra_inferworks_qwen3_asr_model_attestations.v1")
        self.assertEqual(len(manifest["models"]), 18)
        self.assertEqual(
            {entry["model_id"] for entry in manifest["models"]},
            {"Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ForcedAligner-0.6B"},
        )
        self.assertTrue(all(len(entry["sha256"]) == 64 for entry in manifest["models"]))

    def test_public_workflow_examples_use_portable_placeholders(self):
        examples = ROOT / "workflow-examples"
        rendered = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(examples.glob("*.json"))
        )
        self.assertNotIn("__HYDRA_", rendered)
        self.assertNotIn("hydramatrix/", rendered)

        heygem = json.loads(
            (examples / "hydra-heygem-longform-avatar.api.json").read_text(
                encoding="utf-8"
            )
        )
        inputs = heygem["3"]["inputs"]
        self.assertEqual(inputs["service_url"], "__INFERWORKS_HEYGEM_SERVICE_URL__")
        self.assertEqual(inputs["shared_host_root"], "auto")
        self.assertEqual(inputs["container_data_root"], "auto")
        self.assertEqual(inputs["container_name"], "auto")
        self.assertFalse(inputs["release_service_gpu_after"])
        self.assertEqual(inputs["service_gpu_release_path"], "auto")


if __name__ == "__main__":
    unittest.main()
