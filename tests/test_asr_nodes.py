import sys
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import torch


_OUTPUT_ROOT = tempfile.TemporaryDirectory(prefix="hydra-comfyui-audio-node-test-")
sys.modules.setdefault(
    "folder_paths",
    SimpleNamespace(get_output_directory=lambda: _OUTPUT_ROOT.name),
)

_SPEC = spec_from_file_location("hydra_comfyui_audio_nodes", Path(__file__).resolve().parents[1] / "asr_nodes.py")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
HydraQwen3ForcedAlign = _MODULE.HydraQwen3ForcedAlign
HydraQwen3ASRModelLoader = _MODULE.HydraQwen3ASRModelLoader
HydraQwen3LongAsrTranscribe = _MODULE.HydraQwen3LongAsrTranscribe
HydraTranscriptReceipt = _MODULE.HydraTranscriptReceipt


class _Item:
    def __init__(self, text, start_time, end_time):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


class _Aligner:
    def align(self, audio, text, language):
        waveform, sample_rate = audio
        duration = len(waveform) / sample_rate
        characters = list(text)
        items = [
            _Item(character, duration * index / len(characters), duration * (index + 1) / len(characters))
            for index, character in enumerate(characters)
        ]
        return [SimpleNamespace(items=items)]


class _AsrModel:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, language, context, return_time_stamps):
        self.calls += 1
        waveform, sample_rate = audio
        duration = len(waveform) / sample_rate
        index = self.calls
        item = _Item(f"片段{index}", 0.0, duration)
        return [SimpleNamespace(text=f"片段{index}", language=language or "Chinese", time_stamps=[item])]


class HydraAudioNodeTests(unittest.TestCase):
    def test_loader_defaults_to_quality_admitted_bf16_sdpa(self):
        inputs = HydraQwen3ASRModelLoader.INPUT_TYPES()["required"]
        self.assertEqual(inputs["optimization_profile"][1]["default"], "transformers_bf16_sdpa")
        self.assertIn("vllm_bf16", inputs["optimization_profile"][0])

    def test_long_asr_uses_multiple_native_calls_and_restores_global_timeline(self):
        waveform = torch.cat([
            torch.full((900,), 1.0),
            torch.full((900,), 2.0),
            torch.full((200,), 3.0),
        ]).reshape(1, 1, 2000)
        model = _AsrModel()
        text, language, timestamps, metadata_json = HydraQwen3LongAsrTranscribe().transcribe(
            model,
            {"waveform": waveform, "sample_rate": 10},
            language="Chinese",
            max_chunk_seconds=90,
            silence_search_seconds=1,
        )

        metadata = __import__("json").loads(metadata_json)
        self.assertEqual(language, "Chinese")
        self.assertEqual(metadata["actual_chunk_count"], 3)
        self.assertEqual(model.calls, 3)
        self.assertTrue(metadata["automatic_chunking"])
        self.assertEqual(len(timestamps.splitlines()), 3)
        self.assertTrue(timestamps.splitlines()[-1].endswith("片段3"))
        self.assertTrue(text.endswith("片段3"))

    def test_long_locked_script_alignment_chunks_then_restores_global_timeline(self):
        audio = {
            "waveform": torch.zeros((1, 1, 2000), dtype=torch.float32),
            "sample_rate": 10,
        }
        locked_text = "甲乙丙丁戊己庚辛"
        text, language, timestamps, metadata_json = HydraQwen3ForcedAlign().align(
            SimpleNamespace(forced_aligner=_Aligner()),
            audio,
            locked_text,
            "Chinese",
            anchor_text=locked_text,
            anchor_timestamps="0-50: 甲乙\n50-100: 丙丁\n100-150: 戊己\n150-200: 庚辛",
            max_chunk_seconds=80,
        )

        self.assertEqual(text, locked_text)
        self.assertEqual(language, "Chinese")
        lines = timestamps.splitlines()
        self.assertEqual(len(lines), len(locked_text))
        self.assertTrue(lines[0].startswith("0.000-"))
        self.assertTrue(lines[-1].startswith("175.000-200.000:"))
        self.assertEqual(__import__("json").loads(metadata_json)["actual_chunk_count"], 4)

    def test_long_locked_script_alignment_fails_closed_when_asr_anchors_disagree(self):
        audio = {
            "waveform": torch.zeros((1, 1, 2000), dtype=torch.float32),
            "sample_rate": 10,
        }
        with self.assertRaisesRegex(ValueError, "anchor_text_mismatch"):
            HydraQwen3ForcedAlign().align(
                SimpleNamespace(forced_aligner=_Aligner()),
                audio,
                "这是锁定脚本完全不同",
                "Chinese",
                anchor_text="abcdefgh",
                anchor_timestamps="0-100: abcd\n100-200: efgh",
                max_chunk_seconds=150,
                maximum_anchor_character_error_rate=0.2,
            )

    def test_receipt_records_long_audio_chunk_merge_and_strict_alignment(self):
        audio = {
            "waveform": torch.zeros((1, 1, 2000), dtype=torch.float32),
            "sample_rate": 10,
        }
        result = HydraTranscriptReceipt().write_receipt(
            "甲乙",
            "Chinese",
            "0-100: 甲\n100-200: 乙",
            "hydramatrix/test/long-alignment",
            "a" * 64,
            "Qwen/Qwen3-ASR-1.7B",
            "Qwen/Qwen3-ForcedAligner-0.6B",
            audio=audio,
            processing_mode="qwen3_forced_alignment_anchor_chunk_merge",
            max_chunk_seconds=150,
            strict_alignment=True,
        )
        receipt_path = Path(result["result"][0])
        payload = __import__("json").loads(receipt_path.read_text(encoding="utf-8"))
        evidence = payload["long_audio_processing"]
        self.assertTrue(evidence["automatic_chunking"])
        self.assertTrue(evidence["strict_locked_script_alignment"])
        self.assertTrue(evidence["timestamp_offsets_preserved"])
        self.assertEqual(evidence["source_duration_seconds"], 200.0)

    def test_receipt_preserves_runtime_precision_and_acceleration_profile(self):
        metadata = {
            "contract_version": "hydra_qwen3_long_asr_execution.v1",
            "inference_profile": {
                "contract_version": "hydra_inferworks_asr_inference_profile.v1",
                "profile_key": "transformers_bf16_sdpa",
                "backend": "transformers",
                "precision": "bf16",
                "quantization": "none",
                "attention_backend": "sdpa",
                "quality_admission": "production",
            },
        }
        result = HydraTranscriptReceipt().write_receipt(
            "甲",
            "Chinese",
            "0-1: 甲",
            "hydramatrix/test/inference-profile",
            "b" * 64,
            "Qwen/Qwen3-ASR-1.7B",
            "Qwen/Qwen3-ForcedAligner-0.6B",
            processing_metadata=__import__("json").dumps(metadata),
        )
        payload = __import__("json").loads(Path(result["result"][0]).read_text(encoding="utf-8"))
        self.assertEqual(payload["inference_profile"]["precision"], "bf16")
        self.assertEqual(payload["inference_profile"]["attention_backend"], "sdpa")
        self.assertEqual(payload["inference_profile"]["quantization"], "none")


if __name__ == "__main__":
    unittest.main()

