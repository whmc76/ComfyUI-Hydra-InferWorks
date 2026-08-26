import hashlib
import json
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
_MODULE._QWEN_ASR_RUNTIME_VERSION = "0.0.6"
HydraQwen3ForcedAlign = _MODULE.HydraQwen3ForcedAlign
HydraQwen3ASRModelLoader = _MODULE.HydraQwen3ASRModelLoader
HydraQwen3LongAsrTranscribe = _MODULE.HydraQwen3LongAsrTranscribe
HydraTranscriptReceipt = _MODULE.HydraTranscriptReceipt


class _Item:
    def __init__(self, text, start_time, end_time):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


class _Batch(dict):
    def to(self, *_args, **_kwargs):
        return self


class _Thinker:
    def __init__(self, aligner):
        self.aligner = aligner
        self.config = SimpleNamespace(classify_num=5000)

    def __call__(self, **inputs):
        slot_count = inputs["input_ids"].shape[1]
        logits = torch.full((1, slot_count, 5000), -100.0, dtype=torch.float32)
        duration_bins = int(self.aligner.normalized_sample_count / 16000 * 1000 // 80)
        word_count = slot_count // 2
        for word_index in range(word_count):
            start_bin = max(0, int(duration_bins * word_index / word_count))
            end_bin = max(start_bin + 1, int(duration_bins * (word_index + 1) / word_count))
            end_bin = min(4999, end_bin)
            logits[0, word_index * 2, start_bin] = 10.0
            logits[0, word_index * 2 + 1, end_bin] = 10.0
        if self.aligner.raw_overflow:
            logits[0, 0, min(4998, duration_bins + 10)] = 20.0
            logits[0, 1, min(4999, duration_bins + 11)] = 20.0
        return SimpleNamespace(logits=logits)


class _Aligner:
    def __init__(self, raw_overflow=False):
        self.raw_overflow = raw_overflow
        self.normalized_sample_count = 0
        self.timestamp_token_id = 151705
        self.timestamp_segment_time = 80
        self.aligner_processor = SimpleNamespace(
            fix_timestamp=lambda values: list(values),
            encode_timestamp=self._encode_timestamp,
        )
        thinker = _Thinker(self)
        self.model = SimpleNamespace(
            thinker=thinker,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.processor = self._processor

    def _inferworks_normalize_audios(self, audio):
        waveform, sample_rate = audio
        duration_seconds = len(waveform) / sample_rate
        self.normalized_sample_count = max(1, round(duration_seconds * 16000))
        return [torch.zeros(self.normalized_sample_count).numpy()], 16000

    def _processor(self, text, audio, return_tensors, padding):
        del text, audio, return_tensors, padding
        word_count = len(self.aligner_processor.encode_timestamp_result) if hasattr(
            self.aligner_processor, "encode_timestamp_result"
        ) else None
        if word_count is None:
            raise AssertionError("encode_timestamp_result_missing")
        return _Batch({
            "input_ids": torch.full((1, word_count * 2), self.timestamp_token_id, dtype=torch.int64)
        })

    def _encode_timestamp(self, text, _language):
        words = [character for character in text if character.isalnum()]
        self.aligner_processor.encode_timestamp_result = words
        return words, "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"

    def align(self, *_args, **_kwargs):
        raise AssertionError("upstream align fallback must never execute")


class _OverflowAligner(_Aligner):
    def __init__(self):
        super().__init__(raw_overflow=True)


class _AsrModel:
    def __init__(self):
        self.calls = 0
        self.forced_aligner = SimpleNamespace(
            aligner_processor=SimpleNamespace(fix_timestamp=lambda values: list(values))
        )

    def transcribe(self, audio, language, context, return_time_stamps):
        self.calls += 1
        waveform, sample_rate = audio
        duration = len(waveform) / sample_rate
        index = self.calls
        item = _Item(f"片段{index}", 0.0, duration)
        return [SimpleNamespace(text=f"片段{index}", language=language or "Chinese", time_stamps=[item])]


class HydraAudioNodeTests(unittest.TestCase):
    def test_monotonic_logit_decoder_selects_global_best_discrete_path(self):
        logits = torch.full((4, 5), -100.0, dtype=torch.float32)
        logits[0, 0] = 10.0
        logits[1, 3] = 10.0
        logits[1, 2] = 9.0
        logits[2, 2] = 10.0
        logits[2, 3] = 1.0
        logits[3, 4] = 10.0

        bins, evidence = _MODULE._decode_monotonic_timestamp_logits(
            logits,
            word_count=2,
            duration_seconds=0.32,
            timestamp_segment_ms=80,
        )

        self.assertEqual(bins, [0, 2, 2, 4])
        self.assertEqual(evidence["raw_greedy_bins"], [0, 3, 2, 4])
        self.assertEqual(
            evidence["raw_argmax_sha256"],
            hashlib.sha256(b"[0,3,2,4]").hexdigest(),
        )
        self.assertFalse(evidence["raw_constraints_satisfied"])
        self.assertTrue(evidence["selected_constraints_satisfied"])
        self.assertEqual(evidence["constrained_token_sha256"], evidence["selected_bins_sha256"])
        self.assertEqual(evidence["changed_position_count"], 1)
        self.assertFalse(evidence["lis_used"])
        self.assertFalse(evidence["interpolation_used"])
        self.assertFalse(evidence["scale_used"])
        self.assertFalse(evidence["averaging_used"])
        self.assertFalse(evidence["estimated_timestamps"])

    def test_monotonic_logit_decoder_uses_earliest_bin_for_equal_scores(self):
        bins, _evidence = _MODULE._decode_monotonic_timestamp_logits(
            torch.zeros((2, 4), dtype=torch.float32),
            word_count=1,
            duration_seconds=0.24,
            timestamp_segment_ms=80,
        )
        self.assertEqual(bins, [0, 1])

    def test_monotonic_logit_decoder_fails_closed_on_invalid_or_infeasible_inputs(self):
        cases = (
            (
                "timestamp_logits_non_finite",
                torch.tensor([[0.0, float("nan")], [0.0, 1.0]]),
                1,
                0.08,
                80,
            ),
            (
                "timestamp_slot_count_invalid",
                torch.zeros((3, 4)),
                1,
                0.24,
                80,
            ),
            (
                "timestamp_path_infeasible",
                torch.zeros((4, 4)),
                2,
                0.08,
                80,
            ),
            (
                "timestamp_segment_time_invalid",
                torch.zeros((2, 4)),
                1,
                0.24,
                40,
            ),
        )
        for error_code, logits, word_count, duration_seconds, segment_ms in cases:
            with self.subTest(error_code=error_code):
                with self.assertRaisesRegex(ValueError, error_code):
                    _MODULE._decode_monotonic_timestamp_logits(
                        logits,
                        word_count=word_count,
                        duration_seconds=duration_seconds,
                        timestamp_segment_ms=segment_ms,
                    )

    def test_constrained_alignment_fails_closed_on_qwen_version_or_structure_drift(self):
        aligner = _Aligner()
        audio = (torch.zeros(16000).numpy(), 16000)
        original_version = _MODULE._QWEN_ASR_RUNTIME_VERSION
        try:
            _MODULE._QWEN_ASR_RUNTIME_VERSION = "0.0.7"
            with self.assertRaisesRegex(RuntimeError, "qwen_asr_version_invalid"):
                _MODULE._align_with_monotonic_timestamp_logits(
                    aligner, audio, "甲", "Chinese"
                )
        finally:
            _MODULE._QWEN_ASR_RUNTIME_VERSION = original_version
        aligner.model.thinker.config.classify_num = 4999
        with self.assertRaisesRegex(RuntimeError, "runtime_structure_invalid"):
            _MODULE._align_with_monotonic_timestamp_logits(
                aligner, audio, "甲", "Chinese"
            )

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
        text, language, timestamps, metadata_json, evidence = HydraQwen3LongAsrTranscribe().transcribe(
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
        self.assertEqual(evidence.to_payload()["operation"], "transcribe")

    def test_long_locked_script_alignment_chunks_then_restores_global_timeline(self):
        audio = {
            "waveform": torch.zeros((1, 1, 2000), dtype=torch.float32),
            "sample_rate": 10,
        }
        locked_text = "甲乙丙丁戊己庚辛"
        text, language, timestamps, metadata_json, evidence = HydraQwen3ForcedAlign().align(
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
        self.assertTrue(lines[0].startswith("0.000000-"))
        self.assertTrue(lines[-1].endswith("-200.000000: 辛"))
        self.assertEqual(__import__("json").loads(metadata_json)["actual_chunk_count"], 4)
        metadata = __import__("json").loads(metadata_json)
        self.assertEqual(
            metadata["timestamp_provenance"],
            "qwen3_forced_aligner_logits_monotonic_viterbi",
        )
        self.assertFalse(metadata["estimated_timestamps"])
        self.assertTrue(all(chunk["timestamp_transform"] == "offset_only" for chunk in metadata["chunks"]))
        self.assertTrue(all(
            chunk["timestamp_decode"]["selected_constraints_satisfied"]
            for chunk in metadata["chunks"]
        ))
        raw_hashes = [
            chunk["timestamp_decode"]["raw_argmax_sha256"]
            for chunk in metadata["chunks"]
        ]
        constrained_hashes = [
            chunk["timestamp_decode"]["constrained_token_sha256"]
            for chunk in metadata["chunks"]
        ]
        expected_raw_aggregate = hashlib.sha256(
            json.dumps(raw_hashes, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        expected_constrained_aggregate = hashlib.sha256(
            json.dumps(constrained_hashes, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(metadata["raw_argmax_sha256"], expected_raw_aggregate)
        self.assertEqual(metadata["constrained_token_sha256"], expected_constrained_aggregate)
        evidence_payload = evidence.to_payload()
        self.assertEqual(evidence_payload["operation"], "forced_alignment")
        self.assertEqual(evidence_payload["raw_argmax_sha256"], expected_raw_aggregate)
        self.assertEqual(evidence_payload["constrained_token_sha256"], expected_constrained_aggregate)
        expected_language_sha256 = hashlib.sha256(b"Chinese").hexdigest()
        for chunk in metadata["chunks"]:
            decode = chunk["timestamp_decode"]
            self.assertEqual(decode["text_sha256"], chunk["locked_text_sha256"])
            self.assertEqual(decode["language_sha256"], expected_language_sha256)
            self.assertRegex(decode["normalized_audio_content_sha256"], r"^[a-f0-9]{64}$")
            self.assertEqual(
                decode["raw_argmax_sha256"],
                hashlib.sha256(
                    json.dumps(decode["raw_greedy_bins"], separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            )
            self.assertEqual(decode["constrained_token_sha256"], decode["selected_bins_sha256"])

    def test_long_locked_script_alignment_rejects_non_exact_anchor_mapping_even_with_legacy_tolerance(self):
        audio = {
            "waveform": torch.zeros((1, 1, 2000), dtype=torch.float32),
            "sample_rate": 10,
        }
        with self.assertRaisesRegex(ValueError, "anchor_text_not_exact"):
            HydraQwen3ForcedAlign().align(
                SimpleNamespace(forced_aligner=_Aligner()),
                audio,
                "甲乙丙丁",
                "Chinese",
                anchor_text="甲错丙丁",
                anchor_timestamps="0-100: 甲错\n100-200: 丙丁",
                max_chunk_seconds=150,
                maximum_anchor_character_error_rate=1.0,
            )

    def test_forced_alignment_constrains_overflow_logits_without_scaling(self):
        audio = {
            "waveform": torch.zeros((1, 1, 10), dtype=torch.float32),
            "sample_rate": 10,
        }
        _text, _language, timestamps, metadata_json, _evidence = HydraQwen3ForcedAlign().align(
            SimpleNamespace(forced_aligner=_OverflowAligner()),
            audio,
            "甲",
            "Chinese",
        )
        metadata = __import__("json").loads(metadata_json)
        decode = metadata["chunks"][0]["timestamp_decode"]
        self.assertEqual(timestamps, "0.000000-0.960000: 甲")
        self.assertFalse(decode["raw_constraints_satisfied"])
        self.assertTrue(decode["selected_constraints_satisfied"])
        self.assertFalse(decode["scale_used"])
        self.assertTrue(_MODULE._constrained_decode_evidence_valid(decode))
        for field in ("raw_argmax_sha256", "constrained_token_sha256"):
            tampered = dict(decode)
            tampered[field] = "0" * 64
            self.assertFalse(_MODULE._constrained_decode_evidence_valid(tampered), field)
        for field in (
            "normalized_audio_content_sha256",
            "text_sha256",
            "language_sha256",
        ):
            tampered = dict(decode)
            tampered[field] = "invalid"
            self.assertFalse(_MODULE._constrained_decode_evidence_valid(tampered), field)

    def test_upstream_non_monotonic_timestamp_tokens_fail_instead_of_provider_repair(self):
        model = SimpleNamespace(forced_aligner=_Aligner())
        original = model.forced_aligner.aligner_processor.fix_timestamp
        _MODULE._install_strict_timestamp_guard(model)
        self.assertIs(model.forced_aligner.aligner_processor._inferworks_original_fix_timestamp, original)
        with self.assertRaisesRegex(ValueError, "upstream_timestamp_repair_forbidden"):
            model.forced_aligner.aligner_processor.fix_timestamp([0, 80, 40, 120])

    def test_long_locked_script_alignment_fails_closed_when_asr_anchors_disagree(self):
        audio = {
            "waveform": torch.zeros((1, 1, 2000), dtype=torch.float32),
            "sample_rate": 10,
        }
        with self.assertRaisesRegex(ValueError, "anchor_text_not_exact"):
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
        self.assertFalse(evidence["timestamp_offsets_preserved"])
        self.assertEqual(payload["status"], "completed_unverified")
        self.assertEqual(evidence["source_duration_seconds"], 200.0)

    def test_typed_forced_alignment_receipt_admits_only_constrained_logit_timing(self):
        audio = {"waveform": torch.zeros((1, 1, 16), dtype=torch.float32), "sample_rate": 16}
        profile = {
            "contract_version": "hydra_inferworks_asr_inference_profile.v1",
            "profile_key": "transformers_bf16_sdpa",
            "backend": "transformers",
            "precision": "bf16",
            "quantization": "none",
            "attention_backend": "sdpa",
            "quality_admission": "production",
        }
        model = SimpleNamespace(
            forced_aligner=_Aligner(),
            _hydra_model_identity={
                "model_id": "Qwen/Qwen3-ASR-1.7B",
                "aligner_id": "Qwen/Qwen3-ForcedAligner-0.6B",
            },
            _hydra_inference_profile=profile,
        )
        text, language, timestamps, metadata, evidence = HydraQwen3ForcedAlign().align(
            model, audio, "甲乙", "Chinese"
        )
        result = HydraTranscriptReceipt().write_receipt(
            text,
            language,
            timestamps,
            "inferworks/test/exact-forced-alignment",
            "f" * 64,
            "Qwen/Qwen3-ASR-1.7B",
            "Qwen/Qwen3-ForcedAligner-0.6B",
            audio=audio,
            processing_mode="qwen3_forced_alignment_exact_anchor_chunk_merge",
            strict_alignment=True,
            execution_evidence=evidence,
            processing_metadata=metadata,
        )
        payload = __import__("json").loads(Path(result["result"][0]).read_text(encoding="utf-8"))
        timing = payload["long_audio_processing"]
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(
            timing["timestamp_provenance"],
            "qwen3_forced_aligner_logits_monotonic_viterbi",
        )
        self.assertFalse(timing["estimated_timestamps"])
        self.assertEqual(timing["timestamp_transform"], "offset_only")
        self.assertFalse(timing["execution_details"]["interpolation_used"])
        self.assertFalse(timing["execution_details"]["lis_used"])
        chunk_decode = timing["execution_details"]["chunks"][0]["timestamp_decode"]
        self.assertEqual(
            timing["execution_details"]["raw_argmax_sha256"],
            chunk_decode["raw_argmax_sha256"],
        )
        self.assertEqual(
            timing["execution_details"]["constrained_token_sha256"],
            chunk_decode["constrained_token_sha256"],
        )
        self.assertEqual(
            timing["execution_evidence"]["raw_argmax_sha256"],
            chunk_decode["raw_argmax_sha256"],
        )
        self.assertEqual(
            timing["execution_evidence"]["constrained_token_sha256"],
            chunk_decode["constrained_token_sha256"],
        )
        self.assertTrue(timing["timestamp_offsets_preserved"])

        for field, target in (
            ("raw_argmax_sha256", "execution"),
            ("text_sha256", "chunk"),
            ("language_sha256", "chunk"),
        ):
            with self.subTest(field=field):
                tampered_metadata = json.loads(metadata)
                if target == "execution":
                    tampered_metadata[field] = "0" * 64
                else:
                    tampered_metadata["chunks"][0]["timestamp_decode"][field] = "0" * 64
                with self.assertRaisesRegex(
                    ValueError,
                    "constrained_decode_evidence_required",
                ):
                    HydraTranscriptReceipt().write_receipt(
                        text,
                        language,
                        timestamps,
                        f"inferworks/test/tampered-{field}",
                        "f" * 64,
                        "Qwen/Qwen3-ASR-1.7B",
                        "Qwen/Qwen3-ForcedAligner-0.6B",
                        audio=audio,
                        processing_mode="qwen3_forced_alignment_exact_anchor_chunk_merge",
                        strict_alignment=True,
                        execution_evidence=evidence,
                        processing_metadata=json.dumps(tampered_metadata, ensure_ascii=False),
                    )

    def test_receipt_rejects_malformed_overlapping_out_of_bounds_or_text_mismatched_timestamps(self):
        audio = {"waveform": torch.zeros((1, 1, 16), dtype=torch.float32), "sample_rate": 16}
        cases = {
            "timestamp_line_invalid": "not-a-timestamp",
            "timestamp_overlap_invalid": "0-0.75: 甲\n0.5-1: 乙",
            "timestamp_out_of_bounds": "0-2: 甲乙",
            "timestamp_text_not_exact": "0-1: 甲错",
        }
        for error_code, timestamps in cases.items():
            with self.subTest(error_code=error_code):
                with self.assertRaisesRegex(ValueError, error_code):
                    HydraTranscriptReceipt().write_receipt(
                        "甲乙",
                        "Chinese",
                        timestamps,
                        f"inferworks/test/{error_code}",
                        "a" * 64,
                        "Qwen/Qwen3-ASR-1.7B",
                        "Qwen/Qwen3-ForcedAligner-0.6B",
                        audio=audio,
                    )

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
            "timestamp_provenance": "qwen3_forced_aligner_native",
            "estimated_timestamps": False,
            "timestamp_transform": "offset_only",
            "upstream_timestamp_repair_policy": "reject_non_monotonic_raw_tokens",
            "chunks": [{"timestamp_transform": "offset_only", "estimated_timestamps": False}],
        }
        audio = {
            "waveform": torch.zeros((1, 1, 16), dtype=torch.float32),
            "sample_rate": 16,
        }
        waveform, sample_rate = _MODULE._audio_tuple(audio)
        model = SimpleNamespace(
            _hydra_model_identity={
                "model_id": "Qwen/Qwen3-ASR-1.7B",
                "aligner_id": "Qwen/Qwen3-ForcedAligner-0.6B",
            },
            _hydra_inference_profile=metadata["inference_profile"],
        )
        evidence = _MODULE._execution_evidence(
            model,
            waveform,
            sample_rate,
            "甲",
            "Chinese",
            "0-1: 甲",
            "transcribe",
            1.0,
            metadata,
        )
        result = HydraTranscriptReceipt().write_receipt(
            "甲",
            "Chinese",
            "0-1: 甲",
            "hydramatrix/test/inference-profile",
            "b" * 64,
            "Qwen/Qwen3-ASR-1.7B",
            "Qwen/Qwen3-ForcedAligner-0.6B",
            audio=audio,
            execution_evidence=evidence,
            processing_metadata=__import__("json").dumps(metadata),
        )
        payload = __import__("json").loads(Path(result["result"][0]).read_text(encoding="utf-8"))
        self.assertEqual(payload["inference_profile"]["precision"], "bf16")
        self.assertEqual(payload["inference_profile"]["attention_backend"], "sdpa")
        self.assertEqual(payload["inference_profile"]["quantization"], "none")
        self.assertEqual(payload["status"], "completed")

    def test_receipt_rejects_literal_dict_and_operation_mismatch(self):
        audio = {"waveform": torch.zeros((1, 1, 16), dtype=torch.float32), "sample_rate": 16}
        metadata = {"contract_version": "hydra_qwen3_long_asr_execution.v1"}
        with self.assertRaisesRegex(ValueError, "execution_evidence_invalid"):
            HydraTranscriptReceipt().write_receipt(
                "甲", "Chinese", "0-1: 甲", "hydramatrix/test/forged", "d" * 64,
                "Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ForcedAligner-0.6B",
                audio=audio,
                execution_evidence={"quality_admission": "production"},
                processing_metadata=__import__("json").dumps(metadata),
            )
        waveform, sample_rate = _MODULE._audio_tuple(audio)
        profile = {
            "contract_version": "hydra_inferworks_asr_inference_profile.v1",
            "profile_key": "transformers_bf16_sdpa",
            "precision": "bf16",
            "quality_admission": "production",
        }
        model = SimpleNamespace(
            _hydra_model_identity={"model_id": "Qwen/Qwen3-ASR-1.7B", "aligner_id": "Qwen/Qwen3-ForcedAligner-0.6B"},
            _hydra_inference_profile=profile,
        )
        evidence = _MODULE._execution_evidence(
            model, waveform, sample_rate, "甲", "Chinese", "0-1: 甲", "transcribe", 1.0, metadata,
        )
        with self.assertRaisesRegex(ValueError, "execution_operation_mismatch"):
            HydraTranscriptReceipt().write_receipt(
                "甲", "Chinese", "0-1: 甲", "hydramatrix/test/operation", "e" * 64,
                "Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ForcedAligner-0.6B",
                audio=audio,
                execution_evidence=evidence,
                processing_metadata=__import__("json").dumps(metadata),
                processing_mode="qwen3_forced_alignment_anchor_chunk_merge",
                strict_alignment=True,
            )

    def test_receipt_without_typed_execution_evidence_is_unverified(self):
        result = HydraTranscriptReceipt().write_receipt(
            "甲",
            "Chinese",
            "0-1: 甲",
            "hydramatrix/test/unverified-profile",
            "c" * 64,
            "Qwen/Qwen3-ASR-1.7B",
            "Qwen/Qwen3-ForcedAligner-0.6B",
            processing_metadata=__import__("json").dumps({"inference_profile": {"quality_admission": "production"}}),
        )
        payload = __import__("json").loads(Path(result["result"][0]).read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "completed_unverified")
        self.assertIsNone(payload["inference_profile"])


if __name__ == "__main__":
    unittest.main()
