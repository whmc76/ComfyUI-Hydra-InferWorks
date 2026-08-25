import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import folder_paths
import numpy as np
import torch


ASR_OPTIMIZATION_PROFILES = {
    "transformers_bf16_sdpa": {
        "backend": "transformers",
        "precision": "bf16",
        "attention_backend": "sdpa",
        "quantization": "none",
    },
    "transformers_bf16_flash_attention_2": {
        "backend": "transformers",
        "precision": "bf16",
        "attention_backend": "flash_attention_2",
        "quantization": "none",
    },
    "vllm_bf16": {
        "backend": "vllm",
        "precision": "bf16",
        "attention_backend": "vllm_managed",
        "quantization": "none",
    },
    "reference_fp32_eager": {
        "backend": "transformers",
        "precision": "fp32",
        "attention_backend": "eager",
        "quantization": "none",
    },
}
PLUGIN_DIR = Path(__file__).resolve().parent
ASR_MODEL_ATTESTATIONS_PATH = PLUGIN_DIR / "asr-model-attestations.v1.json"
ASR_EXECUTION_EVIDENCE_TYPE = "HYDRA_ASR_EXECUTION_EVIDENCE"
_FILE_HASH_CACHE = {}
_ASR_EVIDENCE_ISSUER = object()
EXPECTED_ASR_ATTESTATION_ROLES = frozenset(
    {
        "asr_model_shard_1",
        "asr_model_shard_2",
        "asr_config",
        "asr_model_index",
        "asr_generation_config",
        "asr_preprocessor_config",
        "asr_tokenizer_config",
        "asr_tokenizer_vocab",
        "asr_tokenizer_merges",
        "asr_chat_template",
        "forced_aligner_model",
        "forced_aligner_config",
        "aligner_generation_config",
        "aligner_preprocessor_config",
        "aligner_tokenizer_config",
        "aligner_tokenizer_vocab",
        "aligner_tokenizer_merges",
        "aligner_chat_template",
    }
)


class HydraAsrExecutionEvidence:
    """Opaque in-process evidence; Comfy API JSON literals cannot construct it."""

    __slots__ = ("_issuer", "_payload")

    def __init__(self, *_args, **_kwargs):
        raise TypeError("HydraAsrExecutionEvidence is issued only by an executed InferWorks node")

    @classmethod
    def _issue(cls, payload):
        instance = object.__new__(cls)
        instance._issuer = _ASR_EVIDENCE_ISSUER
        instance._payload = json.loads(json.dumps(payload, ensure_ascii=False))
        return instance

    def _verified_payload(self):
        if self._issuer is not _ASR_EVIDENCE_ISSUER:
            raise ValueError("hydra_transcript_receipt_execution_evidence_issuer_invalid")
        return json.loads(json.dumps(self._payload, ensure_ascii=False))

    def to_payload(self):
        return self._verified_payload()


def _text(value):
    return str(value or "").strip()


def _resolve_asr_model_directory(value, default_relative_path):
    root = Path(folder_paths.models_dir).resolve()
    raw = _text(value) or default_relative_path
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("hydra_qwen3_asr_model_outside_comfyui_models") from error
    if not candidate.is_dir() or not any(candidate.iterdir()):
        raise FileNotFoundError(f"hydra_qwen3_asr_model_directory_missing:{candidate}")
    return candidate


def _file_sha256(path):
    stat = path.stat()
    cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _FILE_HASH_CACHE.get(cache_key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[cache_key] = value
    return value


def _attest_asr_model_files(model_path, aligner_path):
    try:
        manifest_path = next(
            path
            for path in (
                ASR_MODEL_ATTESTATIONS_PATH,
                Path(sys.prefix) / ASR_MODEL_ATTESTATIONS_PATH.name,
            )
            if path.is_file()
        )
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, StopIteration, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("hydra_qwen3_asr_attestation_manifest_invalid") from error
    root = Path(folder_paths.models_dir).resolve()
    if manifest.get("contract_version") != "hydra_inferworks_qwen3_asr_model_attestations.v1":
        raise RuntimeError("hydra_qwen3_asr_attestation_contract_invalid")
    entries = manifest.get("models", [])
    roles = [_text(entry.get("role")) for entry in entries if isinstance(entry, dict)]
    if len(roles) != len(set(roles)) or set(roles) != EXPECTED_ASR_ATTESTATION_ROLES:
        raise RuntimeError("hydra_qwen3_asr_attestation_roles_invalid")
    expected_model_path = (root / "Qwen3-ASR/Qwen3-ASR-1.7B").resolve()
    expected_aligner_path = (root / "Qwen3-ASR/Qwen3-ForcedAligner-0.6B").resolve()
    if model_path != expected_model_path or aligner_path != expected_aligner_path:
        raise RuntimeError("hydra_qwen3_asr_production_model_identity_untrusted")
    verified = []
    for entry in entries:
        target = (root / _text(entry.get("relative_path"))).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise RuntimeError("hydra_qwen3_asr_attestation_path_escape") from error
        if not target.is_file():
            raise FileNotFoundError(f"hydra_qwen3_asr_attested_file_missing:{target}")
        if target.stat().st_size != int(entry.get("size_bytes", -1)):
            raise RuntimeError(f"hydra_qwen3_asr_attested_file_size_mismatch:{target.name}")
        digest = _file_sha256(target)
        if digest != _text(entry.get("sha256")).lower():
            raise RuntimeError(f"hydra_qwen3_asr_attested_file_sha256_mismatch:{target.name}")
        verified.append({"role": entry.get("role"), "sha256": digest})
    if len(verified) != 18:
        raise RuntimeError("hydra_qwen3_asr_attestation_set_incomplete")
    return {
        "contract_version": "hydra_inferworks_qwen3_asr_runtime_attestation.v1",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "model_id": "Qwen/Qwen3-ASR-1.7B",
        "aligner_id": "Qwen/Qwen3-ForcedAligner-0.6B",
        "files": verified,
    }


def _dtype_name(value):
    normalized = _text(value).replace("torch.", "")
    return {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
    }.get(normalized, normalized or "unknown")


def _config_attention(config):
    return _text(
        getattr(config, "_attn_implementation", None)
        or getattr(config, "_attn_implementation_internal", None)
    ) or "unknown"


def _config_quantization(config):
    value = getattr(config, "quantization_config", None)
    return "none" if value in (None, {}, "") else "configured"


def _observed_asr_runtime(model, profile_key, selected, attestation):
    inner = getattr(model, "model", None)
    inner_config = getattr(inner, "config", None)
    aligner = getattr(model, "forced_aligner", None)
    aligner_inner = getattr(aligner, "model", None)
    aligner_config = getattr(aligner_inner, "config", None)
    aligner_dtype = getattr(aligner_inner, "dtype", None)
    if aligner_dtype is None and aligner_inner is not None:
        try:
            aligner_dtype = next(aligner_inner.parameters()).dtype
        except StopIteration:
            pass
    runtime = {
        "contract_version": "hydra_inferworks_asr_inference_profile.v1",
        "profile_key": profile_key,
        "backend": _text(getattr(model, "backend", None)) or "unknown",
        "precision": _dtype_name(getattr(model, "dtype", None) or getattr(inner, "dtype", None)),
        "aligner_precision": _dtype_name(aligner_dtype),
        "quantization": _config_quantization(inner_config),
        "aligner_quantization": _config_quantization(aligner_config),
        "attention_backend": _config_attention(inner_config),
        "aligner_attention_backend": _config_attention(aligner_config),
        "device": _text(getattr(model, "device", None)),
        "max_inference_batch_size": int(getattr(model, "max_inference_batch_size", 0)),
        "model_attestation": attestation,
        "quality_admission": "candidate",
    }
    production_match = (
        profile_key == "transformers_bf16_sdpa"
        and runtime["backend"] == "transformers"
        and runtime["precision"] == "bf16"
        and runtime["aligner_precision"] == "bf16"
        and runtime["quantization"] == "none"
        and runtime["aligner_quantization"] == "none"
        and runtime["attention_backend"] == "sdpa"
        and runtime["aligner_attention_backend"] == "sdpa"
        and isinstance(attestation, dict)
    )
    if profile_key == "transformers_bf16_sdpa" and not production_match:
        raise RuntimeError("hydra_qwen3_asr_production_runtime_attestation_failed")
    if production_match:
        runtime["quality_admission"] = "production"
    elif profile_key == "reference_fp32_eager":
        runtime["quality_admission"] = "quality_reference"
    return runtime


def _model_inference_profile(model):
    recorded = getattr(model, "_hydra_inference_profile", None)
    if isinstance(recorded, dict):
        return dict(recorded)
    inner = getattr(model, "model", None)
    config = getattr(inner, "config", None)
    dtype = getattr(model, "dtype", None) or getattr(inner, "dtype", None)
    attention = (
        getattr(config, "_attn_implementation", None)
        or getattr(config, "_attn_implementation_internal", None)
        or "unknown"
    )
    return {
        "contract_version": "hydra_inferworks_asr_inference_profile.v1",
        "profile_key": "external_loader_observed",
        "backend": _text(getattr(model, "backend", None)) or "transformers",
        "precision": _text(dtype).replace("torch.", "") or "unknown",
        "quantization": "unknown",
        "attention_backend": _text(attention) or "unknown",
        "quality_admission": "unverified_external_loader",
    }


class HydraQwen3ASRModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_directory": (
                    "STRING",
                    {"default": "Qwen3-ASR/Qwen3-ASR-1.7B"},
                ),
                "forced_aligner_directory": (
                    "STRING",
                    {"default": "Qwen3-ASR/Qwen3-ForcedAligner-0.6B"},
                ),
                "optimization_profile": (
                    list(ASR_OPTIMIZATION_PROFILES),
                    {"default": "transformers_bf16_sdpa"},
                ),
                "max_inference_batch_size": (
                    "INT",
                    {"default": 32, "min": 1, "max": 128},
                ),
            }
        }

    RETURN_TYPES = ("QWEN3_ASR_MODEL", "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load_model"
    CATEGORY = "Hydra InferWorks/ASR"
    DESCRIPTION = "Loads local Qwen3-ASR with an explicit, auditable acceleration profile."

    def load_model(
        self,
        model_directory,
        forced_aligner_directory,
        optimization_profile,
        max_inference_batch_size,
    ):
        from comfy import model_management
        from qwen_asr import Qwen3ASRModel

        profile_key = _text(optimization_profile).lower()
        selected = ASR_OPTIMIZATION_PROFILES.get(profile_key)
        if selected is None:
            raise ValueError(f"hydra_qwen3_asr_optimization_profile_invalid:{profile_key}")
        device = model_management.get_torch_device()
        if selected["precision"] == "bf16":
            if device.type != "cuda" or not torch.cuda.is_bf16_supported():
                raise RuntimeError("hydra_qwen3_asr_bf16_cuda_required")
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        if selected["attention_backend"] == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
            raise RuntimeError("hydra_qwen3_asr_flash_attention_2_required")
        if selected["backend"] == "vllm" and importlib.util.find_spec("vllm") is None:
            raise RuntimeError("hydra_qwen3_asr_vllm_required")

        model_path = _resolve_asr_model_directory(
            model_directory,
            "Qwen3-ASR/Qwen3-ASR-1.7B",
        )
        aligner_path = _resolve_asr_model_directory(
            forced_aligner_directory,
            "Qwen3-ASR/Qwen3-ForcedAligner-0.6B",
        )
        attestation = None
        try:
            attestation = _attest_asr_model_files(model_path, aligner_path)
        except (FileNotFoundError, RuntimeError):
            if profile_key == "transformers_bf16_sdpa":
                raise
        batch_size = max(1, min(128, int(max_inference_batch_size)))
        if selected["backend"] == "vllm":
            model = Qwen3ASRModel.LLM(
                model=str(model_path),
                forced_aligner=str(aligner_path),
                forced_aligner_kwargs={"dtype": dtype, "device_map": str(device)},
                max_inference_batch_size=batch_size,
                dtype="bfloat16",
            )
        else:
            attention = selected["attention_backend"]
            model = Qwen3ASRModel.from_pretrained(
                str(model_path),
                forced_aligner=str(aligner_path),
                forced_aligner_kwargs={
                    "dtype": dtype,
                    "device_map": str(device),
                    "attn_implementation": attention,
                },
                max_inference_batch_size=batch_size,
                max_new_tokens=256,
                dtype=dtype,
                device_map=str(device),
                attn_implementation=attention,
            )
        runtime_profile = _observed_asr_runtime(model, profile_key, selected, attestation)
        model._hydra_inference_profile = runtime_profile
        model._hydra_model_identity = {
            "model_id": attestation.get("model_id") if attestation else None,
            "aligner_id": attestation.get("aligner_id") if attestation else None,
        }
        info = " | ".join(
            (
                (runtime_profile.get("model_attestation") or {}).get("model_id", "unverified-model"),
                profile_key,
                str(device),
                runtime_profile["precision"],
                f"quantization={runtime_profile['quantization']}",
                f"attention={runtime_profile['attention_backend']}",
            )
        )
        return model, info


def _safe_prefix(value):
    raw = _text(value).replace("\\", "/").strip("/")
    parts = [part for part in raw.split("/") if part]
    if not parts or any(part in (".", "..") for part in parts):
        raise ValueError("hydra_transcript_receipt_prefix_invalid")
    clean = [re.sub(r"[^A-Za-z0-9._-]+", "-", part).strip(".-") for part in parts]
    if any(not part for part in clean):
        raise ValueError("hydra_transcript_receipt_prefix_invalid")
    return clean


def _parse_timestamps(value):
    segments = []
    previous_start = -1.0
    for line in _text(value).splitlines():
        match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:\s*(.*?)\s*$", line)
        if not match:
            continue
        start = float(match.group(1))
        end = float(match.group(2))
        if end < start:
            raise ValueError("hydra_transcript_receipt_timestamp_order_invalid")
        if start < previous_start:
            raise ValueError("hydra_transcript_receipt_timestamp_monotonicity_invalid")
        previous_start = start
        segments.append({
            "index": len(segments) + 1,
            "start_seconds": start,
            "end_seconds": end,
            "text": match.group(3),
        })
    return segments


def _audio_tuple(audio):
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("hydra_qwen3_forced_align_audio_required")
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor) or waveform.numel() < 1:
        raise ValueError("hydra_qwen3_forced_align_waveform_invalid")
    wav = waveform.detach().cpu()[0]
    wav = torch.mean(wav, dim=0) if wav.shape[0] > 1 else wav.squeeze(0)
    return (wav.numpy().astype(np.float32), int(audio["sample_rate"]))


def _audio_content_sha256(waveform, sample_rate):
    digest = hashlib.sha256()
    digest.update(b"hydra_audio_f32le_mono.v1\0")
    digest.update(int(sample_rate).to_bytes(8, "little", signed=False))
    digest.update(np.asarray(waveform, dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def _synchronize_model_device(model):
    device = _text(getattr(model, "device", None))
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _execution_evidence(
    model,
    waveform,
    sample_rate,
    text,
    language,
    timestamps,
    operation,
    elapsed_ms,
    execution_metadata,
):
    identity = getattr(model, "_hydra_model_identity", None)
    profile = _model_inference_profile(model)
    identity = identity if isinstance(identity, dict) else {}
    return HydraAsrExecutionEvidence._issue({
        "contract_version": "hydra_inferworks_asr_execution_evidence.v1",
        "operation": operation,
        "model_id": identity.get("model_id"),
        "aligner_id": identity.get("aligner_id"),
        "inference_profile": profile,
        "source_audio_content_sha256": _audio_content_sha256(waveform, sample_rate),
        "text_sha256": hashlib.sha256(_text(text).encode("utf-8")).hexdigest(),
        "language_sha256": hashlib.sha256(_text(language).encode("utf-8")).hexdigest(),
        "timestamps_sha256": hashlib.sha256(_text(timestamps).encode("utf-8")).hexdigest(),
        "execution_metadata_sha256": hashlib.sha256(
            json.dumps(execution_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "execution_elapsed_ms": round(float(elapsed_ms), 3),
    })


def _normalized_characters(value):
    characters = list(_text(value).lower())
    return [(character, index) for index, character in enumerate(characters) if character.isalnum()]


def _character_error_rate(expected, observed):
    left = [entry[0] for entry in _normalized_characters(expected)]
    right = [entry[0] for entry in _normalized_characters(observed)]
    if not left:
        return 0.0 if not right else 1.0
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (0 if left_character == right_character else 1),
            ))
        previous = current
    return previous[-1] / len(left)


def _anchor_groups(segments, max_chunk_seconds):
    groups = []
    current = []
    for segment in segments:
        if current and segment["end_seconds"] - current[0]["start_seconds"] > max_chunk_seconds:
            groups.append(current)
            current = []
        current.append(segment)
        if segment["end_seconds"] - current[0]["start_seconds"] > max_chunk_seconds:
            raise ValueError("hydra_qwen3_forced_align_anchor_segment_exceeds_limit")
    if current:
        groups.append(current)
    return groups


def _map_anchor_boundary(opcodes, anchor_boundary, locked_length):
    for _, anchor_start, anchor_end, locked_start, locked_end in opcodes:
        if anchor_boundary <= anchor_end:
            anchor_span = max(1, anchor_end - anchor_start)
            ratio = max(0.0, min(1.0, (anchor_boundary - anchor_start) / anchor_span))
            return round(locked_start + ratio * (locked_end - locked_start))
    return locked_length


def _locked_text_slices(locked_text, groups, maximum_character_error_rate):
    anchor_texts = ["".join(segment["text"] for segment in group) for group in groups]
    anchor_text = "".join(anchor_texts)
    error_rate = _character_error_rate(locked_text, anchor_text)
    if error_rate > maximum_character_error_rate:
        raise ValueError(f"hydra_qwen3_forced_align_anchor_text_mismatch:{error_rate:.4f}")
    anchor_normalized = _normalized_characters(anchor_text)
    locked_normalized = _normalized_characters(locked_text)
    if not anchor_normalized or not locked_normalized:
        raise ValueError("hydra_qwen3_forced_align_anchor_text_missing")
    matcher = SequenceMatcher(
        None,
        [entry[0] for entry in anchor_normalized],
        [entry[0] for entry in locked_normalized],
        autojunk=False,
    )
    opcodes = matcher.get_opcodes()
    locked_characters = list(_text(locked_text))
    locked_origin_indices = [entry[1] for entry in locked_normalized]
    anchor_cursor = 0
    original_boundaries = [0]
    for group_text in anchor_texts[:-1]:
        anchor_cursor += len(_normalized_characters(group_text))
        locked_boundary = _map_anchor_boundary(opcodes, anchor_cursor, len(locked_normalized))
        if locked_boundary <= 0:
            original_boundary = 0
        elif locked_boundary >= len(locked_origin_indices):
            original_boundary = len(locked_characters)
        else:
            original_boundary = locked_origin_indices[locked_boundary]
        original_boundaries.append(max(original_boundaries[-1], original_boundary))
    original_boundaries.append(len(locked_characters))
    slices = [
        "".join(locked_characters[original_boundaries[index]:original_boundaries[index + 1]])
        for index in range(len(groups))
    ]
    if any(not _normalized_characters(value) for value in slices):
        raise ValueError("hydra_qwen3_forced_align_locked_chunk_empty")
    return slices, error_rate


def _audio_chunk_boundaries(waveform, sample_rate, max_chunk_seconds, search_seconds=8):
    total_samples = len(waveform)
    maximum_samples = max(1, int(max_chunk_seconds * sample_rate))
    search_samples = max(1, int(search_seconds * sample_rate))
    frame_samples = max(1, int(0.05 * sample_rate))
    boundaries = [0]
    cursor = 0
    while total_samples - cursor > maximum_samples:
        hard_end = cursor + maximum_samples
        search_start = max(cursor + frame_samples, hard_end - search_samples)
        candidates = []
        for frame_start in range(search_start, hard_end, frame_samples):
            frame_end = min(hard_end, frame_start + frame_samples)
            frame = waveform[frame_start:frame_end]
            if len(frame):
                candidates.append((float(np.mean(np.square(frame, dtype=np.float64))), frame_end))
        boundary = min(candidates, key=lambda entry: entry[0])[1] if candidates else hard_end
        if boundary <= cursor:
            boundary = hard_end
        boundaries.append(boundary)
        cursor = boundary
    boundaries.append(total_samples)
    return boundaries


class HydraQwen3LongAsrTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3_ASR_MODEL",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "language": ("STRING", {"default": "auto"}),
                "context": ("STRING", {"default": "", "multiline": True}),
                "return_timestamps": ("BOOLEAN", {"default": True}),
                "max_chunk_seconds": ("INT", {"default": 90, "min": 30, "max": 150}),
                "silence_search_seconds": ("INT", {"default": 8, "min": 1, "max": 20}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", ASR_EXECUTION_EVIDENCE_TYPE)
    RETURN_NAMES = ("text", "language", "timestamps", "processing_metadata", "execution_evidence")
    FUNCTION = "transcribe"
    CATEGORY = "Hydra InferWorks/ASR"

    def transcribe(
        self,
        model,
        audio,
        language="auto",
        context="",
        return_timestamps=True,
        max_chunk_seconds=90,
        silence_search_seconds=8,
    ):
        waveform, sample_rate = _audio_tuple(audio)
        _synchronize_model_device(model)
        execution_started = time.perf_counter()
        chunk_limit = max(30, min(150, int(max_chunk_seconds)))
        boundaries = _audio_chunk_boundaries(
            waveform,
            sample_rate,
            chunk_limit,
            max(1, min(20, int(silence_search_seconds))),
        )
        requested_language = _text(language)
        model_language = None if not requested_language or requested_language.lower() == "auto" else requested_language
        resolved_language = model_language or ""
        texts = []
        lines = []
        chunk_receipts = []
        for index in range(len(boundaries) - 1):
            start_sample = boundaries[index]
            end_sample = boundaries[index + 1]
            start_seconds = start_sample / sample_rate
            end_seconds = end_sample / sample_rate
            results = model.transcribe(
                audio=(waveform[start_sample:end_sample], sample_rate),
                language=model_language,
                context=_text(context) or None,
                return_time_stamps=bool(return_timestamps),
            )
            if not results or not _text(getattr(results[0], "text", "")):
                raise ValueError(f"hydra_qwen3_long_asr_chunk_result_missing:{index + 1}")
            result = results[0]
            chunk_text = _text(result.text)
            texts.append(chunk_text)
            if not resolved_language:
                resolved_language = _text(getattr(result, "language", ""))
            time_stamps = getattr(result, "time_stamps", None) or []
            if return_timestamps:
                if not time_stamps:
                    raise ValueError(f"hydra_qwen3_long_asr_chunk_timestamps_missing:{index + 1}")
                lines.extend(
                    f"{start_seconds + float(item.start_time):.3f}-{start_seconds + float(item.end_time):.3f}: {_text(item.text)}"
                    for item in time_stamps
                    if _text(getattr(item, "text", ""))
                )
            chunk_receipts.append({
                "index": index + 1,
                "start_seconds": round(start_seconds, 6),
                "end_seconds": round(end_seconds, 6),
                "text_sha256": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                "timestamp_count": len(time_stamps),
            })
        transcript = "".join(texts)
        timestamp_text = "\n".join(lines)
        _synchronize_model_device(model)
        elapsed_ms = (time.perf_counter() - execution_started) * 1000
        metadata = {
            "contract_version": "hydra_qwen3_long_asr_execution.v1",
            "inference_profile": _model_inference_profile(model),
            "automatic_chunking": len(chunk_receipts) > 1,
            "actual_chunk_count": len(chunk_receipts),
            "max_chunk_seconds": chunk_limit,
            "split_strategy": "lowest_rms_before_hard_limit",
            "chunks": chunk_receipts,
        }
        evidence = _execution_evidence(
            model,
            waveform,
            sample_rate,
            transcript,
            resolved_language,
            timestamp_text,
            "transcribe",
            elapsed_ms,
            metadata,
        )
        return (transcript, resolved_language, timestamp_text, json.dumps(metadata, ensure_ascii=False), evidence)


class HydraQwen3ForcedAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3_ASR_MODEL",),
                "audio": ("AUDIO",),
                "locked_text": ("STRING", {"multiline": True}),
                "language": ("STRING", {"default": "Chinese", "forceInput": True}),
            },
            "optional": {
                "anchor_text": ("STRING", {"multiline": True, "forceInput": True}),
                "anchor_timestamps": ("STRING", {"multiline": True, "forceInput": True}),
                "max_chunk_seconds": ("INT", {"default": 150, "min": 30, "max": 170}),
                "maximum_anchor_character_error_rate": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", ASR_EXECUTION_EVIDENCE_TYPE)
    RETURN_NAMES = ("text", "language", "timestamps", "processing_metadata", "execution_evidence")
    FUNCTION = "align"
    CATEGORY = "Hydra InferWorks/ASR"

    def align(
        self,
        model,
        audio,
        locked_text,
        language,
        anchor_text="",
        anchor_timestamps="",
        max_chunk_seconds=150,
        maximum_anchor_character_error_rate=0.45,
    ):
        transcript = _text(locked_text)
        resolved_language = _text(language)
        if not transcript:
            raise ValueError("hydra_qwen3_forced_align_locked_text_required")
        if not resolved_language or resolved_language.lower() == "auto":
            raise ValueError("hydra_qwen3_forced_align_language_required")
        aligner = getattr(model, "forced_aligner", None)
        if aligner is None:
            raise ValueError("hydra_qwen3_forced_aligner_model_missing")
        waveform, sample_rate = _audio_tuple(audio)
        _synchronize_model_device(model)
        execution_started = time.perf_counter()
        duration_seconds = len(waveform) / sample_rate
        chunk_limit = max(30, min(170, int(max_chunk_seconds)))
        anchor_error_rate = None
        if duration_seconds <= chunk_limit:
            plans = [(0.0, duration_seconds, transcript)]
        else:
            anchors = _parse_timestamps(anchor_timestamps)
            if not _text(anchor_text) or not anchors:
                raise ValueError("hydra_qwen3_forced_align_long_audio_anchors_required")
            groups = _anchor_groups(anchors, chunk_limit)
            locked_slices, anchor_error_rate = _locked_text_slices(
                transcript,
                groups,
                float(maximum_anchor_character_error_rate),
            )
            boundaries = [0.0]
            for index in range(len(groups) - 1):
                current_end = groups[index][-1]["end_seconds"]
                next_start = groups[index + 1][0]["start_seconds"]
                boundaries.append(max(boundaries[-1], min(duration_seconds, (current_end + next_start) / 2)))
            boundaries.append(duration_seconds)
            plans = [
                (boundaries[index], boundaries[index + 1], locked_slices[index])
                for index in range(len(groups))
            ]
        lines = []
        alignment_chunk_receipts = []
        for chunk_index, (start_seconds, end_seconds, chunk_text) in enumerate(plans, start=1):
            start_sample = max(0, min(len(waveform), round(start_seconds * sample_rate)))
            end_sample = max(start_sample + 1, min(len(waveform), round(end_seconds * sample_rate)))
            results = aligner.align(
                audio=(waveform[start_sample:end_sample], sample_rate),
                text=chunk_text,
                language=resolved_language,
            )
            if not results:
                raise ValueError("hydra_qwen3_forced_align_result_missing")
            items = [
                item for item in (getattr(results[0], "items", results[0]) or [])
                if _text(getattr(item, "text", ""))
            ]
            if not items:
                raise ValueError("hydra_qwen3_forced_align_chunk_timestamps_missing")
            local_duration = max(0.001, end_seconds - start_seconds)
            predicted_duration = max(float(item.end_time) for item in items)
            timestamp_scale = min(1.0, local_duration / predicted_duration) if predicted_duration > 0 else 1.0
            lines.extend(
                f"{min(end_seconds, start_seconds + float(item.start_time) * timestamp_scale):.3f}-"
                f"{min(end_seconds, start_seconds + float(item.end_time) * timestamp_scale):.3f}: {_text(item.text)}"
                for item in items
            )
            alignment_chunk_receipts.append({
                "index": chunk_index,
                "start_seconds": round(start_seconds, 6),
                "end_seconds": round(end_seconds, 6),
                "locked_text_sha256": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                "timestamp_scale": round(timestamp_scale, 8),
                "predicted_duration_seconds": round(predicted_duration, 6),
            })
        if not lines:
            raise ValueError("hydra_qwen3_forced_align_timestamps_missing")
        timestamp_text = "\n".join(lines)
        _synchronize_model_device(model)
        elapsed_ms = (time.perf_counter() - execution_started) * 1000
        metadata = {
            "contract_version": "hydra_qwen3_forced_alignment_execution.v1",
            "inference_profile": _model_inference_profile(model),
            "automatic_chunking": len(plans) > 1,
            "actual_chunk_count": len(plans),
            "max_chunk_seconds": chunk_limit,
            "split_strategy": "asr_timestamp_anchor_groups",
            "anchor_character_error_rate": round(anchor_error_rate, 6) if anchor_error_rate is not None else None,
            "chunks": alignment_chunk_receipts,
        }
        evidence = _execution_evidence(
            model,
            waveform,
            sample_rate,
            transcript,
            resolved_language,
            timestamp_text,
            "forced_alignment",
            elapsed_ms,
            metadata,
        )
        return (transcript, resolved_language, timestamp_text, json.dumps(metadata, ensure_ascii=False), evidence)


class HydraTranscriptReceipt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
                "language": ("STRING", {"forceInput": True}),
                "timestamps": ("STRING", {"multiline": True, "forceInput": True}),
                "filename_prefix": ("STRING", {"default": "inferworks/qwen3-asr/transcript"}),
                "source_audio_sha256": ("STRING", {"default": ""}),
                "model_id": ("STRING", {"default": "Qwen/Qwen3-ASR-1.7B"}),
                "aligner_id": ("STRING", {"default": "Qwen/Qwen3-ForcedAligner-0.6B"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "processing_mode": ("STRING", {"default": "qwen3_asr_native_chunk_merge"}),
                "max_chunk_seconds": ("INT", {"default": 180, "min": 30, "max": 1200}),
                "strict_alignment": ("BOOLEAN", {"default": False}),
                "execution_evidence": (ASR_EXECUTION_EVIDENCE_TYPE,),
                "processing_metadata": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("receipt_path",)
    FUNCTION = "write_receipt"
    CATEGORY = "Hydra InferWorks/ASR"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, execution_evidence=None, **_kwargs):
        if execution_evidence is None or isinstance(execution_evidence, list):
            return True
        return "Hydra ASR execution evidence must be connected from the transcribe or forced-align node"

    def write_receipt(
        self,
        text,
        language,
        timestamps,
        filename_prefix,
        source_audio_sha256,
        model_id,
        aligner_id,
        audio=None,
        processing_mode="qwen3_asr_native_chunk_merge",
        max_chunk_seconds=180,
        strict_alignment=False,
        execution_evidence=None,
        processing_metadata="",
    ):
        transcript = _text(text)
        source_hash = _text(source_audio_sha256).lower()
        if not transcript:
            raise ValueError("hydra_transcript_receipt_text_required")
        if not re.fullmatch(r"[a-f0-9]{64}", source_hash):
            raise ValueError("hydra_transcript_receipt_source_hash_invalid")
        segments = _parse_timestamps(timestamps)
        source_duration_seconds = None
        if audio is not None:
            waveform, sample_rate = _audio_tuple(audio)
            source_duration_seconds = len(waveform) / sample_rate
        chunk_limit = max(30, min(1200, int(max_chunk_seconds)))
        execution_details = None
        if _text(processing_metadata):
            try:
                execution_details = json.loads(_text(processing_metadata))
            except json.JSONDecodeError as error:
                raise ValueError("hydra_transcript_receipt_processing_metadata_invalid") from error
        verified_evidence = None
        if execution_evidence is not None:
            if type(execution_evidence) is not HydraAsrExecutionEvidence:
                raise ValueError("hydra_transcript_receipt_execution_evidence_invalid")
            verified_evidence = execution_evidence._verified_payload()
            if verified_evidence.get("contract_version") != "hydra_inferworks_asr_execution_evidence.v1":
                raise ValueError("hydra_transcript_receipt_execution_evidence_contract_invalid")
            expected_hashes = {
                "text_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
                "language_sha256": hashlib.sha256(_text(language).encode("utf-8")).hexdigest(),
                "timestamps_sha256": hashlib.sha256(_text(timestamps).encode("utf-8")).hexdigest(),
            }
            for field, expected in expected_hashes.items():
                if _text(verified_evidence.get(field)).lower() != expected:
                    raise ValueError(f"hydra_transcript_receipt_execution_evidence_{field}_mismatch")
            if audio is None:
                raise ValueError("hydra_transcript_receipt_execution_evidence_audio_required")
            observed_audio_hash = _audio_content_sha256(waveform, sample_rate)
            if _text(verified_evidence.get("source_audio_content_sha256")).lower() != observed_audio_hash:
                raise ValueError("hydra_transcript_receipt_execution_evidence_audio_mismatch")
            evidence_model_id = _text(verified_evidence.get("model_id"))
            evidence_aligner_id = _text(verified_evidence.get("aligner_id"))
            if _text(model_id) and _text(model_id) != evidence_model_id:
                raise ValueError("hydra_transcript_receipt_model_identity_mismatch")
            if _text(aligner_id) and _text(aligner_id) != evidence_aligner_id:
                raise ValueError("hydra_transcript_receipt_aligner_identity_mismatch")
            normalized_mode = _text(processing_mode).lower()
            expected_operation = (
                "forced_alignment"
                if bool(strict_alignment) or "forced_align" in normalized_mode
                else "transcribe"
            )
            if _text(verified_evidence.get("operation")) != expected_operation:
                raise ValueError("hydra_transcript_receipt_execution_operation_mismatch")
            expected_mode = (
                "qwen3_forced_alignment_anchor_chunk_merge"
                if expected_operation == "forced_alignment"
                else "qwen3_asr_native_chunk_merge"
            )
            if normalized_mode != expected_mode:
                raise ValueError("hydra_transcript_receipt_processing_mode_mismatch")
            if bool(strict_alignment) != (expected_operation == "forced_alignment"):
                raise ValueError("hydra_transcript_receipt_strict_alignment_mismatch")
            if not isinstance(execution_details, dict):
                raise ValueError("hydra_transcript_receipt_execution_metadata_required")
            metadata_hash = hashlib.sha256(
                json.dumps(execution_details, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if _text(verified_evidence.get("execution_metadata_sha256")) != metadata_hash:
                raise ValueError("hydra_transcript_receipt_execution_metadata_mismatch")
        verified_profile = verified_evidence.get("inference_profile") if verified_evidence else None
        production_verified = (
            isinstance(verified_profile, dict)
            and verified_profile.get("quality_admission") == "production"
            and _text(verified_evidence.get("model_id")) == "Qwen/Qwen3-ASR-1.7B"
            and _text(verified_evidence.get("aligner_id")) == "Qwen/Qwen3-ForcedAligner-0.6B"
        )
        payload = {
            "contract_version": "hydra_comfyui_qwen3_asr_transcript_receipt.v1",
            "status": "completed" if production_verified else "completed_unverified",
            "source_audio_sha256": source_hash,
            "source_audio_sha256_provenance": "caller_claim_cross_check_required",
            "source_audio_content_sha256": (
                verified_evidence.get("source_audio_content_sha256") if verified_evidence else None
            ),
            "model_id": verified_evidence.get("model_id") if verified_evidence else None,
            "aligner_id": verified_evidence.get("aligner_id") if verified_evidence else None,
            "language": _text(language) or None,
            "text": transcript,
            "timestamps_present": bool(segments),
            "segments": segments,
            "inference_profile": verified_profile,
            "long_audio_processing": {
                "processing_mode": _text(processing_mode) or "qwen3_asr_native_chunk_merge",
                "source_duration_seconds": round(source_duration_seconds, 6) if source_duration_seconds is not None else None,
                "max_chunk_seconds": chunk_limit,
                "automatic_chunking": source_duration_seconds is not None and source_duration_seconds > chunk_limit,
                "chunk_merge": True,
                "timestamp_offsets_preserved": True,
                "timeline_monotonic": True,
                "strict_locked_script_alignment": bool(strict_alignment),
                "last_timestamp_seconds": segments[-1]["end_seconds"] if segments else None,
                "execution_details": execution_details,
                "execution_evidence": verified_evidence,
                "claimed_model_id": _text(model_id) or None,
                "claimed_aligner_id": _text(aligner_id) or None,
            },
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        receipt_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        payload["receipt_sha256"] = receipt_hash

        output_root = Path(folder_paths.get_output_directory()).resolve()
        parts = _safe_prefix(filename_prefix)
        target_dir = output_root.joinpath(*parts[:-1]).resolve()
        if os.path.commonpath([str(output_root), str(target_dir)]) != str(output_root):
            raise ValueError("hydra_transcript_receipt_output_escape")
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{parts[-1]}-{receipt_hash[:16]}.json"
        target = target_dir / filename
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        try:
            with target.open("xb") as stream:
                stream.write(encoded)
        except FileExistsError:
            if target.read_bytes() != encoded:
                raise ValueError("hydra_transcript_receipt_existing_bytes_mismatch")

        subfolder = target_dir.relative_to(output_root).as_posix()
        descriptor = {
            "filename": filename,
            "subfolder": "" if subfolder == "." else subfolder,
            "type": "output",
        }
        return {
            "ui": {
                "text": [transcript],
                "transcripts": [descriptor],
            },
            "result": (str(target),),
        }


NODE_CLASS_MAPPINGS = {
    "HydraQwen3ASRModelLoader": HydraQwen3ASRModelLoader,
    "HydraTranscriptReceipt": HydraTranscriptReceipt,
    "HydraQwen3ForcedAlign": HydraQwen3ForcedAlign,
    "HydraQwen3LongAsrTranscribe": HydraQwen3LongAsrTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HydraQwen3ASRModelLoader": "Hydra InferWorks · Qwen3-ASR · Load Optimized Model",
    "HydraTranscriptReceipt": "Hydra InferWorks · Transcript Receipt",
    "HydraQwen3ForcedAlign": "Hydra InferWorks · Qwen3 Locked-Script Forced Align",
    "HydraQwen3LongAsrTranscribe": "Hydra InferWorks · Qwen3 Long-Audio Transcribe",
}
