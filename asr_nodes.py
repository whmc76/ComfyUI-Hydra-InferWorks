import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import time
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
FORCED_ALIGNMENT_TIMESTAMP_PROVENANCE = "qwen3_forced_aligner_model_token_logits"
FORCED_ALIGNMENT_TIMESTAMP_TRANSFORM = "monotonic_constrained_model_decode"
FORCED_ALIGNMENT_REPAIR_POLICY = "constrained_model_token_decode_no_estimation"
FORCED_ALIGNMENT_DECODER_ALGORITHM_VERSION = "hydra_qwen3_forced_alignment_monotonic_decode.v1"
QWEN_ASR_EXPECTED_VERSION = "0.0.6"
QWEN_ASR_TIMESTAMP_SEGMENT_MS = 80
QWEN_ASR_TIMESTAMP_CLASS_COUNT = 5000
QWEN_ASR_TIMESTAMP_TOKEN_ID = 151705
_FILE_HASH_CACHE = {}
_ASR_EVIDENCE_ISSUER = object()
try:
    _QWEN_ASR_RUNTIME_VERSION = importlib.metadata.version("qwen-asr")
except importlib.metadata.PackageNotFoundError:
    _QWEN_ASR_RUNTIME_VERSION = None
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
        _install_strict_timestamp_guard(model)
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
    previous_end = 0.0
    for line_number, line in enumerate(_text(value).splitlines(), start=1):
        if not line.strip():
            continue
        match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:\s*(.*?)\s*$", line)
        if not match:
            raise ValueError(f"hydra_transcript_receipt_timestamp_line_invalid:{line_number}")
        start = float(match.group(1))
        end = float(match.group(2))
        if end <= start:
            raise ValueError("hydra_transcript_receipt_timestamp_order_invalid")
        if start < previous_end:
            raise ValueError("hydra_transcript_receipt_timestamp_overlap_invalid")
        previous_end = end
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
        "timestamp_provenance": _text(execution_metadata.get("timestamp_provenance")) or None,
        "estimated_timestamps": execution_metadata.get("estimated_timestamps"),
        "timestamp_transform": _text(execution_metadata.get("timestamp_transform")) or None,
        "upstream_timestamp_repair_policy": _text(
            execution_metadata.get("upstream_timestamp_repair_policy")
        ) or None,
        "decoder_algorithm_version": _text(
            execution_metadata.get("decoder_algorithm_version")
        ) or None,
        "raw_argmax_sha256": _text(execution_metadata.get("raw_argmax_sha256")) or None,
        "constrained_token_sha256": _text(
            execution_metadata.get("constrained_token_sha256")
        ) or None,
    })


def _normalized_characters(value):
    characters = list(_text(value).lower())
    return [(character, index) for index, character in enumerate(characters) if character.isalnum()]


def _strict_timestamp_values(data):
    values = data.tolist() if hasattr(data, "tolist") else list(data)
    normalized = [float(value) for value in values]
    if any(not np.isfinite(value) or value < 0 for value in normalized):
        raise ValueError("hydra_qwen3_forced_align_raw_timestamp_invalid")
    if any(current < previous for previous, current in zip(normalized, normalized[1:])):
        raise ValueError("hydra_qwen3_forced_align_upstream_timestamp_repair_forbidden")
    return [int(value) for value in normalized]


def _timestamp_constraint_summary(bins, word_count, max_valid_bin):
    violation_positions = set()
    max_backward_bins = 0
    normalized = [int(value) for value in bins]
    for index, value in enumerate(normalized):
        if value < 0 or value > max_valid_bin:
            violation_positions.add(index)
        if index == 0:
            continue
        minimum = normalized[index - 1] + (1 if index % 2 == 1 else 0)
        if value < minimum:
            violation_positions.add(index)
            max_backward_bins = max(max_backward_bins, minimum - value)
    return {
        "satisfied": len(normalized) == int(word_count) * 2 and not violation_positions,
        "violation_positions": sorted(violation_positions),
        "max_backward_bins": max_backward_bins,
    }


def _stable_prefix_max(values):
    prefix_values = torch.cummax(values, dim=0).values
    indices = torch.arange(values.shape[0], dtype=torch.int64)
    new_best = torch.ones(values.shape[0], dtype=torch.bool)
    if values.shape[0] > 1:
        new_best[1:] = prefix_values[1:] > prefix_values[:-1]
    candidates = torch.where(new_best, indices, torch.full_like(indices, -1))
    prefix_indices = torch.cummax(candidates, dim=0).values
    return prefix_values, prefix_indices


def _decode_monotonic_timestamp_logits(
    timestamp_logits,
    *,
    word_count,
    duration_seconds,
    timestamp_segment_ms,
):
    if int(timestamp_segment_ms) != QWEN_ASR_TIMESTAMP_SEGMENT_MS:
        raise ValueError("hydra_qwen3_forced_align_timestamp_segment_time_invalid")
    try:
        normalized_word_count = int(word_count)
        normalized_duration = float(duration_seconds)
    except (TypeError, ValueError) as error:
        raise ValueError("hydra_qwen3_forced_align_timestamp_decode_shape_invalid") from error
    if normalized_word_count < 1 or not np.isfinite(normalized_duration) or normalized_duration <= 0:
        raise ValueError("hydra_qwen3_forced_align_timestamp_decode_shape_invalid")
    if not isinstance(timestamp_logits, torch.Tensor) or timestamp_logits.ndim != 2:
        raise ValueError("hydra_qwen3_forced_align_timestamp_logits_shape_invalid")
    slot_count, class_count = timestamp_logits.shape
    if slot_count != normalized_word_count * 2:
        raise ValueError("hydra_qwen3_forced_align_timestamp_slot_count_invalid")
    if class_count < 2:
        raise ValueError("hydra_qwen3_forced_align_timestamp_logits_shape_invalid")

    logits = timestamp_logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("hydra_qwen3_forced_align_timestamp_logits_non_finite")
    duration_bound_bin = int(np.floor(
        normalized_duration * 1000.0 / QWEN_ASR_TIMESTAMP_SEGMENT_MS + 1e-9
    ))
    max_valid_bin = min(class_count - 1, duration_bound_bin)
    if max_valid_bin < normalized_word_count:
        raise ValueError("hydra_qwen3_forced_align_timestamp_path_infeasible")

    valid_logits = logits[:, :max_valid_bin + 1]
    parents = torch.full(
        (slot_count, max_valid_bin + 1),
        -1,
        dtype=torch.int16,
    )
    previous = valid_logits[0].clone()
    negative_infinity = torch.tensor(float("-inf"), dtype=torch.float32)
    for slot_index in range(1, slot_count):
        prefix_values, prefix_indices = _stable_prefix_max(previous)
        if slot_index % 2 == 1:
            allowed_values = torch.cat((negative_infinity.reshape(1), prefix_values[:-1]))
            allowed_indices = torch.cat((torch.tensor([-1], dtype=torch.int64), prefix_indices[:-1]))
        else:
            allowed_values = prefix_values
            allowed_indices = prefix_indices
        previous = valid_logits[slot_index] + allowed_values
        parents[slot_index] = allowed_indices.to(dtype=torch.int16)

    best_score = torch.max(previous)
    if not bool(torch.isfinite(best_score)):
        raise ValueError("hydra_qwen3_forced_align_timestamp_path_infeasible")
    best_final_candidates = torch.nonzero(previous == best_score, as_tuple=False).flatten()
    if best_final_candidates.numel() < 1:
        raise ValueError("hydra_qwen3_forced_align_timestamp_path_infeasible")
    selected = [0] * slot_count
    selected[-1] = int(best_final_candidates[0].item())
    for slot_index in range(slot_count - 1, 0, -1):
        parent = int(parents[slot_index, selected[slot_index]].item())
        if parent < 0:
            raise ValueError("hydra_qwen3_forced_align_timestamp_path_infeasible")
        selected[slot_index - 1] = parent

    selected_constraints = _timestamp_constraint_summary(
        selected,
        normalized_word_count,
        max_valid_bin,
    )
    if not selected_constraints["satisfied"]:
        raise ValueError("hydra_qwen3_forced_align_timestamp_path_invalid")
    raw_greedy = torch.argmax(logits, dim=1).to(dtype=torch.int64).tolist()
    raw_constraints = _timestamp_constraint_summary(
        raw_greedy,
        normalized_word_count,
        max_valid_bin,
    )
    logits_sha256 = hashlib.sha256(
        logits.numpy().astype("<f4", copy=False).tobytes(order="C")
    ).hexdigest()
    raw_argmax_sha256 = hashlib.sha256(
        json.dumps(raw_greedy, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    selected_sha256 = hashlib.sha256(
        json.dumps(selected, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    evidence = {
        "timestamp_provenance": FORCED_ALIGNMENT_TIMESTAMP_PROVENANCE,
        "timestamp_transform": FORCED_ALIGNMENT_TIMESTAMP_TRANSFORM,
        "upstream_timestamp_repair_policy": FORCED_ALIGNMENT_REPAIR_POLICY,
        "decoder_algorithm_version": FORCED_ALIGNMENT_DECODER_ALGORITHM_VERSION,
        "estimated_timestamps": False,
        "timestamp_segment_ms": QWEN_ASR_TIMESTAMP_SEGMENT_MS,
        "class_count": class_count,
        "word_count": normalized_word_count,
        "timestamp_position_count": slot_count,
        "duration_seconds": round(normalized_duration, 6),
        "duration_bound_bin": duration_bound_bin,
        "max_valid_bin": max_valid_bin,
        "raw_argmax_sha256": raw_argmax_sha256,
        "raw_argmax_violation_positions": raw_constraints["violation_positions"],
        "raw_max_backward_bins": raw_constraints["max_backward_bins"],
        "constrained_token_sha256": selected_sha256,
        "masked_logits_shape": [slot_count, class_count],
        "masked_logits_fp32_sha256": logits_sha256,
        "deterministic_tie_break": "earliest_bin",
    }
    return selected, evidence


def _install_strict_timestamp_guard(model):
    aligner = getattr(model, "forced_aligner", None)
    processor = getattr(aligner, "aligner_processor", None)
    original = getattr(processor, "fix_timestamp", None)
    if processor is None or not callable(original):
        raise ValueError("hydra_qwen3_forced_align_timestamp_guard_unavailable")
    if getattr(processor, "_inferworks_strict_timestamp_guard", False):
        return

    def reject_repair(data):
        return _strict_timestamp_values(data)

    processor.fix_timestamp = reject_repair
    processor._inferworks_strict_timestamp_guard = True
    processor._inferworks_original_fix_timestamp = original


class _ConstrainedTimestampItem:
    __slots__ = ("text", "start_time", "end_time")

    def __init__(self, text, start_time, end_time):
        self.text = str(text)
        self.start_time = float(start_time)
        self.end_time = float(end_time)


def _normalize_qwen_forced_alignment_audio(audio):
    from qwen_asr.inference.utils import SAMPLE_RATE, normalize_audios

    if int(SAMPLE_RATE) != 16000:
        raise ValueError("hydra_qwen3_forced_align_sample_rate_contract_invalid")
    return normalize_audios(audio), int(SAMPLE_RATE)


def _require_qwen_constrained_decode_contract(aligner):
    if _QWEN_ASR_RUNTIME_VERSION != QWEN_ASR_EXPECTED_VERSION:
        raise RuntimeError(
            "hydra_qwen3_forced_align_qwen_asr_version_invalid:"
            f"{_QWEN_ASR_RUNTIME_VERSION or 'missing'}"
        )
    model = getattr(aligner, "model", None)
    thinker = getattr(model, "thinker", None)
    processor = getattr(aligner, "processor", None)
    aligner_processor = getattr(aligner, "aligner_processor", None)
    encode_timestamp = getattr(aligner_processor, "encode_timestamp", None)
    thinker_config = getattr(thinker, "config", None)
    try:
        classify_num = int(getattr(thinker_config, "classify_num"))
        timestamp_token_id = int(getattr(aligner, "timestamp_token_id"))
        timestamp_segment_ms = int(getattr(aligner, "timestamp_segment_time"))
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("hydra_qwen3_forced_align_runtime_structure_invalid") from error
    if (
        not callable(thinker)
        or not callable(processor)
        or not callable(encode_timestamp)
        or classify_num != QWEN_ASR_TIMESTAMP_CLASS_COUNT
        or timestamp_token_id != QWEN_ASR_TIMESTAMP_TOKEN_ID
        or timestamp_segment_ms != QWEN_ASR_TIMESTAMP_SEGMENT_MS
    ):
        raise RuntimeError("hydra_qwen3_forced_align_runtime_structure_invalid")
    if getattr(model, "device", None) is None or getattr(model, "dtype", None) is None:
        raise RuntimeError("hydra_qwen3_forced_align_runtime_structure_invalid")
    return {
        "model": model,
        "thinker": thinker,
        "processor": processor,
        "aligner_processor": aligner_processor,
        "encode_timestamp": encode_timestamp,
        "classify_num": classify_num,
        "timestamp_token_id": timestamp_token_id,
        "timestamp_segment_ms": timestamp_segment_ms,
    }


def _align_with_monotonic_timestamp_logits(aligner, audio, text, language):
    contract = _require_qwen_constrained_decode_contract(aligner)
    word_list, aligner_input_text = contract["encode_timestamp"](text, language)
    if (
        not isinstance(word_list, list)
        or not word_list
        or any(not _text(word) for word in word_list)
        or not isinstance(aligner_input_text, str)
        or not aligner_input_text
    ):
        raise ValueError("hydra_qwen3_forced_align_word_units_invalid")
    normalizer = getattr(aligner, "_inferworks_normalize_audios", None)
    if callable(normalizer):
        normalized_audios, normalized_sample_rate = normalizer(audio)
    else:
        normalized_audios, normalized_sample_rate = _normalize_qwen_forced_alignment_audio(audio)
    if (
        int(normalized_sample_rate) != 16000
        or not isinstance(normalized_audios, list)
        or len(normalized_audios) != 1
    ):
        raise ValueError("hydra_qwen3_forced_align_normalized_audio_invalid")
    normalized_audio = np.asarray(normalized_audios[0], dtype=np.float32)
    if normalized_audio.ndim != 1 or normalized_audio.size < 1 or not np.isfinite(normalized_audio).all():
        raise ValueError("hydra_qwen3_forced_align_normalized_audio_invalid")
    duration_seconds = normalized_audio.size / normalized_sample_rate

    inputs = contract["processor"](
        text=[aligner_input_text],
        audio=normalized_audios,
        return_tensors="pt",
        padding=True,
    )
    if not callable(getattr(inputs, "to", None)):
        raise RuntimeError("hydra_qwen3_forced_align_processor_output_invalid")
    inputs = inputs.to(contract["model"].device).to(contract["model"].dtype)
    try:
        input_ids = inputs["input_ids"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("hydra_qwen3_forced_align_processor_output_invalid") from error
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError("hydra_qwen3_forced_align_processor_output_invalid")
    with torch.inference_mode():
        output = contract["thinker"](**inputs)
    logits = getattr(output, "logits", None)
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or logits.shape[0] != 1
        or logits.shape[1] != input_ids.shape[1]
        or logits.shape[2] != contract["classify_num"]
    ):
        raise RuntimeError("hydra_qwen3_forced_align_timestamp_logits_shape_invalid")
    timestamp_mask = input_ids[0] == contract["timestamp_token_id"]
    timestamp_logits = logits[0][timestamp_mask]
    selected_bins, decode_evidence = _decode_monotonic_timestamp_logits(
        timestamp_logits,
        word_count=len(word_list),
        duration_seconds=duration_seconds,
        timestamp_segment_ms=contract["timestamp_segment_ms"],
    )
    items = [
        _ConstrainedTimestampItem(
            word,
            round(selected_bins[index * 2] * contract["timestamp_segment_ms"] / 1000.0, 3),
            round(selected_bins[index * 2 + 1] * contract["timestamp_segment_ms"] / 1000.0, 3),
        )
        for index, word in enumerate(word_list)
    ]
    decode_evidence.update({
        "qwen_asr_version": _QWEN_ASR_RUNTIME_VERSION,
        "timestamp_token_id": contract["timestamp_token_id"],
        "normalized_audio_sample_rate": normalized_sample_rate,
        "normalized_audio_sample_count": int(normalized_audio.size),
        "normalized_audio_content_sha256": _audio_content_sha256(
            normalized_audio,
            normalized_sample_rate,
        ),
        "text_sha256": hashlib.sha256(_text(text).encode("utf-8")).hexdigest(),
        "language_sha256": hashlib.sha256(_text(language).encode("utf-8")).hexdigest(),
    })
    return items, decode_evidence


def _constrained_decode_evidence_valid(value):
    if not isinstance(value, dict):
        return False
    try:
        word_count = int(value.get("word_count"))
        position_count = int(value.get("timestamp_position_count"))
        class_count = int(value.get("class_count"))
        duration_seconds = float(value.get("duration_seconds"))
        duration_bound_bin = int(value.get("duration_bound_bin"))
        max_valid_bin = int(value.get("max_valid_bin"))
        normalized_sample_rate = int(value.get("normalized_audio_sample_rate"))
        normalized_sample_count = int(value.get("normalized_audio_sample_count"))
    except (TypeError, ValueError):
        return False
    violation_positions = value.get("raw_argmax_violation_positions")
    sha256_fields = (
        "raw_argmax_sha256",
        "constrained_token_sha256",
        "masked_logits_fp32_sha256",
        "normalized_audio_content_sha256",
        "text_sha256",
        "language_sha256",
    )
    if (
        value.get("timestamp_provenance") != FORCED_ALIGNMENT_TIMESTAMP_PROVENANCE
        or value.get("timestamp_transform") != FORCED_ALIGNMENT_TIMESTAMP_TRANSFORM
        or value.get("upstream_timestamp_repair_policy") != FORCED_ALIGNMENT_REPAIR_POLICY
        or value.get("decoder_algorithm_version") != FORCED_ALIGNMENT_DECODER_ALGORITHM_VERSION
        or value.get("estimated_timestamps") is not False
        or value.get("qwen_asr_version") != QWEN_ASR_EXPECTED_VERSION
        or value.get("timestamp_segment_ms") != QWEN_ASR_TIMESTAMP_SEGMENT_MS
        or value.get("timestamp_token_id") != QWEN_ASR_TIMESTAMP_TOKEN_ID
        or class_count != QWEN_ASR_TIMESTAMP_CLASS_COUNT
        or word_count < 1
        or position_count != word_count * 2
        or normalized_sample_rate != 16000
        or normalized_sample_count < 1
        or not np.isfinite(duration_seconds)
        or abs(duration_seconds - round(normalized_sample_count / normalized_sample_rate, 6)) > 1e-6
        or duration_bound_bin != int(np.floor(
            duration_seconds * 1000.0 / QWEN_ASR_TIMESTAMP_SEGMENT_MS + 1e-9
        ))
        or max_valid_bin != min(class_count - 1, duration_bound_bin)
        or max_valid_bin < word_count
        or value.get("masked_logits_shape") != [position_count, class_count]
        or value.get("deterministic_tie_break") != "earliest_bin"
        or any(not re.fullmatch(r"[a-f0-9]{64}", _text(value.get(field)).lower()) for field in sha256_fields)
        or not isinstance(violation_positions, list)
        or any(type(position) is not int for position in violation_positions)
        or violation_positions != sorted(set(violation_positions))
        or any(position < 0 or position >= position_count for position in violation_positions)
    ):
        return False
    return True


def _aggregate_constrained_decode_hash(chunks, field):
    if any(
        not isinstance(chunk, dict) or not isinstance(chunk.get("timestamp_decode"), dict)
        for chunk in chunks
    ):
        raise ValueError("hydra_qwen3_forced_align_constrained_decode_hash_missing")
    values = [_text(chunk["timestamp_decode"].get(field)).lower() for chunk in chunks]
    if not values or any(not re.fullmatch(r"[a-f0-9]{64}", value) for value in values):
        raise ValueError("hydra_qwen3_forced_align_constrained_decode_hash_missing")
    if len(values) == 1:
        return values[0]
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def _locked_text_slices(locked_text, groups):
    anchor_texts = ["".join(segment["text"] for segment in group) for group in groups]
    anchor_normalized = _normalized_characters("".join(anchor_texts))
    locked_normalized = _normalized_characters(locked_text)
    if not anchor_normalized or not locked_normalized:
        raise ValueError("hydra_qwen3_forced_align_anchor_text_missing")
    if [entry[0] for entry in anchor_normalized] != [entry[0] for entry in locked_normalized]:
        raise ValueError("hydra_qwen3_forced_align_anchor_text_not_exact")
    locked_characters = list(_text(locked_text))
    locked_origin_indices = [entry[1] for entry in locked_normalized]
    normalized_cursor = 0
    original_boundaries = [0]
    for group_text in anchor_texts[:-1]:
        normalized_cursor += len(_normalized_characters(group_text))
        if normalized_cursor >= len(locked_origin_indices):
            original_boundary = len(locked_characters)
        else:
            original_boundary = locked_origin_indices[normalized_cursor]
        original_boundaries.append(max(original_boundaries[-1], original_boundary))
    original_boundaries.append(len(locked_characters))
    slices = [
        "".join(locked_characters[original_boundaries[index]:original_boundaries[index + 1]])
        for index in range(len(groups))
    ]
    if any(not _normalized_characters(value) for value in slices):
        raise ValueError("hydra_qwen3_forced_align_locked_chunk_empty")
    return slices


def _validated_native_items(items, local_duration, sample_rate, error_prefix):
    tolerance = max(1e-6, 1.0 / max(1, int(sample_rate)))
    validated = []
    previous_end = 0.0
    for item in items:
        item_text = _text(getattr(item, "text", ""))
        if not item_text:
            continue
        start = float(getattr(item, "start_time"))
        end = float(getattr(item, "end_time"))
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"{error_prefix}_native_timestamp_invalid")
        if start + tolerance < previous_end:
            raise ValueError(f"{error_prefix}_native_timestamp_non_monotonic")
        if end > local_duration + tolerance:
            raise ValueError(f"{error_prefix}_native_timestamp_out_of_bounds")
        previous_end = end
        validated.append((item_text, start, end))
    if not validated:
        raise ValueError(f"{error_prefix}_chunk_timestamps_missing")
    return validated


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
        if return_timestamps:
            _install_strict_timestamp_guard(model)
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
            if not results or len(results) != 1 or not _text(getattr(results[0], "text", "")):
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
                native_items = _validated_native_items(
                    time_stamps,
                    (end_sample - start_sample) / sample_rate,
                    sample_rate,
                    f"hydra_qwen3_long_asr_chunk_{index + 1}",
                )
                if (
                    [entry[0] for entry in _normalized_characters(chunk_text)] !=
                    [entry[0] for entry in _normalized_characters("".join(item[0] for item in native_items))]
                ):
                    raise ValueError(f"hydra_qwen3_long_asr_chunk_native_text_not_exact:{index + 1}")
                lines.extend(
                    f"{start_seconds + item_start:.6f}-{start_seconds + item_end:.6f}: {item_text}"
                    for item_text, item_start, item_end in native_items
                )
            chunk_receipts.append({
                "index": index + 1,
                "start_seconds": round(start_seconds, 6),
                "end_seconds": round(end_seconds, 6),
                "text_sha256": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                "timestamp_count": len(time_stamps),
                "timestamp_transform": "offset_only",
                "estimated_timestamps": False,
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
            "timestamp_provenance": "qwen3_forced_aligner_native",
            "estimated_timestamps": False,
            "timestamp_transform": "offset_only",
            "upstream_timestamp_repair_policy": "reject_non_monotonic_raw_tokens",
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
                "max_chunk_seconds": ("INT", {"default": 300, "min": 30, "max": 300}),
                "maximum_anchor_character_error_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
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
        max_chunk_seconds=300,
        maximum_anchor_character_error_rate=0.0,
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
        _install_strict_timestamp_guard(model)
        waveform, sample_rate = _audio_tuple(audio)
        _synchronize_model_device(model)
        execution_started = time.perf_counter()
        duration_seconds = len(waveform) / sample_rate
        chunk_limit = max(30, min(300, int(max_chunk_seconds)))
        anchor_text_match = "not_required_single_chunk"
        if duration_seconds <= chunk_limit:
            plans = [(0.0, duration_seconds, transcript)]
        else:
            anchors = _parse_timestamps(anchor_timestamps)
            if not _text(anchor_text) or not anchors:
                raise ValueError("hydra_qwen3_forced_align_long_audio_anchors_required")
            if (
                [entry[0] for entry in _normalized_characters(anchor_text)] !=
                [entry[0] for entry in _normalized_characters("".join(item["text"] for item in anchors))]
            ):
                raise ValueError("hydra_qwen3_forced_align_anchor_timestamp_text_not_exact")
            groups = _anchor_groups(anchors, chunk_limit)
            locked_slices = _locked_text_slices(transcript, groups)
            anchor_text_match = "exact_normalized"
            plans = [
                (groups[index][0]["start_seconds"], groups[index][-1]["end_seconds"], locked_slices[index])
                for index in range(len(groups))
            ]
        lines = []
        alignment_chunk_receipts = []
        previous_global_end = 0.0
        for chunk_index, (start_seconds, end_seconds, chunk_text) in enumerate(plans, start=1):
            if start_seconds < 0 or end_seconds <= start_seconds or end_seconds > duration_seconds:
                raise ValueError("hydra_qwen3_forced_align_anchor_timestamp_out_of_bounds")
            start_sample = max(0, min(len(waveform), round(start_seconds * sample_rate)))
            end_sample = max(start_sample + 1, min(len(waveform), round(end_seconds * sample_rate)))
            slice_start_seconds = start_sample / sample_rate
            slice_end_seconds = end_sample / sample_rate
            items, timestamp_decode = _align_with_monotonic_timestamp_logits(
                aligner,
                audio=(waveform[start_sample:end_sample], sample_rate),
                text=chunk_text,
                language=resolved_language,
            )
            items = [item for item in items if _text(getattr(item, "text", ""))]
            if not items:
                raise ValueError("hydra_qwen3_forced_align_chunk_timestamps_missing")
            local_duration = (end_sample - start_sample) / sample_rate
            native_items = _validated_native_items(
                items,
                local_duration,
                sample_rate,
                "hydra_qwen3_forced_align",
            )
            if (
                [entry[0] for entry in _normalized_characters(chunk_text)] !=
                [entry[0] for entry in _normalized_characters("".join(item[0] for item in native_items))]
            ):
                raise ValueError("hydra_qwen3_forced_align_native_text_not_exact")
            for item_text, item_start, item_end in native_items:
                global_start = slice_start_seconds + item_start
                global_end = slice_start_seconds + item_end
                if global_start + max(1e-6, 1.0 / sample_rate) < previous_global_end:
                    raise ValueError("hydra_qwen3_forced_align_global_timestamp_non_monotonic")
                previous_global_end = global_end
                lines.append(f"{global_start:.6f}-{global_end:.6f}: {item_text}")
            alignment_chunk_receipts.append({
                "index": chunk_index,
                "start_seconds": round(slice_start_seconds, 6),
                "end_seconds": round(slice_end_seconds, 6),
                "locked_text_sha256": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                "timestamp_provenance": FORCED_ALIGNMENT_TIMESTAMP_PROVENANCE,
                "timestamp_transform": FORCED_ALIGNMENT_TIMESTAMP_TRANSFORM,
                "upstream_timestamp_repair_policy": FORCED_ALIGNMENT_REPAIR_POLICY,
                "decoder_algorithm_version": FORCED_ALIGNMENT_DECODER_ALGORITHM_VERSION,
                "estimated_timestamps": False,
                "aligned_item_count": len(native_items),
                "timestamp_decode": timestamp_decode,
            })
        if not lines:
            raise ValueError("hydra_qwen3_forced_align_timestamps_missing")
        timestamp_text = "\n".join(lines)
        _synchronize_model_device(model)
        elapsed_ms = (time.perf_counter() - execution_started) * 1000
        metadata = {
            "contract_version": "hydra_qwen3_forced_alignment_execution.v2",
            "inference_profile": _model_inference_profile(model),
            "automatic_chunking": len(plans) > 1,
            "actual_chunk_count": len(plans),
            "max_chunk_seconds": chunk_limit,
            "split_strategy": (
                "exact_anchor_timestamp_groups"
                if len(plans) > 1
                else "single_chunk_direct_forced_alignment"
            ),
            "anchor_text_match": anchor_text_match,
            "timestamp_provenance": FORCED_ALIGNMENT_TIMESTAMP_PROVENANCE,
            "estimated_timestamps": False,
            "timestamp_transform": FORCED_ALIGNMENT_TIMESTAMP_TRANSFORM,
            "upstream_timestamp_repair_policy": FORCED_ALIGNMENT_REPAIR_POLICY,
            "decoder_algorithm_version": FORCED_ALIGNMENT_DECODER_ALGORITHM_VERSION,
            "raw_argmax_sha256": _aggregate_constrained_decode_hash(
                alignment_chunk_receipts,
                "raw_argmax_sha256",
            ),
            "constrained_token_sha256": _aggregate_constrained_decode_hash(
                alignment_chunk_receipts,
                "constrained_token_sha256",
            ),
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
                "processing_mode": ("STRING", {"default": "qwen3_asr_plus_forced_aligner_chunk_merge"}),
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
        processing_mode="qwen3_asr_plus_forced_aligner_chunk_merge",
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
        if segments:
            if (
                [entry[0] for entry in _normalized_characters(transcript)] !=
                [entry[0] for entry in _normalized_characters("".join(item["text"] for item in segments))]
            ):
                raise ValueError("hydra_transcript_receipt_timestamp_text_not_exact")
            if source_duration_seconds is not None:
                tolerance = max(1e-6, 1.0 / max(1, int(sample_rate)))
                if segments[-1]["end_seconds"] > source_duration_seconds + tolerance:
                    raise ValueError("hydra_transcript_receipt_timestamp_out_of_bounds")
        chunk_limit = max(30, min(1200, int(max_chunk_seconds)))
        execution_details = None
        if _text(processing_metadata):
            try:
                execution_details = json.loads(_text(processing_metadata))
            except json.JSONDecodeError as error:
                raise ValueError("hydra_transcript_receipt_processing_metadata_invalid") from error
        verified_evidence = None
        exact_timestamp_evidence = False
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
                if bool(strict_alignment) or normalized_mode.startswith("qwen3_forced_alignment")
                else "transcribe"
            )
            if _text(verified_evidence.get("operation")) != expected_operation:
                raise ValueError("hydra_transcript_receipt_execution_operation_mismatch")
            expected_mode = (
                "qwen3_forced_alignment_exact_anchor_chunk_merge"
                if expected_operation == "forced_alignment"
                else "qwen3_asr_plus_forced_aligner_chunk_merge"
            )
            compatible_modes = {
                "forced_alignment": {
                    "qwen3_forced_alignment_exact_anchor_chunk_merge",
                    "qwen3_forced_alignment_anchor_chunk_merge",
                },
                "transcribe": {
                    "qwen3_asr_plus_forced_aligner_chunk_merge",
                    "qwen3_asr_native_chunk_merge",
                },
            }
            if normalized_mode not in compatible_modes[expected_operation]:
                raise ValueError("hydra_transcript_receipt_processing_mode_mismatch")
            if bool(strict_alignment) != (expected_operation == "forced_alignment"):
                raise ValueError("hydra_transcript_receipt_strict_alignment_mismatch")
            if not isinstance(execution_details, dict):
                raise ValueError("hydra_transcript_receipt_execution_metadata_required")
            if not segments:
                raise ValueError("hydra_transcript_receipt_precise_timestamps_required")
            constrained_forced_alignment = expected_operation == "forced_alignment"
            expected_provenance = (
                FORCED_ALIGNMENT_TIMESTAMP_PROVENANCE
                if constrained_forced_alignment
                else "qwen3_forced_aligner_native"
            )
            expected_transform = (
                FORCED_ALIGNMENT_TIMESTAMP_TRANSFORM
                if constrained_forced_alignment
                else "offset_only"
            )
            expected_repair_policy = (
                FORCED_ALIGNMENT_REPAIR_POLICY
                if constrained_forced_alignment
                else "reject_non_monotonic_raw_tokens"
            )
            if (
                _text(execution_details.get("timestamp_provenance")) != expected_provenance
                or execution_details.get("estimated_timestamps") is not False
                or _text(execution_details.get("timestamp_transform")) != expected_transform
                or _text(execution_details.get("upstream_timestamp_repair_policy"))
                != expected_repair_policy
            ):
                raise ValueError("hydra_transcript_receipt_exact_timestamp_evidence_required")
            if (
                _text(verified_evidence.get("timestamp_provenance")) != expected_provenance
                or verified_evidence.get("estimated_timestamps") is not False
                or _text(verified_evidence.get("timestamp_transform")) != expected_transform
                or _text(verified_evidence.get("upstream_timestamp_repair_policy"))
                != expected_repair_policy
            ):
                raise ValueError("hydra_transcript_receipt_exact_execution_evidence_required")
            chunks = execution_details.get("chunks") or []
            for chunk in chunks:
                if (
                    not isinstance(chunk, dict)
                    or _text(chunk.get("timestamp_transform")) != expected_transform
                    or chunk.get("estimated_timestamps") is not False
                    or "timestamp_scale" in chunk
                ):
                    raise ValueError("hydra_transcript_receipt_exact_chunk_evidence_required")
            if constrained_forced_alignment:
                exact_decode_fields = {
                    "timestamp_provenance": FORCED_ALIGNMENT_TIMESTAMP_PROVENANCE,
                    "timestamp_transform": FORCED_ALIGNMENT_TIMESTAMP_TRANSFORM,
                    "upstream_timestamp_repair_policy": FORCED_ALIGNMENT_REPAIR_POLICY,
                    "decoder_algorithm_version": FORCED_ALIGNMENT_DECODER_ALGORITHM_VERSION,
                    "estimated_timestamps": False,
                }
                if not chunks or any(
                    not _constrained_decode_evidence_valid(chunk.get("timestamp_decode"))
                    for chunk in chunks
                ):
                    raise ValueError("hydra_transcript_receipt_constrained_decode_evidence_required")
                raw_argmax_sha256 = _aggregate_constrained_decode_hash(
                    chunks,
                    "raw_argmax_sha256",
                )
                constrained_token_sha256 = _aggregate_constrained_decode_hash(
                    chunks,
                    "constrained_token_sha256",
                )
                if (
                    any(execution_details.get(field) != expected for field, expected in exact_decode_fields.items())
                    or any(verified_evidence.get(field) != expected for field, expected in exact_decode_fields.items())
                    or any(chunk.get(field) != expected for chunk in chunks for field, expected in exact_decode_fields.items())
                    or _text(execution_details.get("raw_argmax_sha256")).lower() != raw_argmax_sha256
                    or _text(verified_evidence.get("raw_argmax_sha256")).lower() != raw_argmax_sha256
                    or _text(execution_details.get("constrained_token_sha256")).lower()
                    != constrained_token_sha256
                    or _text(verified_evidence.get("constrained_token_sha256")).lower()
                    != constrained_token_sha256
                ):
                    raise ValueError("hydra_transcript_receipt_constrained_decode_evidence_required")
            if (
                expected_operation == "forced_alignment"
                and execution_details.get("automatic_chunking") is True
                and _text(execution_details.get("anchor_text_match")) != "exact_normalized"
            ):
                raise ValueError("hydra_transcript_receipt_exact_anchor_evidence_required")
            metadata_hash = hashlib.sha256(
                json.dumps(execution_details, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if _text(verified_evidence.get("execution_metadata_sha256")) != metadata_hash:
                raise ValueError("hydra_transcript_receipt_execution_metadata_mismatch")
            exact_timestamp_evidence = True
        verified_profile = verified_evidence.get("inference_profile") if verified_evidence else None
        production_verified = (
            isinstance(verified_profile, dict)
            and verified_profile.get("quality_admission") == "production"
            and _text(verified_evidence.get("model_id")) == "Qwen/Qwen3-ASR-1.7B"
            and _text(verified_evidence.get("aligner_id")) == "Qwen/Qwen3-ForcedAligner-0.6B"
            and bool(segments)
            and exact_timestamp_evidence
        )
        canonical_mode = (
            "qwen3_forced_alignment_exact_anchor_chunk_merge"
            if bool(strict_alignment)
            else "qwen3_asr_plus_forced_aligner_chunk_merge"
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
                "processing_mode": canonical_mode,
                "requested_processing_mode": _text(processing_mode) or canonical_mode,
                "source_duration_seconds": round(source_duration_seconds, 6) if source_duration_seconds is not None else None,
                "max_chunk_seconds": chunk_limit,
                "automatic_chunking": (
                    execution_details.get("automatic_chunking")
                    if isinstance(execution_details, dict)
                    else source_duration_seconds is not None and source_duration_seconds > chunk_limit
                ),
                "chunk_merge": True,
                "timestamp_provenance": (
                    execution_details.get("timestamp_provenance") if isinstance(execution_details, dict) else None
                ),
                "estimated_timestamps": (
                    execution_details.get("estimated_timestamps") if isinstance(execution_details, dict) else None
                ),
                "timestamp_transform": (
                    execution_details.get("timestamp_transform") if isinstance(execution_details, dict) else None
                ),
                "upstream_timestamp_repair_policy": (
                    execution_details.get("upstream_timestamp_repair_policy")
                    if isinstance(execution_details, dict)
                    else None
                ),
                "timestamp_offsets_preserved": production_verified,
                "timeline_monotonic": bool(segments),
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
