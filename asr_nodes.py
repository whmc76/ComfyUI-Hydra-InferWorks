import hashlib
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

import folder_paths
import numpy as np
import torch


def _text(value):
    return str(value or "").strip()


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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "language", "timestamps", "processing_metadata")
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
        metadata = {
            "contract_version": "hydra_qwen3_long_asr_execution.v1",
            "automatic_chunking": len(chunk_receipts) > 1,
            "actual_chunk_count": len(chunk_receipts),
            "max_chunk_seconds": chunk_limit,
            "split_strategy": "lowest_rms_before_hard_limit",
            "chunks": chunk_receipts,
        }
        return ("".join(texts), resolved_language, "\n".join(lines), json.dumps(metadata, ensure_ascii=False))


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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "language", "timestamps", "processing_metadata")
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
        metadata = {
            "contract_version": "hydra_qwen3_forced_alignment_execution.v1",
            "automatic_chunking": len(plans) > 1,
            "actual_chunk_count": len(plans),
            "max_chunk_seconds": chunk_limit,
            "split_strategy": "asr_timestamp_anchor_groups",
            "anchor_character_error_rate": round(anchor_error_rate, 6) if anchor_error_rate is not None else None,
            "chunks": alignment_chunk_receipts,
        }
        return (transcript, resolved_language, "\n".join(lines), json.dumps(metadata, ensure_ascii=False))


class HydraTranscriptReceipt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
                "language": ("STRING", {"forceInput": True}),
                "timestamps": ("STRING", {"multiline": True, "forceInput": True}),
                "filename_prefix": ("STRING", {"default": "hydramatrix/qwen3-asr/transcript"}),
                "source_audio_sha256": ("STRING", {"default": ""}),
                "model_id": ("STRING", {"default": "Qwen/Qwen3-ASR-1.7B"}),
                "aligner_id": ("STRING", {"default": "Qwen/Qwen3-ForcedAligner-0.6B"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "processing_mode": ("STRING", {"default": "qwen3_asr_native_chunk_merge"}),
                "max_chunk_seconds": ("INT", {"default": 180, "min": 30, "max": 1200}),
                "strict_alignment": ("BOOLEAN", {"default": False}),
                "processing_metadata": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("receipt_path",)
    FUNCTION = "write_receipt"
    CATEGORY = "Hydra InferWorks/ASR"
    OUTPUT_NODE = True

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
        payload = {
            "contract_version": "hydra_comfyui_qwen3_asr_transcript_receipt.v1",
            "status": "completed",
            "source_audio_sha256": source_hash,
            "model_id": _text(model_id),
            "aligner_id": _text(aligner_id) or None,
            "language": _text(language) or None,
            "text": transcript,
            "timestamps_present": bool(segments),
            "segments": segments,
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
    "HydraTranscriptReceipt": HydraTranscriptReceipt,
    "HydraQwen3ForcedAlign": HydraQwen3ForcedAlign,
    "HydraQwen3LongAsrTranscribe": HydraQwen3LongAsrTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HydraTranscriptReceipt": "Hydra InferWorks · Transcript Receipt",
    "HydraQwen3ForcedAlign": "Hydra InferWorks · Qwen3 Locked-Script Forced Align",
    "HydraQwen3LongAsrTranscribe": "Hydra InferWorks · Qwen3 Long-Audio Transcribe",
}

