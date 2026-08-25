from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Optional
from urllib.request import Request, urlopen

import torch
from comfy_api.latest import ComfyExtension, Input, InputImpl, Types, io
from typing_extensions import override

import comfy.model_management

from .hydra_heygem.client import HeyGemClient
from .hydra_heygem.config import resolve_endpoint_config
from .hydra_heygem.lifecycle import DockerContainerLifecycle
from .hydra_heygem.job_identity import resolve_job_code
from .hydra_heygem.paths import map_result_to_host, prefer_final_muxed_artifact


CONTRACT_VERSION = "hydra_comfyui_heygem_longform_avatar_receipt.v1"


def _text(value: object) -> str:
    return str(value or "").strip()


def _is_auto(value: object) -> bool:
    return _text(value).lower() in {"", "auto", "default", "environment", "env"}


def _first_environment_value(*keys: str) -> str:
    for key in keys:
        value = _text(os.environ.get(key))
        if value:
            return value
    return ""


def _resolve_shared_host_root(value: object) -> Path:
    configured = "" if _is_auto(value) else _text(value)
    configured = configured or _first_environment_value(
        "HYDRA_HEYGEM_SHARED_HOST_ROOT",
        "HYDRA_AVATAR_STAGE_DIR",
        "AVATAR_STAGE_DIR",
        "EXTERNAL_HEYGEM_STAGE_DIR",
        "HEYGEM_DATA_DIR",
    )
    if not configured:
        raise ValueError("hydra_heygem_shared_host_root_required")
    root = Path(configured).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_container_data_root(value: object) -> str:
    configured = "" if _is_auto(value) else _text(value)
    configured = configured or _first_environment_value(
        "HYDRA_HEYGEM_CONTAINER_DATA_ROOT",
        "HYDRA_AVATAR_CONTAINER_DATA_DIR",
        "AVATAR_CONTAINER_DATA_DIR",
        "HEYGEM_CONTAINER_DATA_DIR",
    )
    return (configured or "/code/data").replace("\\", "/").rstrip("/")


def _resolve_container_name(value: object) -> str:
    configured = "" if _is_auto(value) else _text(value)
    return configured or _first_environment_value("HYDRA_HEYGEM_CONTAINER_NAME") or "hm-heygem"


def _container_path(container_root: str, relative: PurePosixPath) -> str:
    return f"{container_root.rstrip('/')}/{relative.as_posix().lstrip('/')}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic_json(value: object, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target.resolve()


def _save_audio(audio: Input.Audio, target: Path) -> None:
    waveform = audio["waveform"]
    if waveform is None or waveform.ndim not in {2, 3}:
        raise ValueError("hydra_heygem_audio_waveform_invalid")
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.shape[-1] <= 0:
        raise ValueError("hydra_heygem_audio_waveform_empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    pcm = (
        waveform.detach()
        .cpu()
        .float()
        .clamp(-1.0, 1.0)
        .mul(32767.0)
        .round()
        .to(dtype=torch.int16)
        .transpose(0, 1)
        .contiguous()
        .numpy()
        .tobytes()
    )
    with wave.open(str(target), "wb") as output:
        output.setnchannels(int(waveform.shape[0]))
        output.setsampwidth(2)
        output.setframerate(int(audio["sample_rate"]))
        output.writeframes(pcm)


def _save_reference_video(reference_video: Input.Video, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    reference_video.save_to(
        str(target),
        format=Types.VideoContainer.MP4,
        codec=Types.VideoCodec.AUTO,
    )


def _download_result(url: str, target: Path, timeout_seconds: float) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, method="GET", headers={"accept": "video/*,application/octet-stream"})
    with urlopen(request, timeout=max(float(timeout_seconds), 0.1)) as response:
        with target.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    return target


def _wait_for_artifact(path: Path, timeout_seconds: float) -> Path:
    deadline = time.monotonic() + max(float(timeout_seconds), 0.1)
    while time.monotonic() <= deadline:
        comfy.model_management.throw_exception_if_processing_interrupted()
        if path.is_file() and path.stat().st_size > 0:
            return path.resolve()
        time.sleep(0.25)
    raise ValueError(f"hydra_heygem_result_artifact_missing_or_empty:{path}")


class HydraHeyGemLongformAvatar(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HydraHeyGemLongformAvatar",
            display_name="Hydra HeyGem Long-form Avatar",
            category="Hydra InferWorks/Avatar",
            description=(
                "Generate a long-form HeyGem avatar through a configurable service endpoint. "
                "The result remains file-backed and is never expanded into an IMAGE frame tensor."
            ),
            inputs=[
                io.Audio.Input("audio", tooltip="The exact authoritative speech waveform."),
                io.Video.Input("reference_video", tooltip="The locked HeyGem presenter reference video."),
                io.String.Input(
                    "job_code",
                    default="auto",
                    tooltip=(
                        "Optional caller-owned correlation identity. Safe explicit values make the "
                        "durable receipt path deterministic; auto generates a unique identity."
                    ),
                ),
                io.String.Input(
                    "service_url",
                    default="auto",
                    tooltip="Full http(s) service URL. Overrides host/port and environment when set.",
                ),
                io.String.Input(
                    "service_host",
                    default="auto",
                    tooltip="Editable service host. Used with service_port when service_url is auto.",
                ),
                io.Int.Input(
                    "service_port",
                    default=0,
                    min=0,
                    max=65535,
                    step=1,
                    tooltip="Editable service port. 0 resolves from environment or compatibility default.",
                ),
                io.String.Input("submit_path", default="/easy/submit"),
                io.String.Input("query_path", default="/easy/query"),
                io.String.Input(
                    "health_path",
                    default="auto",
                    tooltip="Health route, or auto to probe query_path with code=healthcheck.",
                ),
                io.String.Input(
                    "shared_host_root",
                    default="auto",
                    tooltip="Host directory mounted into the HeyGem container. Required via input or environment.",
                ),
                io.String.Input("container_data_root", default="auto"),
                io.Combo.Input(
                    "lifecycle_mode",
                    options=["external", "docker_existing_container"],
                    default="external",
                ),
                io.String.Input("container_name", default="auto"),
                io.Boolean.Input("release_comfyui_models", default=True),
                io.Boolean.Input("stop_container_after", default=False),
                io.Int.Input("ready_timeout_seconds", default=90, min=1, max=900, step=1),
                io.Int.Input("generation_timeout_seconds", default=7200, min=10, max=86400, step=10),
                io.Float.Input("poll_interval_seconds", default=3.0, min=0.1, max=60.0, step=0.1),
                io.Int.Input("artifact_wait_seconds", default=60, min=1, max=600, step=1),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
                io.String.Output(display_name="artifact_path"),
                io.String.Output(display_name="receipt_json"),
            ],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(
        cls,
        audio: Input.Audio,
        reference_video: Input.Video,
        job_code: str,
        service_url: str,
        service_host: str,
        service_port: int,
        submit_path: str,
        query_path: str,
        health_path: str,
        shared_host_root: str,
        container_data_root: str,
        lifecycle_mode: str,
        container_name: str,
        release_comfyui_models: bool,
        stop_container_after: bool,
        ready_timeout_seconds: int,
        generation_timeout_seconds: int,
        poll_interval_seconds: float,
        artifact_wait_seconds: int,
    ) -> io.NodeOutput:
        endpoint = resolve_endpoint_config(
            service_url=service_url,
            service_host=service_host,
            service_port=service_port,
        )
        shared_root = _resolve_shared_host_root(shared_host_root)
        container_root = _resolve_container_data_root(container_data_root)
        resolved_container_name = _resolve_container_name(container_name)
        job_code = resolve_job_code(job_code)
        audio_relative = PurePosixPath("inputs") / "audio" / f"{job_code}.wav"
        video_relative = PurePosixPath("inputs") / "video" / f"{job_code}.mp4"
        audio_host_path = shared_root.joinpath(*audio_relative.parts)
        video_host_path = shared_root.joinpath(*video_relative.parts)
        _save_audio(audio, audio_host_path)
        _save_reference_video(reference_video, video_host_path)

        if release_comfyui_models:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache(force=True)

        lifecycle = DockerContainerLifecycle(
            lifecycle_mode,
            resolved_container_name,
        )
        prepare_receipt = lifecycle.prepare()
        release_receipt = None
        generation_receipt = None
        ready_receipt = None
        output_path: Optional[Path] = None
        try:
            resolved_health_path = (
                f"{query_path}{'&' if '?' in query_path else '?'}code=healthcheck"
                if _is_auto(health_path)
                else health_path
            )
            client = HeyGemClient(
                endpoint,
                submit_path=submit_path,
                query_path=query_path,
                interrupt_check=comfy.model_management.throw_exception_if_processing_interrupted,
            )
            ready_receipt = client.wait_until_ready(
                health_path=resolved_health_path,
                timeout_seconds=ready_timeout_seconds,
            )
            generation_receipt = client.generate(
                code=job_code,
                audio_container_path=_container_path(container_root, audio_relative),
                video_container_path=_container_path(container_root, video_relative),
                timeout_seconds=generation_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                extra_payload={
                    "watermark_switch": 0,
                    "digital_auth": 0,
                    "chaofen": 0,
                    "pn": 0,
                },
            )
            mapped_result = map_result_to_host(
                generation_receipt.result,
                shared_host_root=shared_root,
                container_data_root=container_root,
            )
            mapped_result = prefer_final_muxed_artifact(
                mapped_result,
                code=job_code,
                shared_host_root=shared_root,
            )
            if isinstance(mapped_result, str):
                output_path = _download_result(
                    mapped_result,
                    shared_root / "temp" / f"{job_code}-download.mp4",
                    artifact_wait_seconds,
                )
            else:
                output_path = mapped_result
            output_path = _wait_for_artifact(output_path, artifact_wait_seconds)
        finally:
            release_receipt = lifecycle.release(stop_after_job=stop_container_after)

        if generation_receipt is None or output_path is None:
            raise ValueError("hydra_heygem_generation_receipt_missing")
        receipt_path = (shared_root / "receipts" / f"{job_code}.json").resolve()
        receipt = {
            "contract_version": CONTRACT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "job_code": job_code,
            "receipt_path": str(receipt_path),
            "service": {
                "base_url": endpoint.base_url,
                "host": endpoint.host,
                "port": endpoint.port,
                "resolution_source": endpoint.source,
                "submit_path": submit_path,
                "query_path": query_path,
                "health_path": resolved_health_path,
                "ready": ready_receipt,
            },
            "lifecycle": {
                "mode": prepare_receipt.mode,
                "container_name": prepare_receipt.container_name,
                "was_running": prepare_receipt.was_running,
                "started": prepare_receipt.started,
                "stopped": release_receipt.stopped if release_receipt else False,
                "release_comfyui_models": bool(release_comfyui_models),
                "stop_container_after": bool(stop_container_after),
            },
            "inputs": {
                "audio_host_path": str(audio_host_path),
                "audio_container_path": _container_path(container_root, audio_relative),
                "audio_sha256": _sha256(audio_host_path),
                "reference_video_host_path": str(video_host_path),
                "reference_video_container_path": _container_path(container_root, video_relative),
                "reference_video_sha256": _sha256(video_host_path),
            },
            "output": {
                "artifact_path": str(output_path),
                "artifact_size_bytes": output_path.stat().st_size,
                "artifact_sha256": _sha256(output_path),
            },
            "provider": {
                "poll_count": generation_receipt.poll_count,
                "elapsed_seconds": generation_receipt.elapsed_seconds,
                "result": generation_receipt.result,
                "final_response": generation_receipt.final_response,
            },
        }
        _write_atomic_json(receipt, receipt_path)
        return io.NodeOutput(
            InputImpl.VideoFromFile(str(output_path)),
            str(output_path),
            json.dumps(receipt, ensure_ascii=False, indent=2),
        )


class HydraHeyGemExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [HydraHeyGemLongformAvatar]


async def comfy_entrypoint() -> HydraHeyGemExtension:
    return HydraHeyGemExtension()


NODE_CLASS_MAPPINGS = {"HydraHeyGemLongformAvatar": HydraHeyGemLongformAvatar}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HydraHeyGemLongformAvatar": "Hydra InferWorks · HeyGem Long-form Avatar"
}
