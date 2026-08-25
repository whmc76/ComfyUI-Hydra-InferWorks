# Changelog

## 1.1.0 - 2026-08-25

- Add explicit IndexTTS 2.5 quality, compiled, maximum-acceleration, and FP32 reference profiles.
- Make requested CUDA-kernel and FlashAttention acceleration fail closed when it is not actually available.
- Add the Hydra-owned Qwen3-ASR model loader with BF16 SDPA, FlashAttention 2, vLLM, and FP32 reference profiles.
- Record precision, quantization, attention backend, acceleration flags, and quality-admission state in model information and transcript receipts.
- Keep unverified INT8/INT4 checkpoints out of production profiles until task-specific voice and transcription quality gates pass.

## 1.0.0 - 2026-08-25

- Rename the unified public node pack to Hydra InferWorks.
- Merge the production IndexTTS 2.5 implementation from ComfyUI Top TTS.
- Make deterministic IndexTTS worker cleanup and dependency-ordered unload part of the released plugin.
- Add Hydra long-audio Qwen3-ASR, locked-script forced alignment, and immutable transcript receipt nodes.
- Preserve all existing Top TTS and HeyGem class types for workflow compatibility.
- Isolate TTS, ASR, and HeyGem imports so an unavailable optional capability does not disable the others.
- Retain the file-backed, configurable-endpoint HeyGem node and its durable receipt contract.

## 0.2.0

- Add a caller-owned, path-safe `job_code` input for deterministic prompt, receipt, and artifact correlation.
- Preserve automatic UUID generation for interactive ComfyUI use.

## 0.1.0 - 2026-08-03

- Add the `HydraHeyGemLongformAvatar` ComfyUI node.
- Support configurable service URL, host, port, submit/query/health paths.
- Support external and existing-Docker-container lifecycle modes.
- Keep long video inputs and outputs file-backed.
- Add shared mount path validation, artifact hashes, and durable receipts.
- Persist each generation receipt atomically and verify Docker release state after stop.
- Mark generation as an uncached ComfyUI output node so API workflows execute it directly.
- Write staged WAV inputs with standard PCM encoding so TorchCodec is not required.
