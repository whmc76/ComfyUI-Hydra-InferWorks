# Changelog

## 1.2.3 - 2026-08-26

- Preserve the complete v1.2.2 native-bin constrained-decode evidence contract while binding every
  alignment chunk to its normalized audio content, exact locked text, and language hashes.
- Add reproducible raw-argmax and constrained-token path hashes to every chunk and aggregate them
  across multi-chunk alignments into the typed execution evidence.
- Fail closed when chunk bindings, path hashes, or aggregate hashes drift; native 80 ms bins,
  offset-only timeline merging, and the explicit no-repair policy remain unchanged.

## 1.2.2 - 2026-08-26

- Decode Qwen3 ForcedAligner timestamp logits with a deterministic global maximum-logit monotonic
  dynamic program over the model's native 80 ms classes.
- Require every word or character to have a positive native-bin span, keep consecutive units
  non-overlapping, and bound all selected bins to the normalized audio duration.
- Bypass upstream `fix_timestamp` and `parse_timestamp` completely on the production forced-alignment
  path; LIS, nearest-neighbor repair, interpolation, averaging, scaling, clamping, and duration
  projection remain forbidden.
- Bind typed execution evidence to the raw greedy bins, selected path, logits and path digests,
  model/runtime structure, constraints, score penalty, and explicit no-repair flags.
- Fail closed on qwen-asr version or structure drift, invalid logits, timestamp-slot mismatch, and
  infeasible paths; old raw-native provenance cannot satisfy the new forced-alignment receipt gate.

## 1.2.1 - 2026-08-26

- Make Qwen3 ForcedAligner native units the only production ASR timing authority.
- Reject non-monotonic raw timestamp tokens before upstream nearest-neighbor or linear-interpolation
  repair can run.
- Remove fuzzy anchor mapping, character-ratio projection, timestamp scaling, clamping, and averaged
  chunk boundaries from the alignment path; exact normalized text coverage is now mandatory.
- Bind receipts to native timestamp provenance, offset-only chunk merging, and the strict upstream
  repair policy. Missing or estimated timing evidence cannot receive `completed` status.
- Raise the direct forced-alignment window to the upstream-supported five-minute limit so shorter
  inputs do not require needless anchor chunking.

## 1.2.0 - 2026-08-26

- Make Hydra InferWorks installable in any ComfyUI project without a HydraMatrix runtime.
- Use `INFERWORKS_*` and upstream `HEYGEM_*` variables as the canonical configuration surface,
  while retaining existing `HYDRA_*` names as compatibility aliases.
- Remove implicit HeyGem host/port, `/code/data`, `hm-heygem`, and GPU-release route defaults.
- Resolve the shared directory independently in the ComfyUI and HeyGem service namespaces,
  so host, container, remote, Windows, and Linux deployments can provide their own mount paths.
- Keep all existing `TopTTS25*`, `HydraQwen3*`, and `HydraHeyGem*` class types so published
  workflows remain compatible.
- Replace Hydra-only placeholders and output prefixes in the public workflow examples.

## 1.1.4 - 2026-08-26

- Reject a HeyGem GPU release response when the provider reports a non-empty cleanup error.
- When the provider reports CUDA as available, require both CUDA cache and IPC cleanup to be
  explicitly accepted before emitting a successful GPU release receipt.

## 1.1.3 - 2026-08-25

- Let externally supervised HeyGem deployments release service GPU memory after artifact
  materialization without exposing Docker control to the ComfyUI container.
- Persist the requested path and exact provider response in the durable HeyGem receipt.

## 1.1.2 - 2026-08-25

- Install the lightweight `soundfile` import dependency in the public CPU CI matrix so
  the TTS loader and worker runtime-profile contracts are exercised on Python 3.10–3.12.

## 1.1.1 - 2026-08-25

- Preserve v1.0 TTS precision and acceleration inputs when an older workflow has no optimization-profile field.
- Fail BF16 profiles closed unless CUDA BF16 actually activates, and distinguish requested, wrapped, and executed acceleration.
- Report the accelerated GPT stage as FP16/mixed precision instead of mislabeling the full maximum profile as BF16.
- Write the observed TTS runtime profile into ComfyUI history as `hydra_inferworks_tts_execution.v1` evidence.
- Attest the exact official Qwen3-ASR 1.7B and ForcedAligner 0.6B weights, configs, processors, and tokenizers before granting production admission.
- Bind typed ASR execution evidence to the audio content, transcript, language, timestamps, actual runtime profile, and attested model identity.
- Downgrade legacy caller-provided ASR metadata to `completed_unverified` instead of accepting it as production runtime truth.

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
