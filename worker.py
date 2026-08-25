import json
import importlib.util
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


PLUGIN_DIR = Path(__file__).resolve().parent
PRIVATE_DEPS = PLUGIN_DIR / "python_deps"
PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(PRIVATE_DEPS))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def reply(payload: dict) -> None:
    PROTOCOL_OUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    PROTOCOL_OUT.flush()


def load_model(command: dict):
    from top_tts_vendor.indextts.infer_v2_5 import IndexTTS2

    if command["use_gpt_acceleration"] and importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError("hydra_indextts25_flash_attention_required")
    if command["use_torch_compile"] and not hasattr(torch, "compile"):
        raise RuntimeError("hydra_indextts25_torch_compile_unavailable")

    model = IndexTTS2(
        cfg_path=command["config_path"],
        model_dir=command["model_dir"],
        use_bf16=command["use_bf16"],
        device=command["device"],
        use_cuda_kernel=command["use_cuda_kernel"],
        use_deepspeed=False,
        use_accel=command["use_gpt_acceleration"],
        use_torch_compile=command["use_torch_compile"],
        use_qwen_emo=command["use_qwen_emo"],
    )
    if command["use_cuda_kernel"] and not model.use_cuda_kernel:
        raise RuntimeError("hydra_indextts25_cuda_kernel_activation_failed")
    if command["optimization_profile"] != "legacy_custom" and command["use_bf16"]:
        if not str(model.device).startswith("cuda") or not bool(model.use_bf16):
            raise RuntimeError("hydra_indextts25_bf16_cuda_activation_failed")
    if command["use_gpt_acceleration"]:
        if getattr(getattr(model, "gpt", None), "accel_engine", None) is None:
            raise RuntimeError("hydra_indextts25_gpt_acceleration_activation_failed")
    model._hydra_optimization_profile = str(command["optimization_profile"])
    return model


def _parameter_dtype(module) -> str:
    try:
        value = str(next(module.parameters()).dtype).replace("torch.", "")
        return {
            "bfloat16": "bf16",
            "float16": "fp16",
            "float32": "fp32",
        }.get(value, value)
    except (AttributeError, StopIteration, TypeError):
        return "unknown"


def runtime_profile(model, command: dict, *, synthesis_completed: bool) -> dict:
    gpt = getattr(model, "gpt", None)
    accel_engine = getattr(gpt, "accel_engine", None)
    s2mel_models = getattr(getattr(model, "s2mel", None), "models", None)
    cfm = s2mel_models["cfm"] if s2mel_models is not None and "cfm" in s2mel_models else None
    estimator = getattr(cfm, "estimator", None)
    compile_wrapped = bool(model.use_torch_compile and hasattr(estimator, "_orig_mod"))
    gpt_accel_active = accel_engine is not None
    stage_precisions = {
        "gpt": _parameter_dtype(getattr(accel_engine, "model", None) if gpt_accel_active else gpt),
        "s2mel": _parameter_dtype(getattr(model, "s2mel", None)),
        "bigvgan": _parameter_dtype(getattr(model, "bigvgan", None)),
    }
    observed = {value for value in stage_precisions.values() if value != "unknown"}
    if "fp16" in observed and bool(model.use_bf16) and "fp32" in observed:
        precision = "mixed_bf16_fp16_fp32"
    elif "fp16" in observed and bool(model.use_bf16):
        precision = "mixed_bf16_fp16"
    elif bool(model.use_bf16) and "fp32" in observed:
        precision = "mixed_bf16_fp32"
    elif bool(model.use_bf16):
        precision = "bf16"
    else:
        precision = "fp32"
    profile_key = str(command.get("optimization_profile") or getattr(model, "_hydra_optimization_profile", "unverified"))
    quality_admission = {
        "quality_bf16": "production",
        "compiled_bf16": "candidate",
        "maximum_bf16": "candidate",
        "reference_fp32": "quality_reference",
        "legacy_custom": "unverified_legacy",
    }.get(profile_key, "unverified")
    return {
        "contract_version": "hydra_inferworks_tts_inference_profile.v1",
        "profile_key": profile_key,
        "device": str(model.device),
        "precision": precision,
        "quantization": "none",
        "stage_precisions": stage_precisions,
        "acceleration": {
            "gpt_flash_attention": gpt_accel_active,
            "torch_compile": bool(compile_wrapped and synthesis_completed),
            "bigvgan_cuda_kernel": bool(model.use_cuda_kernel),
        },
        "acceleration_state": {
            "gpt_flash_attention": "active" if gpt_accel_active else "disabled",
            "torch_compile": (
                "executed"
                if compile_wrapped and synthesis_completed
                else "wrapped_not_executed"
                if compile_wrapped
                else "disabled"
            ),
            "bigvgan_cuda_kernel": "active" if model.use_cuda_kernel else "disabled",
        },
        "quality_admission": quality_admission,
    }


def require_production_profile_truth(profile: dict) -> None:
    if profile.get("profile_key") != "quality_bf16":
        return
    expected_stages = {"gpt": "bf16", "s2mel": "fp32", "bigvgan": "fp32"}
    if (
        profile.get("precision") != "mixed_bf16_fp32"
        or profile.get("quantization") != "none"
        or profile.get("stage_precisions") != expected_stages
        or any(bool(value) for value in profile.get("acceleration", {}).values())
        or profile.get("quality_admission") != "production"
        or not str(profile.get("device", "")).startswith("cuda")
    ):
        raise RuntimeError("hydra_indextts25_quality_bf16_runtime_attestation_failed")


def synthesize(model, command: dict) -> str:
    kwargs = command["kwargs"]
    seed = int(kwargs["seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    result = model.infer(
        spk_audio_prompt=command["speaker_path"],
        text=command["text"],
        output_path=None,
        lang=command["language"],
        emo_audio_prompt=command.get("emotion_path"),
        emo_alpha=kwargs["emotion_strength"],
        emo_vector=command.get("emotion"),
        use_emo_text=bool(command.get("emotion_text")),
        emo_text=command.get("emotion_text") or None,
        use_random=kwargs["randomize_emotion"],
        interval_silence=kwargs["interval_silence_ms"],
        verbose=False,
        max_text_tokens_per_segment=kwargs["max_text_tokens_per_segment"],
        duration_factor=kwargs["duration_factor"],
        text_normalization=kwargs["text_normalization"],
        temperature=kwargs["temperature"],
        top_p=kwargs["top_p"],
        top_k=kwargs["top_k"],
        num_beams=kwargs["num_beams"],
        repetition_penalty=kwargs["repetition_penalty"],
        max_mel_tokens=kwargs["max_mel_tokens"],
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(f"IndexTTS returned no audio: {result!r}")
    sample_rate, samples = result
    sf.write(command["output_path"], samples, int(sample_rate), subtype="PCM_16")
    return command["output_path"]


def main() -> None:
    model = None
    for line in sys.stdin:
        try:
            command = json.loads(line)
            action = command["action"]
            if action == "probe":
                import transformers
                from top_tts_vendor.indextts.infer_v2_5 import IndexTTS2

                reply({"ok": True, "transformers": transformers.__version__, "engine": IndexTTS2.__name__})
            elif action == "load":
                model = load_model(command)
                profile = runtime_profile(model, command, synthesis_completed=False)
                require_production_profile_truth(profile)
                reply({
                    "ok": True,
                    "device": str(model.device),
                    "precision": profile["precision"],
                    "quantization": "none",
                    "optimization_profile": command["optimization_profile"],
                    "acceleration": profile["acceleration"],
                    "runtime_profile": profile,
                })
            elif action == "synthesize":
                if model is None:
                    raise RuntimeError("Model is not loaded")
                output_path = synthesize(model, command)
                profile = runtime_profile(model, command, synthesis_completed=True)
                require_production_profile_truth(profile)
                reply({
                    "ok": True,
                    "output_path": output_path,
                    "runtime_profile": profile,
                })
            elif action == "shutdown":
                reply({"ok": True})
                return
            else:
                raise ValueError(f"Unknown worker action: {action}")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
