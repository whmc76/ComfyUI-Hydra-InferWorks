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
    return model


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
                reply({
                    "ok": True,
                    "device": str(model.device),
                    "precision": "bf16" if model.use_bf16 else "fp32",
                    "quantization": "none",
                    "optimization_profile": command["optimization_profile"],
                    "acceleration": {
                        "gpt_flash_attention": bool(model.use_accel),
                        "torch_compile": bool(model.use_torch_compile),
                        "bigvgan_cuda_kernel": bool(model.use_cuda_kernel),
                    },
                })
            elif action == "synthesize":
                if model is None:
                    raise RuntimeError("Model is not loaded")
                output_path = synthesize(model, command)
                reply({"ok": True, "output_path": output_path})
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
