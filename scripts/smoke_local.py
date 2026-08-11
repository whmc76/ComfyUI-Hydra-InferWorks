"""Run a real local-weight synthesis through the same classes used by ComfyUI."""

import argparse
import importlib.util
import sys
from pathlib import Path

import soundfile as sf
import torch


PLUGIN_DIR = Path(__file__).resolve().parents[1]
COMFYUI_DIR = PLUGIN_DIR.parents[1]


def load_nodes_module():
    sys.path.insert(0, str(COMFYUI_DIR))
    spec = importlib.util.spec_from_file_location("top_tts_smoke_nodes", PLUGIN_DIR / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--text", default="你好，这是 ComfyUI Top TTS 的本地推理验证。")
    args = parser.parse_args()

    nodes = load_nodes_module()
    samples, sample_rate = sf.read(args.reference_audio, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T.copy())
    reference_audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    model, model_info = nodes.TopTTS25ModelLoader().load(
        model_directory="IndexTTS-2.5",
        device="cuda",
        precision="bf16",
        load_emotion_text_model=False,
        use_cuda_kernel=False,
        use_torch_compile=False,
    )
    try:
        audio, generation_info = nodes.TopTTS25Synthesize().synthesize(
            model=model,
            text=args.text,
            reference_audio=reference_audio,
            language="ZH",
            duration_factor=1.0,
            emotion_strength=1.0,
            seed=20260811,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(args.output), audio["waveform"][0].T.numpy(), audio["sample_rate"], subtype="PCM_16")
        print(model_info)
        print(generation_info)
        print(args.output.resolve())
    finally:
        model.unload()


if __name__ == "__main__":
    main()
