import atexit
import json
import os
import subprocess
import sys
import tempfile
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import soundfile as sf

import comfy.model_management
import folder_paths


PLUGIN_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PLUGIN_DIR / "top_tts_vendor"
PRIVATE_DEPS_DIR = PLUGIN_DIR / "python_deps"
DEFAULT_MODEL_DIRECTORY = "IndexTTS-2.5"
MODEL_TYPE = "TOP_TTS_2_5_MODEL"
EMOTION_TYPE = "TOP_TTS_2_5_EMOTION"
TTS_OPTIMIZATION_PROFILES = (
    "quality_bf16",
    "compiled_bf16",
    "maximum_bf16",
    "reference_fp32",
    "legacy_custom",
)

MAIN_MODEL_FILES = (
    "config.yaml",
    "codec.pth",
    "feat1.pt",
    "feat2.pt",
    "gpt.pth",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
)
AUXILIARY_MODEL_FILES = (
    "hf_cache/campplus_cn_common.bin",
    "hf_cache/w2v-bert-2.0/config.json",
    "hf_cache/w2v-bert-2.0/model.safetensors",
    "hf_cache/w2v-bert-2.0/preprocessor_config.json",
    "hf_cache/bigvgan/config.json",
    "hf_cache/bigvgan/bigvgan_generator.pt",
)


def _model_directory_names() -> list[str]:
    root = Path(folder_paths.models_dir)
    names = [entry.name for entry in root.iterdir() if entry.is_dir()]
    names = sorted(set(names), key=str.casefold)
    if DEFAULT_MODEL_DIRECTORY in names:
        names.remove(DEFAULT_MODEL_DIRECTORY)
    return [DEFAULT_MODEL_DIRECTORY, *names]


def _resolve_model_directory(name: str) -> Path:
    root = Path(folder_paths.models_dir).resolve()
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Model directory must be inside ComfyUI/models") from exc
    if candidate.parent != root:
        raise ValueError("Choose a direct child directory of ComfyUI/models")
    return candidate


def _missing_model_files(model_dir: Path, load_emotion_model: bool) -> list[str]:
    required = [*MAIN_MODEL_FILES, *AUXILIARY_MODEL_FILES]
    if load_emotion_model:
        required.extend(
            (
                "qwen0.6bemo4-merge/config.json",
                "qwen0.6bemo4-merge/model.safetensors",
                "qwen0.6bemo4-merge/tokenizer.json",
            )
        )
    return [relative for relative in required if not (model_dir / relative).is_file()]


def _ensure_runtime() -> None:
    source = VENDOR_DIR / "indextts" / "infer_v2_5.py"
    if not source.is_file():
        raise RuntimeError(
            "IndexTTS 2.5 inference source is missing from this installation. "
            "Reinstall Hydra InferWorks from its official GitHub repository."
        )
    private_transformers = PRIVATE_DEPS_DIR / "transformers" / "__init__.py"
    private_tokenizers = PRIVATE_DEPS_DIR / "tokenizers" / "__init__.py"
    if not private_transformers.is_file() or not private_tokenizers.is_file():
        raise RuntimeError(
            "Private IndexTTS dependencies are missing. Run install.py with ComfyUI's Python, "
            "then restart ComfyUI."
        )


def _write_reference_audio(audio: dict[str, Any], prefix: str) -> str:
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Reference audio must be a ComfyUI AUDIO value")
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    if waveform.ndim == 3:
        waveform = waveform[0]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"Unsupported reference waveform shape: {tuple(waveform.shape)}")
    waveform = waveform.detach().float().cpu()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    temp_dir = Path(folder_paths.get_temp_directory())
    temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".wav", dir=temp_dir, delete=False) as handle:
        path = handle.name
    sf.write(path, waveform.squeeze(0).numpy(), int(audio["sample_rate"]), subtype="PCM_16")
    return path


def _load_comfy_audio(path: str) -> dict[str, Any]:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T.copy())
    return {"waveform": waveform.unsqueeze(0).contiguous(), "sample_rate": int(sample_rate)}


_WORKERS = weakref.WeakSet()


class TopTTSWorker:
    def __init__(self, load_command: dict[str, Any]):
        log_dir = Path(folder_paths.get_temp_directory())
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / "comfyui_top_tts_worker.log"
        self.log_handle = self.log_path.open("a", encoding="utf-8")
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            [sys.executable, str(PLUGIN_DIR / "worker.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_handle,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=creationflags,
        )
        self.lock = threading.Lock()
        _WORKERS.add(self)
        try:
            self.info = self.request(load_command)
        except Exception:
            self.close()
            raise

    def request(self, command: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError(f"Top TTS worker exited with code {self.process.returncode}. Log: {self.log_path}")
            assert self.process.stdin is not None
            assert self.process.stdout is not None
            self.process.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"Top TTS worker stopped unexpectedly. Log: {self.log_path}")
            response = json.loads(line)
            if not response.get("ok"):
                raise RuntimeError(f"Top TTS worker error: {response.get('error', 'unknown error')}. Log: {self.log_path}")
            return response

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            try:
                self.request({"action": "shutdown"})
                process.wait(timeout=15)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        log_handle = getattr(self, "log_handle", None)
        if log_handle is not None and not log_handle.closed:
            log_handle.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _close_workers() -> None:
    for worker in list(_WORKERS):
        worker.close()


atexit.register(_close_workers)


@dataclass
class TopTTSModel:
    worker: TopTTSWorker
    model_dir: Path
    device: str
    precision: str
    quantization: str
    optimization_profile: str
    acceleration: dict[str, bool]
    runtime_profile: dict[str, Any]
    emotion_model_loaded: bool

    def unload(self) -> None:
        self.worker.close()
        comfy.model_management.soft_empty_cache()

    def require_worker(self) -> TopTTSWorker:
        if self.worker.process.poll() is not None:
            raise RuntimeError("This Top TTS model was unloaded. Run the model loader again.")
        return self.worker


class TopTTS25ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_directory": (_model_directory_names(), {"default": DEFAULT_MODEL_DIRECTORY}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "load_emotion_text_model": ("BOOLEAN", {"default": False}),
                "use_cuda_kernel": ("BOOLEAN", {"default": False}),
                "use_torch_compile": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "optimization_profile": (TTS_OPTIMIZATION_PROFILES, {"default": "quality_bf16"}),
            },
        }

    RETURN_TYPES = (MODEL_TYPE, "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load"
    CATEGORY = "Hydra InferWorks/TTS"
    DESCRIPTION = "Loads official IndexTTS 2.5 weights for fully local inference."

    def load(
        self,
        model_directory: str,
        device: str,
        precision: str,
        load_emotion_text_model: bool,
        use_cuda_kernel: bool,
        use_torch_compile: bool,
        optimization_profile: str = "legacy_custom",
    ):
        model_dir = _resolve_model_directory(model_directory)
        missing = _missing_model_files(model_dir, load_emotion_text_model)
        if missing:
            preview = "\n".join(f"  - {name}" for name in missing[:12])
            extra = f"\n  ... and {len(missing) - 12} more" if len(missing) > 12 else ""
            raise FileNotFoundError(
                f"IndexTTS 2.5 is incomplete in {model_dir}. Missing:\n{preview}{extra}\n"
                "Run download_models.py --accept-license from the plugin directory, then restart ComfyUI."
            )

        _ensure_runtime()
        # Older serialized workflows do not pass newly-added optional inputs.
        # Keep the UI default for new graphs while preserving every v1.0
        # precision and acceleration input when this field is absent.
        profile = str(optimization_profile or "legacy_custom").strip().lower()
        if profile not in TTS_OPTIMIZATION_PROFILES:
            raise ValueError(f"hydra_indextts25_optimization_profile_invalid:{profile}")
        if profile == "quality_bf16":
            precision = "bf16"
            use_cuda_kernel = False
            use_torch_compile = False
            use_gpt_acceleration = False
        elif profile == "compiled_bf16":
            precision = "bf16"
            use_cuda_kernel = False
            use_torch_compile = True
            use_gpt_acceleration = False
        elif profile == "maximum_bf16":
            precision = "bf16"
            use_cuda_kernel = True
            use_torch_compile = True
            use_gpt_acceleration = True
        elif profile == "reference_fp32":
            precision = "fp32"
            use_cuda_kernel = False
            use_torch_compile = False
            use_gpt_acceleration = False
        else:
            use_gpt_acceleration = False
        resolved_device = None if device == "auto" else ("cuda:0" if device == "cuda" else device)
        if resolved_device == "cuda:0" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected, but PyTorch cannot access a CUDA device")
        worker = TopTTSWorker(
            {
                "action": "load",
                "config_path": str(model_dir / "config.yaml"),
                "model_dir": str(model_dir),
                "use_bf16": precision == "bf16",
                "device": resolved_device,
                "use_cuda_kernel": bool(use_cuda_kernel),
                "use_torch_compile": bool(use_torch_compile),
                "use_gpt_acceleration": bool(use_gpt_acceleration),
                "optimization_profile": profile,
                "use_qwen_emo": bool(load_emotion_text_model),
            }
        )
        handle = TopTTSModel(
            worker=worker,
            model_dir=model_dir,
            device=worker.info["device"],
            precision=worker.info["precision"],
            quantization=worker.info["quantization"],
            optimization_profile=worker.info["optimization_profile"],
            acceleration=worker.info["acceleration"],
            runtime_profile=worker.info["runtime_profile"],
            emotion_model_loaded=load_emotion_text_model,
        )
        enabled = ",".join(key for key, value in handle.acceleration.items() if value) or "eager"
        info = (
            f"IndexTTS 2.5 | {handle.device} | {handle.precision} | "
            f"quantization={handle.quantization} | profile={handle.optimization_profile} | "
            f"acceleration={enabled} | {model_dir}"
        )
        return handle, info


class TopTTS25EmotionVector:
    @classmethod
    def INPUT_TYPES(cls):
        slider = {"default": 0.0, "min": 0.0, "max": 1.2, "step": 0.01}
        return {
            "required": {
                "happy": ("FLOAT", dict(slider)),
                "angry": ("FLOAT", dict(slider)),
                "sad": ("FLOAT", dict(slider)),
                "afraid": ("FLOAT", dict(slider)),
                "disgusted": ("FLOAT", dict(slider)),
                "melancholic": ("FLOAT", dict(slider)),
                "surprised": ("FLOAT", dict(slider)),
                "calm": ("FLOAT", {**slider, "default": 1.0}),
            }
        }

    RETURN_TYPES = (EMOTION_TYPE,)
    RETURN_NAMES = ("emotion",)
    FUNCTION = "build"
    CATEGORY = "Hydra InferWorks/TTS"

    def build(self, happy, angry, sad, afraid, disgusted, melancholic, surprised, calm):
        return ([float(happy), float(angry), float(sad), float(afraid), float(disgusted), float(melancholic), float(surprised), float(calm)],)


class TopTTS25Synthesize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "text": ("STRING", {"multiline": True, "default": "你好，欢迎使用 Hydra InferWorks。"}),
                "reference_audio": ("AUDIO",),
                "language": (["ZH", "EN", "JA", "ES", "AR"], {"default": "ZH"}),
                "duration_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "emotion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "emotion": (EMOTION_TYPE,),
                "emotion_audio": ("AUDIO",),
                "emotion_text": ("STRING", {"multiline": True, "default": ""}),
                "randomize_emotion": ("BOOLEAN", {"default": False}),
                "interval_silence_ms": ("INT", {"default": 200, "min": 0, "max": 2000, "step": 10}),
                "max_text_tokens_per_segment": ("INT", {"default": 120, "min": 20, "max": 600, "step": 5}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "max_mel_tokens": ("INT", {"default": 1500, "min": 100, "max": 4000, "step": 10}),
                "text_normalization": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_info")
    FUNCTION = "synthesize"
    CATEGORY = "Hydra InferWorks/TTS"
    DESCRIPTION = "Clones the reference voice and synthesizes speech locally with IndexTTS 2.5."

    def synthesize(
        self,
        model: TopTTSModel,
        text: str,
        reference_audio: dict[str, Any],
        language: str,
        duration_factor: float,
        emotion_strength: float,
        seed: int,
        emotion=None,
        emotion_audio=None,
        emotion_text: str = "",
        randomize_emotion: bool = False,
        interval_silence_ms: int = 200,
        max_text_tokens_per_segment: int = 120,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        max_mel_tokens: int = 1500,
        text_normalization: bool = True,
    ):
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")
        worker = model.require_worker()
        emotion_text = emotion_text.strip()
        if emotion_text and not model.emotion_model_loaded:
            raise RuntimeError("Emotion text requires load_emotion_text_model=True in the Top TTS model loader")

        speaker_path = _write_reference_audio(reference_audio, "top_tts_speaker_")
        emotion_path = None
        output_path = None
        try:
            if emotion_audio is not None and emotion is None and not emotion_text:
                emotion_path = _write_reference_audio(emotion_audio, "top_tts_emotion_")
            with tempfile.NamedTemporaryFile(
                prefix="top_tts_output_",
                suffix=".wav",
                dir=folder_paths.get_temp_directory(),
                delete=False,
            ) as handle:
                output_path = handle.name
            response = worker.request(
                {
                    "action": "synthesize",
                    "speaker_path": speaker_path,
                    "emotion_path": emotion_path,
                    "output_path": output_path,
                    "text": text,
                    "language": language,
                    "emotion": emotion,
                    "emotion_text": emotion_text,
                    "kwargs": {
                        "seed": int(seed),
                        "duration_factor": float(duration_factor),
                        "emotion_strength": float(emotion_strength),
                        "randomize_emotion": bool(randomize_emotion),
                        "interval_silence_ms": int(interval_silence_ms),
                        "max_text_tokens_per_segment": int(max_text_tokens_per_segment),
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "top_k": int(top_k),
                        "num_beams": int(num_beams),
                        "repetition_penalty": float(repetition_penalty),
                        "max_mel_tokens": int(max_mel_tokens),
                        "text_normalization": bool(text_normalization),
                    },
                }
            )
            audio = _load_comfy_audio(response["output_path"])
        finally:
            for path in (speaker_path, emotion_path, output_path):
                if path and os.path.isfile(path):
                    os.unlink(path)
        duration = audio["waveform"].shape[-1] / audio["sample_rate"]
        runtime_profile = response.get("runtime_profile")
        if not isinstance(runtime_profile, dict):
            raise RuntimeError("hydra_indextts25_runtime_profile_missing")
        model.runtime_profile = runtime_profile
        model.precision = str(runtime_profile.get("precision") or model.precision)
        model.acceleration = dict(runtime_profile.get("acceleration") or {})
        info = json.dumps(
            {
                "contract_version": "hydra_inferworks_tts_execution.v1",
                "status": "completed",
                "language": language,
                "duration_seconds": round(duration, 6),
                "seed": int(seed),
                "sample_rate": int(audio["sample_rate"]),
                "runtime_profile": runtime_profile,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "ui": {"generation_info": [info]},
            "result": (audio, info),
        }


class TopTTS25UnloadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": (MODEL_TYPE,)},
            "optional": {"after": ("AUDIO",)},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "Hydra InferWorks/TTS"
    OUTPUT_NODE = True

    def unload(self, model: TopTTSModel, after=None):
        model.unload()
        return ("IndexTTS 2.5 unloaded and cache cleanup requested",)


NODE_CLASS_MAPPINGS = {
    "TopTTS25ModelLoader": TopTTS25ModelLoader,
    "TopTTS25EmotionVector": TopTTS25EmotionVector,
    "TopTTS25Synthesize": TopTTS25Synthesize,
    "TopTTS25UnloadModel": TopTTS25UnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TopTTS25ModelLoader": "Hydra InferWorks · IndexTTS 2.5 · Load Model",
    "TopTTS25EmotionVector": "Hydra InferWorks · IndexTTS 2.5 · Emotion Vector",
    "TopTTS25Synthesize": "Hydra InferWorks · IndexTTS 2.5 · Synthesize",
    "TopTTS25UnloadModel": "Hydra InferWorks · IndexTTS 2.5 · Unload Model",
}
