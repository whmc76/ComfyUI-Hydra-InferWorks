# ComfyUI Top TTS

[中文](#中文) · [English](#english)

## 中文

ComfyUI Top TTS 是面向 ComfyUI 的 **IndexTTS 2.5 本地推理插件**。它直接加载本地官方权重，
不接入任何云端 TTS API，不上传文本、参考音频或生成音频。模型准备完成后可以断网推理。

### 功能

- 官方 IndexTTS 2.5 本地权重推理
- 零样本音色克隆
- 中文、英文、日语、西班牙语、阿拉伯语
- 独立情感参考音频、八维情感向量、情感文本控制
- 语速/时长、随机种子和生成参数控制
- ComfyUI 原生 `AUDIO` 输入输出
- BF16、FP32、CPU/CUDA 和显式显存卸载

### 安装

在 `ComfyUI/custom_nodes` 中执行：

```bash
git clone https://github.com/whmc76/ComfyUI-Top-TTS.git
```

安装依赖。Windows 便携版示例：

```powershell
..\..\python_embeded\python.exe -m pip install -r .\ComfyUI-Top-TTS\requirements.txt
..\..\python_embeded\python.exe .\ComfyUI-Top-TTS\install.py
```

Linux/macOS：

```bash
python -m pip install -r ComfyUI-Top-TTS/requirements.txt
python ComfyUI-Top-TTS/install.py
```

插件不会替换 ComfyUI 已安装的 PyTorch、torchaudio 或 transformers。官方 IndexTTS 2.5 当前要求
`transformers==4.52.1`；`install.py` 会把它安装到插件私有目录，并让本地推理工作进程隔离使用，
避免和其他 ComfyUI 节点的 transformers 版本冲突。

### 下载本地权重

先阅读仓库中的 `UPSTREAM_MODEL_LICENSE.txt`。接受后，在插件目录运行：

```powershell
# Hugging Face
..\..\python_embeded\python.exe .\download_models.py --source huggingface --accept-license

# 中国大陆可选 ModelScope
..\..\python_embeded\python.exe .\download_models.py --source modelscope --accept-license
```

脚本只下载官方 `IndexTeam/IndexTTS-2.5` 权重和官方推理需要的辅助模型，默认保存到：

```text
ComfyUI/models/IndexTTS-2.5/
```

也可以手动下载后放到该目录。运行时节点会在加载前检查主权重和辅助权重是否完整；缺失时直接报错，
不会在推理过程中偷偷联网。

### 基本工作流

```text
Load Audio ───────────────┐
                          ▼
Top TTS 2.5 - Load Model → Top TTS 2.5 - Synthesize → Save Audio
                          ▲
Top TTS 2.5 - Emotion Vector（可选）
```

情感控制优先级为：情感向量 > 情感文本 > 情感参考音频。使用情感文本时，加载器中的
`load_emotion_text_model` 必须开启，这会额外占用显存。

### 节点

- `Top TTS 2.5 - Load Model`：加载本地模型，选择设备、精度和可选优化。
- `Top TTS 2.5 - Emotion Vector`：生成八维情感控制向量。
- `Top TTS 2.5 - Synthesize`：音色克隆和语音合成。
- `Top TTS 2.5 - Unload Model`：释放模型并请求清理显存缓存。

### 说明

- 参考音频建议使用 5–15 秒、单人、清晰、低噪声的人声。
- `duration_factor` 大于 1 会变慢，小于 1 会变快。
- CUDA kernel 与 `torch.compile` 默认关闭，以优先保证首次安装稳定；确认环境支持后可在加载器开启。
- 只使用你有权使用的声音，并遵守当地隐私、肖像权、声音权和合成内容标识要求。

## English

ComfyUI Top TTS provides **fully local IndexTTS 2.5 inference for ComfyUI**. It loads official
weights from disk and never sends text, reference audio, or generated audio to a cloud TTS API.
Once the model files are prepared, inference works offline.

### Highlights

- Official IndexTTS 2.5 local-weight inference
- Zero-shot voice cloning
- Chinese, English, Japanese, Spanish, and Arabic
- Emotion audio, eight-dimensional emotion vectors, and emotion-text guidance
- Duration/speed, seed, and advanced generation controls
- Native ComfyUI `AUDIO` input and output
- BF16/FP32, CPU/CUDA, and explicit model unloading

### Quick install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/whmc76/ComfyUI-Top-TTS.git
python -m pip install -r ComfyUI-Top-TTS/requirements.txt
python ComfyUI-Top-TTS/install.py
```

Read `UPSTREAM_MODEL_LICENSE.txt`, then download the official local weights:

```bash
cd ComfyUI-Top-TTS
python download_models.py --source huggingface --accept-license
# or: python download_models.py --source modelscope --accept-license
```

The default model location is `ComfyUI/models/IndexTTS-2.5`. Runtime loading is offline-only and
fails with a clear missing-file list instead of downloading during inference.

The plugin keeps ComfyUI's existing PyTorch, torchaudio, and transformers installations. `install.py`
places the upstream-required `transformers==4.52.1` in a plugin-private directory used only by the
local inference worker process.

### License and upstream

The integration code in this repository is MIT licensed. The vendored official IndexTTS inference
source and separately downloaded model weights use the upstream bilibili Model Use License
Agreement. See `THIRD_PARTY_NOTICES.md` and `UPSTREAM_MODEL_LICENSE.txt`.

Upstream source: [`index-tts/index-tts`](https://github.com/index-tts/index-tts), pinned to commit
`b5ea881bec284b72f0b1cc04e0a724ff0c6b93e9`.
