# Hydra InferWorks for ComfyUI

[中文](#中文) · [English](#english)

Hydra InferWorks 是 HydraMatrix 的统一 ComfyUI 推理插件。一个插件提供三组彼此隔离的能力：

- **TTS**：IndexTTS 2.5 本地推理、零样本音色克隆、多语言与情感控制；
- **ASR**：Qwen3-ASR 长音频转写、锁定脚本强制对齐和不可变 JSON 回执；
- **Avatar**：通过可配置服务端点执行 HeyGem 长视频数字人生成，并保留文件与哈希回执。

某个可选能力缺少依赖时，只禁用该能力；其他已满足依赖的节点仍可加载。插件不包含模型权重、HeyGem 服务或容器镜像。

## 中文

### 安装

在 `ComfyUI/custom_nodes` 中执行：

```bash
git clone https://github.com/whmc76/ComfyUI-Hydra-InferWorks.git
python -m pip install -r ComfyUI-Hydra-InferWorks/requirements.txt
python ComfyUI-Hydra-InferWorks/install.py
```

Windows 便携版可以把 `python` 替换为 `..\..\python_embeded\python.exe`。
`install.py` 将 IndexTTS 2.5 要求的 `transformers==4.52.1` 安装到插件私有目录，不替换 ComfyUI 的全局 Transformers。

从旧插件迁移时，不要同时加载 `ComfyUI-Top-TTS`、`ComfyUI-Hydra-HeyGem` 与 Hydra InferWorks；它们保留相同节点 class type，重复加载会产生节点冲突。旧工作流无需改节点 ID。

### IndexTTS 2.5

先阅读 `UPSTREAM_MODEL_LICENSE.txt`，接受上游模型协议后下载官方权重：

```bash
cd ComfyUI-Hydra-InferWorks
python download_models.py --source huggingface --accept-license
# 中国大陆也可以使用：
python download_models.py --source modelscope --accept-license
```

默认模型目录为 `ComfyUI/models/IndexTTS-2.5/`。推理阶段只读取本地文件；文件不完整时会列出缺失项并失败，不会静默联网下载。

TTS 节点保持现有工作流 class type：

- `TopTTS25ModelLoader`
- `TopTTS25EmotionVector`
- `TopTTS25Synthesize`
- `TopTTS25UnloadModel`

卸载节点可以接收可选 `after` 音频依赖，确保生成完成后再终止隔离 worker、释放模型并请求清理显存。

模型加载器提供四个明确档位：`quality_bf16` 是质量优先的生产基线；
`compiled_bf16` 在 BF16 基础上启用 `torch.compile`；`maximum_bf16` 还要求
FlashAttention GPT 加速和 BigVGAN CUDA kernel，缺少任一能力都会失败而不是静默降级；
`reference_fp32` 仅用于质量对照。旧工作流仍可通过 `legacy_custom` 保留原来的独立开关。
模型信息会报告实际精度、量化格式和加速后端。BF16 属于低精度推理而不是权重量化；
未经音色相似度、情感控制和可懂度验证的 INT8/INT4 权重不会冒充生产档。

### Qwen3-ASR 与强制对齐

Hydra InferWorks 现在直接提供 `HydraQwen3ASRModelLoader`，不再依赖另一个
ComfyUI 节点包来加载模型。安装依赖中固定 `qwen-asr==0.0.6`，模型仍然只从
ComfyUI 本地模型目录读取。它提供：

- `HydraQwen3ASRModelLoader`：显式选择 BF16 SDPA、BF16 FlashAttention 2、BF16 vLLM 或 FP32 对照档；
- `HydraQwen3LongAsrTranscribe`：在低能量边界切分长音频并恢复全局时间线；
- `HydraQwen3ForcedAlign`：使用 ASR 锚点分块对齐调用方锁定文本；
- `HydraTranscriptReceipt`：把源音频哈希、模型身份、时间戳和分块证据写入不可变 JSON 回执。

模型权重应放在工作流配置的本地路径。生产工作流固定使用 `Qwen/Qwen3-ASR-1.7B` 与 `Qwen/Qwen3-ForcedAligner-0.6B`。
默认生产档是 `transformers_bf16_sdpa`；FlashAttention 2 与 vLLM 只有在依赖存在并完成
本机质量/吞吐验证后才切换。当前官方未发布经过 Qwen3-ASR 任务质量验证的量化权重，
因此 INT8/INT4 不在生产档，回执会明确记录 `quantization=none`，避免把 BF16 错称为量化。

### HeyGem

`HydraHeyGemLongformAvatar` 保留原有节点 ID，支持：

- 原生 ComfyUI `AUDIO` 与文件型 `VIDEO`；
- 运行时可配置 URL、host 与 port；
- 外部服务或既有 Docker 容器生命周期；
- 调用方提供的安全 `job_code`；
- 文件型结果、SHA-256、服务响应和生命周期回执；
- 任务结束后可选停止精确容器并释放 GPU。

它不包含 HeyGem 本体。部署方仍需准备 HeyGem 兼容的 submit/query 服务和共享挂载目录。

### 能力隔离

插件入口分别加载 `tts_nodes.py`、`asr_nodes.py` 和 `heygem_nodes.py`。加载结果公开在 `CAPABILITY_STATUS` 中。缺少 ASR provider、HeyGem 新版 Comfy API 或 TTS Python 依赖时，错误会绑定到对应能力，而不会删除其他成功加载的节点。

### 开发与验证

```bash
python -m pip install -e ".[test]"
python -m pytest
```

发布验收还要求在真实 ComfyUI 中读取 `/object_info`，并分别执行代表性的 IndexTTS 2.5、Qwen3-ASR/ForcedAligner 与 HeyGem 队列任务。

### 许可证

Hydra InferWorks 自研集成代码使用 Apache-2.0。仓库中固定的 IndexTTS 2.5 上游推理源码及另外下载的模型权重受 bilibili Model Use License Agreement 约束，不属于 Apache-2.0；详见 `THIRD_PARTY_NOTICES.md` 与 `UPSTREAM_MODEL_LICENSE.txt`。第三方 ASR provider、HeyGem 和模型分别遵循其上游许可。

## English

Hydra InferWorks is a unified ComfyUI inference node pack for HydraMatrix. It combines fully local IndexTTS 2.5 speech synthesis, Hydra-owned long-audio Qwen3-ASR/forced-alignment receipt nodes, and the file-backed HeyGem long-form avatar adapter. Capability imports are isolated so one missing optional dependency does not disable the remaining modules.

Install the repository and Python requirements, run `install.py` for the private IndexTTS compatibility environment, then download the official IndexTTS 2.5 weights only after accepting `UPSTREAM_MODEL_LICENSE.txt`. Hydra InferWorks owns the Qwen3-ASR loader and pins the official `qwen-asr` package; HeyGem still requires an existing compatible service.

Production profiles prefer BF16 and explicit optimized attention/compilation backends. Requested acceleration fails closed when unavailable, and every TTS model description or ASR receipt records the actual precision, quantization state, and backend. INT8/INT4 remains outside production admission until model-specific quality evidence exists.

Existing `TopTTS25*`, `HydraQwen3*`, `HydraTranscriptReceipt`, and `HydraHeyGemLongformAvatar` class types are preserved for workflow compatibility; `HydraQwen3ASRModelLoader` is the new unified loader.
