# Nano-vLLM-VoxCPM

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

An inference engine for VoxCPM based on Nano-vLLM.

### Features
- Faster than the pytorch implementation
- Support concurrent requests
- Friendly async API, easy to use in FastAPI (see [fastapi/app.py](fastapi/app.py))

### Supported Models

This project supports inference for both versions of VoxCPM weights:
- [VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B)
- [VoxCPM1.5](https://huggingface.co/openbmb/VoxCPM1.5)

### Installation

Nano-vLLM-VoxCPM is not available on PyPI yet, you need to install it from source.

```bash
git clone https://github.com/a710128/nanovllm-voxcpm.git
cd nanovllm-voxcpm
pip install -e .
```

### Quick Start (FastAPI Server)

You can start the API server using the following command. Make sure to specify the correct `MODEL_PATH`.

```bash
MODEL_PATH="./VoxCPM-0.5B" fastapi run fastapi/app.py
```

### Configuration (Environment Variables)

The server can be configured using environment variables. You can set them in a `.env` file or pass them directly to the command.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MODEL_PATH` | `~/VoxCPM-0.5B` | Path to the VoxCPM model directory (supports both 0.5B and 1.5 versions). |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Maximum number of tokens processed in a batch. |
| `MAX_NUM_SEQS` | `16` | Maximum number of concurrent sequences (requests). |
| `MAX_MODEL_LEN` | `4096` | Maximum sequence length for the model. |
| `GPU_MEMORY_UTILIZATION` | `0.95` | Fraction of GPU memory to be used. |
| `ENFORCE_EAGER` | `False` | Whether to enforce eager execution mode. |
| `DEVICES` | `[0]` | List of GPU device IDs to use. |
| `DB_PATH` | `prompts.db` | Path to the SQLite database for storing prompts. |
| `INFERENCE_TIMESTEPS` | `10` | Number of diffusion inference timesteps. |
| `SCHEDULER_LOG_INTERVAL` | `5.0` | Interval (in seconds) for scheduler logging. |
| `SCHEDULER_LOG_ENABLE` | `True` | Enable or disable scheduler logging. |

### Basic Usage

See the [example.py](example.py) for a usage example.

### Known Issue

If you encounter an error like `ValueError: Missing parameters: ...` followed by `destroy_process_group()`, it is likely because `nanovllm` expects model parameters in `.safetensors` format, but the original VoxCPM model uses `.pt`.

Solution: Download the `.safetensors` file manually and place it in your model folder.
- [VoxCPM-0.5B-Safetensors](https://huggingface.co/euphoricpenguin22/VoxCPM-0.5B-Safetensors/blob/main/model.safetensors)

### Acknowledgments
- [VoxCPM](https://github.com/OpenBMB/VoxCPM)
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)

### License
MIT License

---

<a name="chinese"></a>
## 中文 (Chinese)

基于 Nano-vLLM 的 VoxCPM 推理引擎。

### 特性
- 比 PyTorch 实现更快
- 支持并发请求
- 友好的异步 API，易于在 FastAPI 中使用（参见 [fastapi/app.py](fastapi/app.py)）

### 支持的模型

本项目支持两个版本的 VoxCPM 权重推理：
- [VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B)
- [VoxCPM1.5](https://huggingface.co/openbmb/VoxCPM1.5)

### 安装

Nano-vLLM-VoxCPM 尚未发布到 PyPI，需要从源码安装。

```bash
git clone https://github.com/a710128/nanovllm-voxcpm.git
cd nanovllm-voxcpm
pip install -e .
```

### 快速开始 (FastAPI 服务)

使用以下命令启动 API 服务。请确保指定了正确的模型路径 (`MODEL_PATH`)。

```bash
MODEL_PATH="./VoxCPM-0.5B" fastapi run fastapi/app.py
```

### 配置 (环境变量)

服务器可以通过环境变量进行配置。您可以在 `.env` 文件中设置它们，或者直接传递给命令。

| 变量名 | 默认值 | 描述 |
| :--- | :--- | :--- |
| `MODEL_PATH` | `~/VoxCPM-0.5B` | VoxCPM 模型目录的路径（支持 0.5B 和 1.5 版本）。 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 批处理中处理的最大 token 数。 |
| `MAX_NUM_SEQS` | `16` | 最大并发序列（请求）数。 |
| `MAX_MODEL_LEN` | `4096` | 模型的最大序列长度。 |
| `GPU_MEMORY_UTILIZATION` | `0.95` | GPU 显存占用比例。 |
| `ENFORCE_EAGER` | `False` | 是否强制使用 eager 执行模式。 |
| `DEVICES` | `[0]` | 使用的 GPU 设备 ID 列表。 |
| `DB_PATH` | `prompts.db` | 用于存储 prompt 的 SQLite 数据库路径。 |
| `INFERENCE_TIMESTEPS` | `10` | 扩散推理的时间步数。 |
| `SCHEDULER_LOG_INTERVAL` | `5.0` | 调度器日志记录间隔（秒）。 |
| `SCHEDULER_LOG_ENABLE` | `True` | 启用或禁用调度器日志记录。 |

### 基本用法

请参考 [example.py](example.py) 查看使用示例。

### 已知问题

如果您遇到类似 `ValueError: Missing parameters: ...` 的错误，通常是因为 `nanovllm` 读取模型参数需要 `.safetensors` 格式，而 VoxCPM 原始格式为 `.pt`。

解决方案：手动下载 `.safetensors` 文件并将其放入模型文件夹中。
- [VoxCPM-0.5B-Safetensors](https://huggingface.co/euphoricpenguin22/VoxCPM-0.5B-Safetensors/blob/main/model.safetensors)

### 致谢
- [VoxCPM](https://github.com/OpenBMB/VoxCPM)
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)

### 许可证
MIT License
