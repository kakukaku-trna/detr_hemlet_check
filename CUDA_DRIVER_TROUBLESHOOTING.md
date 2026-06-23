# CUDA 驱动版本不匹配问题解决方案

## 问题诊断

### 错误信息
```
RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED 
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 575.57
```

### 根本原因

你的系统配置：
- **GPU 驱动**: 575.57（新版本）
- **CUDA 工具包**: 11.4（旧版本，发布于 2021.8）
- **兼容性问题**: CUDA 11.4 与新的 GPU 驱动不兼容

CUDA 11.4 支持的最新驱动版本通常是 **470.x**，但你的系统是 **575.57**，导致版本不匹配。

## 解决方案

### ✅ 方案1：使用 CPU 训练（最快，临时方案）

如果 GPU 训练不紧急，可以先用 CPU 训练：

```bash
python main.py \
  --device cpu \
  --backbone convnext_tiny \
  --backbone_weights ./temp/model.safetensors \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --batch_size 1 \
  --epochs 1 \
  --output_dir ./output_cpu_test
```

**优点：** 立即可用，无需改动系统  
**缺点：** 训练速度慢（~50-100 倍）

### ⚡ 方案2：升级 CUDA 工具包到 12.4（推荐）

这是最完整的解决方案，升级 CUDA 以匹配新驱动。

#### 步骤1：检查 CUDA 12.4 兼容性

CUDA 12.4 支持驱动版本 550+，完全兼容你的 575.57 驱动。

#### 步骤2：卸载旧 CUDA

```bash
sudo apt-get --purge remove cuda-*
sudo apt-get --purge remove cuda-toolkit-*
sudo rm -rf /usr/local/cuda*
```

#### 步骤3：安装 CUDA 12.4

```bash
# 添加 NVIDIA 仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

# 下载并安装 CUDA 12.4
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2004-12-4-local_12.4.1-550.54.15-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-4-local_12.4.1-550.54.15-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-4
```

#### 步骤4：更新环境变量

编辑 `~/.bashrc`，添加或修改：

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.4
```

```bash
source ~/.bashrc
nvcc --version  # 验证安装
```

#### 步骤5：重新安装 PyTorch（CUDA 12.4 版本）

```bash
# 卸载旧版本
pip uninstall torch torchvision torchaudio -y

# 安装 CUDA 12.4 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### 步骤6：验证

```bash
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.is_available()}')"
nvidia-smi  # 应该正常显示
```

### 🔄 方案3：降级 GPU 驱动（不推荐）

如果升级 CUDA 有困难，也可以尝试降级驱动，但这样可能影响系统其他功能。

```bash
# 不推荐此方案，因为新驱动通常有更好的性能和稳定性
```

### ⚙️ 方案4：使用容器（Docker）

如果系统配置复杂，可以使用 NVIDIA 官方 Docker 镜像：

```bash
docker run --gpus all -it \
  -v /home/jiangyang.li2/detr_hemlet_check:/workspace \
  --workdir /workspace \
  nvcr.io/nvidia/pytorch:24.02-py3
```

这会提供完全配置好的 CUDA 12.4 + PyTorch 环境。

## 临时方案：代码级降级处理

我已经在 `main.py` 中添加了自动回退逻辑：

```python
# Handle CUDA driver/library version mismatch
if 'cuda' in args.device.lower():
    try:
        test_tensor = torch.zeros(1, device=args.device)
    except RuntimeError as e:
        if 'NVML_SUCCESS' in str(e) or 'Driver' in str(e) or 'CUDA' in str(e):
            print(f"⚠️  CUDA initialization failed: {e}")
            print("Falling back to CPU. To use GPU, update CUDA drivers.")
            args.device = 'cpu'
        else:
            raise
```

这样即使 GPU 初始化失败，也会自动转到 CPU。

## 故障排查

### 检查当前配置

```bash
# 1. GPU 驱动版本
nvidia-smi | grep "Driver Version"

# 2. CUDA 工具包版本
nvcc --version

# 3. PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 4. cuDNN 版本
python -c "import torch; print(torch.backends.cudnn.version())"

# 5. GPU 可用性
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

### NVIDIA 驱动兼容性表

| GPU 驱动版本 | 支持 CUDA 版本 | 发布日期 |
|----------|------------|--------|
| 550+ | 12.4+ | 2024 |
| 525-549 | 12.0-12.3 | 2023 |
| 470-524 | 11.0-11.8 | 2020-2022 |
| 418-469 | 10.0-10.2 | 2018-2020 |

### 常见错误和原因

| 错误 | 原因 | 解决方案 |
|------|------|--------|
| `NVML_SUCCESS == DriverAPI` | 驱动与 CUDA 不兼容 | 升级 CUDA 或驱动 |
| `CUDA out of memory` | GPU 内存不足 | 减小 batch_size 或用 CPU |
| `libcuda.so.1 not found` | CUDA 库未安装 | 重新安装 CUDA |
| `GPU not found` | GPU 驱动未安装 | 安装 NVIDIA 驱动 |

## 对训练的影响

### 使用 CPU 的性能影响

如果暂时用 CPU 训练：

| 配置 | 速度 | 内存 |
|------|------|------|
| GPU (V100/A100) | 基准 | 32GB |
| CPU (高端多核) | ~20-50x 慢 | 256GB+ |
| GPU fallback | 不可用 | 不可用 |

**建议：** 如果用 CPU，将 batch_size 降至 1，仅用于快速验证而非完整训练。

## 推荐行动步骤

### 短期（今天）
1. ✅ 使用 CPU 验证代码是否正常运行
   ```bash
   python main.py --device cpu --epochs 1 ...
   ```
2. ✅ 确认训练脚本、数据加载等无其他问题

### 中期（这周）
3. ⚡ 执行方案2，升级 CUDA 工具包到 12.4
4. ⚡ 重新安装 PyTorch CUDA 12.4 版本
5. ⚡ 验证 GPU 可用性

### 长期（之后）
6. 💪 使用 GPU 进行完整训练
7. 📊 监控 GPU 使用率和温度

## 技术参考

- [NVIDIA CUDA 发布说明](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [NVIDIA 驱动兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)

