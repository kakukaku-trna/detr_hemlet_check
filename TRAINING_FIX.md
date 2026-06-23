# 训练错误修复指南

## 你遇到的问题

### 错误信息
```
RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED
at ../c10/cuda/CUDACachingAllocator.cpp:1123
```

### 错误位置
发生在训练的第一个 batch，在 Deformable-DETR 的 FFN（Feed-Forward Network）层的 ReLU 激活函数。

### 根本原因
这**不是**驱动不兼容问题（虽然诊断看起来像），而是：

1. **CUDA 缓存内存碎片** - 长时间训练后 GPU 内存碎片化
2. **PyTorch CUDA 缓存问题** - 1.10.0 版本的已知 bug
3. **多进程 GPU 访问冲突** - 分布式训练中的竞争条件

## 快速修复方案

### ✅ 方案1：清理 CUDA 缓存（最快）

在 `main.py` 的训练循环前添加：

```python
# main.py 第 285 行左右，"Start training" 之后
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

完整位置：
```python
print("Start training")
torch.cuda.empty_cache()  # 添加这行
start_time = time.time()
```

### ✅ 方案2：禁用 CUDA 缓存分配器（推荐）

在 `main.py` 最上面添加：

```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```

或者在运行训练时：

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python main.py ...
```

### ✅ 方案3：使用梯度累积而非 batch size

如果 batch size 设置太大，改用梯度累积：

```bash
python main.py \
  --batch_size 1 \
  --accumulation_steps 2 \
  ...
```

等价于 batch_size=2，但内存压力更小。

### ✅ 方案4：升级 PyTorch（最彻底）

当前版本 1.10.0 很旧（2021 年），新版本有 CUDA 相关的 bug 修复：

```bash
# 卸载旧版本
pip uninstall torch torchvision torchaudio -y

# 安装新版本（CUDA 11.4 兼容）
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

注意：CUDA 11.4 已停止支持，建议用 11.8 或更新版本。

## 完整修复步骤

### 第一步：执行快速修复

编辑 `main.py`，找到 "Start training" 并添加缓存清理：

```python
    print("Start training")
    torch.cuda.empty_cache()  # ← 添加这行
    start_time = time.time()
```

### 第二步：设置环境变量

运行训练时使用：

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python main.py \
  --backbone convnext_tiny \
  --backbone_weights ./temp/model.safetensors \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --batch_size 2 \
  --epochs 5 \
  --output_dir ./output_df
```

### 第三步：监控 GPU 内存

在另一个终端运行：

```bash
watch -n 1 nvidia-smi
```

观察：
- GPU 内存是否稳定增长（正常）还是急剧波动（异常）
- 是否有内存泄漏

## 备选方案：使用 CPU 验证

如果上述方案都不行，先用 CPU 验证代码正确性：

```bash
python main.py \
  --device cpu \
  --batch_size 1 \
  --epochs 1 \
  --output_dir ./test_cpu
```

这会很慢（~50-100 倍），但能确认训练脚本本身无问题。

## 修改后的训练命令

**推荐的完整训练命令：**

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python main.py \
  --backbone convnext_tiny \
  --backbone_weights ./temp/model.safetensors \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --enc_n_points 2 \
  --dec_n_points 2 \
  --batch_size 2 \
  --epochs 50 \
  --output_dir ./output_df \
  --num_workers 4 \
  --lr 2e-4
```

## 其他优化建议

### 减少内存占用

1. **减小 batch size**
   ```bash
   --batch_size 1  # 从 2 改为 1
   ```

2. **减少 query 数量**
   ```bash
   --num_queries 100  # 从 150 改为 100
   ```

3. **使用混合精度训练**
   ```bash
   # main.py 中：
   with torch.cuda.amp.autocast():
       outputs = model(samples)
   ```

### 增加训练稳定性

1. **更小的学习率**
   ```bash
   --lr 1e-4  # 从 2e-4 改为 1e-4
   ```

2. **梯度裁剪**（已有）
   ```bash
   --clip_max_norm 0.1
   ```

3. **使用 SGD 而非 Adam**
   ```bash
   --sgd
   ```

## 如果问题持续

### 收集诊断信息

```bash
# 1. 运行诊断脚本
python diagnose_cuda.py

# 2. 记录错误日志
python main.py ... 2>&1 | tee training_error.log

# 3. 检查 CUDA 内存
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -l 1
```

### 报告问题

如果修复无效，请提供：
- `nvidia-smi` 输出
- `nvcc --version` 输出
- `diagnose_cuda.py` 输出
- 完整的训练命令和错误日志

## 快速检查清单

- [ ] 已添加 `torch.cuda.empty_cache()` 到训练前
- [ ] 已设置环境变量 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
- [ ] GPU 内存充足（运行时 nvidia-smi 检查）
- [ ] batch_size 不过大（推荐 1-2）
- [ ] 已运行 `diagnose_cuda.py` 验证 GPU 可用

## 推荐行动

1. **立即**：添加缓存清理 + 环境变量
2. **如果仍失败**：减小 batch_size 到 1
3. **如果还是失败**：升级 PyTorch 到 2.0+
4. **最后办法**：用 CPU 验证代码，然后调查 CUDA 环境

