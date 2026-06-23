# 使用本地预训练权重加速训练

## 问题背景

当使用 ConvNeXt 作为 backbone 时，模型会在训练时从官网自动下载预训练权重。如果网络不稳定或想加快启动速度，可以使用本地权重文件。

## 解决方案

### 1. 安装依赖

```bash
pip install safetensors
```

### 2. 下载/准备权重文件

将 ConvNeXt 的预训练权重文件放在本地，支持的格式：
- `.safetensors` 格式（推荐，安全且快速）
- `.pth` 格式（PyTorch checkpoint）

例如：`/home/jiangyang.li2/detr_hemlet_check/Deformable-DETR/temp/model.safetensors`

### 3. 训练时指定本地权重

在训练命令中添加 `--backbone_weights` 参数：

```bash
python main.py \
  --backbone convnext_tiny \
  --backbone_weights ./temp/model.safetensors \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --enc_n_points 2 \
  --dec_n_points 2 \
  --epochs 50 \
  --output_dir ./checkpoints/convnext_baseline
```

### 4. 工作原理

- **本地权重优先加载**：如果提供了 `--backbone_weights` 参数且文件存在，会直接从本地加载
- **自动回退**：如果本地加载失败，会自动尝试从 timm 官网下载
- **无网络依赖**：指定本地权重后，完全不需要访问官网

## 完整训练命令示例

```bash
# 方式1：使用本地 safetensors 权重
python main.py \
  --backbone convnext_tiny \
  --backbone_weights /home/jiangyang.li2/detr_hemlet_check/Deformable-DETR/temp/model.safetensors \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --epochs 50 \
  --batch_size 2 \
  --output_dir ./output_df

# 方式2：使用相对路径
python main.py \
  --backbone convnext_tiny \
  --backbone_weights ./temp/model.safetensors \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --epochs 50 \
  --batch_size 2 \
  --output_dir ./output_df

# 方式3：不指定本地权重（使用官网下载）
python main.py \
  --backbone convnext_tiny \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --epochs 50 \
  --batch_size 2 \
  --output_dir ./output_df
```

## 常见问题

### Q: 权重文件格式有什么区别？

- **safetensors**：安全格式，无法执行恶意代码，加载速度快
- **pth**：PyTorch 原生格式，文件通常更大

### Q: 权重不匹配会怎样？

如果权重与模型架构不匹配，会显示警告并跳过不匹配的层：
```
Warning: Failed to load weights from ...: size mismatch...
```

加载会使用 `strict=False`，所以不会报错，不匹配的层会随机初始化。

### Q: 如何获取 ConvNeXt 的预训练权重？

推荐的来源：
1. **Hugging Face Model Hub**：https://huggingface.co/models?search=convnext
2. **timm 官方**：https://github.com/rwightman/pytorch-image-models
3. **Meta/FAIR 官方发布**

下载方法：
```python
# 直接下载 safetensors 格式
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="facebook/convnext-tiny",
    filename="model.safetensors",
    local_dir="./temp"
)
```

### Q: 为什么还是会下载官网权重？

检查以下几点：
1. 确认 `--backbone_weights` 参数正确指定
2. 确认文件路径存在（检查文件大小）：`ls -lh ./temp/model.safetensors`
3. 查看日志输出，应该看到 "Loading pretrained weights from:" 的信息
4. 如果提示文件不存在，检查路径是绝对路径还是相对路径

### Q: 支持混合架构吗？

不支持。权重必须与 backbone 名称对应：
- `--backbone convnext_tiny` 需要 convnext_tiny 的权重
- `--backbone convnext_small` 需要 convnext_small 的权重

## 性能收益

使用本地权重的优势：

| 场景 | 时间 |
|------|------|
| 首次训练（无缓存） | 节省 1-5 分钟（避免官网下载） |
| 多次训练（有缓存） | 基本相同（都使用缓存） |
| 有限带宽环境 | 避免网络中断 |
| 离线训练 | 必需 |

## 轻量化配置参考

当前项目使用的 ConvNeXt-Tiny 轻量化参数：

```bash
--backbone convnext_tiny           # 64.6M params
--num_decoder_layers 4             # 默认 6
--num_queries 150                  # 默认 300
--enc_n_points 2                   # 默认 4
--dec_n_points 2                   # 默认 4
```

预期收益：
- 参数量减少 36%
- FLOPs 减少 60%
- 延迟减少 20-55%（取决于硬件）
- mAP 下降 < 1

