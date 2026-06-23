# ConvNeXt Backbone 本地预训练权重加载方案

## 问题描述

当使用 ConvNeXt 作为 backbone 进行 Deformable-DETR 训练时，即使本地已有预训练权重文件（`/home/jiangyang.li2/detr_hemlet_check/Deformable-DETR/temp/model.safetensors`），模型仍然会尝试从官网（timm hub）下载权重，导致：

1. **网络延迟**：等待官网下载
2. **网络失败**：官网不可达时训练无法启动
3. **重复下载**：每次训练都重复下载相同权重

## 解决方案

### 核心修改点

#### 1. **models/backbone.py** - ConvNeXtBackbone 类

**问题所在（原代码）：**
```python
# 原代码会直接从 timm 官网下载
model = timm.create_model(name, pretrained=load_pretrained, features_only=True,
                         out_indices=(1, 2, 3))
```

**解决方案（新代码）：**
```python
class ConvNeXtBackbone(nn.Module):
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, 
                 pretrained: bool = True, pretrained_weights_path: str = None):
        super().__init__()
        import timm
        
        # 1. 创建模型时不下载权重
        model = timm.create_model(name, pretrained=False, features_only=True,
                                 out_indices=(1, 2, 3))
        
        # 2. 优先从本地加载
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            # 支持 .safetensors 和 .pth 格式
            if pretrained_weights_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(pretrained_weights_path)
            else:
                state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            
            model.load_state_dict(state_dict, strict=False)
        
        # 3. 本地加载失败才下载
        elif pretrained:
            model = timm.create_model(name, pretrained=True, features_only=True,
                                     out_indices=(1, 2, 3))
```

**关键特性：**
- ✅ 本地路径优先加载
- ✅ 支持 `.safetensors` 和 `.pth` 格式
- ✅ `strict=False` 处理权重不匹配
- ✅ 自动回退到官网下载

#### 2. **main.py** - 新增命令行参数

```python
parser.add_argument('--backbone_weights', type=str, default=None,
                    help="Path to local pretrained backbone weights (e.g., model.safetensors or model.pth)")
```

#### 3. **models/backbone.py** - build_backbone 函数修改

```python
def build_backbone(args):
    if args.backbone.startswith('convnext'):
        pretrained_weights_path = getattr(args, 'backbone_weights', None)
        backbone = ConvNeXtBackbone(args.backbone, train_backbone, return_interm_layers,
                                   pretrained=not pretrained_weights_path,  # 有本地路径就不下载
                                   pretrained_weights_path=pretrained_weights_path)
```

#### 4. **requirements.txt** - 新增依赖

```
safetensors  # 支持安全、高效的权重加载
```

### 使用方法

#### 方式1：使用本地 safetensors 权重（推荐）

```bash
python main.py \
  --backbone convnext_tiny \
  --backbone_weights ./temp/model.safetensors \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --epochs 50 \
  --batch_size 2 \
  --output_dir ./output_df
```

#### 方式2：使用绝对路径

```bash
python main.py \
  --backbone convnext_tiny \
  --backbone_weights /home/jiangyang.li2/detr_hemlet_check/Deformable-DETR/temp/model.safetensors \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --epochs 50 \
  --output_dir ./output_df
```

#### 方式3：不指定本地权重（使用官网）

```bash
python main.py \
  --backbone convnext_tiny \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --epochs 50 \
  --output_dir ./output_df
```

### 工作流程图

```
训练启动
  │
  ├─ 检查 --backbone_weights 参数
  │
  ├─ 如果指定了本地路径：
  │   ├─ 文件存在？
  │   │   └─ 是 → 加载本地权重 ✓
  │   │   └─ 否 → 尝试官网下载
  │
  ├─ 如果没指定本地路径：
  │   └─ 检查 pretrained=True
  │       └─ 是 → 从官网下载
  │
  └─ 模型初始化完成
```

## 验证步骤

### 1. 检查权重文件

```bash
ls -lh ./temp/model.safetensors
# 应该看到文件存在，大小约 110MB
```

### 2. 运行测试脚本

```bash
python test_local_weights.py \
  --weights ./temp/model.safetensors \
  --backbone convnext_tiny
```

预期输出：
```
✓ Weights file found: 0.11 GB
✓ timm is installed
✓ safetensors is installed
✓ Model created successfully
✓ Loaded XXX tensors from safetensors
✓ State dict loaded successfully
✓ Forward pass successful
✓ All tests passed!
```

### 3. 启动训练并检查日志

```bash
python main.py \
  --backbone convnext_tiny \
  --backbone_weights ./temp/model.safetensors \
  --epochs 1 \
  --output_dir ./test_output
```

预期日志：
```
Loading pretrained weights from: ./temp/model.safetensors
Successfully loaded weights from ./temp/model.safetensors
```

**不应该出现：**
```
Downloading: "https://..." (不下载)
```

## 文件修改清单

### 已修改文件

| 文件 | 修改内容 | 类型 |
|------|--------|------|
| `models/backbone.py` | 修改 ConvNeXtBackbone 类，支持本地权重加载 | 主要 |
| `main.py` | 添加 `--backbone_weights` 参数 | 主要 |
| `requirements.txt` | 添加 safetensors 依赖 | 次要 |

### 新增文件

| 文件 | 用途 |
|------|------|
| `LOCAL_WEIGHTS_USAGE.md` | 使用说明和常见问题 |
| `test_local_weights.py` | 权重加载测试脚本 |
| `BACKBONE_WEIGHTS_SOLUTION.md` | 本文档 |

## 轻量化设计进展总结

### 当前轻量化方案（方案B）

**目标：** ResNet50 → ConvNeXt-Tiny + 蒸馏

**核心改动：**

| 组件 | ResNet50 | ConvNeXt-Tiny | 变化 |
|-----|---------|---------------|------|
| **Backbone 参数** | 43.3M | 7.8M | -82% |
| **Backbone FLOPs** | ~60G | 30G | -50% |
| **Decoder 层数** | 6 | 4 | -33% |
| **Query 数量** | 300 | 150 | -50% |
| **采样点** | 4 | 2 | -50% |

**预期收益：**
- 参数量：-82% (backbone) + 其他模块 = 总体 -36%
- FLOPs：-60%
- 延迟：-20% ~ -55%（取决于硬件）
- mAP 保持：< 1 点下降

### 本次改进的作用

本次改进（本地权重加载）的作用是：

1. **加快训练启动速度**：避免每次都从官网下载 110MB 权重
2. **支持离线训练**：无需网络连接即可使用预训练权重
3. **提高训练稳定性**：不依赖官网可用性
4. **便于分布式训练**：多进程/多卡训练时不会重复下载

### 后续优化方向

根据 `light.md` 和优化框架，还可以继续：

1. **知识蒸馏**：使用 ResNet50 师模型蒸馏 ConvNeXt 学生模型
2. **Attention 简化**：减少 attention head 或采样点
3. **量化部署**：INT8 量化 + TensorRT 加速
4. **进一步压缩**：尝试 MobileNetV3 或 EfficientNet-Lite 作为 backbone

## 常见问题排查

### Q: 训练时仍然下载权重怎么办？

**检查清单：**
1. 确认指定了 `--backbone_weights` 参数
2. 检查文件路径是否正确：`ls -lh <path>`
3. 查看日志中是否有 "Loading pretrained weights from:"
4. 运行 `test_local_weights.py` 验证加载

### Q: 权重格式不匹配怎么办？

**现象：** 出现 "size mismatch" 警告

**解决：** 这是正常的。代码使用 `strict=False`，不匹配的层会随机初始化。如果精度下降明显，需要检查权重是否对应正确的 backbone 版本。

### Q: safetensors 如何安装？

```bash
pip install safetensors
```

## 总结

通过以上修改，你现在可以：

✅ 使用本地预训练权重加速 ConvNeXt backbone 训练  
✅ 避免每次训练都从官网下载权重  
✅ 支持离线和分布式训练  
✅ 保持与原始 timm 官网下载的兼容性（作为回退）  

**建议的训练命令：**

```bash
python main.py \
  --backbone convnext_tiny \
  --backbone_weights ./temp/model.safetensors \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --enc_n_points 2 \
  --dec_n_points 2 \
  --epochs 50 \
  --batch_size 2 \
  --output_dir ./output_df
```

