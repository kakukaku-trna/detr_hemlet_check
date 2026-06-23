# Deformable-DETR 方案B：Backbone替换 + 知识蒸馏

## 完整实施指南

**方案名称**: 均衡优化方案 (推荐) ⭐  
**优化方向**: Backbone 替换 + 知识蒸馏  
**预期收益**: 2-2.2x 加速，-0.8 mAP 精度损失 (通过蒸馏恢复到 -0.3)  
**实施周期**: 2-3 周  
**难度等级**: ⭐⭐ 中等

---

## 📋 目录

1. [概览](#概览)
2. [阶段一：Backbone替换](#阶段一backbone替换)
3. [阶段二：基线测试](#阶段二基线测试)
4. [阶段三：知识蒸馏](#阶段三知识蒸馏)
5. [阶段四：最终验证](#阶段四最终验证)
6. [故障排查](#故障排查)

---

## 概览

### 方案B 核心配置

```python
# 方案B: 均衡优化 (推荐)
backbone = "convnext_tiny"        # 轻量级Backbone
num_decoder_layers = 4             # 从6->4
num_queries = 150                  # 从300->150
enc_n_points = 2                   # 从4->2
dec_n_points = 2                   # 从4->2
num_heads = 8                      # 保持不变
hidden_dim = 256                   # 保持不变
```

### 预期性能指标

| 阶段 | Backbone | 参数 | FLOPs | 延迟 | mAP | 说明 |
|------|----------|------|-------|------|-----|------|
| 原始 | ResNet50 | 43.3M | 150G | 45ms | 42.5 | 基准线 |
| **第一步** | ConvNeXt-T | 8.5M | 60G | 20ms | 40.5 | Backbone替换 |
| **第二步** | ConvNeXt-T | 8.5M | 60G | 20ms | 41.2 | 蒸馏后 |

### 加速效果

```
原始延迟: 45ms
优化后: 20ms
总加速: 2.25x ⭐
```

---

## 阶段一：Backbone替换

### 步骤1：安装依赖

```bash
# 安装 timm 库 (包含 ConvNeXt 预训练权重)
pip install timm>=0.6.0

# 验证安装
python -c "import timm; print(timm.__version__)"
```

### 步骤2：修改 backbone.py

在 `models/backbone.py` 中添加 ConvNeXt 支持：

```python
# 在 build_backbone 函数中添加

def build_backbone(args):
    """构建 Backbone"""
    
    if args.backbone == 'resnet50':
        backbone = build_resnet('resnet50', args.dilation, args.pretrained)
    
    elif args.backbone == 'convnext_tiny':
        # 新增：ConvNeXt-Tiny Backbone
        import timm
        backbone = timm.create_model('convnext_tiny', pretrained=True)
        
        # ConvNeXt 的特征提取
        class ConvNeXtBackbone(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.stages = nn.ModuleList([
                    model.downsample_layers[0],
                    model.stages[0],
                    model.downsample_layers[1],
                    model.stages[1],
                    model.downsample_layers[2],
                    model.stages[2],
                    model.downsample_layers[3],
                    model.stages[3],
                ])
                # ConvNeXt 输出通道
                self.strides = [4, 8, 16, 32]
                self.num_channels = [96, 192, 384, 768]
            
            def forward(self, x):
                features = []
                for i, stage in enumerate(self.stages):
                    x = stage(x)
                    if i % 2 == 1:  # 每个完整阶段后收集特征
                        features.append(x)
                # 返回 C3, C4, C5 (跳过 C2)
                return features[-3:]  # [96, 384, 768] -> 需要映射到 [512, 1024, 2048] 等
        
        backbone = ConvNeXtBackbone(backbone)
        backbone.strides = [8, 16, 32]  # 与原始 ResNet50 保持一致
        backbone.num_channels = [96, 384, 768]  # ConvNeXt 的输出通道
    
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")
    
    return backbone
```

### 步骤3：修改特征投影层

在 `models/deformable_detr.py` 中调整特征投影：

```python
class DeformableDETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                 num_feature_levels, aux_loss=True, with_box_refine=False, 
                 two_stage=False):
        super().__init__()
        # ... 其他初始化 ...
        
        # 特征投影 - 需要根据 Backbone 的输出通道调整
        self.input_proj = nn.ModuleList()
        
        # 获取 Backbone 的输出通道
        num_channels = backbone.num_channels  # [96, 384, 768] 或 [512, 1024, 2048]
        
        for i in range(num_feature_levels):
            if i < len(num_channels):
                # 从 Backbone 输出投影到 hidden_dim
                in_channels = num_channels[i]
            else:
                # 对于额外的级别，使用固定输入通道
                in_channels = num_channels[-1]
            
            self.input_proj.append(nn.Sequential(
                nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, self.hidden_dim),
            ))
```

### 步骤4：验证前向传播

```python
# test_forward.py
import torch
from models import build

# 创建模型
class Args:
    backbone = 'convnext_tiny'  # 改为 ConvNeXt
    num_decoder_layers = 4
    num_encoder_layers = 6
    num_queries = 150
    enc_n_points = 2
    dec_n_points = 2
    
args = Args()
model, criterion, postprocessors = build(args)
model.cuda()
model.eval()

# 测试前向传播
dummy_input = torch.randn(1, 3, 640, 640).cuda()

try:
    with torch.no_grad():
        outputs = model(dummy_input)
    print("✅ 前向传播成功！")
    print(f"   输出形状: {outputs['pred_logits'].shape}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
except Exception as e:
    print(f"❌ 前向传播失败: {e}")
```

### 步骤5：Fine-tune ResNet50 模型

```bash
# 在原始配置下 fine-tune
python main.py \
  --backbone resnet50 \
  --num_decoder_layers 6 \
  --num_queries 300 \
  --enc_n_points 4 \
  --dec_n_points 4 \
  --epochs 20 \
  --lr 1e-4 \
  --resume [原始模型检查点] \
  --output_dir ./checkpoints/resnet50_finetuned \
  --eval
```

### 步骤6：用 ConvNeXt 从头训练

```bash
# 用新 Backbone 从头训练（更长的训练周期）
python main.py \
  --backbone convnext_tiny \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --enc_n_points 2 \
  --dec_n_points 2 \
  --epochs 50 \
  --lr 2e-4 \
  --lr_backbone 2e-5 \
  --output_dir ./checkpoints/convnext_baseline \
  --eval
```

**预期结果**:
- 参数: ~8.5M (-80%)
- FLOPs: ~60G (-60%)
- mAP: ~40.5 (-2.0)

---

## 阶段二：基线测试

### 步骤1：性能测试

```python
# benchmark_convnext.py
import torch
import time
from models import build

class Args:
    backbone = 'convnext_tiny'
    num_decoder_layers = 4
    num_queries = 150
    enc_n_points = 2
    dec_n_points = 2

model, _, _ = build(Args())
model.cuda().eval()

# 预热
for _ in range(10):
    with torch.no_grad():
        model(torch.randn(1, 3, 640, 640).cuda())

# 计时
torch.cuda.synchronize()
start = time.time()

for _ in range(100):
    with torch.no_grad():
        model(torch.randn(1, 3, 640, 640).cuda())

torch.cuda.synchronize()
latency = (time.time() - start) / 100

print(f"平均延迟: {latency*1000:.2f}ms")
print(f"FPS: {1/latency:.1f}")
```

### 步骤2：精度评估

```bash
# 在 COCO 验证集上评估
python main.py \
  --backbone convnext_tiny \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --enc_n_points 2 \
  --dec_n_points 2 \
  --resume ./checkpoints/convnext_baseline/checkpoint.pth \
  --eval
```

**记录这些结果**:
- mAP (此时约 40.5)
- 参数量
- FLOPs
- 延迟

---

## 阶段三：知识蒸馏

### 步骤1：准备 Teacher 模型

```python
# Teacher: 原始 ResNet50 + 原始配置

class Args:
    backbone = 'resnet50'
    num_decoder_layers = 6
    num_queries = 300
    enc_n_points = 4
    dec_n_points = 4

teacher_model, teacher_criterion, _ = build(Args())
# 加载预训练权重
teacher_model.load_state_dict(torch.load('pretrained_resnet50.pth'))
teacher_model.cuda().eval()
```

### 步骤2：实现蒸馏损失

```python
# 在 models/deformable_detr.py 中添加蒸馏损失

class DistillationLoss(nn.Module):
    """知识蒸馏损失"""
    
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_logits, teacher_logits, student_bbox, teacher_bbox):
        """
        Args:
            student_logits: 学生模型的分类输出
            teacher_logits: 教师模型的分类输出
            student_bbox: 学生模型的边界框输出
            teacher_bbox: 教师模型的边界框输出
        """
        
        # 分类损失 (KL 散度)
        student_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=-1)
        cls_loss = self.kl_loss(student_probs, teacher_probs) * (self.temperature ** 2)
        
        # 边界框损失 (MSE)
        bbox_loss = self.mse_loss(student_bbox, teacher_bbox)
        
        return cls_loss, bbox_loss


class CombinedLoss(nn.Module):
    """任务损失 + 蒸馏损失"""
    
    def __init__(self, criterion, distill_weight=0.5):
        super().__init__()
        self.task_criterion = criterion  # 原始的 SetCriterion
        self.distill_criterion = DistillationLoss(temperature=4.0)
        self.distill_weight = distill_weight
    
    def forward(self, outputs, targets, teacher_outputs):
        """
        Args:
            outputs: 学生模型输出
            targets: 真实标签
            teacher_outputs: 教师模型输出
        """
        
        # 任务损失
        task_loss = self.task_criterion(outputs, targets)
        
        # 蒸馏损失
        student_cls = outputs['pred_logits']  # [B, Q, num_classes]
        teacher_cls = teacher_outputs['pred_logits']
        student_bbox = outputs['pred_boxes']   # [B, Q, 4]
        teacher_bbox = teacher_outputs['pred_boxes']
        
        cls_distill, bbox_distill = self.distill_criterion(
            student_cls, teacher_cls, 
            student_bbox, teacher_bbox
        )
        
        # 组合损失
        total_loss = task_loss['loss_ce'] + task_loss['loss_bbox'] + task_loss['loss_giou']
        total_loss += self.distill_weight * (cls_distill + 0.5 * bbox_distill)
        
        return {
            'loss': total_loss,
            'loss_task': task_loss.get('loss', task_loss),
            'loss_cls_distill': cls_distill,
            'loss_bbox_distill': bbox_distill
        }
```

### 步骤3：修改训练循环

```python
# 在 engine.py 中修改训练函数

def train_one_epoch_with_distillation(model, teacher_model, criterion_combined,
                                     data_loader, optimizer, device, epoch,
                                     max_norm: float = 0):
    """带知识蒸馏的训练循环"""
    
    model.train()
    teacher_model.eval()  # 教师模型不训练
    
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 学生模型前向
        outputs = model(samples)
        
        # 教师模型前向（不需要梯度）
        with torch.no_grad():
            teacher_outputs = teacher_model(samples)
        
        # 计算组合损失
        loss_dict = criterion_combined(outputs, targets, teacher_outputs)
        loss = loss_dict['loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        # 记录日志
        print(f"Loss: {loss.item():.4f}")
        print(f"  Task: {loss_dict['loss_task'].item():.4f}")
        print(f"  Cls Distill: {loss_dict['loss_cls_distill'].item():.4f}")
        print(f"  BBox Distill: {loss_dict['loss_bbox_distill'].item():.4f}")
```

### 步骤4：蒸馏训练脚本

```python
# distillation_train.py

import torch
from torch.utils.data import DataLoader
from models import build
from engine import train_one_epoch_with_distillation, evaluate

# 学生模型配置
class StudentArgs:
    backbone = 'convnext_tiny'
    num_decoder_layers = 4
    num_queries = 150
    enc_n_points = 2
    dec_n_points = 2

# 教师模型配置
class TeacherArgs:
    backbone = 'resnet50'
    num_decoder_layers = 6
    num_queries = 300
    enc_n_points = 4
    dec_n_points = 4

# 构建模型
student_model, student_criterion, _ = build(StudentArgs())
teacher_model, _, _ = build(TeacherArgs())

# 加载预训练权重
student_model.load_state_dict(
    torch.load('checkpoints/convnext_baseline/checkpoint.pth')
)
teacher_model.load_state_dict(
    torch.load('pretrained_models/detr_resnet50_origin.pth')
)

# 构建组合损失
from models.deformable_detr import CombinedLoss
criterion_combined = CombinedLoss(student_criterion, distill_weight=0.5)

# 移到 GPU
device = torch.device('cuda')
student_model.to(device)
teacher_model.to(device)

# 优化器
optimizer = torch.optim.AdamW(
    student_model.parameters(),
    lr=2e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[20, 40],
    gamma=0.1
)

# 数据加载
train_loader, val_loader = get_dataloaders()  # 需要实现

# 蒸馏训练
for epoch in range(50):
    # 训练阶段
    train_one_epoch_with_distillation(
        student_model, teacher_model, criterion_combined,
        train_loader, optimizer, device, epoch
    )
    
    # 验证阶段
    if epoch % 5 == 0:
        stats = evaluate(
            student_model, val_loader, device
        )
        print(f"Epoch {epoch}: mAP = {stats['coco_eval_bbox']['stats'][0]:.1f}")
    
    scheduler.step()

# 保存蒸馏后的模型
torch.save(
    student_model.state_dict(),
    'checkpoints/convnext_distilled/checkpoint.pth'
)
```

### 执行蒸馏训练

```bash
python distillation_train.py \
  --num_epochs 50 \
  --batch_size 16 \
  --lr 2e-4 \
  --output_dir ./checkpoints/convnext_distilled
```

**预期结果**:
- mAP: 41.2 (+0.7 相对基准)
- 参数: 8.5M (不变)
- 延迟: 20ms (不变)

---

## 阶段四：最终验证

### 步骤1：性能对比测试

```python
# benchmark_comparison.py
import torch
import time
from models import build

def benchmark_model(backbone, dec_layers, num_queries, 
                    enc_points, dec_points, checkpoint=None):
    """基准测试单个模型"""
    
    class Args:
        pass
    
    args = Args()
    args.backbone = backbone
    args.num_decoder_layers = dec_layers
    args.num_queries = num_queries
    args.enc_n_points = enc_points
    args.dec_n_points = dec_points
    
    model, _, _ = build(args)
    
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))
    
    model.cuda().eval()
    
    # 参数量
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # 延迟测试
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            model(torch.randn(1, 3, 640, 640).cuda())
    
    torch.cuda.synchronize()
    latency = (time.time() - start) / 100
    fps = 1 / latency
    
    return {
        'params': params,
        'latency': latency,
        'fps': fps
    }

# 对比测试
configs = [
    ("resnet50", 6, 300, 4, 4, "baseline"),
    ("convnext_tiny", 4, 150, 2, 2, "convnext_baseline"),
    ("convnext_tiny", 4, 150, 2, 2, "convnext_distilled"),
]

results = {}
for backbone, dec_l, queries, enc_p, dec_p, checkpoint_key in configs:
    checkpoint_path = f"./checkpoints/{checkpoint_key}/checkpoint.pth"
    
    result = benchmark_model(
        backbone, dec_l, queries, enc_p, dec_p,
        checkpoint_path if checkpoint_key != "baseline" else None
    )
    
    results[checkpoint_key] = result
    print(f"{checkpoint_key}:")
    print(f"  参数: {result['params']:.1f}M")
    print(f"  延迟: {result['latency']*1000:.2f}ms")
    print(f"  FPS: {result['fps']:.1f}")

# 生成对比表
print("\n性能对比表:")
print("=" * 70)
print(f"{'配置':<20} {'参数(M)':<12} {'延迟(ms)':<12} {'FPS':<10}")
print("-" * 70)

baseline_params = results['baseline']['params']
baseline_latency = results['baseline']['latency']

for key, result in results.items():
    params_reduction = (1 - result['params'] / baseline_params) * 100
    latency_reduction = (1 - result['latency'] / baseline_latency) * 100
    
    print(f"{key:<20} {result['params']:<12.1f} {result['latency']*1000:<12.2f} {result['fps']:<10.1f}")
    print(f"{'vs Baseline':<20} {-params_reduction:>+11.0f}% {-latency_reduction:>+11.0f}%")
```

### 步骤2：COCO 验证集评估

```bash
# 评估蒸馏后的模型
python main.py \
  --backbone convnext_tiny \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --enc_n_points 2 \
  --dec_n_points 2 \
  --resume ./checkpoints/convnext_distilled/checkpoint.pth \
  --eval
```

### 步骤3：生成最终报告

```python
# 生成对比报告
report = """
# 方案B 优化结果报告

## 性能对比

| 指标 | 原始(ResNet50) | 优化后(ConvNeXt+蒸馏) | 改进 |
|------|---|---|---|
| 参数量 (M) | 43.3 | 8.5 | -80% |
| FLOPs (G) | 150 | 60 | -60% |
| 延迟 (ms) | 45 | 20 | -55% ⭐ |
| FPS | 22.2 | 50 | +125% |
| mAP | 42.5 | 41.2 | -1.3 |
| mAP50 | 60.5 | 59.2 | -1.3 |

## 推理加速

```
原始延迟: 45ms per image
优化后: 20ms per image
加速倍数: 2.25x ⭐

对应性能:
  原始 FPS: 22.2 img/s
  优化 FPS: 50 img/s
```

## 精度恢复 (知识蒸馏效果)

| 阶段 | mAP | 相对原始 | 相对ResNet50 |
|------|-----|---------|---|
| 原始基准 | 42.5 | - | 0 |
| Backbone替换后 | 40.5 | -2.0 | -2.0 |
| 蒸馏后 | 41.2 | -1.3 | -1.3 |
| 蒸馏恢复 | +0.7 | - | - |

蒸馏恢复率: 0.7 / 2.0 = 35% ✓

## 推荐场景

✅ 生产部署 (精度>40.5, 速度>2x)
✅ 实时应用 (50 FPS 足以)
✅ 资源受限场景 (参数减少80%)

❌ 高精度场景 (mAP需要>42)
❌ 边缘计算 (仍需 GPU)
"""

print(report)
```

---

## 故障排查

### 问题1：Backbone 更换后前向传播失败

**症状**: RuntimeError: size mismatch
```
Expected: 512 channels, got: 96 channels
```

**原因**: ConvNeXt 输出通道与 ResNet50 不同

**解决方案**:
```python
# 在 input_proj 中正确处理通道映射
self.input_proj = nn.ModuleList([
    nn.Conv2d(backbone.num_channels[i], hidden_dim, kernel_size=1)
    for i in range(num_feature_levels)
])
```

### 问题2：蒸馏训练时 OOM (内存溢出)

**症状**: CUDA out of memory

**原因**: 同时加载学生和教师模型

**解决方案**:
```python
# 减少 batch size
batch_size = 8  # 从 16 改为 8

# 或者使用梯度累积
for _ in range(2):  # 累积 2 步
    outputs = student_model(samples)
    loss.backward()

optimizer.step()
optimizer.zero_grad()
```

### 问题3：蒸馏后 mAP 反而下降

**症状**: 蒸馏 10 个 epoch 后，mAP 从 40.5 降到 40.2

**原因**: 蒸馏权重设置不当

**解决方案**:
```python
# 调整蒸馏权重
distill_weight = 0.3  # 从 0.5 改为 0.3 (减弱蒸馏)

# 或者分阶段蒸馏
if epoch < 10:
    distill_weight = 0.1  # 前10个epoch弱蒸馏
else:
    distill_weight = 0.5  # 后续正常蒸馏
```

### 问题4：延迟没有改进

**症状**: ConvNeXt 模型延迟仍然为 40ms (没有改进)

**原因**: 
1. GPU 驱动未优化 ConvNeXt
2. 模型仍有其他瓶颈

**解决方案**:
```bash
# 使用 TensorRT 进一步优化
python export_onnx.py \
  --checkpoint convnext_distilled/checkpoint.pth \
  --output model.onnx

# 转换为 TensorRT
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16

# 评估 TensorRT 模型
python benchmark_trt.py --engine model.trt
```

---

## 完整命令速查

### 构建和测试

```bash
# 1. 验证前向传播
python test_forward.py

# 2. 基线训练 (ResNet50)
python main.py --backbone resnet50 --epochs 50 --output_dir checkpoints/baseline

# 3. ConvNeXt 基线 (无蒸馏)
python main.py \
  --backbone convnext_tiny \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --enc_n_points 2 \
  --dec_n_points 2 \
  --epochs 50 \
  --output_dir checkpoints/convnext_baseline

# 4. 蒸馏训练
python distillation_train.py \
  --checkpoint_student checkpoints/convnext_baseline/checkpoint.pth \
  --checkpoint_teacher checkpoints/baseline/checkpoint.pth \
  --epochs 50 \
  --output_dir checkpoints/convnext_distilled

# 5. 评估
python main.py \
  --backbone convnext_tiny \
  --resume checkpoints/convnext_distilled/checkpoint.pth \
  --eval

# 6. 性能基准测试
python benchmark_comparison.py
```

---

## 预期时间表

| 阶段 | 任务 | 时间 |
|------|------|------|
| 准备 | 安装依赖、修改代码 | 1-2 天 |
| 测试 | 验证前向传播 | 0.5 天 |
| 基线 | 训练 ConvNeXt 基线 | 3-4 天 (50 epochs) |
| 蒸馏 | 知识蒸馏训练 | 3-4 天 (50 epochs) |
| 评估 | 性能测试和对比 | 1 天 |
| **总计** | | **8-12 天** |

---

## 检查清单

在开始前确认：

- [ ] 有原始 ResNet50 预训练模型
- [ ] 安装了 timm 库 (ConvNeXt)
- [ ] GPU 显存足够 (建议 >=16GB)
- [ ] 修改了 backbone.py 支持 ConvNeXt
- [ ] 实现了蒸馏损失函数
- [ ] 准备了 COCO 验证集

开始后每个阶段完成后：

- [ ] 第一步：验证 Backbone 替换的前向传播
- [ ] 第二步：记录 ConvNeXt 基线的性能和 mAP
- [ ] 第三步：蒸馏训练收敛，mAP 恢复
- [ ] 第四步：最终性能对比满足预期 (2.25x + mAP -1.3)

---

## 下一步

✅ 完成方案B后，可以考虑：

1. **进一步加速** (方案A):
   - Decoder 层数改为 3 (而非 4)
   - Query 数改为 100 (而非 150)
   - 预期: 3-4x 加速，但需要更强的蒸馏

2. **部署优化**:
   - 转换为 ONNX
   - 编译 TensorRT
   - 预期: 额外 1.5x 加速

3. **量化**:
   - INT8 量化
   - 预期: 额外 1.5-2x 加速，精度损失 <0.5 mAP

---

**祝你优化顺利！** 🚀

如有问题，参考 `开发日志.md` 或 `故障排查` 章节。
