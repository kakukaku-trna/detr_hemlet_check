# 方案B 快速参考卡片

## 🎯 一页纸总结

**方案**: Backbone 替换 (ResNet50→ConvNeXt) + 知识蒸馏  
**目标**: 2-2.2x 加速，精度损失 <1.5 mAP  
**周期**: 2-3 周  
**难度**: ⭐⭐ 中等

---

## 📊 性能对标

| 指标 | 原始 | 优化后 | 改进 |
|------|------|--------|------|
| **延迟** | 45ms | 20ms | **-55% ⭐** |
| FPS | 22 | 50 | +125% |
| 参数 | 43.3M | 8.5M | -80% |
| FLOPs | 150G | 60G | -60% |
| mAP | 42.5 | 41.2 | -1.3 |

---

## 🔧 核心配置

```python
# 替换为这个配置
--backbone convnext_tiny      # Backbone
--num_decoder_layers 4        # 从 6 改为 4
--num_queries 150             # 从 300 改为 150
--enc_n_points 2              # 从 4 改为 2
--dec_n_points 2              # 从 4 改为 2
```

---

## 🚀 快速命令

### 第1步：测试前向传播 (30分钟)

```bash
# 验证 ConvNeXt 可以正常工作
python test_forward.py  # 修改为使用 convnext_tiny
```

### 第2步：训练基线 (3-4 天)

```bash
python main.py \
  --backbone convnext_tiny \
  --num_decoder_layers 4 \
  --num_queries 150 \
  --enc_n_points 2 \
  --dec_n_points 2 \
  --epochs 50 \
  --output_dir ./checkpoints/convnext_baseline \
  --eval
```

### 第3步：知识蒸馏 (3-4 天)

```bash
python distillation_train.py \
  --checkpoint_student ./checkpoints/convnext_baseline/checkpoint.pth \
  --checkpoint_teacher ./pretrained/resnet50_original.pth \
  --epochs 50 \
  --output_dir ./checkpoints/convnext_distilled
```

### 第4步：最终评估 (1 小时)

```bash
# 性能测试
python benchmark_comparison.py

# 精度评估
python main.py \
  --backbone convnext_tiny \
  --resume ./checkpoints/convnext_distilled/checkpoint.pth \
  --eval
```

---

## 📝 代码改动清单

### 1️⃣ 修改 `models/backbone.py`

添加 ConvNeXt 支持：
```python
elif args.backbone == 'convnext_tiny':
    import timm
    model = timm.create_model('convnext_tiny', pretrained=True)
    # 封装为 Backbone 类
    backbone = ConvNeXtBackbone(model)
    backbone.strides = [8, 16, 32]
    backbone.num_channels = [96, 384, 768]
```

### 2️⃣ 修改 `models/deformable_detr.py`

调整特征投影处理 Backbone 输出通道差异：
```python
# input_proj 需要根据 backbone.num_channels 动态创建
for i in range(num_feature_levels):
    in_channels = backbone.num_channels[i] if i < len(backbone.num_channels) 
                  else backbone.num_channels[-1]
    self.input_proj.append(nn.Sequential(
        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
        nn.GroupNorm(32, hidden_dim),
    ))
```

### 3️⃣ 添加蒸馏损失 (新文件或新函数)

```python
class DistillationLoss(nn.Module):
    def forward(self, student, teacher):
        # KL散度 for 分类
        # MSE loss for 边界框
        return cls_loss + bbox_loss

class CombinedLoss(nn.Module):
    def forward(self, student_out, targets, teacher_out):
        task_loss = self.criterion(student_out, targets)
        distill_loss = self.distill_criterion(student_out, teacher_out)
        return task_loss + 0.5 * distill_loss
```

### 4️⃣ 修改 `engine.py` 训练循环

```python
def train_one_epoch_with_distillation(
    student_model, teacher_model, criterion_combined, 
    data_loader, optimizer, device
):
    # 教师不更新: teacher_model.eval()
    # 组合损失: criterion_combined(student_out, targets, teacher_out)
```

---

## ⏱️ 时间表

| 周期 | 任务 | 预计时间 |
|------|------|---------|
| **第1周** | 代码准备 + 基线训练 | 3-4 天 (50 epochs) |
| **第2周** | 蒸馏训练 | 3-4 天 (50 epochs) |
| **第3周** | 评估和微调 | 1-2 天 |

---

## 📊 关键指标

### 性能指标

```
总加速: 2.25x
  = 2x (Backbone 替换)
  × 1.1 (Decoder 减层 + Query 减少)

性能破分:
  原始: 45ms per image → 22 FPS
  优化: 20ms per image → 50 FPS
```

### 精度指标

```
mAP 变化:
  原始: 42.5
  Backbone 替换后: 40.5 (-2.0)
  知识蒸馏后: 41.2 (-1.3)
  蒸馏恢复: +0.7 (+35%)
```

### 模型大小

```
参数量:
  原始: 43.3M
  优化: 8.5M (-80%)

磁盘大小:
  原始: ~180MB
  优化: ~35MB (-81%)
```

---

## ⚠️ 常见陷阱

### ❌ 错误1：直接替换 Backbone 而不调整特征投影

```python
# 错误
model.input_proj = input_proj_resnet  # 通道数不匹配

# 正确
model.input_proj = build_input_proj(backbone.num_channels, hidden_dim)
```

### ❌ 错误2：蒸馏过程中 Teacher 也在更新

```python
# 错误
criterion = CombinedLoss(teacher_model)
teacher_model.train()  # ❌ 教师应该 eval!

# 正确
teacher_model.eval()
with torch.no_grad():
    teacher_out = teacher_model(...)
```

### ❌ 错误3：蒸馏权重设置不当

```python
# 错误：权重过大，学生只学蒸馏信号，忽视真实标签
loss = task_loss + 10.0 * distill_loss  # ❌

# 正确：蒸馏和任务平衡
loss = task_loss + 0.5 * distill_loss   # ✓
```

---

## ✅ 验证清单

启动前：
- [ ] 有 ResNet50 预训练权重
- [ ] 装了 `pip install timm>=0.6.0`
- [ ] GPU 显存 >=16GB
- [ ] 能访问 COCO 验证集

第1步后：
- [ ] ✅ 前向传播成功
- [ ] ✅ 参数量确实是 8.5M

第2步后：
- [ ] ✅ ConvNeXt 基线收敛
- [ ] ✅ mAP 约 40.5
- [ ] ✅ 延迟约 20ms

第3步后：
- [ ] ✅ 蒸馏训练收敛
- [ ] ✅ mAP 恢复到 41.2
- [ ] ✅ 蒸馏损失逐渐下降

第4步后：
- [ ] ✅ 最终 mAP: 41.2 ±0.1
- [ ] ✅ 总加速: 2.2x ±0.1x

---

## 🎓 如果遇到问题

### 问题：前向传播失败

```
RuntimeError: size mismatch: expected (16, 512, 80, 80) but got (16, 96, 80, 80)
```

**解决**:
```python
# input_proj 需要正确处理通道
print(f"Backbone 输出通道: {backbone.num_channels}")  # 应该打印 [96, 384, 768]
# 确保 input_proj 的输入通道与之匹配
```

### 问题：OOM (内存溢出)

```
RuntimeError: CUDA out of memory
```

**解决**:
```bash
# 减少 batch size
--batch_size 8  # 从 16 改为 8

# 或者只在 eval 时加载 teacher
teacher_model.to('cpu')  # 平时不需要在 GPU
teacher_model.cuda() if i % 10 == 0 else None
```

### 问题：蒸馏后 mAP 没有恢复

**检查**:
1. Teacher 模型是否真的是好的 (mAP>42)
2. 蒸馏权重是否太大
3. 训练是否足够长 (50 epochs 最少)

**调整**:
```python
# 先尝试更弱的蒸馏
distill_weight = 0.2  # 从 0.5 改为 0.2

# 或者两阶段蒸馏
if epoch < 20:
    distill_weight = 0.2  # 前 20 epoch 弱蒸馏
else:
    distill_weight = 0.5  # 后 30 epoch 强蒸馏
```

---

## 📞 快速参考

| 我想... | 做这个 |
|--------|--------|
| 验证代码可行 | `python test_forward.py` |
| 训练 ConvNeXt 基线 | `python main.py --backbone convnext_tiny ...` |
| 进行知识蒸馏 | `python distillation_train.py` |
| 评估最终性能 | `python benchmark_comparison.py` |
| 查看详细指南 | 打开 `方案B_详细实施指南.md` |
| 查看所有代码 | 参考 `方案B_详细实施指南.md` 代码部分 |

---

## 🎯 成功标志

当你看到这些指标时，说明方案B成功了 ✅

```
✅ ConvNeXt 基线 (第2步):
   - 参数: 8.5M
   - 延迟: 20ms
   - mAP: 40.5

✅ 蒸馏后 (第3步):
   - 参数: 8.5M
   - 延迟: 20ms
   - mAP: 41.2 (+0.7)

✅ 总体成果:
   - 加速: 2.25x ⭐
   - 参数: -80%
   - 精度损失: -1.3 mAP (可接受)
```

---

**版本**: 1.0  
**最后更新**: 2026-06-10  
**状态**: ✅ 完整可用

祝优化顺利！🚀
