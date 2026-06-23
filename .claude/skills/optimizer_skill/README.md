# Deformable-DETR Optimizer Skill

一个系统化的 Deformable-DETR 模型轻量化研发 Skill，基于6个核心优化方向。

## 📋 Skill 概况

| 项目 | 说明 |
|------|------|
| **Skill 名称** | deformable_detr_optimizer |
| **目标** | 优化 Deformable-DETR 的 FLOPs、延迟、参数量，同时保持 mAP |
| **支持的优化方向** | 6 个 |
| **包含的 Prompts** | 3 个（分析、优化、基准测试） |
| **提供的模板** | 2 个（实验、消融研究） |

## 🎯 核心优化方向

### 方向1: Query数量压缩
- **默认**：`num_queries = 300`
- **目标**：降低至 100-200
- **收益**：Decoder FLOPs ↓60%、延迟显著下降
- **实施难度**：⭐ 简单

### 方向2: Decoder层数减少
- **默认**：6 层 decoder
- **目标**：3-4 层
- **收益**：速度提升 30%+，mAP下降 <1
- **实施难度**：⭐ 简单

### 方向3: Backbone替换
- **当前**：ResNet50
- **候选**：
  - MobileNetV3-Large
  - EfficientNet-Lite0/1/2
  - ConvNeXt-Tiny
  - FasterNet
  - RepViT
- **收益**：速度提升 50%+ ⭐ 收益最大
- **实施难度**：⭐⭐ 中等

### 方向4: Attention简化
- **默认**：8 heads, 4 sampling points
- **改进**：4 heads, 2-4 sampling points
- **收益**：延迟提升 10-20%
- **实施难度**：⭐ 简单

### 方向5: 知识蒸馏
- **Teacher**：Full Deformable-DETR (ResNet50)
- **Student**：轻量级变体
- **收益**：恢复 0.5-1.5 mAP 精度点
- **实施难度**：⭐⭐⭐ 复杂

### 方向6: 部署优化 (ONNX/TensorRT)
- **流程**：PyTorch → ONNX → TensorRT
- **收益**：3倍速度提升 ⭐⭐ 收益巨大
- **实施难度**：⭐⭐ 中等

## 📂 Skill 结构

```
optimizer_skill/
├── README.md                 # 本文件
├── SKILL.md                  # Skill 定义和优化优先级
├── prompts/
│   ├── analyze.md           # 分析当前实现
│   ├── optimize.md          # 生成优化方案
│   └── benchmark.md         # 基准测试和对比
└── templates/
    ├── experiment.md        # 单个实验记录模板
    └── ablation.md          # 消融研究模板
```

## 🚀 使用工作流

### 第一步：分析（Analyze）

使用 `prompts/analyze.md` 中的指导：

```
Claude，请基于我的 Deformable-DETR 实现进行分析：
- 参数分布
- FLOPs 分布
- 内存使用
- 优化机会排序
```

**输出**：结构化分析表，ROI 排名

---

### 第二步：规划优化（Plan Optimization）

选择最佳 ROI 的优化方向：

- 🥇 **Backbone 替换** — 收益最大
- 🥈 **Decoder 层数减少** — 实施简单，收益大
- 🥉 **Query 数量压缩** — 实施简单，收益中等

---

### 第三步：实施修改（Implement）

使用 `prompts/optimize.md` 中的指导生成代码：

```
Claude，基于 [方向]，生成优化代码：
- 修改文件列表
- 具体代码差异
- 预期收益
- 风险评估
```

**输出**：完整的代码修改建议

---

### 第四步：基准测试（Benchmark）

使用 `prompts/benchmark.md` 验证效果：

```
Claude，请基准测试修改后的模型：
- 参数量
- FLOPs
- 延迟
- mAP
```

**输出**：详细对比表格

---

### 第五步：记录结果（Document）

使用提供的模板记录实验：

- **单个实验**：使用 `templates/experiment.md`
- **系统消融**：使用 `templates/ablation.md`

## 📊 关键度量指标

### 大小指标
- Parameters (M) — 模型参数量
- Model size (MB) — 模型文件大小

### 计算指标
- FLOPs (G) — 单张图计算量
- Memory (MB) — 峰值 GPU 显存

### 性能指标
- **FPS** — 吞吐量
- **Latency (ms)** — 推理延迟
- **Batch latency** — 不同 batch size 的延迟

### 精度指标
- **mAP** — 标准 COCO 检测指标
- **mAP50, mAP75** — 不同 IoU 阈值
- **mAP_S, mAP_M, mAP_L** — 不同物体尺度

## 💡 最佳实践

### 1. 系统性优化
- 不要一次改多个方向
- 逐个验证每个优化的效果
- 使用消融研究理解交互效应

### 2. 度量驱动
- 始终基准测试，不凭感觉
- 记录所有实验数据
- 对比基准线，计算 ROI

### 3. 准确性优先
- 轻量化的目标是 speed-up，不是替换
- mAP 下降 < 1 是可接受的底线
- 考虑知识蒸馏恢复精度

### 4. 硬件感知
- 不同硬件的优化不同
- 在目标硬件上测试
- 记录硬件规格和 CUDA 版本

### 5. 部署联动
- 轻量化后用 TensorRT 进一步优化
- 组合效果可达 3-5 倍加速
- 考虑量化（INT8）

## 📝 实验流程

```
1. 分析 (analyze.md)
   ↓
2. 选择优化方向
   ↓
3. 生成优化方案 (optimize.md)
   ↓
4. 实施修改
   ↓
5. 基准测试 (benchmark.md)
   ↓
6. 记录结果 (experiment.md)
   ↓
7. 评估 ROI
   ↓
8. 集成或继续优化
```

## 🎓 示例工作流

### Example 1: Query 压缩

```markdown
## 实验：Query 数量压缩

### 假设
Decoder FLOPs 占比 30%，通过将 num_queries 从 300 减少到 100，
可以将 Decoder FLOPs 减少 60%，总体 FLOPs 减少 18%。

### 配置变更
- 修改 config：num_queries = 100
- 修改模型 forward pass

### 预期收益
- FLOPs: -18%
- Latency: -15%
- mAP: -0.5

### 实验结果
[实际运行后填写]
```

### Example 2: 对比基准线

```markdown
| 指标 | Baseline | Modified | Change | ROI |
|------|----------|----------|--------|-----|
| Params | 43.3M | 19.4M | -55% | ✓ |
| FLOPs | 215G | 127G | -41% | ✓ |
| Latency | 45ms | 28ms | -38% | ✓ |
| mAP | 40.3 | 39.1 | -1.2 | ⚠ |
```

## 🔧 工具链集成

与以下工具协同使用：

- **fvcore** — FLOPs 计算
- **thop** — 模型分析
- **torch.utils.benchmark** — 延迟测试
- **TensorRT** — 部署优化
- **wandb** — 实验追踪

## 📚 参考资源

### 论文
- Deformable DETR: https://arxiv.org/abs/2010.04159
- Knowledge Distillation: https://arxiv.org/abs/1503.02531
- Model Compression: https://arxiv.org/abs/2104.14294

### 代码库
- Deformable-DETR: https://github.com/fundamentalvision/Deformable-DETR
- MMDetection: https://github.com/open-mmlab/mmdetection

## ✅ 检查清单

在开始优化前，确保：

- [ ] 有基准线模型和基准线精度
- [ ] 明确硬件目标（GPU 型号、部署场景）
- [ ] 理解当前模型的性能瓶颈
- [ ] 有足够的计算资源进行实验
- [ ] 能够在目标数据集上验证精度

## 🤝 提问和支持

### 常见问题

**Q: 应该从哪个方向开始？**
A: 建议按以下顺序：
1. Decoder 层数减少（简单且有效）
2. Query 压缩（组合使用效果更好）
3. Backbone 替换（需要重新训练）

**Q: 如何平衡速度和精度？**
A: 使用知识蒸馏。Teacher 用原始模型，Student 用轻量化模型，
联合训练可以恢复 0.5-1.5 mAP 精度。

**Q: 为什么轻量化后反而更慢？**
A: 可能原因：
- 内存访问模式改变，缓存命中率下降
- 不同硬件对不同操作的优化不同
- 需要在目标硬件上测试

---

**最后更新**: 2026-06-10
