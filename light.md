Skill目标

例如：

Skill Name:
deformable_detr_optimizer

Purpose:
帮助优化和轻量化Deformable-DETR模型，
包括Backbone压缩、Transformer剪枝、
Attention简化、蒸馏训练、
参数量/FLOPs分析、
推理速度优化等。
Skill目录
.claude/
└── skills/
    └── deformable_detr_optimizer/
        ├── SKILL.md
        ├── prompts/
        │   ├── analyze.md
        │   ├── optimize.md
        │   └── benchmark.md
        └── templates/
            ├── experiment.md
            └── ablation.md
SKILL.md

核心内容：

# Deformable-DETR Optimizer

## Objective

Optimize Deformable-DETR for:

- lower FLOPs
- lower latency
- smaller model size
- maintain mAP

## Optimization Priorities

1. Backbone compression
2. Encoder simplification
3. Decoder simplification
4. Query reduction
5. Knowledge distillation
6. Quantization
7. Deployment acceleration

---

## Required Analysis

When modifying a model:

1. Calculate parameter count.
2. Estimate FLOPs.
3. Identify latency bottlenecks.
4. Explain impact on accuracy.

Always provide:

- changed files
- code diff
- expected gains
- risks
分析Prompt

analyze.md

Analyze current Deformable-DETR implementation.

Focus on:

1. Backbone
2. Encoder
3. Decoder
4. Multi-scale feature fusion
5. Deformable attention

Output:

- parameter distribution
- FLOPs distribution
- memory usage
- optimization opportunities

Rank opportunities by ROI.
优化Prompt

optimize.md

Given a Deformable-DETR implementation:

Generate code modifications that reduce computation.

Priority:

1. reduce FLOPs
2. reduce latency
3. preserve mAP

For each modification:

- rationale
- expected parameter reduction
- expected FLOPs reduction
- implementation diff
Benchmark Prompt

benchmark.md

After modification:

Report:

- Params
- FLOPs
- FPS
- Latency
- mAP

Compare with baseline.

Output markdown table.
针对Deformable-DETR最值得让Claude研究的方向

如果是自动化优化，我会把下面这些写进 Skill。

方向1 Query数量压缩

默认：

num_queries = 300

改：

num_queries = 100

收益：

Decoder FLOPs ↓ 60%
Latency ↓

Claude可以自动评估：

100
150
200
300
方向2 Decoder层数减少

默认：

6 layers

尝试：

3 layers
4 layers

收益很大。

很多实际项目：

6 → 3

mAP下降不到1
速度提升30%以上
方向3 Backbone替换

让Claude自动搜索：

ResNet50

→ MobileNetV3
→ EfficientNet-Lite
→ ConvNeXt-Tiny
→ FasterNet
→ RepViT

这是收益最大的地方。

方向4 Attention简化

Deformable Attention：

8 heads
4 sampling points

尝试：

4 heads
2 sampling points

参数变化不大：

但速度提升明显。

方向5 Distillation

让Claude自动构建：

Teacher:
Deformable-DETR-R50

Student:
Light-Deformable-DETR

增加：

cls_loss_distill
bbox_loss_distill
feature_loss

保持精度。

方向6 ONNX/TensorRT部署优化

Skill里要求：

Always generate:

export_onnx.py
benchmark_trt.py

自动做：

PyTorch
→ ONNX
→ TensorRT

因为很多时候：

模型轻量化收益 20%

TensorRT收益 300%