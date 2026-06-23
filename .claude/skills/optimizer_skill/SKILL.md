# Deformable-DETR Optimizer

## Objective

Optimize Deformable-DETR for:

- Lower FLOPs
- Lower latency
- Smaller model size
- Maintain or improve mAP

## Optimization Priorities

1. **Query数量压缩** — Reduce num_queries (300 → 100-200)
2. **Decoder层数减少** — Reduce decoder layers (6 → 3-4)
3. **Backbone替换** — Replace with efficient backbones (ResNet50 → MobileNetV3/EfficientNet/ConvNeXt-Tiny)
4. **Attention简化** — Reduce heads/sampling points in deformable attention
5. **知识蒸馏** — Knowledge distillation from larger teacher model
6. **量化部署** — Quantization and TensorRT optimization

## Required Analysis

When modifying a model, always provide:

1. **Changed files** — List all modified source files
2. **Code diff** — Show exact modifications
3. **Expected gains**:
   - Parameter count reduction (%)
   - FLOPs reduction (%)
   - Latency/speed improvement (%)
4. **Accuracy impact** — Expected mAP change
5. **Risks** — Potential issues and mitigation

## Benchmark Metrics

Track these metrics:

- **Params** (M) — Model parameters
- **FLOPs** (G) — Theoretical computation
- **FPS** — Frames per second
- **Latency** (ms) — Per-image inference time
- **mAP** — COCO detection accuracy
- **mAP50** — Detection @ IoU=0.5

## Implementation Workflow

1. **Analyze** — Current architecture profiling
2. **Plan** — Select optimization strategy
3. **Implement** — Modify model/training code
4. **Benchmark** — Measure performance and accuracy
5. **Report** — Compare with baseline
