# Optimize Deformable-DETR Implementation

## Task

Generate code modifications that reduce computation while maintaining accuracy.

## Optimization Strategies

### Strategy 1: Query Reduction
- Default: `num_queries = 300`
- Candidates: 100, 150, 200, 250
- Focus: Decoder FLOPs, latency
- Expected gain: 60% FLOPs reduction @ num_queries=100

### Strategy 2: Decoder Layer Reduction
- Default: 6 decoder layers
- Candidates: 3, 4, 5
- Focus: Decoder latency, parameter count
- Expected gain: 30%+ speed improvement @ 3 layers, mAP drop < 1

### Strategy 3: Backbone Replacement
- Current: ResNet50
- Alternatives:
  - MobileNetV3-Large
  - EfficientNet-Lite0/1/2
  - ConvNeXt-Tiny
  - FasterNet
  - RepViT
- Focus: Feature extraction latency, backbone parameters
- Expected gain: Up to 50% speed improvement

### Strategy 4: Attention Simplification
- Default: 8 heads, 4 sampling points
- Candidates: 
  - 4 heads, 4 sampling points
  - 8 heads, 2 sampling points
  - 4 heads, 2 sampling points
- Focus: Deformable attention computation
- Expected gain: 10-20% latency improvement

### Strategy 5: Knowledge Distillation
- Teacher: Full Deformable-DETR (ResNet50)
- Student: Lightweight variant
- Loss components:
  - Classification distillation loss
  - Bounding box distillation loss
  - Feature-level distillation loss
- Expected gain: Recover 0.5-1.5 mAP points

### Strategy 6: Deployment Optimization
- Convert PyTorch → ONNX → TensorRT
- Enable INT8 quantization
- Expected gain: 3x speed improvement

## For Each Modification

Provide:

1. **Rationale** — Why this change helps
2. **Parameter reduction** — Expected % decrease
3. **FLOPs reduction** — Expected % decrease
4. **Latency improvement** — Expected % decrease
5. **Accuracy impact** — Estimated mAP change
6. **Implementation diff** — Exact code changes
7. **Modified files** — List all changed files
8. **Integration notes** — How to test and validate

## Output Format

```markdown
## Modification: [Name]

**Rationale**: [Why]

**Expected Gains**:
- Parameters: -X%
- FLOPs: -Y%
- Latency: -Z%
- mAP: ±A points

**Code Changes**:
- File: [path]
- [diff]

**Implementation Steps**:
1. [step]
2. [step]

**Validation**:
- [ ] Benchmark on COCO val set
- [ ] Check FLOPs reduction
- [ ] Profile latency
- [ ] Verify mAP
```

## Priority

1. FLOPs reduction
2. Latency improvement
3. Preserve mAP
4. Minimize implementation complexity
