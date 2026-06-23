# Deformable-DETR Ablation Study Template

## Ablation Study: [Title]

Study systematic impact of components on performance.

## Study Design

| Component | Baseline | Variant 1 | Variant 2 | Variant 3 |
|-----------|----------|-----------|-----------|-----------|
| [Component A] | [config] | [config] | [config] | [config] |
| [Component B] | [config] | [config] | [config] | [config] |
| [Component C] | [config] | [config] | [config] | [config] |

## Experiments

### Experiment 1: [Component A] Impact

**Configs to test**:
- [Variant A1]
- [Variant A2]
- [Variant A3]

### Experiment 2: [Component B] Impact

**Configs to test**:
- [Variant B1]
- [Variant B2]
- [Variant B3]

### Experiment 3: [Component C] Impact

**Configs to test**:
- [Variant C1]
- [Variant C2]
- [Variant C3]

## Results Comparison

### Parameters & FLOPs

| Config | Params (M) | FLOPs (G) | Backbone | Decoder | Attention |
|--------|-----------|----------|----------|---------|-----------|
| Baseline | [X] | [X] | [X]% | [X]% | [X]% |
| Variant 1 | [Y] | [Y] | [Y]% | [Y]% | [Y]% |
| Variant 2 | [Z] | [Z] | [Z]% | [Z]% | [Z]% |

### Performance Metrics

| Config | Latency (ms) | FPS | Memory (MB) |
|--------|-------------|-----|------------|
| Baseline | [X] | [X] | [X] |
| Variant 1 | [Y] | [Y] | [Y] |
| Variant 2 | [Z] | [Z] | [Z] |

### Accuracy Metrics

| Config | mAP | mAP50 | mAP75 | mAP_S | mAP_M | mAP_L |
|--------|-----|-------|-------|-------|-------|-------|
| Baseline | [X] | [X] | [X] | [X] | [X] | [X] |
| Variant 1 | [Y] | [Y] | [Y] | [Y] | [Y] | [Y] |
| Variant 2 | [Z] | [Z] | [Z] | [Z] | [Z] | [Z] |

## Key Findings

### Component A Impact

**Finding**: [What changed as we modified this component]

- Param reduction: [X]% per unit
- FLOPs reduction: [Y]% per unit
- Accuracy impact: [Z] mAP per unit
- Optimal value: [recommendation]

### Component B Impact

**Finding**: [What changed as we modified this component]

- Param reduction: [X]% per unit
- FLOPs reduction: [Y]% per unit
- Accuracy impact: [Z] mAP per unit
- Optimal value: [recommendation]

### Component C Impact

**Finding**: [What changed as we modified this component]

- Param reduction: [X]% per unit
- FLOPs reduction: [Y]% per unit
- Accuracy impact: [Z] mAP per unit
- Optimal value: [recommendation]

## Interaction Effects

### A × B Interaction

**Question**: Does effect of A depend on B?

- A alone: [effect]
- A + B variant 1: [effect]
- A + B variant 2: [effect]

**Conclusion**: [Synergy/Independence/Conflict]

### A × C Interaction

[Same format]

### B × C Interaction

[Same format]

## Recommendations

### Optimal Configuration

```python
config = {
    "component_a": [recommended_value],
    "component_b": [recommended_value],
    "component_c": [recommended_value],
    # ... other configs
}
```

**Expected performance**:
- Parameters: [X]M (-[Y]% vs baseline)
- FLOPs: [X]G (-[Y]% vs baseline)
- Latency: [X]ms (-[Y]% vs baseline)
- mAP: [X] ([±Y] vs baseline)

### Trade-off Analysis

| Config | Speed | Size | mAP | Use Case |
|--------|-------|------|-----|----------|
| Aggressive | ↑↑↑ | ↓↓↓ | [-Z] | [Mobile/Edge] |
| Balanced | ↑↑ | ↓↓ | [-Z/2] | [Server] |
| Conservative | ↑ | ↓ | [-Z/4] | [High-accuracy] |

## Visualization

### Performance Curves

```
FLOPs vs mAP:
    mAP ^
        |     ●(best)
        |    ●
        |   ●
        |  ●
        | ●
        |●
        +------- FLOPs>
```

### Pareto Front

[Plot showing optimal trade-offs between metrics]

## Next Steps

- [ ] Validate optimal config on full test set
- [ ] Test on different hardware (GPU/CPU/EdgeDevices)
- [ ] Compare with competing methods
- [ ] Deploy recommended variant
- [ ] Monitor performance in production

## References

- Paper: [link to related paper]
- Prior ablations: [[exp-id]](link/to/previous/ablation.md)
- Follow-up work: [[exp-id]](link/to/followup/ablation.md)
