# Hash Pipeline - Quick Reference

```bash
# Full analysis
python tools/hash_pipeline/hash_pipeline.py

# With visualization
python tools/hash_pipeline/hash_pipeline.py --visualize --elements 8

# Code generation hints
python tools/hash_pipeline/hash_pipeline.py --codegen
```

**Key insight**: tmp1 and tmp2 in each stage are INDEPENDENT - can run in parallel!

**Per stage**: `a = (a op1 const) op2 (a op3 shift)` = 2 parallel ops + 1 combine = 2 cycles

**6 stages x 2 cycles = 12 cycles per element** (critical path)

**With 6 VALU slots**: theoretical min = max(12, ceil(18N/6)) = max(12, 3N) cycles for N elements
