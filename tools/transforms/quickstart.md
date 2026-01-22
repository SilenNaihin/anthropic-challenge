# Transforms - Quick Start

## TL;DR

Codified transformations: loop unroll, vectorize, pipeline, hoist invariants.

```bash
python tools/transforms/transforms.py           # Analyze kernel
python tools/transforms/transforms.py --demo    # See examples
```

## The Four Transforms

### 1. Loop Unroll

```python
from tools.transforms.transforms import unroll_loop

result = unroll_loop(body, unroll_factor=4)
```

### 2. Vectorize Batch

```python
from tools.transforms.transforms import vectorize_batch, ScalarOp

ops = [ScalarOp("alu", "+", 10, [20, 30], 0), ...]  # 8 similar ops
result = vectorize_batch(ops)
```

### 3. Software Pipeline

```python
from tools.transforms.transforms import software_pipeline

result = software_pipeline(prologue, body, epilogue, overlap_cycles=12)
```

### 4. Hoist Invariants

```python
from tools.transforms.transforms import hoist_invariants

result = hoist_invariants(loop_body, iterations=16)
```

## Hash Helper

```python
from tools.transforms.transforms import generate_vectorized_hash_stage

# Generates 2-cycle hash stage using tmp1 || tmp2 ILP
instrs = generate_vectorized_hash_stage(0, val, tmp1, tmp2, const1, const3)
```

## Quick Analysis

```python
from tools.transforms.transforms import analyze_transform_opportunities

analysis = analyze_transform_opportunities(instructions)
print(analysis['summary']['primary_bottleneck'])  # 'hash' or 'memory'
```

## Workflow

1. Analyze -> 2. Hoist invariants -> 3. Unroll -> 4. Pack

See `README.md` for full documentation.
