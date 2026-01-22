# Transformation Library

Codified transformations to reduce manual errors in mechanical transforms for VLIW SIMD optimization.

## Overview

This library provides four key transformations:

1. **Loop Unrolling** - Replicate loop body N times with register renaming
2. **Vectorize Batch** - Convert scalar ops to vector ops (VLEN=8)
3. **Software Pipelining** - Overlap iteration N+1 prep with N execution
4. **Hoist Invariants** - Move loop-invariant code outside loops

## Installation

No additional dependencies required. Optional: `pip install rich` for colored output.

## Quick Start

```bash
# Analyze current kernel for transformation opportunities
python tools/transforms/transforms.py

# Run demonstration
python tools/transforms/transforms.py --demo

# JSON output for scripting
python tools/transforms/transforms.py --json
```

## API Reference

### Loop Unrolling

```python
from tools.transforms.transforms import unroll_loop

# Basic unroll 4x
result = unroll_loop(loop_body, unroll_factor=4)

# With custom register renaming
result = unroll_loop(
    body=loop_body,
    unroll_factor=4,
    rename_map_fn=lambda addr, iter_num: addr + iter_num * 100
)

# Check result
if result.success:
    print(f"Speedup: {result.speedup:.2f}x")
    new_code = result.transformed_code
```

**Parameters:**
- `body`: List of instruction bundles forming the loop body
- `unroll_factor`: Replication count (2, 4, 8, etc.)
- `induction_var`: Optional scratch address of loop counter
- `rename_map_fn`: Function(addr, iteration) -> new_addr for register renaming

**Notes:**
- Register renaming prevents WAW/WAR hazards between unrolled iterations
- Without renaming, iterations are serialized (no speedup)
- Increases code size but reduces loop overhead
- Best combined with VLIW packing after unrolling

### Vectorize Batch

```python
from tools.transforms.transforms import vectorize_batch, ScalarOp, analyze_vectorization_opportunity

# Define scalar operations
ops = [
    ScalarOp("alu", "+", dest=10, sources=[20, 30], cycle=0),
    ScalarOp("alu", "+", dest=11, sources=[21, 31], cycle=1),
    ScalarOp("alu", "+", dest=12, sources=[22, 32], cycle=2),
    # ... up to VLEN (8) similar ops
]

# Vectorize
result = vectorize_batch(ops, batch_size=8)

# Or analyze existing code for opportunities
analysis = analyze_vectorization_opportunity(instructions)
print(f"Potential savings: {analysis['potential_savings']} cycles")
```

**Requirements for vectorization:**
- All ops must have same operation type
- Destinations should be consecutive (efficient vstore)
- Sources should have regular stride (efficient vload)

**Analysis output:**
```
{
    "scalar_alu_count": 234,
    "vectorizable_sequences": 15,
    "total_vectorizable_ops": 120,
    "potential_vector_ops": 15,
    "potential_savings": 105
}
```

### Software Pipelining

```python
from tools.transforms.transforms import (
    software_pipeline,
    generate_pipeline_schedule,
    analyze_pipeline_opportunity,
    PipelineStage
)

# Analyze loop for pipelining potential
analysis = analyze_pipeline_opportunity(loop_body)
print(f"Overlap opportunity: {analysis['overlap_opportunity']}")
print(f"Recommended overlap: {analysis['recommended_overlap']} cycles")

# Define pipeline stages
stages = [
    PipelineStage(
        name="setup",
        instructions=[...],
        cycles=4,
        produces={10, 11},  # scratch addresses produced
        consumes={0}        # scratch addresses consumed
    ),
    PipelineStage(
        name="hash",
        instructions=[...],
        cycles=16,
        produces={20},
        consumes={10}
    ),
    PipelineStage(
        name="finish",
        instructions=[...],
        cycles=6,
        produces={30},
        consumes={20}
    ),
]

# Generate schedule visualization
schedule = generate_pipeline_schedule(stages, iterations=3)
print(schedule['schedule_visualization'])

# Apply pipelining
result = software_pipeline(
    prologue=setup_code,
    body=hash_loop,
    epilogue=finish_code,
    overlap_cycles=12  # How much N+1 overlaps with N
)
```

**Key insight for hash function:**
During 16 hash cycles, LOAD/STORE engines are mostly free. Use them for:
- Loading next batch indices
- Computing next batch addresses
- Prefetching tree nodes

### Hoist Loop Invariants

```python
from tools.transforms.transforms import hoist_invariants, identify_invariants

# Identify invariants automatically
invariants = identify_invariants(loop_body, entry_values={0, 1, 2})

# Hoist them
result = hoist_invariants(
    loop_body=loop_body,
    iterations=16,
    known_invariants=invariants
)

print(f"Hoisted {result.metadata['hoisted_count']} instructions")
print(f"Saved {result.cycles_saved} cycles")
```

**What gets hoisted:**
- `const` loads (always invariant)
- `vbroadcast` of constants
- Address computations using only invariants
- Any instruction that only reads invariant values

## Hash-Specific Helpers

### Generate Vectorized Hash Stage

```python
from tools.transforms.transforms import generate_vectorized_hash_stage

# Generate optimized code for hash stage 0
instrs = generate_vectorized_hash_stage(
    stage_idx=0,
    val_base=100,     # Vector of values being hashed
    tmp1_base=200,    # Temp vector 1
    tmp2_base=300,    # Temp vector 2
    const1_vec=400,   # Pre-broadcast constant 1
    const3_vec=500    # Pre-broadcast constant 3
)

# Output: 2 cycles using 2-way ILP (tmp1 || tmp2)
# Cycle 0: valu[(op1, tmp1, val, const1), (op3, tmp2, val, const3)]
# Cycle 1: valu[(op2, val, tmp1, tmp2)]
```

**Hash stage structure:**
Each of 6 stages has 3 ops: op1, op2, op3
- op1 and op3 are independent (compute tmp1 and tmp2)
- op2 depends on both (combines tmp1 and tmp2)

2-way ILP within each stage = 2 cycles/stage = 12 cycles total for hash.

### Generate Unroll Code Hint

```python
from tools.transforms.transforms import generate_unroll_code_hint

hint = generate_unroll_code_hint(
    op_template="valu: ('+', dest, src1, src2)",
    unroll_factor=4,
    vector_base=100,
    scalar_inputs=[0, 1, 2]
)
print(hint)
```

## Complete Workflow Example

```python
from tools.transforms.transforms import (
    analyze_transform_opportunities,
    unroll_loop,
    hoist_invariants,
    identify_invariants
)
from tools.vliw_packer.vliw_packer import pack_kernel

# 1. Analyze opportunities
analysis = analyze_transform_opportunities(instructions)
print(f"Primary bottleneck: {analysis['summary']['primary_bottleneck']}")

# 2. Apply transformations
# Step 2a: Hoist invariants first
invariants = identify_invariants(loop_body, entry_values={...})
hoist_result = hoist_invariants(loop_body, iterations=16, invariants)

# Step 2b: Unroll remaining body
unroll_result = unroll_loop(hoist_result.transformed_code, unroll_factor=4)

# 3. Pack the result
packed, stats = pack_kernel(unroll_result.transformed_code)
print(f"Final speedup: {stats.speedup:.2f}x")
```

## TransformResult Object

All transformations return a `TransformResult`:

```python
@dataclass
class TransformResult:
    success: bool              # Did transformation succeed?
    original_cycles: int       # Cycles before transformation
    transformed_cycles: int    # Cycles after transformation
    cycles_saved: int          # Difference
    speedup: float             # original / transformed
    description: str           # Human-readable description
    warnings: List[str]        # Any warnings generated
    transformed_code: List[dict]  # The transformed instructions
    metadata: Dict[str, Any]   # Additional info
```

## Best Practices

### Order of Transformations

1. **Hoist invariants first** - Reduces work for other transforms
2. **Unroll loops** - Exposes more ILP
3. **Apply software pipelining** - Overlap iterations
4. **Pack with VLIW packer** - Maximize slot utilization

### Register Allocation

When unrolling or pipelining:
- Use separate register sets for overlapped iterations
- Plan scratch space allocation carefully
- Check for scratch overflow with constraint_validator

### Validation

After transformations:
```bash
# Validate constraints
python tools/constraint_validator/constraint_validator.py

# Check slot utilization
python tools/slot_analyzer.py --packing

# Verify correctness
python tests/submission_tests.py
```

## Output Formats

### Human-Readable (default)

```
TRANSFORMATION RESULT
=====================
Success:             True
Description:         Unrolled loop 4x with register renaming
Original cycles:     100
Transformed cycles:  100
Cycles saved:        0
Speedup:             1.00x
```

### JSON (--json flag)

```json
{
    "success": true,
    "original_cycles": 100,
    "transformed_cycles": 100,
    "cycles_saved": 0,
    "speedup": 1.0,
    "description": "Unrolled loop 4x with register renaming",
    "warnings": [],
    "metadata": {
        "unroll_factor": 4,
        "body_cycles": 25,
        "register_offset": "auto"
    }
}
```

## Limitations

- **Not a compiler**: Manual integration required
- **Heuristic analysis**: May miss some opportunities
- **Register allocation**: User must manage scratch space
- **No correctness verification**: Use tests/submission_tests.py

## See Also

- `tools/vliw_packer/` - Pack instructions after transforming
- `tools/slot_analyzer.py` - Analyze utilization
- `tools/hash_pipeline/` - Detailed hash analysis
- `tools/constraint_validator/` - Validate constraints
