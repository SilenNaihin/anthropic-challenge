# Code Emitter for VLIW SIMD Optimization

**Transform analysis into action.** The Code Emitter takes transformation specifications and generates actual Python code compatible with `KernelBuilder` that can be copy-pasted directly into `perf_takehome.py`.

## Why This Tool Matters

Other analysis tools tell you WHAT to do:
- "Unroll this loop 4x"
- "Vectorize these scalar ops"
- "Pipeline hash with prep"

This tool actually DOES it:
- Generates working Python code
- Handles register allocation automatically
- Validates output with constraint_validator
- Outputs instruction lists ready for testing

**Result:** Dramatically speeds up iteration by eliminating manual translation errors.

## Features

| Feature | Description |
|---------|-------------|
| Unrolled Loops | Generate N-way unrolled loops with automatic register renaming |
| Vectorized Hash | Single, dual, and triple-batch hash function code |
| Pipelined Stages | Interleaved hash + prep code generation |
| Batch Setup/Finish | Complete setup and teardown code |
| Register Allocation | Automatic scratch address management |
| Validation | Constraint checking before runtime |
| Rich Output | Colorful terminal display (with fallback) |
| JSON Output | Machine-readable format for tooling |

## Installation

No additional dependencies required. Rich is optional for colored output:

```bash
pip install rich  # Optional, for colored output
```

## Usage

### Command Line

```bash
# Run demonstration
python tools/code_emitter/code_emitter.py --demo

# Emit vectorized hash (single batch)
python tools/code_emitter/code_emitter.py --emit-hash

# Emit dual-batch hash (default)
python tools/code_emitter/code_emitter.py --emit-dual-hash

# Emit triple-batch hash (max VALU utilization)
python tools/code_emitter/code_emitter.py --emit-triple-hash

# With validation
python tools/code_emitter/code_emitter.py --emit-dual-hash --validate

# JSON output (for tooling integration)
python tools/code_emitter/code_emitter.py --emit-hash --json

# Plain text output (no colors)
python tools/code_emitter/code_emitter.py --emit-hash --no-color
```

### Python API

```python
from tools.code_emitter.code_emitter import CodeEmitter, EmittedCode

# Create emitter with base scratch address
emitter = CodeEmitter(base_scratch_addr=100)

# Emit vectorized hash for single batch
const_vectors = [(200 + i*16, 208 + i*16) for i in range(6)]
result = emitter.emit_vectorized_hash(
    val_base=100,
    tmp1_base=108,
    tmp2_base=116,
    const_vectors=const_vectors
)

print(f"Cycles: {result.cycle_count}")
print(f"Code:\n{result.python_code}")

# Get instruction list for direct use
instructions = result.instruction_list

# Emit dual-batch hash (4 VALU slots per cycle)
result = emitter.emit_dual_hash(
    val_a=100, tmp1_a=108, tmp2_a=116,
    val_b=124, tmp1_b=132, tmp2_b=140,
    const_vectors=const_vectors
)

# Emit triple-batch hash (all 6 VALU slots)
result = emitter.emit_triple_hash(
    val_a=100, tmp1_a=108, tmp2_a=116,
    val_b=124, tmp1_b=132, tmp2_b=140,
    val_c=148, tmp1_c=156, tmp2_c=164,
    const_vectors=const_vectors
)

# Validate generated code
result = emitter.validate_code(result)
if result.validation_passed:
    print("Code is valid!")
else:
    print(f"Errors: {result.validation_errors}")
```

## Transformation Types

### 1. Unrolled Loops

Generate N-way unrolled loops with automatic register renaming to avoid WAW/WAR hazards:

```python
body_template = [
    {"alu": [("+", 10, 20, 30)]},
    {"valu": [("*", 40, 50, 60)]},
    {"store": [("store", 70, 80)]},
]

result = emitter.emit_unrolled_loop(
    body_template,
    factor=4,
    register_prefix="u"
)

# Result: 12 cycles (3 * 4) with renamed registers
# Iteration 0: regs 10, 20, 30, 40, 50, 60, 70, 80
# Iteration 1: regs 160, 170, 180, ... (offset by 150)
# etc.
```

### 2. Vectorized Hash

The hash function has 6 stages, each with 3 operations that can leverage 2-way ILP:

```
Stage pattern: val = op2(op1(val, const1), op3(val, const3))
ILP: op1 || op3 (cycle 1), then op2 (cycle 2)
```

**Single batch (2 VALU slots):**
```python
result = emitter.emit_vectorized_hash(val, tmp1, tmp2, const_vectors)
# 12 cycles (6 stages * 2 cycles each)
```

**Dual batch (4 VALU slots):**
```python
result = emitter.emit_dual_hash(
    val_a, tmp1_a, tmp2_a,
    val_b, tmp1_b, tmp2_b,
    const_vectors
)
# 12 cycles for 2 batches = 6 cycles/batch equivalent
```

**Triple batch (6 VALU slots - maximum utilization):**
```python
result = emitter.emit_triple_hash(
    val_a, tmp1_a, tmp2_a,
    val_b, tmp1_b, tmp2_b,
    val_c, tmp1_c, tmp2_c,
    const_vectors
)
# 12 cycles for 3 batches = 4 cycles/batch equivalent
```

### 3. Pipelined Iterations

Overlap preparation of batch N+1 with hash computation of batch N:

```python
result = emitter.emit_pipelined_iteration(
    hash_regs_cur=current_batch_registers,
    prep_regs_next=next_batch_registers,
    const_vectors=hash_constants,
    inp_indices_p=..., inp_values_p=..., forest_p=...,
    offset_a=batch_offset_a, offset_b=batch_offset_b
)

# Pipeline schedule during 16 hash cycles:
# Cycle 0:  Hash stage 0 cycle A + Address computation (ALU)
# Cycle 1:  Hash stage 0 cycle B + (nothing)
# Cycle 2:  Hash stage 1 cycle A + Vector loads (LOAD)
# Cycle 3:  Hash stage 1 cycle B + Vector loads (LOAD)
# Cycle 4:  Hash stage 2 cycle A + Tree addresses (2 extra VALU)
# Cycle 5:  Hash stage 2 cycle B + Extract addresses (12 ALU)
# Cycles 6-14: Hash stages 3-5 + Scattered loads (LOAD)
```

### 4. Batch Setup/Finish

**Setup (non-pipelined, for first iteration):**
```python
result = emitter.emit_batch_setup(
    regs=batch_registers,
    inp_indices_p=..., inp_values_p=..., forest_p=...,
    offset_a=0, offset_b=8,
    zero_const=zero_addr
)
# ~15 cycles for address comp, vloads, scattered loads, XOR
```

**Finish (with optional overlapped next-batch loads):**
```python
result = emitter.emit_batch_finish(
    regs=batch_registers,
    v_one=..., v_two=..., v_zero=..., v_n_nodes=...,
    has_next=True,
    next_regs=next_batch_registers
)
# 5 cycles for index update, bounds check, stores
# Uses multiply_add for efficient bounds: idx = idx * in_bounds + 0
```

## Output Format

### EmittedCode Structure

```python
@dataclass
class EmittedCode:
    success: bool              # Whether emission succeeded
    python_code: str           # Copy-pasteable Python code
    instruction_list: List[dict]  # Direct instruction list
    cycle_count: int           # Total cycles
    scratch_usage: int         # Scratch addresses used
    description: str           # Human-readable description
    warnings: List[str]        # Any warnings
    validation_passed: bool    # Constraint validation result
    validation_errors: List[str]  # Validation errors if any
    metadata: Dict[str, Any]   # Additional info (ILP, batches, etc.)
```

### JSON Output

```json
{
  "success": true,
  "python_code": "# Dual-batch vectorized hash...",
  "cycle_count": 12,
  "scratch_usage": 48,
  "description": "Dual-batch hash: 6 stages, 12 cycles, 4 VALU slots/cycle",
  "warnings": [],
  "validation_passed": true,
  "validation_errors": [],
  "metadata": {
    "batches": 2,
    "stages": 6,
    "cycles_per_stage": 2,
    "max_valu_slots": 4
  },
  "instruction_list": [...]
}
```

## Integration with Other Tools

### With slot_analyzer

```bash
# Generate code, save to file
python tools/code_emitter/code_emitter.py --emit-dual-hash --json > emitted.json

# Analyze generated instructions
# (Would need to extract instruction_list and analyze)
```

### With constraint_validator

```python
from tools.code_emitter.code_emitter import CodeEmitter
from tools.constraint_validator.constraint_validator import validate_kernel

emitter = CodeEmitter()
result = emitter.emit_dual_hash(...)

# Built-in validation
result = emitter.validate_code(result)

# Or direct validation
validation = validate_kernel(result.instruction_list)
```

## Example Workflow

1. **Analyze** current bottleneck with `slot_analyzer`
2. **Identify** transformation opportunity
3. **Generate** code with `code_emitter`
4. **Validate** with built-in constraint checking
5. **Copy-paste** into `perf_takehome.py`
6. **Test** with `python tests/submission_tests.py`
7. **Iterate** as needed

```bash
# Step 1: Find bottleneck
python tools/slot_analyzer.py --recommendations

# Step 2: Generate optimized code
python tools/code_emitter/code_emitter.py --emit-triple-hash --validate

# Step 3: Copy generated code to perf_takehome.py
# (Manual step - copy from output)

# Step 4: Validate result
python tests/submission_tests.py
```

## Architecture Reference

| Engine | Slots/cycle | Common Ops |
|--------|-------------|------------|
| alu    | 12          | +, -, *, //, %, ^, &, \|, <<, >>, <, == |
| valu   | 6           | vbroadcast, multiply_add, same as alu |
| load   | 2           | load, vload, const, load_offset |
| store  | 2           | store, vstore |
| flow   | 1           | select, vselect, cond_jump, jump, halt, pause |

**Key**: Effects apply at END of cycle (parallel reads before writes)

## Tips for Best Results

1. **Use dual/triple hash** when you have enough registers - maximizes VALU utilization
2. **Pipeline aggressively** - 16 hash cycles can hide 10+ prep cycles
3. **Use multiply_add** for bounds check instead of vselect (VALU vs FLOW)
4. **Pre-broadcast constants** outside loops - saves cycles per iteration
5. **Validate before testing** - catch errors early

## Limitations

- Generated code may need manual tuning for specific register layouts
- Pipelined code assumes specific memory layout
- Some transformations require manual register assignment
- Not all edge cases handled (e.g., unaligned batches)
