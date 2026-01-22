# Code Emitter - Quick Start

## TL;DR

**Generate copy-pasteable kernel code instead of writing it manually.**

```bash
# Demo all features
python tools/code_emitter/code_emitter.py --demo

# Generate dual-batch hash (most common)
python tools/code_emitter/code_emitter.py --emit-dual-hash

# With validation
python tools/code_emitter/code_emitter.py --emit-dual-hash --validate
```

## Common Tasks

| Task | Command |
|------|---------|
| Demo | `python tools/code_emitter/code_emitter.py --demo` |
| Single hash | `--emit-hash` |
| Dual hash (4 VALU) | `--emit-dual-hash` |
| Triple hash (6 VALU) | `--emit-triple-hash` |
| JSON output | `--json` |
| Validate | `--validate` |

## Python API (30 seconds)

```python
from tools.code_emitter.code_emitter import CodeEmitter

emitter = CodeEmitter(base_scratch_addr=100)
const_vectors = [(200 + i*16, 208 + i*16) for i in range(6)]

# Emit dual-batch hash
result = emitter.emit_dual_hash(
    val_a=100, tmp1_a=108, tmp2_a=116,
    val_b=124, tmp1_b=132, tmp2_b=140,
    const_vectors=const_vectors
)

# Use the code
print(result.python_code)      # Copy-paste this
instructions = result.instruction_list  # Or use directly
print(f"Cycles: {result.cycle_count}")
```

## Output

- `python_code` - Copy-paste into perf_takehome.py
- `instruction_list` - Direct instruction list for testing
- `cycle_count` - Total cycles
- `validation_passed` - Whether code passes constraint checks

## See Also

- Full docs: `README.md`
- Transforms library: `tools/transforms/transforms.py`
- Constraint validator: `tools/constraint_validator/`
