# DSL Compiler - Quick Start

## TL;DR

Write algorithms in Python-like DSL, compile to optimized VLIW instructions.

## Usage

```bash
# Compile DSL file
python tools/dsl_compiler/dsl_compiler.py myfile.dsl

# Demo with example
python tools/dsl_compiler/dsl_compiler.py --demo

# JSON output
python tools/dsl_compiler/dsl_compiler.py myfile.dsl --json
```

## DSL Cheat Sheet

```python
# Variables
var x              # Scalar
vec v[8]           # Vector (VLEN=8)

# Arithmetic (works on scalars and vectors)
y = x + 1
z = a ^ b
w = c << 12

# Memory
val = load(addr)
store(addr, val)
v = vload(base)    # Vector load
vstore(base, v)    # Vector store
v = broadcast(x)   # Scalar -> vector

# Hash (expands to 6-stage implementation)
h = hash(input)

# Loops (unrolled at compile time)
for i in range(8):
    process(i)

# Vectorize hint
@vectorize(8)
def process_batch(offset):
    vec v = vload(base + offset)
    v = hash(v)
    vstore(base + offset, v)
```

## From Python

```python
from tools.dsl_compiler.dsl_compiler import DSLCompiler

compiler = DSLCompiler()
result = compiler.compile(dsl_source)

if result.success:
    instructions = result.instructions  # VLIW instruction stream
    scratch_map = result.scratch_map    # Variable -> address mapping
```

## Key Features

| Feature | Description |
|---------|-------------|
| Auto-vectorize | Scalar ops -> vector ops (VLEN=8) |
| Hash expansion | `hash(x)` -> 6-stage optimal implementation |
| VLIW packing | Independent ops run in parallel |
| Dependency tracking | Correct scheduling order |

## Example

```python
# my_algorithm.dsl
@vectorize(8)
def process(offset):
    vec v_data = vload(data_p + offset)
    vec v_result = hash(v_data ^ broadcast(key))
    vstore(output_p + offset, v_result)
```

Compiles to ~20 VLIW cycles with automatic:
- 6 hash stages expanded
- Dependencies resolved
- Parallel operations packed

## Output

```
Success:        True
Cycles:         234
Scratch used:   156

Scratch Allocation:
   0: v_data (8 words)
   8: v_result (8 words)
  16: key (1 word)
```

## See Also

- Full docs: `tools/dsl_compiler/README.md`
- Example: `tools/dsl_compiler/hash_tree.dsl`
