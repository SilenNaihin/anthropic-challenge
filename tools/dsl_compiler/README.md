# DSL Compiler for VLIW SIMD Architecture

A high-level domain-specific language (DSL) compiler that transforms algorithm descriptions into optimized VLIW instruction streams.

## Overview

Instead of manually writing low-level VLIW instructions, you can express your algorithm in a Python-like DSL and let the compiler handle:

- **Automatic vectorization**: Scalar ops become vector ops (VLEN=8)
- **Hash expansion**: High-level `hash(x)` expands to optimal 6-stage implementation
- **Dependency analysis**: Tracks data dependencies for correct scheduling
- **VLIW packing**: Schedules independent operations in parallel
- **Memory allocation**: Automatically allocates scratch space

## Installation

No additional dependencies required. Rich is optional for colored output:

```bash
pip install rich  # Optional
```

## Quick Start

```bash
# Compile a DSL file
python tools/dsl_compiler/dsl_compiler.py my_algorithm.dsl

# Run demo with example DSL
python tools/dsl_compiler/dsl_compiler.py --demo

# Output JSON
python tools/dsl_compiler/dsl_compiler.py my_algorithm.dsl --json
```

## DSL Syntax

### Variables

```python
# Scalar variable
var x

# Vector variable (VLEN=8 elements)
vec v_data

# Vector with custom size
vec v_big[16]
```

### Arithmetic Operations

```python
# All standard operators work on scalars and vectors
result = a + b      # Add
result = a - b      # Subtract
result = a * b      # Multiply
result = a // b     # Integer divide
result = a % b      # Modulo
result = a ^ b      # XOR
result = a & b      # AND
result = a | b      # OR
result = a << b     # Left shift
result = a >> b     # Right shift
result = a < b      # Less than (returns 0 or 1)
result = a == b     # Equal (returns 0 or 1)
```

### Memory Operations

```python
# Scalar load/store
val = load(addr)
store(addr, val)

# Vector load/store (contiguous memory)
v_data = vload(base_addr)
vstore(base_addr, v_data)

# Broadcast scalar to vector
v_const = broadcast(scalar_val)
```

### Hash Function

```python
# High-level hash operation (expands to 6-stage implementation)
hashed = hash(input_value)
```

### Control Flow

```python
# For loops (compile-time unrolled)
for i in range(8):
    process(i)

# Range with start, end
for i in range(0, 16):
    work(i)

# Range with start, end, step
for i in range(0, 32, 8):
    batch_process(i)

# Conditionals
if condition:
    do_something()
else:
    do_other()
```

### Functions and Decorators

```python
# Simple function
def process_element(idx):
    val = load(data_p + idx)
    result = val * 2
    store(data_p + idx, result)

# Vectorized function (hint to compiler)
@vectorize(8)
def process_batch(offset):
    vec v_data
    v_data = vload(base + offset)
    v_result = v_data * 2
    vstore(base + offset, v_result)
```

### Comments

```python
# Single-line comments with hash
x = 1  # End-of-line comment
```

## Example: Hash+Tree Algorithm

```python
# hash_tree.dsl
# Express the core algorithm from the optimization challenge

@vectorize(8)
def process_batch(batch_offset):
    vec v_idx
    vec v_val
    vec v_node_val

    # Load batch data
    v_idx = vload(indices_p + batch_offset)
    v_val = vload(values_p + batch_offset)

    # Compute tree addresses
    v_addr = forest_p + v_idx

    # Scattered loads (handled specially by compiler)
    for lane in range(8):
        v_node_val[lane] = load(v_addr[lane])

    # XOR and hash (fully vectorizable)
    v_mixed = v_val ^ v_node_val
    v_hashed = hash(v_mixed)

    # Index computation
    v_parity = v_hashed & broadcast(1)
    v_new_idx = v_idx * broadcast(2) + v_parity + broadcast(1)

    # Bounds check using multiply trick
    v_in_bounds = v_new_idx < broadcast(n_nodes)
    v_final = v_new_idx * v_in_bounds

    # Store results
    vstore(values_p + batch_offset, v_hashed)
    vstore(indices_p + batch_offset, v_final)
```

## Using from Python

```python
from tools.dsl_compiler.dsl_compiler import DSLCompiler

# Compile DSL source
compiler = DSLCompiler()
result = compiler.compile(dsl_source)

if result.success:
    print(f"Compiled to {result.cycles} cycles")
    print(f"Scratch used: {result.scratch_used}")

    # Get instructions for KernelBuilder
    instructions = result.instructions
    scratch_map = result.scratch_map

# Or use convenience method
instructions, scratch_map = compiler.compile_to_kernel_builder(dsl_source)
```

## Compilation Pipeline

1. **Lexing**: Tokenize source into tokens (numbers, identifiers, operators)
2. **Parsing**: Build Abstract Syntax Tree (AST) from tokens
3. **IR Building**: Convert AST to Intermediate Representation (IR) DAG
4. **Hash Expansion**: Expand `hash(x)` to 6-stage implementation
5. **Vectorization**: Convert scalar ops to vector ops where applicable
6. **Scratch Allocation**: Assign scratch addresses to all values
7. **Scheduling**: Schedule IR nodes into VLIW instruction bundles
8. **Output**: Emit KernelBuilder-compatible instruction stream

## Intermediate Representation (IR)

The IR is a DAG of operations:

```python
@dataclass
class IRNode:
    id: int             # Unique identifier
    op: OpType          # Operation (ADD, MUL, HASH, etc.)
    dest: IRValue       # Destination value
    sources: List[IRValue]  # Source operands
    scheduled_cycle: int    # Assigned cycle (-1 if not scheduled)
    engine: str         # Assigned engine (alu, valu, load, store, flow)
```

Operations types:
- **Arithmetic**: ADD, SUB, MUL, DIV, MOD, XOR, AND, OR, SHL, SHR, LT, EQ
- **Memory**: LOAD, STORE, VLOAD, VSTORE, CONST
- **Vector**: VBROADCAST, MULTIPLY_ADD, VSELECT
- **High-level**: HASH (expands during compilation)

## Scheduling Algorithm

The compiler uses list scheduling with dependency tracking:

1. Build dependency graph (RAW hazards)
2. Find initially ready nodes (no dependencies)
3. For each cycle:
   - Pack as many ready nodes as possible respecting slot limits
   - Emit instruction bundle
   - Update ready list with newly unblocked nodes
4. Continue until all nodes scheduled

Slot limits respected:
- ALU: 12 slots
- VALU: 6 slots
- LOAD: 2 slots
- STORE: 2 slots
- FLOW: 1 slot

## Output Formats

### Terminal Output (Rich)

```
+==============================+
|  DSL COMPILATION RESULT      |
+==============================+

| Metric       | Value   |
|--------------|---------|
| Success      | True    |
| Cycles       | 234     |
| Scratch used | 156     |

Scratch Allocation:
| Address | Name     |
|---------|----------|
|       0 | idx      |
|       1 | val      |
|       8 | v_data   |

Instructions:
   0: {'load': [('const', 0, 0)]}
   1: {'alu': [('+', 1, 0, 2)]}
   ...
```

### JSON Output

```json
{
  "success": true,
  "cycles": 234,
  "scratch_used": 156,
  "errors": [],
  "warnings": [],
  "stats": {
    "ir_nodes": 89,
    "values": 23,
    "constants": 12
  }
}
```

## Integration with KernelBuilder

```python
from perf_takehome import KernelBuilder
from tools.dsl_compiler.dsl_compiler import DSLCompiler

class MyKernel(KernelBuilder):
    def build_kernel(self, ...):
        # Compile DSL
        compiler = DSLCompiler()
        compiler.scratch_ptr = self.scratch_ptr  # Share scratch allocator

        result = compiler.compile(my_dsl_source)

        # Merge instructions
        self.instrs.extend(result.instructions)

        # Update scratch pointer
        self.scratch_ptr = compiler.scratch_ptr
```

## Limitations

Current limitations (future improvements):

1. **No backward jumps**: Loops must be compile-time unrollable
2. **Simple scheduling**: List scheduling, not optimal ILP extraction
3. **Limited vector scatter**: Scattered loads require manual handling
4. **No register renaming**: May create false dependencies
5. **Basic type inference**: Explicit type hints recommended

## Files

- `dsl_compiler.py` - Main compiler implementation
- `hash_tree.dsl` - Example DSL for the hash+tree algorithm
- `README.md` - This documentation
- `quickstart.md` - Quick reference

## Future Work

- [ ] Software pipelining support
- [ ] Better ILP scheduling (critical path heuristics)
- [ ] Register allocation with spilling
- [ ] Loop-invariant code motion
- [ ] Common subexpression elimination
- [ ] Gather/scatter instruction generation
- [ ] Peephole optimizations

## See Also

- `tools/transforms/` - Manual transformation helpers
- `tools/vliw_packer/` - Auto-packing for existing code
- `tools/slot_analyzer.py` - Analyze slot utilization
- `tools/dependency_graph/` - Dependency analysis
