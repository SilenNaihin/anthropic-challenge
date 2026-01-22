# Anthropic Performance Engineering Challenge

## Goal

Optimize the kernel in `KernelBuilder.build_kernel` (`perf_takehome.py`) to minimize clock cycles on a simulated VLIW SIMD architecture.

---

## Target Architecture

Custom **VLIW (Very Large Instruction Word) SIMD** processor:

### Engine Slot Limits (operations per cycle)

| Engine | Slots | Description |
|--------|-------|-------------|
| `alu`  | 12    | Scalar ALU operations |
| `valu` | 6     | Vector ALU operations (VLEN=8 elements each) |
| `load` | 2     | Memory loads |
| `store`| 2     | Memory stores |
| `flow` | 1     | Control flow operation |

### Key Properties

- All effects apply at **END of cycle** (parallel reads before writes)
- Vector length `VLEN = 8`
- Scratch space = 1536 words (like registers + cache)
- 32-bit words throughout

---

## The Problem

**Parallel tree traversal kernel:**

- **Input**: batch_size=256 indices/values, rounds=16, tree height=10
- **Each iteration**:
  1. Load node value from tree at current index
  2. `val = myhash(val ^ node_val)` (6-stage hash function)
  3. `idx = 2*idx + (1 if val%2==0 else 2)` (branch left/right)
  4. Wrap to root if past tree bottom
- **Output**: Store updated indices and values

---

## Performance Benchmarks

| Cycles | Achievement |
|--------|-------------|
| **147,734** | Baseline (current naive starter code) |
| 18,532 | 2-hour version starting point (7.97x faster) |
| 2,164 | Claude Opus 4 after many hours |
| 1,790 | Claude Opus 4.5 casual session |
| **1,487** | Claude Opus 4.5 after 11.5 hours **(BEAT THIS!)** |
| ??? | Best human (undisclosed, substantially better) |

---

## Optimization Opportunities

The starter code is **maximally naive**:

1. **NO VLIW parallelism** - only 1 op per instruction bundle
2. **NO vector operations** - all scalar despite valu/vload/vstore existing
3. **NO instruction scheduling** - dependent ops back-to-back
4. **NO memory access optimization** - scattered loads

### Key Techniques to Explore

- Pack independent operations into same cycle (use all 12 ALU slots)
- Vectorize batch processing (process 8 elements per valu op)
- Pipeline/interleave independent computations
- Optimize memory layout and access patterns
- Unroll and software pipeline loops

---

## Validation

```bash
# Verify tests folder unchanged
git diff origin/main tests/

# Run submission tests
python tests/submission_tests.py
```

---

## Reward

Beat **1,487 cycles** → Email `performance-recruiting@anthropic.com` with code + resume
