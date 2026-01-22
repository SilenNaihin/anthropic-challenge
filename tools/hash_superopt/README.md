# Hash Stage Superoptimizer

A tool that exhaustively enumerates ALL legal schedules for hash operations to find THE provably optimal schedule. Unlike heuristic-based approaches, this guarantees minimum cycle counts through exhaustive search.

## Why Superoptimize?

The hash function is the hottest code path in the VLIW SIMD challenge:
- **4096 calls** (256 batch x 16 rounds)
- Each hash has 6 stages, each with 3 operations
- Total: 18 operations per hash, executed 4096 times

For small operation counts like this, exhaustive enumeration is feasible and removes all guesswork about what's achievable.

## Key Insights

### Single Stage Structure
Each hash stage computes:
```
a = (a OP1 const1) OP2 (a OP3 shift_amount)
```

This breaks down into 3 operations:
- **tmp1**: `a OP1 const1` (e.g., `a + 0x7ED55D16`)
- **tmp2**: `a OP3 shift_amount` (e.g., `a << 12`)
- **combine**: `tmp1 OP2 tmp2` (e.g., `tmp1 + tmp2`)

**Critical dependencies**:
- `tmp1` and `tmp2` are **independent** (both only read `a`)
- `combine` depends on **both** `tmp1` and `tmp2`

### Single Stage Minimum
- **2 cycles** per stage (tmp1||tmp2 in parallel, then combine)
- **12 cycles** for full 6-stage hash
- Only **25% VALU utilization** (using 2-3 of 6 slots)

### Multi-Batch Pipelining
The key optimization is interleaving multiple independent hash computations:

| Batches | Total Cycles | Cycles/Hash | Speedup | VALU Util |
|---------|-------------|-------------|---------|-----------|
| 1       | 12          | 12.00       | 1.00x   | 25.0%     |
| 2       | 12          | 6.00        | 2.00x   | 50.0%     |
| 3       | 12          | 4.00        | 3.00x   | 75.0%     |
| 4       | 13          | 3.25        | 3.69x   | 92.3%     |

**Sweet spot: 3 batches** gives perfect VALU packing for tmp1+tmp2 cycles.

## Usage

### Basic Analysis
```bash
# Single stage (3 ops) - demonstrates the concept
python hash_superopt.py

# Full 6-stage hash (18 ops)
python hash_superopt.py --stages 6

# Multi-batch pipelining (2 hashes interleaved)
python hash_superopt.py --stages 6 --batches 2

# Optimal configuration (3 batches, 75% utilization)
python hash_superopt.py --stages 6 --batches 3

# Analyze batch scaling
python hash_superopt.py --stages 6 --scale
```

### Output Options
```bash
# Emit VLIW instruction sequence
python hash_superopt.py --stages 6 --emit

# JSON output
python hash_superopt.py --stages 6 --json

# Show all Pareto-optimal schedules
python hash_superopt.py --pareto

# Plain text (no colors)
python hash_superopt.py --no-color
```

## Example Output

```
Multi-Batch Hash Superoptimizer Results
Stages: 6 | Batches: 3 | Operations: 54

Optimization Summary:
  Total cycles:           12
  Cycles per hash:        4.00
  Single hash baseline:   12
  Speedup from pipelining: 3.00x

Average VALU Utilization: 75.0%

Optimal Schedule (cycle-by-cycle):
  Cycle  0: B0:[S0.tmp1, S0.tmp2] | B1:[S0.tmp1, S0.tmp2] | B2:[S0.tmp1, S0.tmp2] (6/6 VALU)
  Cycle  1: B0:[S0.combine] | B1:[S0.combine] | B2:[S0.combine] (3/6 VALU)
  ...

This schedule is PROVABLY OPTIMAL for 3 batches.
```

## Implementation Notes

### Architecture Constraints
- **6 VALU slots per cycle** (for vector operations)
- **VLEN=8** (each vector op processes 8 elements)
- Effects apply at **end of cycle** (parallel reads before writes)

### Search Algorithm
1. **Single stage**: Direct enumeration (only 3 valid schedules)
2. **Multi-stage**: Branch and bound with pruning:
   - Critical path lower bound
   - Throughput lower bound
   - Greedy packing for efficiency

### Optimality Guarantee
The tool explores all valid schedules respecting:
- Data dependencies (combine after tmp1 and tmp2)
- Slot limits (max 6 VALU per cycle)
- Cycle semantics (writes apply at end of cycle)

Any schedule it reports is **provably optimal** for the given configuration.

## Practical Application

### For Full Kernel Optimization
With 4096 hash calls and 3-batch pipelining:
- **Hash cycles**: 4096 / 3 * 12 = 16,384 cycles (vs 49,152 sequential)
- **3x reduction** in hash compute time

### Integration Strategy
1. Process elements in groups of 24 (3 batches x 8 VLEN)
2. Pre-broadcast all constants before the hash loop
3. Follow the optimal schedule exactly

### Remaining Overhead (not in this model)
- vbroadcast for constants (can be hoisted)
- Memory loads/stores (2 slots/cycle, can overlap)
- Loop control (minimal with unrolling)
- XOR with node value before hash
- Index calculation after hash

## Files
- `hash_superopt.py` - Main superoptimizer
- `README.md` - This documentation
- `quickstart.md` - Quick reference

## See Also
- `../hash_pipeline/` - ILP analysis (theoretical bounds)
- `../transforms/` - Transformation library for implementation
- `../slot_analyzer.py` - Verify actual instruction streams
