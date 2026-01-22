# Hash Pipeline Analyzer

Analyzes the 6-stage hash function dependency structure and generates optimal pipeline schedules for maximum throughput on the VLIW SIMD architecture.

## Why This Tool?

The hash function runs **4096 times** per kernel execution (256 batch x 16 rounds). It is THE hottest code path. Even small improvements here have massive impact on total cycle count.

## Key Insights

### 1. Intra-Stage Parallelism
Each hash stage computes: `a = (a op1 const) op2 (a op3 shift)`

This can be broken down as:
```
tmp1 = a op1 const    # Independent!
tmp2 = a op3 shift    # Independent!
a = tmp1 op2 tmp2     # Depends on both
```

**tmp1 and tmp2 are independent** - they can execute in parallel!

### 2. Inter-Element Parallelism
Stage N for element B doesn't depend on stage N for element A. This enables **software pipelining** where multiple elements progress through the pipeline simultaneously.

### 3. Architecture Limits
- 6 VALU slots per cycle (vectorized)
- 12 ALU slots per cycle (scalar)
- VLEN = 8 elements per vector operation

## Usage

```bash
# Full analysis with defaults (8 elements)
python tools/hash_pipeline/hash_pipeline.py

# Analyze specific number of elements
python tools/hash_pipeline/hash_pipeline.py --elements 16

# Show cycle-by-cycle schedule visualization
python tools/hash_pipeline/hash_pipeline.py --visualize

# Get VLIW code generation hints
python tools/hash_pipeline/hash_pipeline.py --codegen

# JSON output for scripting
python tools/hash_pipeline/hash_pipeline.py --json

# Compare different batch sizes
python tools/hash_pipeline/hash_pipeline.py --batch-analysis

# Use scalar ALU instead of vector VALU
python tools/hash_pipeline/hash_pipeline.py --scalar
```

## Output Explained

### Stage Analysis
Shows each hash stage's formula and identifies parallelism:
```
Stage 0:
  Formula: a = (a + 0x7ED55D16) + (a << 12)
  tmp1: a + const     (1 ALU op)
  tmp2: a << 12       (1 ALU op)
  combine: tmp1 + tmp2 (1 ALU op)
  Parallelism: tmp1 || tmp2  (2-way ILP within stage)
```

### Scheduling Comparison
Compares different scheduling strategies:
```
Strategy                                Cycles    Speedup
------------------------------------------------------------
Sequential (baseline)                      144       1.00x
Intra-stage parallel (tmp1||tmp2)           96       1.50x
Software pipelined                          24       6.00x
Maximally pipelined                         24       6.00x
------------------------------------------------------------
Theoretical minimum                         24
```

### Batch Size Analysis
Shows optimal batch sizes for pipelining:
```
  Elements     Max Pipe    Cycles/Elem   Efficiency
----------------------------------------------------
         8           24           3.00       100.0%
        16           48           3.00       100.0%
```

## Theoretical Analysis

For N elements through 6 stages:
- **Total operations**: N x 6 stages x 3 ops = 18N ops
- **Critical path (1 element)**: 12 cycles (6 stages x 2 cycles each)
- **Throughput limit**: ceil(18N / 6 VALU) = 3N cycles
- **Theoretical minimum**: max(critical_path, throughput_limit)

## Code Generation Hints

The `--codegen` flag provides:
1. **Constant preloading** - All 6 stage constants
2. **Register allocation** - Suggested vector register layout
3. **Loop structure** - Optimal per-stage instruction sequence
4. **Software pipelining** - How to overlap multiple batches
5. **Advanced optimizations** - LUTs, unrolling, etc.

## Integration with Other Tools

```bash
# Use slot_analyzer to verify your implementation
python tools/slot_analyzer.py --packing --deps

# Full optimization workflow:
# 1. Analyze hash pipeline
python tools/hash_pipeline/hash_pipeline.py --codegen > hash_plan.txt
# 2. Implement optimized hash
# 3. Verify with slot_analyzer
python tools/slot_analyzer.py --recommendations
```

## Hash Stage Reference

| Stage | op1 | const1     | op2 | op3 | shift |
|-------|-----|------------|-----|-----|-------|
| 0     | +   | 0x7ED55D16 | +   | <<  | 12    |
| 1     | ^   | 0xC761C23C | ^   | >>  | 19    |
| 2     | +   | 0x165667B1 | +   | <<  | 5     |
| 3     | +   | 0xD3A2646C | ^   | <<  | 9     |
| 4     | +   | 0xFD7046C5 | +   | <<  | 3     |
| 5     | ^   | 0xB55A4F09 | ^   | >>  | 16    |

## Advanced Optimization Ideas

### 1. Lookup Tables
For some hash patterns, precomputed LUTs might be faster than computation, especially if memory bandwidth allows.

### 2. Algebraic Simplification
Some stages might be combinable or simplifiable based on algebraic properties.

### 3. SIMD-Friendly Restructuring
Consider restructuring the hash to be more SIMD-friendly (e.g., processing 8 values through one stage before moving to next).

### 4. Multi-Batch Pipelining
Process 2+ batches of 8 elements simultaneously, hiding latencies between stages.

## Files

- `hash_pipeline.py` - Main analyzer tool
- `README.md` - This documentation
- `quickstart.md` - Quick reference (10-15 lines)
