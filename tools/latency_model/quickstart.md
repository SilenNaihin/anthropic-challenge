# Latency Model - Quick Start

## TL;DR

**ALL operations take exactly 1 cycle.** No multi-cycle latencies. No stalls.

## Quick Commands

```bash
# Full analysis
python tools/latency_model/latency_model.py

# JSON output
python tools/latency_model/latency_model.py --json

# Plain text
python tools/latency_model/latency_model.py --no-color
```

## Key Facts

| Engine | Slots | Latency | Notes |
|--------|-------|---------|-------|
| alu    | 12    | 1 cycle | Scalar math |
| valu   | 6     | 1 cycle | Vector math (8 elements) |
| load   | 2     | 1 cycle | Memory read |
| store  | 2     | 1 cycle | Memory write |
| flow   | 1     | 1 cycle | Control flow |

## Critical Semantics

```
Same cycle:   write r1, then read r1 -> gets OLD value
Next cycle:   write r1, then read r1 -> gets NEW value
```

## What This Means

1. **Latency is NOT your problem** - all ops are 1 cycle
2. **Dependencies ARE your problem** - RAW hazards force serialization
3. **Throughput limits matter** - load/store = 2/cycle, flow = 1/cycle
4. **Pack instructions** - most cycles waste 90%+ of slots

## Bottom Line

Focus on:
- Breaking dependency chains (unroll, pipeline, vectorize)
- Filling unused slots (VLIW packing)
- Avoiding flow operations (only 1/cycle)

NOT on:
- Latency hiding
- Out-of-order scheduling
- Speculation
