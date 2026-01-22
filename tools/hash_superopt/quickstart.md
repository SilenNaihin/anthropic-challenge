# Hash Superoptimizer - Quick Start

## TL;DR
Exhaustively finds THE optimal schedule for hash operations. No guesswork.

## Key Results

| Config | Cycles/Hash | Speedup | Utilization |
|--------|-------------|---------|-------------|
| Single hash | 12 | 1x | 25% |
| 3 batches | 4 | **3x** | **75%** |

**Use 3 batches** - perfect packing, 3x speedup.

## Quick Commands

```bash
# Single hash (baseline)
python hash_superopt.py --stages 6

# Optimal: 3 batches pipelined
python hash_superopt.py --stages 6 --batches 3

# See scaling analysis
python hash_superopt.py --stages 6 --scale

# Get instruction sequence
python hash_superopt.py --stages 6 --emit

# JSON output
python hash_superopt.py --stages 6 --batches 3 --json
```

## Optimal Schedule Pattern

With 3 batches (B0, B1, B2):
```
Cycle 0:  [B0:tmp1,tmp2] [B1:tmp1,tmp2] [B2:tmp1,tmp2]  (6/6 VALU)
Cycle 1:  [B0:combine]   [B1:combine]   [B2:combine]    (3/6 VALU)
Cycle 2:  [B0:tmp1,tmp2] [B1:tmp1,tmp2] [B2:tmp1,tmp2]  (6/6 VALU)
...repeat for all 6 stages...
```

## Impact on Full Kernel

- **Without pipelining**: 4096 hashes x 12 cycles = 49,152 cycles
- **With 3-batch pipelining**: 4096 hashes x 4 cycles = 16,384 cycles
- **Savings**: ~33,000 cycles

## Why It's Optimal

- Exhaustive enumeration guarantees no better schedule exists
- Dependencies: tmp1||tmp2 -> combine (2 cycles minimum per stage)
- 3 batches fills all 6 VALU slots for tmp1+tmp2 cycles
