# ILP Scheduler - Quick Reference

## TL;DR

Find theoretically optimal schedule using ILP. Proves what's achievable vs greedy.

## Install

```bash
pip install ortools  # Required for optimal solving
```

## Usage

```bash
# Basic (find optimal)
python tools/ilp_scheduler/ilp_scheduler.py

# JSON output
python tools/ilp_scheduler/ilp_scheduler.py --json

# Faster (30s limit)
python tools/ilp_scheduler/ilp_scheduler.py -t 30
```

## Key Output

```
Current Cycles:      8,500
Optimal Cycles:      6,234
Gap:                 2,266 (26.7%)
Speedup Potential:   1.36x
```

## Interpretation

| Gap % | Action |
|-------|--------|
| < 5% | Near optimal. Need algorithmic changes, not scheduling. |
| 5-20% | Try aggressive VLIW packing. |
| > 20% | Major scheduling gains possible. |

## Python API

```python
from tools.ilp_scheduler.ilp_scheduler import solve_optimal_schedule

result = solve_optimal_schedule(bundles, time_limit_seconds=60)
print(f"Optimal: {result.optimal_cycles}, Speedup: {result.speedup_potential:.2f}x")
```

## Status Values

- `OPTIMAL` - Proven best possible
- `FEASIBLE` - Found solution, maybe not optimal
- `TIMEOUT` - Hit time limit

## When to Use

1. **Before optimizing** - Know the theoretical bound
2. **After packing** - Check if greedy found optimal
3. **Stuck on progress** - Determine if scheduling or algorithm is the issue
