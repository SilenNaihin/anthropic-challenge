# ILP Optimal Scheduler

**Status**: Completed | **Priority**: P1

An Integer Linear Programming (ILP) based optimal scheduler for VLIW SIMD instruction streams. Uses OR-Tools CP-SAT solver to find the theoretically optimal schedule, proving what's achievable vs greedy heuristics.

## Why This Tool Matters

Greedy list scheduling (used by VLIW packer) may not find optimal. This tool:

1. **Proves optimality** - Gives the theoretical minimum cycles achievable
2. **Measures the gap** - Shows how far current schedule is from optimal
3. **Guides optimization** - If gap is small, wins must come from algorithmic changes, not scheduling

## Quick Usage

```bash
# Basic analysis
python tools/ilp_scheduler/ilp_scheduler.py

# JSON output for scripting
python tools/ilp_scheduler/ilp_scheduler.py --json

# Limit solver time (default 120s)
python tools/ilp_scheduler/ilp_scheduler.py --time-limit 60

# Show solver progress
python tools/ilp_scheduler/ilp_scheduler.py --verbose
```

## Installation

OR-Tools is required for optimal solving. The tool falls back to greedy scheduling if unavailable.

```bash
pip install ortools
```

## ILP Formulation

The scheduling problem is formulated as:

### Variables
- `cycle[i]` = integer in `[0, n-1]` - which cycle instruction `i` is scheduled

### Constraints

1. **Dependencies**: For each RAW dependency (i, j) where j reads what i writes:
   ```
   cycle[j] >= cycle[i] + 1
   ```

2. **Slot Limits**: For each cycle `c` and engine `e`:
   ```
   count(instructions assigned to cycle c for engine e) <= slot_limit[e]
   ```

   Where limits are: alu=12, valu=6, load=2, store=2, flow=1

### Objective
```
minimize makespan = max(cycle[i] for all i)
```

## Output Explanation

### Status Values

| Status | Meaning |
|--------|---------|
| `OPTIMAL` | Proven optimal solution found |
| `FEASIBLE` | Solution found, may not be optimal (timeout/fallback) |
| `INFEASIBLE` | No valid schedule exists (shouldn't happen) |
| `TIMEOUT` | Time limit reached, best found returned |
| `ERROR` | Solver error or OR-Tools not available |

### Key Metrics

| Metric | Description |
|--------|-------------|
| Current Cycles | Cycles in the original instruction stream |
| Optimal Cycles | Minimum achievable cycles |
| Gap | Difference between current and optimal |
| Gap % | Percentage improvement possible |
| Speedup Potential | Current / Optimal ratio |
| Optimal Utilization | Slot usage in optimal schedule |

### Interpretation Guide

| Gap % | Interpretation |
|-------|----------------|
| < 5% | **Excellent** - Already near optimal. Focus on algorithmic changes. |
| 5-20% | **Good** - Some room for scheduling improvement. Try aggressive packing. |
| > 20% | **Significant** - Major gains possible via better scheduling. |

## Example Output

```
ILP OPTIMAL SCHEDULER RESULTS
======================================================================

Status:              OPTIMAL (proven optimal)
Solve Time:          4.53 seconds

----------------------------------------------------------------------
SCHEDULE COMPARISON
----------------------------------------------------------------------
Current Cycles:      8,500
Optimal Cycles:      6,234
Gap:                 2,266 cycles (26.7%)
Speedup Potential:   1.36x

----------------------------------------------------------------------
ANALYSIS
----------------------------------------------------------------------
Total Instructions:  45,678
Total Dependencies:  89,012
Optimal Utilization: 32.4%

----------------------------------------------------------------------
INTERPRETATION
----------------------------------------------------------------------
SIGNIFICANT GAP: Current schedule is far from optimal.
Major scheduling improvements are possible.
Run VLIW packer with aggressive mode or manual optimization.
```

## Python API

```python
from tools.ilp_scheduler.ilp_scheduler import solve_optimal_schedule

# Load your instruction bundles
bundles = [
    {"alu": [("+", 1, 2, 3)]},
    {"load": [("load", 4, 5)]},
    # ...
]

# Solve
result = solve_optimal_schedule(
    bundles,
    time_limit_seconds=120,
    verbose=False
)

# Access results
print(f"Optimal: {result.optimal_cycles}")
print(f"Current: {result.current_cycles}")
print(f"Speedup: {result.speedup_potential:.2f}x")
print(f"Status: {result.status.value}")

# Get the optimal schedule
for instr_id, cycle in result.schedule.items():
    print(f"Instruction {instr_id} -> Cycle {cycle}")
```

## Complexity

The ILP scheduling problem is NP-hard in general. However:

- CP-SAT solver is highly optimized for scheduling problems
- Typical kernels (10k-100k instructions) solve in seconds to minutes
- Falls back to greedy O(n log n) if timeout or OR-Tools unavailable

### Scaling

| Instructions | Typical Solve Time |
|--------------|-------------------|
| < 1,000 | < 1 second |
| 1,000 - 10,000 | 1-30 seconds |
| 10,000 - 50,000 | 30s - 2 minutes |
| > 50,000 | May timeout |

## Comparison with VLIW Packer

| Aspect | ILP Scheduler | VLIW Packer |
|--------|---------------|-------------|
| **Guarantee** | Proven optimal (if OPTIMAL status) | Heuristic only |
| **Speed** | Slower (seconds-minutes) | Fast (milliseconds) |
| **Use Case** | Analysis, proving bounds | Production scheduling |
| **Output** | Optimal cycle count + schedule | Packed instruction bundles |

## Limitations

1. **Memory Usage**: Creates boolean variables for each (instruction, cycle) pair. May use significant memory for large kernels.

2. **Flow Control**: Conservatively orders all instructions after flow control. May not find optimal if more parallelism is safe.

3. **Register Renaming**: Doesn't consider WAW/WAR hazards that could be eliminated by renaming.

4. **Instruction Latency**: Assumes all instructions complete in 1 cycle (which is correct for this architecture).

## Files

- `ilp_scheduler.py` - Main scheduler implementation
- `README.md` - This documentation
- `quickstart.md` - Quick reference

## Dependencies

- Python 3.8+
- `ortools` (optional but recommended)
- `rich` (optional, for colored output)

## See Also

- `tools/vliw_packer/` - Greedy list scheduling for production use
- `tools/dependency_graph/` - Dependency analysis without solving
- `tools/slot_analyzer.py` - Utilization analysis
