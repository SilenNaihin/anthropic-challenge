# Cycle Profiler

Break down cycles by code section to understand WHERE time is spent, not just how many cycles total.

## Overview

The Cycle Profiler categorizes each instruction into phases (hash, memory, index calculation, etc.) and provides:

- **Hotspot identification**: Which phases dominate execution time
- **Phase breakdown**: Detailed statistics per phase
- **Per-round analysis**: How cycles distribute across rounds
- **Optimization recommendations**: Targeted suggestions based on profile

## Quick Start

```bash
# Basic profiling
python tools/cycle_profiler/cycle_profiler.py

# Full analysis with recommendations
python tools/cycle_profiler/cycle_profiler.py --all

# JSON output for scripting
python tools/cycle_profiler/cycle_profiler.py --json
```

## Phases

The profiler categorizes instructions into these phases:

| Phase | Description | Typical Source |
|-------|-------------|----------------|
| `hash` | Hash computation (6-stage mixing) | VALU ops: +, ^, <<, >> |
| `memory_load` | Load operations | load, vload |
| `memory_store` | Store operations | store, vstore |
| `index_calc` | Index/address computation | ALU add, multiply_add |
| `bounds_check` | Bounds checking | <, >, vselect |
| `xor_mix` | XOR with node values | ^ after loads |
| `broadcast` | Vector broadcasts | vbroadcast |
| `flow` | Control flow | pause, jumps |
| `init` | Initialization | const loading |

## Usage

### Basic Profiling

```bash
python tools/cycle_profiler/cycle_profiler.py
```

Shows:
- Summary statistics (total cycles, init vs main loop)
- Hotspot table (phases ranked by cycle count)
- ASCII bar visualization

### Detailed Phase Breakdown

```bash
python tools/cycle_profiler/cycle_profiler.py --detailed
```

For each phase shows:
- Total cycles and slots
- Number of occurrences
- Average slots per occurrence
- Engine breakdown (which engines execute this phase)

### Per-Round Analysis

```bash
python tools/cycle_profiler/cycle_profiler.py --per-round
```

Shows how cycles distribute across detected rounds (useful for loop analysis).

### Optimization Recommendations

```bash
python tools/cycle_profiler/cycle_profiler.py --recommendations
```

Provides targeted suggestions based on detected hotspots:
- If hash dominates -> suggests hash pipelining
- If loads dominate -> suggests prefetching/overlapping
- If index calc dominates -> suggests strength reduction

### All Analyses

```bash
python tools/cycle_profiler/cycle_profiler.py --all
```

Runs all analyses together.

### JSON Output

```bash
python tools/cycle_profiler/cycle_profiler.py --json > profile.json
```

Machine-readable output for scripting or further analysis.

## Output Example

```
================================================================================
CYCLE PROFILER - WHERE IS TIME SPENT?
================================================================================

Total Cycles:        792
Total Slots Used:    5,432
Init Cycles:         47
Main Loop Cycles:    745

----------------------------------------------------------------------
HOTSPOTS (phases by cycle count)
----------------------------------------------------------------------

Phase                   Cycles   % of Total   Exclusive
----------------------------------------------------------------------
hash                       512        64.6%       45.2%
memory_load                234        29.5%       12.3%
index_calc                 156        19.7%        8.1%
xor_mix                     48         6.1%        2.4%
memory_store                32         4.0%        4.0%
...
```

## Interpreting Results

### Hotspots

- **% of Total**: Percentage of cycles where this phase is active
- **Exclusive**: Cycles where ONLY this phase runs (no overlap)

High exclusive % means the phase has dedicated cycles (opportunity for packing).
Low exclusive % means good overlap with other work.

### Optimization Guidance

| Hotspot | Implication | Action |
|---------|-------------|--------|
| hash > 50% | Hash-bound | Use hash pipelining (see hash_pipeline tool) |
| load > 30% | Memory-bound | Prefetch, overlap with compute |
| index > 20% | Compute overhead | Pre-compute, strength reduction |
| init > 10% | Startup cost | Hoist constants, reduce setup |

### Per-Round Consistency

If per-round cycles vary significantly:
- Check for unrolling opportunities (uniform rounds = unroll-friendly)
- Look for first/last round special cases

## Programmatic Usage

```python
from tools.cycle_profiler.cycle_profiler import profile_instructions, analyze_kernel

# Profile the current kernel
instructions = analyze_kernel()
result = profile_instructions(instructions)

# Access results
print(f"Total cycles: {result.total_cycles}")
print(f"Init cycles: {result.init_cycles}")

# Get hotspots
for phase, percentage in result.hotspots[:5]:
    print(f"{phase.value}: {percentage:.1f}%")

# Get specific phase stats
hash_stats = result.phase_stats[Phase.HASH]
print(f"Hash cycles: {hash_stats.total_cycles}")
print(f"Hash slots: {hash_stats.total_slots}")
```

## Comparison with Other Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| **cycle_profiler** | WHERE time is spent | First: identify bottleneck area |
| slot_analyzer | HOW slots are used | Second: analyze utilization in hotspot |
| dependency_graph | WHY slots are blocked | Third: find blocking dependencies |
| hash_pipeline | Hash-specific ILP | When hash is the hotspot |

## Tips

1. **Start here**: Use cycle_profiler first to identify which area needs attention
2. **Follow recommendations**: The tool suggests next steps based on profile
3. **Track progress**: Re-profile after optimizations to see impact
4. **Combine with slot_analyzer**: Once you know WHERE, use slot_analyzer to see HOW WELL that area uses available slots

## Command Line Options

```
-j, --json           Output JSON instead of human-readable
-d, --detailed       Show detailed phase breakdown
-r, --per-round      Show per-round breakdown
    --recommendations Show optimization recommendations
-a, --all            Show all analyses
    --no-color       Disable colored output
```
