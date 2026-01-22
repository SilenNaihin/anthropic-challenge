# Register Pressure Analyzer

Analyzes scratch memory (register file) usage patterns to understand register pressure throughout kernel execution. Essential for determining whether further unrolling/pipelining is feasible.

## Why This Matters

The VLIW SIMD architecture has 1536 scratch memory slots that serve as registers. Understanding pressure helps:

1. **Determine optimization headroom** - If pressure is low, we can be more aggressive with unrolling/pipelining
2. **Identify blockers** - If pressure is high (>90%), further optimization may be blocked
3. **Find reuse opportunities** - Dead values can be reused to reduce total address count
4. **Guide register renaming** - Long-lived values may benefit from renaming to reduce conflicts

## Installation

No additional dependencies required. Rich library provides colored output (optional).

```bash
# Optional: Install Rich for better output
pip install rich
```

## Quick Usage

```bash
# Basic analysis
python tools/register_pressure/register_pressure.py

# Full analysis with all features
python tools/register_pressure/register_pressure.py --all

# JSON output for scripting
python tools/register_pressure/register_pressure.py --json

# Specific analyses
python tools/register_pressure/register_pressure.py --peaks      # Pressure peaks
python tools/register_pressure/register_pressure.py --reuse      # Reuse opportunities
python tools/register_pressure/register_pressure.py --visualize  # Pressure chart
```

## Features

### 1. Live Range Tracking

For each scratch address, tracks:
- **def_cycle**: When the value is written (defined)
- **last_use_cycle**: When the value is last read
- **live_length**: Number of cycles the value is live
- **use_count**: How many times the value is read

### 2. Pressure Computation

Computes how many values are simultaneously live at each cycle:
- **max_simultaneous_live**: Peak pressure during execution
- **avg_live_per_cycle**: Average pressure
- **peak_pressure_pct**: Peak as percentage of 1536 limit

### 3. Peak Detection

Identifies cycles with notably high register pressure:
- Finds top 5% of pressure cycles or anything >75% of limit
- Reports which addresses contribute to pressure at each peak

### 4. Reuse Opportunities

Finds addresses that become dead and could be reused:
- Shows when each address dies (last use)
- Suggests where it could be reused (next allocation)
- Helps reduce total address count

### 5. Pressure Visualization

ASCII art chart showing pressure over time:
- `#` = >90% of limit (critical)
- `*` = 75-90% (high)
- `+` = 50-75% (medium)
- `.` = <50% (healthy)

## Output Interpretation

### Summary Section

```
Max Simultaneous Live:   372
Peak Cycle:              320
Peak Pressure:           24.2%
```

- **<25%**: Healthy - plenty of room for optimization
- **25-50%**: Moderate - watch growth with unrolling
- **50-75%**: Elevated - be cautious with more optimization
- **75-90%**: High - consider register optimization first
- **>90%**: Critical - may block further optimization

### Address Breakdown

| Category | Description |
|----------|-------------|
| Scalar | Single-element values (1 address) |
| Vector | Vector values (VLEN=8 addresses) |
| Constants | Values loaded once, live entire program |
| Dead After Def | Written but never read (waste) |
| Long-Lived (>50) | Live for many cycles, may block reuse |

### Pressure Distribution

Shows what percentage of cycles have each pressure level:
```
0-10%:     53 cycles      (initialization)
10-25%:  4,383 cycles **** (steady state)
```

### Reuse Opportunities

```
Address    Dead After    Could Reuse At    Gap
  2        cycle 1       cycle 3           2 cycles
```

Addresses with larger gaps are better candidates for manual reuse.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Healthy pressure (<90%) |
| 1 | Warning (90-95%) |
| 2 | Critical (>95%) |

## Options Reference

```
--json, -j       Output as JSON
--peaks, -p      Show pressure peaks
--reuse, -r      Show reuse opportunities
--visualize, -v  Show pressure chart
--all, -a        Show everything
--kernel FILE    Load kernel from JSON file
--no-color       Plain text output (no Rich)
```

## JSON Output Schema

```json
{
  "summary": {
    "total_cycles": 4436,
    "total_addresses_used": 405,
    "utilization_pct": 26.37,
    "max_simultaneous_live": 370,
    "max_live_cycle": 320,
    "peak_pressure_pct": 24.09,
    "avg_live_per_cycle": 363.17,
    "scratch_limit": 1536,
    "headroom": 1131
  },
  "breakdown": {...},
  "peaks": [...],
  "reuse_opportunities": [...],
  "pressure_histogram": {...}
}
```

## Programmatic Usage

```python
from tools.register_pressure.register_pressure import analyze_register_pressure

# Get kernel instructions (from KernelBuilder or JSON file)
instructions = [...]

# Analyze
result = analyze_register_pressure(instructions)

# Access metrics
print(f"Peak pressure: {result.peak_pressure_pct:.1f}%")
print(f"Headroom: {result.headroom} addresses")

# Check if safe to add more unrolling
if result.peak_pressure_pct < 50:
    print("Safe to add more unrolling")
elif result.peak_pressure_pct < 75:
    print("Proceed with caution")
else:
    print("Consider register optimization first")
```

## Common Workflows

### 1. Before Adding Unrolling

```bash
# Check current pressure
python tools/register_pressure/register_pressure.py

# If headroom > 200 addresses, unrolling is likely safe
# If peak_pressure_pct > 75%, consider alternatives
```

### 2. Understanding Pressure Growth

```bash
# Run with visualization to see pressure over time
python tools/register_pressure/register_pressure.py --visualize

# Look for:
# - Ramp-up during initialization
# - Steady state during main loop
# - Any unexpected spikes
```

### 3. Finding Optimization Opportunities

```bash
# Full analysis
python tools/register_pressure/register_pressure.py --all

# Check:
# - "Dead After Def" count (wasted allocations)
# - Long-lived values (may block reuse)
# - Reuse opportunities with large gaps
```

## Integration with Other Tools

| Tool | Use Case |
|------|----------|
| constraint_validator | Validates scratch addresses don't overflow |
| dependency_graph | Understand why values stay live |
| transforms | Apply optimizations that affect register use |

## Current Kernel Analysis (Example)

```
Summary:
- Total Addresses Used: 405 (26.4% of 1536 limit)
- Peak Pressure: 24.2% at cycle 320
- Headroom: 1,131 addresses

Assessment: HEALTHY
- Plenty of room for additional unrolling/pipelining
- Current implementation uses registers efficiently
- 91.8% of values are long-lived (hash constants, batch data)
```

## Troubleshooting

### High Pressure (>75%)

1. Check for unnecessary duplicate allocations
2. Look for values that could share addresses (non-overlapping live ranges)
3. Consider reducing unroll factor
4. Hoist loop-invariant allocations outside loop

### Many Dead-After-Def Values

1. May indicate debug/development code left in
2. Check for unused variables in kernel builder
3. May be intentional (placeholder allocations)

### Low Reuse Opportunities

1. Current implementation may already be well-optimized
2. Most values have overlapping live ranges
3. Consider different scheduling to create reuse windows
