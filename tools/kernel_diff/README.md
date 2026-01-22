# Kernel Diff Tool

**Status**: Completed | **File**: `tools/kernel_diff/kernel_diff.py`

Compare two VLIW SIMD kernel versions to track optimization impact. Essential for understanding what changed between optimization iterations.

## Features

- **Cycle Comparison**: Before/after cycle counts with speedup calculation
- **Utilization Diff**: Slot usage changes per engine
- **Instruction Diff**: Operations added, removed, or modified
- **Side-by-Side View**: Per-cycle slot comparison
- **Rich Output**: Colored terminal output (with fallback to plain text)
- **JSON Output**: Machine-readable output for scripting

## Installation

No additional dependencies required. Rich library is optional for colored output:

```bash
pip install rich  # Optional, for colored output
```

## Usage

### Basic Comparison

Compare two saved kernel JSON files:

```bash
python tools/kernel_diff/kernel_diff.py kernel1.json kernel2.json
```

### Named Comparison

Use explicit before/after flags:

```bash
python tools/kernel_diff/kernel_diff.py --before v1.json --after v2.json
```

### Compare Against Baseline

Compare current kernel implementation against a saved baseline:

```bash
python tools/kernel_diff/kernel_diff.py --baseline baseline.json
```

### Save Current Kernel

Save the current kernel from `perf_takehome.py` to JSON:

```bash
python tools/kernel_diff/kernel_diff.py --save current.json
```

### JSON Output

Get machine-readable output:

```bash
python tools/kernel_diff/kernel_diff.py --json kernel1.json kernel2.json
```

### Verbose Mode

Include side-by-side cycle comparison:

```bash
python tools/kernel_diff/kernel_diff.py -v kernel1.json kernel2.json
```

### Disable Colors

For plain text output:

```bash
python tools/kernel_diff/kernel_diff.py --no-color kernel1.json kernel2.json
```

## Output Sections

### Summary

Quick overview of whether the change was an improvement or regression:

```
IMPROVEMENT: 1.45x faster (-234 cycles)
```

### Cycle Comparison

Before/after cycle counts with calculated speedup:

```
CYCLE COMPARISON
------------------------------------------------------------
  Before:  1000 cycles
  After:   690 cycles
  Delta:   -310 cycles
  Speedup: 1.45x FASTER
```

### Utilization Comparison

Overall slot utilization change:

```
UTILIZATION COMPARISON
------------------------------------------------------------
  Before:  35.2%
  After:   51.8%
  Delta:   +16.6%
```

### Per-Engine Changes

Detailed breakdown by execution engine:

```
PER-ENGINE CHANGES
----------------------------------------------------------------------
Engine        Before       After       Delta   Util Delta
----------------------------------------------------------------------
alu            1,234       1,456        +222       +2.3%
valu             567         890        +323      +12.5%
load             234         234           0        0.0%
store            123         100         -23       -1.2%
flow              45          32         -13       -0.8%
```

### Operation Changes

Which instructions were added, removed, or changed:

```
SIGNIFICANT OPERATION CHANGES
----------------------------------------------------------------------
Operation                    Before      After      Delta         Type
----------------------------------------------------------------------
valu:multiply_add                 0         48        +48        added
alu:+                           156        124        -32     modified
load:vload                       24         48        +24     modified
```

## Workflow Integration

### Optimization Tracking

1. **Save baseline before optimization**:
   ```bash
   python tools/kernel_diff/kernel_diff.py --save snapshots/v1.json
   ```

2. **Make optimization changes to `perf_takehome.py`**

3. **Compare against baseline**:
   ```bash
   python tools/kernel_diff/kernel_diff.py --baseline snapshots/v1.json
   ```

4. **If improvement, save new version**:
   ```bash
   python tools/kernel_diff/kernel_diff.py --save snapshots/v2.json
   ```

### A/B Testing

Compare two different optimization approaches:

```bash
# Save approach A
python tools/kernel_diff/kernel_diff.py --save approach_a.json

# Switch to approach B, save it
python tools/kernel_diff/kernel_diff.py --save approach_b.json

# Compare approaches
python tools/kernel_diff/kernel_diff.py approach_a.json approach_b.json
```

### CI/CD Integration

Use JSON output for automated testing:

```bash
python tools/kernel_diff/kernel_diff.py --json baseline.json current.json | jq '.cycles.speedup'
```

Check for regressions:

```bash
speedup=$(python tools/kernel_diff/kernel_diff.py --json baseline.json current.json | jq '.cycles.speedup')
if (( $(echo "$speedup < 1.0" | bc -l) )); then
    echo "REGRESSION DETECTED"
    exit 1
fi
```

## JSON Output Format

```json
{
  "names": {
    "before": "v1.json",
    "after": "v2.json"
  },
  "cycles": {
    "before": 1000,
    "after": 690,
    "delta": -310,
    "speedup": 1.449,
    "speedup_pct": 44.9
  },
  "slots_used": {
    "before": 8120,
    "after": 8234,
    "delta": 114
  },
  "utilization": {
    "before": 35.26,
    "after": 51.84,
    "delta": 16.58
  },
  "per_engine": {
    "alu": {
      "before_util": 12.3,
      "after_util": 14.5,
      "delta_util": 2.2,
      "before_total": 1234,
      "after_total": 1456,
      "delta_total": 222
    }
  },
  "operation_changes": {
    "added": ["valu:multiply_add"],
    "removed": ["alu:nop"],
    "changes": [
      {
        "engine": "valu",
        "opcode": "multiply_add",
        "before": 0,
        "after": 48,
        "delta": 48,
        "type": "added"
      }
    ]
  },
  "summary": "IMPROVEMENT: 1.45x faster (-310 cycles)"
}
```

## Key Metrics to Watch

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| Speedup | > 1.0 | < 1.0 |
| Utilization Delta | Positive | Negative |
| Cycle Delta | Negative | Positive |
| New valu ops | More vectorization | - |
| More load ops | - | Memory bound |

## Related Tools

- **slot_analyzer.py**: Deep-dive into a single kernel's slot utilization
- **cycle_profiler**: Understand where time is spent by phase
- **dependency_graph**: Analyze what prevents further optimization

## Command Reference

```
usage: kernel_diff.py [-h] [--before FILE] [--after FILE] [--baseline FILE]
                      [--save FILE] [--json] [--verbose] [--no-color]
                      [files ...]

positional arguments:
  files                 Kernel JSON files to compare (before after)

optional arguments:
  -h, --help            show this help message and exit
  --before FILE, -b FILE
                        Before kernel JSON file
  --after FILE, -a FILE
                        After kernel JSON file
  --baseline FILE       Compare current kernel against baseline
  --save FILE           Save current kernel to JSON file
  --json                Output JSON instead of human-readable
  --verbose, -v         Show side-by-side cycle comparison
  --no-color            Disable colored output
```
