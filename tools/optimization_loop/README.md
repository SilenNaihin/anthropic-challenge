# Optimization Loop Runner

Meta-tool that automates the profile->analyze->transform->validate optimization loop for VLIW SIMD kernel optimization.

## Overview

The optimization loop runner orchestrates all analysis tools to provide a comprehensive view of performance bottlenecks and actionable optimization suggestions. It runs:

1. **Profiling**: slot_analyzer, dependency_graph, cycle_profiler
2. **Bottleneck Detection**: Identifies what's limiting performance
3. **Transform Suggestions**: Recommends specific optimizations
4. **Validation**: Ensures changes don't break correctness

## Installation

No additional dependencies required. Uses existing tools from the `tools/` directory.

For Rich output (recommended):
```bash
pip install rich
```

## Quick Start

```bash
# Full optimization loop
python tools/optimization_loop/optimize.py

# Profile only
python tools/optimization_loop/optimize.py --profile

# Profile and get suggestions (skip validation)
python tools/optimization_loop/optimize.py --suggest

# JSON output
python tools/optimization_loop/optimize.py --json
```

## Usage

### Full Optimization Loop

```bash
python tools/optimization_loop/optimize.py
```

Runs all four steps:
1. Profiling (slot_analyzer, dependency_graph, cycle_profiler)
2. Bottleneck detection
3. Transform suggestions
4. Validation

### Profile Only

```bash
python tools/optimization_loop/optimize.py --profile
```

Runs only the profiling tools and reports their status.

### Suggest Transforms

```bash
python tools/optimization_loop/optimize.py --suggest
```

Profiles the kernel and suggests optimizations without running validation (faster).

### Validate Only

```bash
python tools/optimization_loop/optimize.py --validate
```

Runs constraint validation and measures cycles.

### Dry Run

```bash
python tools/optimization_loop/optimize.py --dry-run
```

Shows what would be done without actually running tools. Useful for understanding the workflow.

### JSON Output

```bash
python tools/optimization_loop/optimize.py --json
```

Outputs machine-readable JSON for scripting and integration.

## Output Sections

### Summary

```
Tools Run: slot_analyzer, dependency_graph, cycle_profiler
Tools Succeeded: 3/3
Current Cycles: 792
Target Cycles: 1,487
Slot Utilization: 45.2%
Critical Path: 42 cycles
Theoretical Speedup: 18.86x
```

### Bottlenecks

Bottlenecks are identified and ranked by severity:

- **HIGH**: Critical issues that significantly impact performance
- **MEDIUM**: Issues worth addressing
- **LOW**: Minor optimization opportunities

Example:
```
[HIGH] slot_utilization
  Very low slot utilization (25.3%). Most execution units idle.
  Metric: utilization_pct = 25.3 (threshold: 30)
```

### Transforms

Suggested transformations are ranked by priority:

- **P1**: High impact, address first
- **P2**: Medium impact
- **P3**: Nice to have

Each transform includes:
- Description of what to do
- Potential savings estimate
- Difficulty level (easy/medium/hard)
- Prerequisites

Example:
```
P1 [EASY] Pack more instructions per cycle
  Many cycles have unused slots. Use VLIW packer or manually combine.
  Potential: Up to 75% cycle reduction
  Prerequisites: Run constraint_validator to ensure valid packing
```

### Validation

```
Status: PASSED
Cycles: 792
Speedup: 186.54x over baseline
```

## Bottleneck Types

| Type | Description |
|------|-------------|
| `slot_utilization` | Low overall slot usage (<30% critical, <50% medium) |
| `dependency_chain` | Long critical path with many wasted cycles |
| `memory_bound` | Load/store operations dominate (>50%) |
| `hash_bound` | Hash computation dominates (>50%) |
| `engine_imbalance` | Some engines saturated while others idle |
| `packing_missed` | Easy packing opportunities not taken |

## Transform Types

| Type | Description |
|------|-------------|
| `instruction_packing` | Combine independent instructions into single cycle |
| `loop_unrolling` | Unroll loops to expose more parallelism |
| `software_pipelining` | Overlap iterations to hide latencies |
| `vectorization` | Use VALU to process multiple elements |
| `dependency_breaking` | Restructure to reduce dependencies |
| `hoisting` | Move invariants outside loops |

## Command Line Options

| Option | Description |
|--------|-------------|
| `--profile`, `-p` | Only run profiling tools |
| `--suggest`, `-s` | Profile and suggest (no validation) |
| `--validate`, `-v` | Only run validation |
| `--no-validation` | Skip validation in full loop |
| `--json`, `-j` | Output JSON format |
| `--dry-run`, `-n` | Show what would be done |
| `--verbose` | Show verbose output |
| `--no-color` | Disable colored output |

## Integration with Other Tools

The optimization loop uses these tools internally:

- `tools/slot_analyzer.py` - Slot utilization analysis
- `tools/dependency_graph/dependency_graph.py` - Dependency DAG analysis
- `tools/cycle_profiler/cycle_profiler.py` - Phase-based profiling
- `tools/constraint_validator/constraint_validator.py` - Validation

For deeper analysis, use these tools directly:

```bash
# Detailed slot analysis
python tools/slot_analyzer.py --packing --deps --recommendations

# Dependency graph with hot registers
python tools/dependency_graph/dependency_graph.py --top 20

# Cycle profiler with per-round breakdown
python tools/cycle_profiler/cycle_profiler.py --all
```

## Typical Workflow

1. **Initial Analysis**:
   ```bash
   python tools/optimization_loop/optimize.py --suggest
   ```

2. **Address Top Bottleneck**:
   - Read the suggested transforms
   - Apply the highest priority, easiest transform
   - Re-run analysis to verify improvement

3. **Iterate**:
   ```bash
   python tools/optimization_loop/optimize.py
   ```

4. **Track Progress**:
   ```bash
   python tools/optimization_loop/optimize.py --json > analysis_v1.json
   # Make changes
   python tools/optimization_loop/optimize.py --json > analysis_v2.json
   # Compare cycles in JSON files
   ```

## Example Output

```
======================================================================
                      OPTIMIZATION LOOP REPORT
======================================================================
Timestamp: 2024-01-22T10:30:00

Tools Run: slot_analyzer, dependency_graph, cycle_profiler
Tools Succeeded: 3/3
Current Cycles: 792
Target Cycles: 1,487
Slot Utilization: 45.2%
Critical Path: 42 cycles
Theoretical Speedup: 18.86x

----------------------------------------------------------------------
BOTTLENECKS DETECTED (2)
----------------------------------------------------------------------

1. [HIGH] dependency_chain
   Long dependency chains. 94.7% of cycles could be eliminated.
   Metric: wasted_cycles_pct = 94.7 (threshold: 50)

2. [MEDIUM] slot_utilization
   Low slot utilization (45.2%). Room for more parallelism.
   Metric: utilization_pct = 45.2 (threshold: 50)

----------------------------------------------------------------------
SUGGESTED TRANSFORMS (3)
----------------------------------------------------------------------

1. P1 [HARD] Break dependency chains
   Critical path is 42 cycles. Use register renaming or reorder.
   Potential: 18.86x theoretical speedup
   Prerequisites: Identify hot registers, Map dependency DAG

2. P2 [HARD] Apply software pipelining
   Overlap iterations of loops to hide dependency latencies.
   Potential: 2-3x for memory-bound loops
   Prerequisites: Identify loop boundaries

3. P3 [EASY] Pack more instructions per cycle
   Many cycles have unused slots.
   Potential: Up to 55% cycle reduction
   Prerequisites: Run constraint_validator

----------------------------------------------------------------------
VALIDATION RESULTS
----------------------------------------------------------------------
Status: PASSED
Cycles: 792
Speedup: 186.54x over baseline
```

## JSON Schema

```json
{
  "timestamp": "2024-01-22T10:30:00",
  "profiles": {
    "slot_analyzer": {
      "tool_name": "slot_analyzer",
      "success": true,
      "duration_ms": 1234.5,
      "data": { ... }
    }
  },
  "bottlenecks": [
    {
      "type": "slot_utilization",
      "severity": "high",
      "description": "...",
      "metric_name": "utilization_pct",
      "metric_value": 25.3,
      "threshold": 30,
      "evidence": { ... }
    }
  ],
  "transforms": [
    {
      "type": "instruction_packing",
      "priority": 1,
      "title": "...",
      "description": "...",
      "potential_savings": "...",
      "difficulty": "easy",
      "prerequisites": [...]
    }
  ],
  "validation": {
    "passed": true,
    "cycles": 792,
    "baseline_cycles": 147734,
    "speedup": 186.54,
    "errors": [],
    "warnings": []
  },
  "summary": {
    "total_bottlenecks": 2,
    "high_severity_bottlenecks": 1,
    "total_transforms": 3,
    "current_cycles": 792,
    "target_cycles": 1487
  }
}
```

## Troubleshooting

### Tools Not Found

Ensure you're running from the project root:
```bash
cd /path/to/anthropic-challenge
python tools/optimization_loop/optimize.py
```

### JSON Parse Errors

Some tools output progress messages to stderr. These are ignored when parsing JSON output. If you see parse errors, run the individual tool directly to debug:
```bash
python tools/slot_analyzer.py --json
```

### Timeout Errors

The optimization loop has a 60-second timeout per tool. If a tool times out:
1. Run it directly to see where it hangs
2. Check for infinite loops in the kernel
3. Reduce batch size for testing

## See Also

- `tools/tools.md` - Overview of all tools
- `tools/prd.json` - Tool tracking and priorities
- Individual tool READMEs for detailed documentation
