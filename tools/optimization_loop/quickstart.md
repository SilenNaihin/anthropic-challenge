# Optimization Loop - Quick Start

## TL;DR

Automated optimization analysis. Profiles kernel, finds bottlenecks, suggests fixes.

```bash
python tools/optimization_loop/optimize.py
```

## Common Commands

```bash
# Full analysis (profile + bottlenecks + suggestions + validation)
python tools/optimization_loop/optimize.py

# Quick analysis (skip validation)
python tools/optimization_loop/optimize.py --suggest

# Profile only
python tools/optimization_loop/optimize.py --profile

# JSON output
python tools/optimization_loop/optimize.py --json
```

## What It Does

1. **Profiles** using slot_analyzer, dependency_graph, cycle_profiler
2. **Detects bottlenecks**: slot utilization, dependencies, memory/hash bound
3. **Suggests transforms**: packing, pipelining, vectorization, etc.
4. **Validates**: ensures correctness and measures cycles

## Reading the Output

### Bottlenecks (fix these)
- `[HIGH]` = Critical, fix first
- `[MEDIUM]` = Worth addressing
- `[LOW]` = Minor

### Transforms (how to fix)
- `P1` = High priority
- `[EASY]` = Quick win
- `[HARD]` = Significant effort

## Example Workflow

```bash
# 1. See what's wrong
python tools/optimization_loop/optimize.py --suggest

# 2. Fix top bottleneck (usually packing or dependencies)

# 3. Verify improvement
python tools/optimization_loop/optimize.py

# 4. Repeat
```

## Flags

| Flag | What it does |
|------|-------------|
| `--suggest` | Quick mode, no validation |
| `--profile` | Only run profilers |
| `--validate` | Only check correctness |
| `--json` | Machine-readable output |
| `--dry-run` | Preview without running |
| `--no-validation` | Full loop minus validation |
