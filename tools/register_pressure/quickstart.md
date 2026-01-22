# Register Pressure Analyzer - Quick Start

## TL;DR

Analyze scratch memory usage to determine if we have room for more unrolling/pipelining.

```bash
# Basic analysis
python tools/register_pressure/register_pressure.py

# Full analysis with charts and suggestions
python tools/register_pressure/register_pressure.py --all

# JSON output
python tools/register_pressure/register_pressure.py --json
```

## Key Metrics

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Peak Pressure | <50% | 50-75% | >75% |
| Headroom | >500 | 200-500 | <200 |

## Quick Assessment

```
Peak Pressure:  24.2%  -> HEALTHY, room for optimization
Headroom:       1,131  -> PLENTY of addresses available
```

## What It Shows

| Section | Purpose |
|---------|---------|
| Summary | Overall pressure stats |
| Breakdown | Scalar vs vector vs constants |
| Distribution | Pressure histogram over cycles |
| Peaks | Cycles with highest pressure |
| Reuse | Addresses that could be reused |

## Options

```
--all        Show everything
--peaks      Show pressure peaks
--reuse      Show reuse opportunities
--visualize  Show ASCII pressure chart
--json       Machine-readable output
--no-color   Plain text (no Rich)
```

## When to Use

1. **Before adding unrolling** - Check if headroom exists
2. **After optimization** - Verify pressure didn't spike
3. **Debugging failures** - May be out of scratch space

## Exit Codes

- `0` = Healthy (<90%)
- `1` = Warning (90-95%)
- `2` = Critical (>95%)

## Current Status

```
Addresses Used:  405/1536 (26.4%)
Peak Pressure:   24.2%
Headroom:        1,131 addresses

-> Safe to add more unrolling/pipelining
```
