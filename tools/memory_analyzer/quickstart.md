# Memory Analyzer - Quick Start

## TL;DR

Reveals why `vload`/`vstore` can't be used in certain places (tree lookups = scattered addresses).

## Basic Usage

```bash
# Full analysis
python tools/memory_analyzer/memory_analyzer.py

# JSON output
python tools/memory_analyzer/memory_analyzer.py --json
```

## Key Metrics

| Metric | What It Means |
|--------|--------------|
| **Vectorization Rate** | % of memory ops using vload/vstore |
| **Scattered Loads** | Tree lookups - NOT vectorizable |
| **Sequential Opportunity** | Scalar loads that COULD be vloads |

## Pattern Types

| Pattern | Vectorizable? |
|---------|--------------|
| Sequential | Yes - use vload |
| Strided | Maybe - gather |
| **Scattered** | **NO** - tree lookups |

## The Core Insight

Tree lookups produce scattered addresses:
```
Element 0: tree[42]    \
Element 1: tree[17]     > Can't use vload - addresses not consecutive
Element 2: tree[891]   /
```

This is **inherent** to tree traversal - can't be fixed with simple vectorization.

## Options

```
--json      JSON output
--verbose   More details
--no-color  Plain text
```
