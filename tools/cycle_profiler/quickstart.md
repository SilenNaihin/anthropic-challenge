# Cycle Profiler - Quick Reference

## TL;DR

Find out WHERE cycles are spent (hash vs memory vs index calc).

```bash
# Basic
python tools/cycle_profiler/cycle_profiler.py

# Full analysis
python tools/cycle_profiler/cycle_profiler.py --all

# JSON
python tools/cycle_profiler/cycle_profiler.py --json
```

## Key Output

```
HOTSPOTS (phases by cycle count)
--------------------------------
hash           512    64.6%   <- Focus here first!
memory_load    234    29.5%
index_calc     156    19.7%
```

## What Each Phase Means

| Phase | What it is |
|-------|------------|
| hash | 6-stage hash mixing |
| memory_load | vload, load ops |
| memory_store | vstore, store ops |
| index_calc | Address math |
| bounds_check | Comparisons, vselect |
| xor_mix | XOR with node values |

## Options

| Flag | What it does |
|------|--------------|
| `--all` | Everything |
| `--detailed` | Phase details |
| `--per-round` | Per-iteration breakdown |
| `--recommendations` | Suggestions |
| `--json` | Machine output |

## Next Steps

- hash > 50%? -> Run `hash_pipeline.py`
- load > 30%? -> Check prefetching
- Then run `slot_analyzer.py` on hotspot area
