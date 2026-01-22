# Slot Analyzer Quickstart

## TL;DR

```bash
# Full analysis with recommendations
python tools/slot_analyzer.py --packing --deps --recommendations

# JSON output for scripting
python tools/slot_analyzer.py --json

# Compare before/after
python tools/slot_analyzer.py --save before.json
# ... make changes ...
python tools/slot_analyzer.py --compare before.json after.json
```

## Key Flags

| Flag | What it does |
|------|--------------|
| `-p` | Packing opportunities |
| `-d` | Dependency analysis |
| `-r` | Recommendations |
| `--json` | Machine-readable output |

## What to Look For

1. **Utilization %** - Below 50% = lots of room to improve
2. **Critical path vs actual** - Big gap = parallelism available
3. **Packable pairs** - Free cycle savings
4. **RAW-blocked pairs** - Harder but bigger wins
