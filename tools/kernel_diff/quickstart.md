# Kernel Diff - Quick Start

## TL;DR

Compare two kernel versions to track optimization impact.

```bash
# Compare two kernels
python tools/kernel_diff/kernel_diff.py kernel1.json kernel2.json

# Compare current vs baseline
python tools/kernel_diff/kernel_diff.py --baseline baseline.json

# Save current kernel
python tools/kernel_diff/kernel_diff.py --save current.json
```

## Common Workflows

### Track optimization progress:
```bash
python tools/kernel_diff/kernel_diff.py --save v1.json
# ... make changes ...
python tools/kernel_diff/kernel_diff.py --baseline v1.json
```

### JSON output:
```bash
python tools/kernel_diff/kernel_diff.py --json k1.json k2.json
```

### With side-by-side view:
```bash
python tools/kernel_diff/kernel_diff.py -v k1.json k2.json
```

## Key Output

- **Speedup > 1.0**: Improvement
- **Speedup < 1.0**: Regression
- **+Utilization**: Better slot usage
- **-Cycles**: Faster execution
