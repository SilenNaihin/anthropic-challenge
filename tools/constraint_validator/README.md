# Constraint Validator

Static constraint checking for VLIW SIMD kernels. Catches errors before slow runtime failures.

## Overview

The Constraint Validator performs static analysis on kernel instruction streams to detect violations of VLIW architecture constraints before runtime. This is essential for catching errors early in the development cycle, avoiding time-consuming debugging sessions with the full simulator.

## Features

### 1. Slot Limit Validation
Verifies that each cycle respects the per-engine slot limits:
- **ALU**: 12 slots max
- **VALU**: 6 slots max
- **Load**: 2 slots max
- **Store**: 2 slots max
- **Flow**: 1 slot max

### 2. Scratch Memory Overflow Detection
- Detects accesses beyond `SCRATCH_SIZE` (1536)
- Detects negative scratch addresses
- Reports high water mark usage
- Warns when approaching the limit (>75%, >90%)

### 3. Same-Cycle Hazard Detection
Detects RAW (Read-After-Write) hazards within the same cycle:
- In VLIW, all slots execute "simultaneously"
- Reads happen before writes are committed
- Reading a value written in the same cycle gets the OLD value
- This is often unintentional and indicates a bug

### 4. Register Usage Validation
- **Uninitialized reads**: Reading from scratch addresses before any write
- **Dead writes**: Writing without subsequent reads (informational only)

## Installation

No additional dependencies required. For colored output, install `rich`:

```bash
pip install rich
```

## Usage

### Basic Validation

```bash
# Validate the current kernel from perf_takehome.py
python tools/constraint_validator/constraint_validator.py

# With verbose output (includes info messages)
python tools/constraint_validator/constraint_validator.py --verbose
```

### JSON Output

```bash
# For scripting or integration
python tools/constraint_validator/constraint_validator.py --json
```

### Strict Mode

```bash
# Exit non-zero on warnings (for CI/CD)
python tools/constraint_validator/constraint_validator.py --strict
```

### Validate Saved Kernel

```bash
# Validate a kernel saved as JSON
python tools/constraint_validator/constraint_validator.py --kernel saved_kernel.json
```

### Disable Colors

```bash
# Plain text output (for logs, piping)
python tools/constraint_validator/constraint_validator.py --no-color
```

## Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--json` | `-j` | Output results as JSON |
| `--strict` | `-s` | Treat warnings as errors |
| `--kernel FILE` | `-k FILE` | Load kernel from JSON file |
| `--verbose` | `-v` | Show all issues including info |
| `--no-color` | | Disable colored output |

## Output Format

### Console Output (Rich)

```
+====================+
| CONSTRAINT VALIDATION RESULTS |
+====================+

+----------------+
| PASS - All constraints satisfied |
+----------------+

Errors:   0
Warnings: 3
Info:     2
Total Cycles: 598
Scratch Usage: 485 / 1536 (31.6%)

WARNINGS (3)
+-------+------------------+--------------------------------+
| Cycle | Category         | Message                        |
+-------+------------------+--------------------------------+
| 15    | same_cycle_hazard| Same-cycle RAW hazard at...   |
+-------+------------------+--------------------------------+
```

### JSON Output

```json
{
  "valid": true,
  "error_count": 0,
  "warning_count": 3,
  "info_count": 2,
  "total_cycles": 598,
  "scratch_high_water": 485,
  "scratch_limit": 1536,
  "issues": [
    {
      "severity": "warning",
      "category": "same_cycle_hazard",
      "message": "Same-cycle RAW hazard at scratch[42]...",
      "cycle": 15,
      "engine": null,
      "details": {...}
    }
  ],
  "statistics": {
    "total_cycles": 598,
    "total_slots_used": 2456,
    "avg_slots_per_cycle": 4.11,
    "engine_usage": {...}
  }
}
```

## API Usage

The validator can be imported and used programmatically:

```python
from tools.constraint_validator.constraint_validator import validate_kernel, ValidationResult

# Your instruction list
instructions = [
    {"alu": [(...), (...)], "load": [(...)]},
    # ...
]

# Validate
result: ValidationResult = validate_kernel(instructions)

# Check results
if result.is_valid:
    print("Kernel is valid!")
else:
    print(f"Found {result.error_count} errors")
    for issue in result.issues:
        if issue.severity.value == "error":
            print(f"  {issue.message}")
```

## Severity Levels

| Level | Description | Exit Code |
|-------|-------------|-----------|
| **ERROR** | Will definitely cause runtime failure | 1 |
| **WARNING** | May cause issues or is suboptimal | 0 (1 with --strict) |
| **INFO** | Informational message | 0 |

## Issue Categories

| Category | Severity | Description |
|----------|----------|-------------|
| `slot_limit` | ERROR | Exceeded slot limit for an engine |
| `scratch_overflow` | ERROR/WARNING | Scratch address out of bounds |
| `same_cycle_hazard` | WARNING | RAW hazard within same cycle |
| `register_usage` | WARNING/INFO | Uninitialized reads or dead writes |

## Integration with Other Tools

### With slot_analyzer.py

```bash
# First validate, then analyze
python tools/constraint_validator/constraint_validator.py && \
python tools/slot_analyzer.py --packing
```

### With vliw_packer

```bash
# Pack, then validate the result
python tools/vliw_packer/vliw_packer.py --output packed.json
python tools/constraint_validator/constraint_validator.py --kernel packed.json
```

### In Scripts

```bash
#!/bin/bash
# Validate before running expensive tests
if python tools/constraint_validator/constraint_validator.py --strict --json > /dev/null 2>&1; then
    echo "Validation passed, running tests..."
    python tests/submission_tests.py
else
    echo "Validation failed!"
    python tools/constraint_validator/constraint_validator.py
    exit 1
fi
```

## Common Issues and Solutions

### Slot Limit Exceeded

**Problem**: More instructions in an engine than allowed.

**Solution**: Split instructions across multiple cycles or use different engines.

### Scratch Overflow

**Problem**: Accessing scratch addresses beyond 1536.

**Solution**: Review scratch allocation, reuse temporary registers, or restructure data layout.

### Same-Cycle RAW Hazard

**Problem**: Reading a value that's being written in the same cycle.

**Solution**: Move the read to the next cycle, or restructure to avoid the dependency.

### Uninitialized Reads

**Problem**: Reading from scratch before any write.

**Solution**: Either this is intentional (relying on zero-initialization) or add initialization.

## Architecture Reference

```
VLIW Bundle (per cycle):
+-------+-------+------+-------+------+
|  ALU  | VALU  | LOAD | STORE | FLOW |
| (12)  |  (6)  |  (2) |  (2)  |  (1) |
+-------+-------+------+-------+------+
         All execute simultaneously
         Reads before Writes committed

Scratch Memory: 1536 words (32-bit each)
Vector Length (VLEN): 8 elements
```

## Related Tools

- **slot_analyzer.py**: Detailed utilization analysis
- **dependency_graph/**: Full dependency DAG analysis
- **vliw_packer/**: Automatic instruction packing
