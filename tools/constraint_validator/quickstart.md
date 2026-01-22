# Constraint Validator - Quick Start

## TL;DR

Static validation to catch kernel errors before runtime.

```bash
# Validate current kernel
python tools/constraint_validator/constraint_validator.py

# JSON output
python tools/constraint_validator/constraint_validator.py --json

# Strict mode (fail on warnings)
python tools/constraint_validator/constraint_validator.py --strict
```

## What It Checks

| Check | Catches |
|-------|---------|
| Slot limits | >12 alu, >6 valu, >2 load, >2 store, >1 flow |
| Scratch overflow | Address >= 1536 or negative |
| Same-cycle RAW | Write + read same address in one cycle |
| Register usage | Reads before writes |

## Exit Codes

- `0` = Valid (or warnings only without --strict)
- `1` = Errors found (or warnings with --strict)

## Options

```
--json       JSON output
--strict     Warnings = errors
--kernel F   Validate saved kernel JSON
--verbose    Show info messages
--no-color   Plain text
```

## Common Workflow

```bash
# 1. Validate before testing
python tools/constraint_validator/constraint_validator.py --strict

# 2. If pass, run tests
python perf_takehome.py Tests.test_kernel_cycles
```
