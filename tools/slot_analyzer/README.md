# Slot Utilization Analyzer

Comprehensive analysis tool for VLIW SIMD kernel instruction streams. Identifies optimization opportunities by analyzing slot utilization, dependencies, and packing potential.

## Features

| Feature | Description |
|---------|-------------|
| **Slot Utilization** | Overall and per-engine utilization percentages |
| **Dependency Analysis** | RAW hazard detection between instructions |
| **Critical Path** | Estimates theoretical minimum cycles |
| **Packing Opportunities** | Identifies instruction pairs that can be combined |
| **Recommendations** | Prioritized optimization suggestions with difficulty ratings |
| **Kernel Comparison** | Before/after diff for tracking improvements |
| **Rich Output** | Colorful terminal output with tables (optional) |

## Installation

No installation required. Uses standard Python libraries plus optional `rich` for colored output:

```bash
pip install rich  # Optional, for colored output
```

## Usage

### Basic Analysis
```bash
python tools/slot_analyzer.py
```

### With All Features
```bash
python tools/slot_analyzer.py --packing --deps --recommendations
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--packing`, `-p` | Show packing opportunities |
| `--deps`, `-d` | Show dependency analysis |
| `--recommendations`, `-r` | Show optimization recommendations |
| `--json` | Output in JSON format |
| `--top N`, `-n N` | Number of worst/best cycles to show (default: 10) |
| `--save FILE` | Save kernel instructions to JSON |
| `--compare FILE1 FILE2` | Compare two kernel snapshots |
| `--no-color` | Disable Rich colored output |
| `--verbose`, `-v` | Show per-cycle details |

## Output Sections

### 1. Slot Utilization Summary
- Total cycles and slots used
- Overall utilization percentage
- Theoretical speedup if 100% utilized

### 2. Per-Engine Breakdown
Shows for each engine (alu, valu, load, store, flow):
- Total ops used
- Maximum possible
- Utilization percentage
- Average ops per cycle

### 3. Histogram
Distribution of slots used per cycle - shows where waste is concentrated.

### 4. Dependency Analysis (`--deps`)
- RAW (Read-After-Write) hazard count
- Blocking pairs that prevent packing
- Critical path length
- Overhead vs critical path

### 5. Packing Opportunities (`--packing`)
- Instruction pairs that CAN be combined
- Pairs blocked by dependencies
- Breakdown by engine combination type

### 6. Recommendations (`--recommendations`)
Prioritized suggestions:
- **HIGH**: Critical issues (dependency chains, low utilization)
- **MEDIUM**: Easy wins (packable instructions)
- **LOW**: Engine balance issues

## Example Output

```
╔══════════════════════════════════════════════════════════════════╗
║ SLOT UTILIZATION ANALYSIS                                        ║
╚══════════════════════════════════════════════════════════════════╝

  Total Cycles          8,517
  Overall Utilization   12.3%
  Avg Slots/Cycle       2.83 / 23

>>> Theoretical speedup if 100% utilized: 8.1x
>>> That would be: 1050 cycles
```

## Workflow: Tracking Optimizations

```bash
# 1. Save baseline
python tools/slot_analyzer.py --save baseline.json

# 2. Make optimizations to perf_takehome.py

# 3. Compare
python tools/slot_analyzer.py --compare baseline.json optimized.json
```

## Architecture

The analyzer works by:
1. Building instruction list from `KernelBuilder`
2. Extracting read/write sets for each instruction
3. Computing slot utilization statistics
4. Detecting RAW hazards between adjacent cycles
5. Identifying packing opportunities
6. Generating prioritized recommendations

## Key Insights This Tool Provides

1. **Utilization Gap**: How far from theoretical maximum?
2. **Bottleneck Engine**: Which engine limits throughput?
3. **Easy Wins**: Packable pairs with no dependencies
4. **Hard Problems**: Pairs blocked by RAW hazards
5. **Priority Order**: What to optimize first

## Files

- `slot_analyzer.py` - Main analyzer (in parent tools/ folder)
- `README.md` - This documentation
- `quickstart.md` - Quick reference

## Future Improvements

- Full DAG for critical path (currently adjacent-only)
- Hot register detection (which addresses cause most RAW hazards)
- Instruction latency model
- Modularize into separate files
