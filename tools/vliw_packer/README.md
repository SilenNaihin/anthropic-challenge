# VLIW Auto-Packer

Automatically packs independent instructions into VLIW bundles, respecting slot limits and data dependencies.

## Features

- **Dependency-aware scheduling**: Tracks RAW, WAW, WAR hazards
- **Slot limit enforcement**: 12 ALU, 6 VALU, 2 load, 2 store, 1 flow
- **Priority-based scheduling**: Critical path aware instruction ordering
- **Detailed statistics**: Before/after comparison with speedup metrics

## Usage

```bash
# Basic packing
python tools/vliw_packer/vliw_packer.py

# With verbose output
python tools/vliw_packer/vliw_packer.py --verbose

# Output packed kernel to file
python tools/vliw_packer/vliw_packer.py --output packed_kernel.json
```

## How It Works

1. **Build Dependency Graph**: Analyzes all instructions for data dependencies
   - RAW (Read-After-Write): True dependencies
   - WAW (Write-After-Write): Output dependencies
   - WAR (Write-After-Read): Anti-dependencies

2. **Calculate Priorities**: Uses reverse topological order to assign priorities based on critical path

3. **List Scheduling**: Greedily schedules ready instructions into cycles while respecting:
   - Slot limits per engine
   - Data dependencies (instruction must wait for all inputs)

4. **Output Statistics**: Reports cycles saved and utilization improvement

## Example Output

```
============================================================
VLIW PACKING RESULTS
============================================================

Original cycles:    4,947
Packed cycles:      3,493
Cycles saved:       1,454
Speedup:            1.42x

Original utilization: 19.9%
Packed utilization:   28.1%
Improvement:          +8.3%
```

## Limitations

- Flow control is handled conservatively (may miss some packing opportunities)
- No register renaming (WAR/WAW deps could be broken with renaming)
- Memory aliasing not tracked (assumes scratch addresses are exact)

## Future Improvements

- Relax flow control dependencies
- Implement register renaming for WAR/WAW
- Use heap for ready queue (performance)
- Add intra-bundle hazard detection
