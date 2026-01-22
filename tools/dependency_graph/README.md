# Dependency Graph Builder

Builds a complete dependency DAG (Directed Acyclic Graph) for VLIW SIMD instruction streams to enable accurate critical path analysis and parallelism detection.

## Why This Tool?

The slot_analyzer's dependency checking only looks at **adjacent cycles**. This misses long-range dependencies:

```
Cycle 0: write to register 5
Cycle 1: unrelated work
...
Cycle 100: read from register 5  <-- This is a RAW dependency spanning 100 cycles!
```

This tool tracks **ALL** dependencies to find:
- **TRUE critical path** through the entire program
- **Hot registers** causing the most blocking
- **Parallelism potential** (how much speedup is theoretically possible)

## Quick Start

```bash
# Basic analysis
python tools/dependency_graph.py

# JSON output for scripting
python tools/dependency_graph.py --json

# Export visualization
python tools/dependency_graph.py --dot graph.dot
dot -Tpng graph.dot -o graph.png
```

## Features

### 1. Full DAG Construction
Uses efficient O(n + edges) algorithm with last-writer/last-reader maps instead of O(n^2) pairwise comparison.

### 2. Critical Path Analysis
Finds the longest dependency chain through the program - this is the theoretical minimum runtime.

**Key Insight**: If your critical path is 2321 cycles but your kernel runs in 6987 cycles, you have 4666 wasted cycles (66.8%) that could be eliminated with better scheduling.

### 3. Hot Register Detection
Identifies which scratch addresses cause the most dependencies. These are your optimization targets:

```
Addr  Name        Deps    Writes   Reads
20    val_vec[0]  1,152   1,024    1,152   <-- Heavy dependency traffic
21    val_vec[1]  1,152   1,024    1,152
...
```

### 4. Parallelism Metrics

- **Parallelism Potential**: cycles / critical_path = how many instructions could theoretically run in parallel
- **Max Parallel Width**: widest level in the DAG - peak concurrent independent instructions
- **Dependency Density**: edges / max_possible_edges - how interconnected the graph is

### 5. Hazard Types

| Hazard | Name | Description | Eliminable? |
|--------|------|-------------|-------------|
| RAW | Read-After-Write | True dependency - must wait for write | No |
| WAW | Write-After-Write | Output dependency - preserve write order | Yes (renaming) |
| WAR | Write-After-Read | Anti-dependency - can't overwrite before read | Yes (renaming) |

By default, only RAW hazards are tracked since WAW/WAR can be eliminated through register renaming.

## Output Format

### Human-Readable
```
CRITICAL PATH ANALYSIS
------------------------------------------------------------
Critical Path Length:   2321 cycles
Current Total Cycles:   6987
Wasted Cycles:          4666 (66.8%)
Theoretical Speedup:    3.01x
```

### JSON
```json
{
  "total_cycles": 6987,
  "total_dependencies": 171544,
  "dependency_breakdown": {
    "RAW": 171544,
    "WAW": 0,
    "WAR": 0
  },
  "critical_path_length": 2321,
  "critical_path_cycles": [10, 11, 39, 41, ...],
  "hot_registers": [
    {"addr": 20, "deps": 1152, "writes": 1024, "reads": 1152, "name": null},
    ...
  ],
  "parallelism_potential": 3.01,
  "max_parallel_width": 54,
  "theoretical_speedup": 3.01,
  "wasted_cycles": 4666
}
```

### DOT Visualization
Export to DOT format for Graphviz:

```bash
python tools/dependency_graph.py --dot graph.dot
dot -Tpng graph.dot -o graph.png

# Or use online viewer: https://dreampuf.github.io/GraphvizOnline/
```

Critical path nodes are highlighted in red.

## Command-Line Options

```
usage: dependency_graph.py [-h] [--json] [--dot FILE] [--top N] [--all-hazards] [--no-color]

Options:
  --json           Output JSON instead of human-readable
  --dot FILE       Export graph to DOT file for Graphviz
  --top N, -n N    Number of hot registers to show (default: 10)
  --all-hazards    Include WAR and WAW hazards (not just RAW)
  --no-color       Disable colored output
```

## Algorithm

### Complexity
- **Time**: O(n + e) where n = cycles, e = dependencies
- **Space**: O(n + e) for adjacency list and stats

### Last-Writer Map Pattern
Instead of checking all pairs O(n^2), we maintain:
- `last_writer[reg]` = cycle that last wrote to reg
- `last_readers[reg]` = set of cycles that read since last write

For each cycle:
1. For each read: add RAW edge from `last_writer[reg]`
2. For each write: add WAW edge from `last_writer[reg]`, WAR edges from `last_readers[reg]`
3. Update maps

### Critical Path (Longest Path)
Since the graph is a DAG with cycles in topological order:
```python
dist[0] = 1
for each dependency (from -> to):
    dist[to] = max(dist[to], dist[from] + 1)
critical_path = max(dist)
```

## Integration with Other Tools

### With slot_analyzer
```python
from tools.dependency_graph import analyze_dependency_graph, DependencyGraphResult
from tools.slot_analyzer import analyze_kernel

result, instructions = analyze_kernel()
dep_result = analyze_dependency_graph(instructions)

# Now you have both slot utilization AND true critical path
```

### With NetworkX (optional)
```python
from tools.dependency_graph import build_networkx_graph

G = build_networkx_graph(instructions)
# Now you can use any NetworkX algorithm:
# - nx.dag_longest_path(G)
# - nx.all_simple_paths(G, source, target)
# - etc.
```

## Interpreting Results

### High Critical Path vs Total Cycles
- **Ratio close to 1**: Good scheduling, dependencies are the bottleneck
- **Ratio > 2**: Poor scheduling, lots of parallelism being wasted
- **Action**: Use vliw_packer to automatically schedule better

### Hot Registers
- Registers with many deps are serialization points
- Consider:
  - Using more registers (reduce reuse)
  - Reordering operations to break chains
  - Software pipelining to overlap dependencies

### Low Parallelism Potential
- If potential < 4x, the algorithm itself is sequential
- May need algorithmic changes, not just scheduling
