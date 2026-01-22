# Memory Access Pattern Analyzer

Analyzes load/store patterns in the VLIW SIMD kernel to identify vectorization opportunities and blockers.

## Purpose

This tool reveals **WHY** `vload`/`vstore` can't be used in certain places. Understanding memory access patterns is critical for optimization because:

1. **Vector loads (vload)** require contiguous memory addresses
2. **Tree lookups** produce scattered/random addresses that can't be vectorized
3. **Stride patterns** might allow gather operations
4. **Address computation** chains determine vectorizability

## Quick Start

```bash
# Full analysis
python tools/memory_analyzer/memory_analyzer.py

# JSON output for scripting
python tools/memory_analyzer/memory_analyzer.py --json

# Verbose output with all details
python tools/memory_analyzer/memory_analyzer.py --verbose

# Plain text (no colors)
python tools/memory_analyzer/memory_analyzer.py --no-color
```

## Features

### Access Pattern Detection

Classifies each memory access into patterns:

| Pattern | Description | Vectorizable? |
|---------|-------------|---------------|
| Sequential | Consecutive addresses (addr, addr+1, ...) | Yes - use vload/vstore |
| Strided | Regular stride (addr, addr+s, addr+2s, ...) | Potentially - gather |
| Scattered | Computed/random addresses | No - inherent limitation |
| Broadcast | Same address loaded multiple times | Use vbroadcast |

### Stride Analysis

Analyzes consecutive memory accesses to detect stride patterns:

```
stride=1     234  ################################ vectorizable!
stride=2      12  ####
stride=8       5  ##
```

### Address Source Tracking

Traces where memory addresses come from:

| Source | Description | Implication |
|--------|-------------|-------------|
| Constant | Fixed offset (header loads) | Always vectorizable |
| Linear | base + offset * i | Vectorizable with proper alignment |
| Computed | ALU operation result | Often tree index - scattered |
| Indirect | Loaded from another address | Pointer chase - not vectorizable |

### Vectorization Blockers

Identifies specific reasons why vectorization isn't possible:

1. **Scattered tree lookups**: Tree node addresses are computed from hash results
2. **Non-contiguous writes**: Output patterns don't align with vector width
3. **Data-dependent addresses**: Address comes from conditional or computed values

## Output Sections

### Summary Statistics

```
Metric                              Count     Elements
------------------------------------------------------------
Scalar Loads                          234          234
Vector Loads (vload)                   45          360
Scalar Stores                          12           12
Vector Stores (vstore)                 20          160
------------------------------------------------------------
TOTAL                                 311          766

Vectorization Rate: 67.9% (of elements)
Scattered Loads (tree lookups): 156
Sequential Opportunity: ~23 potential vloads
```

### Load Access Patterns

Shows how load addresses are distributed:

```
Pattern                  Count      Description
----------------------------------------------------------------------
scattered                  156      Computed addresses (not vectorizable)
sequential                  78      Contiguous addresses (vectorizable)
unknown                     45      Cannot determine pattern
```

### Address Source Breakdown

```
Source                   Count      Implication
----------------------------------------------------------------------
computed                   156      ALU result (often tree index)
constant                    45      Fixed offset (header/init)
linear                      33      Array traversal (vectorizable)
indirect                    12      Pointer chase (not vectorizable)
```

### Vectorization Blockers

```
1. Scattered tree lookups
   156 scalar loads use computed addresses from tree index calculation.
   These are inherently non-contiguous (random access into tree values array).

   Affected cycles: 156
   Potential fix: Consider: 1) Prefetching tree nodes, 2) Batch tree walks,
                  3) Software managed cache for hot nodes
```

### Recommendations

The tool provides prioritized recommendations:

- **HIGH**: Critical issues affecting performance
- **MEDIUM**: Opportunities for improvement
- **LOW**: Minor optimizations

## Understanding Tree Lookup Scattering

The kernel performs tree traversals where:

1. Each element has an index into the tree
2. After hashing, the index is updated: `idx = 2*idx + (hash % 2 == 0 ? 1 : 2)`
3. Tree node values are loaded from `forest_values[idx]`

This creates **inherently scattered** access patterns because:
- Different elements traverse different tree paths
- Hash results determine next index (data-dependent)
- Tree indices are not contiguous across elements

### Why vload Can't Help Here

```
Element 0: needs tree[42]
Element 1: needs tree[17]
Element 2: needs tree[891]
...
```

These addresses are **not consecutive**, so `vload` is impossible. Each element must use a scalar `load` with its computed address.

## Optimization Strategies for Scattered Accesses

While scattered loads can't be directly vectorized, you can:

1. **Software Prefetching**: Overlap address computation with prior loads
2. **Batch Tree Walks**: Process multiple tree levels in one kernel invocation
3. **Hot Node Caching**: Keep frequently accessed nodes in scratch memory
4. **Tree Layout Optimization**: Reorganize tree for better cache behavior

## JSON Output Format

```json
{
  "total_loads": 234,
  "total_stores": 32,
  "total_vloads": 45,
  "total_vstores": 20,
  "load_patterns": {
    "scattered": 156,
    "sequential": 78
  },
  "store_patterns": {
    "sequential": 32
  },
  "vectorization_rate": 67.89,
  "sequential_opportunity": 23,
  "scattered_loads": 156,
  "addr_source_breakdown": {
    "computed": 156,
    "constant": 45,
    "linear": 33
  },
  "stride_histogram": {
    "1": 234,
    "8": 12
  },
  "vectorization_blockers": [
    {
      "reason": "Scattered tree lookups",
      "description": "...",
      "affected_cycles": 156,
      "potential_fix": "..."
    }
  ]
}
```

## Integration with Other Tools

### With Slot Analyzer

```bash
# Check if memory operations are the bottleneck
python tools/slot_analyzer.py --recommendations
```

If load/store utilization is high while ALU is low, memory patterns are limiting performance.

### With Dependency Graph

```bash
# Check if memory addresses cause dependency chains
python tools/dependency_graph/dependency_graph.py --top 20
```

Hot registers that are memory addresses indicate potential memory optimization opportunities.

### With Hash Pipeline

The hash computation determines tree indices, which determines memory access patterns. Optimizing hash throughput affects when addresses are ready for memory operations.

## Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--json` | `-j` | Output JSON instead of human-readable |
| `--verbose` | `-v` | Show detailed access information |
| `--no-color` | | Disable colored output |
| `--top N` | `-n` | Number of items to show in lists |
| `--help` | `-h` | Show help message |

## Algorithm Details

### Address Source Detection

The analyzer traces backwards through instructions to find the origin of each address register:

1. Scan backwards from the memory access cycle
2. Find the instruction that wrote to the address register
3. Classify based on the writing instruction type:
   - `const` -> CONSTANT
   - `load` -> INDIRECT
   - ALU operation -> COMPUTED
   - `vbroadcast` from scalar -> LINEAR

### Pattern Classification

Access patterns are determined by:
1. The address source type
2. Whether vector operations are used
3. Consecutive access analysis for stride detection

### Stride Detection

For consecutive scalar loads:
1. Sort loads by cycle
2. For loads within 2 cycles of each other
3. Calculate destination register stride
4. Build histogram of detected strides

## Limitations

1. **Static Analysis**: Analyzes instruction stream, not runtime behavior
2. **Conservative Estimates**: Some patterns may be better than detected
3. **No Value Tracking**: Can't determine actual addresses at runtime
4. **Single Kernel**: Analyzes the current kernel implementation only

## Contributing

When adding features:
1. Follow the existing pattern classification system
2. Add new blockers to `identify_vectorization_blockers()`
3. Update recommendations in `_print_recommendations()`
4. Add tests for new patterns

## See Also

- `tools/slot_analyzer.py` - Slot utilization analysis
- `tools/dependency_graph/` - Dependency chain analysis
- `tools/hash_pipeline/` - Hash computation analysis
- `CLAUDE.md` - Project overview and optimization strategies
