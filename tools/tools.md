# VLIW SIMD Optimization Tools

Reference documentation for all analysis and optimization tools.

## Tool Status

| Tool | Status | Location | Priority |
|------|--------|----------|----------|
| [Slot Analyzer](#slot-analyzer) | **Completed** | `tools/slot_analyzer.py` | P0 |
| [VLIW Auto-Packer](#vliw-auto-packer) | **Completed** | `tools/vliw_packer/` | P0 |
| [Dependency Graph](#dependency-graph) | **Completed** | `tools/dependency_graph.py` | P0 |
| [Hash Pipeline](#hash-pipeline) | **Completed** | `tools/hash_pipeline/` | P1 |
| Cycle Profiler | Not Started | - | P1 |
| Memory Analyzer | Not Started | - | P1 |

See `tools/prd.json` for full tracking.

---

## Slot Analyzer

**Status**: Completed | **File**: `tools/slot_analyzer.py`

Comprehensive VLIW slot utilization analysis with dependency detection and optimization recommendations.

### Quick Usage
```bash
python tools/slot_analyzer.py --packing --deps --recommendations
```

### Features
- Overall/per-engine utilization
- RAW hazard detection
- Critical path estimation
- Packing opportunity analysis
- Prioritized recommendations
- Kernel comparison (before/after)
- Rich colored output

### Documentation
- Full docs: `tools/slot_analyzer/README.md`
- Quick ref: `tools/slot_analyzer/quickstart.md`

---

## VLIW Auto-Packer

**Status**: Completed | **Folder**: `tools/vliw_packer/`

Automatically packs independent instructions into VLIW bundles.

### Quick Usage
```bash
python tools/vliw_packer/vliw_packer.py
python tools/vliw_packer/vliw_packer.py --output packed.json
```

### Features
- Dependency-aware scheduling (RAW, WAW, WAR hazards)
- Respects slot limits per engine (12 alu, 6 valu, 2 load, 2 store, 1 flow)
- Priority-based list scheduling
- Outputs packed kernel with statistics
- **Achieves ~1.4x speedup automatically**

### Documentation
- Full docs: `tools/vliw_packer/README.md`
- Quick ref: `tools/vliw_packer/quickstart.md`

---

## Dependency Graph

**Status**: Completed | **File**: `tools/dependency_graph.py`

Builds full dependency DAG for accurate critical path analysis.

### Quick Usage
```bash
python tools/dependency_graph.py
python tools/dependency_graph.py --json
```

### Features
- Full DAG construction (not just adjacent cycles)
- True critical path calculation with topological DP
- Parallelism potential analysis
- Hot register identification (which addresses cause most blocking)
- Rich colored output
- O(n + edges) efficient algorithm

### Key Metrics
- Critical path length vs actual cycles
- Wasted cycles percentage
- Top blocking registers
- Dependency density

### Documentation
- Full docs: `tools/dependency_graph/README.md`
- Quick ref: `tools/dependency_graph/quickstart.md`

---

## Hash Pipeline

**Status**: Completed | **Folder**: `tools/hash_pipeline/`

Analyzes the 6-stage hash function for software pipelining opportunities.

### Quick Usage
```bash
python tools/hash_pipeline/hash_pipeline.py
python tools/hash_pipeline/hash_pipeline.py --elements 4 --verbose
```

### Features
- Stage dependency mapping (6 stages with intra-stage parallelism)
- Pipeline schedule generation
- Interleaving plans for multiple elements
- Cycle-accurate simulation
- Code generation hints
- **Key finding**: 2-way ILP within each stage (tmp1 || tmp2 independent)

### Documentation
- Full docs: `tools/hash_pipeline/README.md`
- Quick ref: `tools/hash_pipeline/quickstart.md`

---

## Common Patterns

### Analyzing Current Kernel
```bash
# Quick check
python tools/slot_analyzer.py

# Full analysis
python tools/slot_analyzer.py --packing --deps --recommendations
```

### Tracking Optimization Progress
```bash
# Save baseline
python tools/slot_analyzer.py --save snapshots/v1.json

# After changes
python tools/slot_analyzer.py --compare snapshots/v1.json snapshots/v2.json
```

### JSON Output for Scripting
```bash
python tools/slot_analyzer.py --json > analysis.json
```

---

## Adding New Tools

1. Create folder: `tools/<tool_name>/`
2. Add `README.md` with full documentation
3. Add `quickstart.md` with quick reference
4. Update `tools/prd.json` with status
5. Update this file (`tools/tools.md`)
6. Update `CLAUDE.md` if user-facing
