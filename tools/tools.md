# VLIW SIMD Optimization Tools

Reference documentation for all analysis and optimization tools.

## Tool Status

| Tool | Status | Location | Priority |
|------|--------|----------|----------|
| [Slot Analyzer](#slot-analyzer) | **Completed** | `tools/slot_analyzer.py` | P0 |
| [VLIW Auto-Packer](#vliw-auto-packer) | In Progress | `tools/vliw_packer/` | P0 |
| [Dependency Graph](#dependency-graph) | In Progress | `tools/dependency_graph/` | P0 |
| [Hash Pipeline](#hash-pipeline) | In Progress | `tools/hash_pipeline/` | P1 |
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

**Status**: In Progress | **Folder**: `tools/vliw_packer/`

Automatically packs independent instructions into VLIW bundles.

### Purpose
Take instruction pairs identified as "packable" by slot_analyzer and automatically combine them into single cycles, respecting slot limits and dependencies.

### Expected Features
- Dependency-aware scheduling
- Respects slot limits per engine
- Topological sorting
- Outputs packed kernel

---

## Dependency Graph

**Status**: In Progress | **Folder**: `tools/dependency_graph/`

Builds full dependency DAG for accurate critical path analysis.

### Purpose
Current slot_analyzer only checks adjacent cycles. This tool builds a complete dependency graph to find the true critical path and identify all parallelization opportunities.

### Expected Features
- Full DAG construction
- True critical path calculation
- Parallelism potential analysis
- Hot register identification
- Visualization (optional)

---

## Hash Pipeline

**Status**: In Progress | **Folder**: `tools/hash_pipeline/`

Analyzes the 6-stage hash function for software pipelining opportunities.

### Purpose
The hash function is the hottest code path (4096 calls per kernel run). This tool analyzes stage dependencies and generates optimal pipeline schedules.

### Expected Features
- Stage dependency mapping
- Pipeline schedule generation
- Interleaving plans for multiple elements
- Cycle-accurate simulation

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
