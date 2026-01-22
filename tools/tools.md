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

ILP analysis tool for the 6-stage hash function - the hottest code path (4096 calls per kernel).

### Important Note
This is an **ILP analysis tool**, not a cycle-accurate simulator. Cycle counts are theoretical lower bounds. Always validate with `slot_analyzer.py` on actual instruction streams.

### Quick Usage
```bash
# Full analysis (includes realistic estimates)
python tools/hash_pipeline/hash_pipeline.py

# With visualization
python tools/hash_pipeline/hash_pipeline.py --visualize --elements 8

# Code generation hints
python tools/hash_pipeline/hash_pipeline.py --codegen

# JSON output
python tools/hash_pipeline/hash_pipeline.py --json
```

### Features
- Stage dependency analysis (identifies tmp1 || tmp2 independence)
- Multiple scheduling strategies (sequential, intra-parallel, pipelined, max-pipelined)
- Theoretical minimum calculations (throughput vs latency limited)
- **Realistic estimates** with overhead (vbroadcast, memory, loop control)
- Batch size analysis for optimal vectorization
- VLIW code generation hints
- Schedule visualization
- **Key finding**: 2-way ILP within each stage (tmp1 || tmp2 independent)

### Key Output
```
Per-Batch Breakdown (8 elements via VLEN):
  Hash computation (6 stages x 2):  12 cycles
  vbroadcast overhead:              0-12 cycles (pre-load vs inline)
  Memory/loop/index:                6+ cycles
  TOTAL (optimized):                ~20 cycles/batch
```

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
