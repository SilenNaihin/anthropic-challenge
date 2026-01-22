# VLIW SIMD Optimization Tools

Reference documentation for all analysis and optimization tools.

## Tool Status

| Tool | Status | Location | Priority |
|------|--------|----------|----------|
| [Slot Analyzer](#slot-analyzer) | **Completed** | `tools/slot_analyzer.py` | P0 |
| [VLIW Auto-Packer](#vliw-auto-packer) | **Completed** | `tools/vliw_packer/` | P0 |
| [Dependency Graph](#dependency-graph) | **Completed** | `tools/dependency_graph/` | P0 |
| [Hash Pipeline](#hash-pipeline) | **Completed** | `tools/hash_pipeline/` | P1 |
| [Cycle Profiler](#cycle-profiler) | **Completed** | `tools/cycle_profiler/` | P1 |
| [Memory Analyzer](#memory-analyzer) | **Completed** | `tools/memory_analyzer/` | P1 |
| [Constraint Validator](#constraint-validator) | **Completed** | `tools/constraint_validator/` | P2 |

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

**Status**: Completed | **Folder**: `tools/dependency_graph/`

Builds full dependency DAG for accurate critical path analysis.

### Quick Usage
```bash
python tools/dependency_graph/dependency_graph.py
python tools/dependency_graph/dependency_graph.py --json
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

## Cycle Profiler

**Status**: Completed | **Folder**: `tools/cycle_profiler/`

Breaks down cycles by code section (hash, memory, index calc) to understand WHERE time is spent.

### Quick Usage
```bash
# Basic profiling
python tools/cycle_profiler/cycle_profiler.py

# Full analysis with all features
python tools/cycle_profiler/cycle_profiler.py --all

# JSON output
python tools/cycle_profiler/cycle_profiler.py --json
```

### Features
- Phase tagging (hash, memory_load, memory_store, index_calc, bounds_check, xor_mix, broadcast, flow)
- Per-round breakdown (cycle distribution across rounds)
- Hotspot identification (phases ranked by cycle count)
- Exclusive cycle tracking (cycles where only one phase runs)
- Optimization recommendations based on profile
- Rich colored output

### Key Output
```
HOTSPOTS (phases by cycle count)
--------------------------------
Phase                Cycles   % of Total   Exclusive
hash                    512       64.6%       45.2%
memory_load             234       29.5%       12.3%
index_calc              156       19.7%        8.1%
```

### Optimization Guidance
- hash > 50%? -> Focus on hash pipelining (use hash_pipeline tool)
- load > 30%? -> Consider prefetching or overlapping with compute
- index > 20%? -> Pre-compute addresses, use strength reduction

### Documentation
- Full docs: `tools/cycle_profiler/README.md`
- Quick ref: `tools/cycle_profiler/quickstart.md`

---

## Memory Analyzer

**Status**: Completed | **Folder**: `tools/memory_analyzer/`

Analyzes load/store patterns to identify vectorization opportunities and blockers. Reveals **WHY** `vload`/`vstore` can't be used in certain places.

### Quick Usage
```bash
# Full analysis
python tools/memory_analyzer/memory_analyzer.py

# JSON output
python tools/memory_analyzer/memory_analyzer.py --json

# Verbose with details
python tools/memory_analyzer/memory_analyzer.py --verbose
```

### Features
- Access pattern detection (sequential, strided, scattered, broadcast)
- Stride analysis for consecutive memory accesses
- Vectorization blocker identification
- Address source tracking (constant, linear, computed, indirect)
- Rich colored output
- Recommendations for optimization

### Key Insight

Tree lookups produce **scattered addresses** that CANNOT be vectorized:

```
Element 0: needs tree[42]   \
Element 1: needs tree[17]    > Not consecutive - vload impossible
Element 2: needs tree[891]  /
```

This is inherent to tree traversal where hash results determine next index.

### Key Output
```
Metric                              Count     Elements
------------------------------------------------------------
Scalar Loads                          234          234
Vector Loads (vload)                   45          360
Scattered Loads (tree lookups)        156            -

Vectorization Rate: 67.9%
Sequential Opportunity: ~23 potential vloads
```

### Access Patterns

| Pattern | Vectorizable? | Description |
|---------|--------------|-------------|
| Sequential | Yes | Contiguous addresses (use vload/vstore) |
| Strided | Maybe | Regular stride (gather possible) |
| **Scattered** | **No** | Computed addresses (tree lookups) |
| Broadcast | Use vbroadcast | Same address multiple times |

### Documentation
- Full docs: `tools/memory_analyzer/README.md`
- Quick ref: `tools/memory_analyzer/quickstart.md`

---

## Constraint Validator

**Status**: Completed | **Folder**: `tools/constraint_validator/`

Static constraint checking for VLIW SIMD kernels. Catches errors before slow runtime failures.

### Quick Usage
```bash
# Validate current kernel
python tools/constraint_validator/constraint_validator.py

# JSON output
python tools/constraint_validator/constraint_validator.py --json

# Strict mode (fail on warnings)
python tools/constraint_validator/constraint_validator.py --strict
```

### Features
- Slot limit validation (12 alu, 6 valu, 2 load, 2 store, 1 flow)
- Scratch memory overflow detection
- Same-cycle RAW hazard detection
- Register usage validation (uninitialized reads)
- Rich colored output
- JSON output for scripting
- Strict mode for CI/CD integration

### What It Catches
| Check | Error Type |
|-------|------------|
| Slot limits exceeded | ERROR |
| Scratch address >= 1536 | ERROR |
| Negative scratch address | ERROR |
| Same-cycle RAW hazard | WARNING |
| Scratch usage > 90% | WARNING |
| Uninitialized reads | WARNING |

### Documentation
- Full docs: `tools/constraint_validator/README.md`
- Quick ref: `tools/constraint_validator/quickstart.md`

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
