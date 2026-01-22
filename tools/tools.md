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
| [Transforms](#transforms) | **Completed** | `tools/transforms/` | P2 |
| [Kernel Diff](#kernel-diff) | **Completed** | `tools/kernel_diff/` | P2 |
| [Optimization Loop](#optimization-loop) | **Completed** | `tools/optimization_loop/` | P3 |

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

## Transforms

**Status**: Completed | **Folder**: `tools/transforms/`

Codified transformations to reduce manual errors in mechanical transforms for VLIW SIMD optimization.

### Quick Usage
```bash
# Analyze kernel for transformation opportunities
python tools/transforms/transforms.py

# Run demonstration
python tools/transforms/transforms.py --demo

# JSON output
python tools/transforms/transforms.py --json
```

### Features
- **Loop unrolling** - Replicate loop body N times with automatic register renaming
- **Vectorize batch** - Convert scalar ops to vector ops (VLEN=8)
- **Software pipelining** - Overlap iteration N+1 prep with N execution
- **Hoist invariants** - Move loop-invariant code outside loops
- Hash-specific helpers (generate vectorized hash stages)
- Analysis tools for identifying transform opportunities
- Rich colored output
- JSON output for scripting

### The Four Transforms

| Transform | Purpose | Typical Speedup |
|-----------|---------|-----------------|
| Loop Unroll | Reduce loop overhead, expose ILP | 1.2-2x |
| Vectorize | VLEN=8 elements per op | Up to 8x |
| Software Pipeline | Hide latency via overlap | 1.5-2x |
| Hoist Invariants | Remove redundant computation | 1.1-1.5x |

### Example: Unroll + Pack Workflow
```python
from tools.transforms.transforms import unroll_loop, hoist_invariants
from tools.vliw_packer.vliw_packer import pack_kernel

# 1. Hoist invariants first
hoist_result = hoist_invariants(loop_body, iterations=16)

# 2. Unroll remaining body
unroll_result = unroll_loop(hoist_result.transformed_code, unroll_factor=4)

# 3. Pack the result
packed, stats = pack_kernel(unroll_result.transformed_code)
print(f"Speedup: {stats.speedup:.2f}x")
```

### Hash Helper
```python
from tools.transforms.transforms import generate_vectorized_hash_stage

# Generates 2-cycle hash stage using tmp1 || tmp2 ILP
instrs = generate_vectorized_hash_stage(0, val_base, tmp1, tmp2, const1, const3)
```

### Documentation
- Full docs: `tools/transforms/README.md`
- Quick ref: `tools/transforms/quickstart.md`

---

## Kernel Diff

**Status**: Completed | **Folder**: `tools/kernel_diff/`

Compare two kernel versions to track optimization impact. Essential for understanding what changed between optimization iterations.

### Quick Usage
```bash
# Compare two kernels
python tools/kernel_diff/kernel_diff.py kernel1.json kernel2.json

# Compare current vs baseline
python tools/kernel_diff/kernel_diff.py --baseline baseline.json

# Save current kernel
python tools/kernel_diff/kernel_diff.py --save current.json
```

### Features
- Cycle comparison (before/after cycles, speedup)
- Utilization diff (slot usage changes per engine)
- Instruction diff (operations added, removed, modified)
- Side-by-side comparison view
- Rich colored output
- JSON output for scripting
- Baseline comparison mode

### Key Output
```
SUMMARY: IMPROVEMENT: 1.45x faster (-310 cycles)

CYCLE COMPARISON
  Before:  1000 cycles
  After:   690 cycles
  Speedup: 1.45x FASTER

PER-ENGINE CHANGES
Engine        Before       After       Delta   Util Delta
alu            1,234       1,456        +222       +2.3%
valu             567         890        +323      +12.5%
```

### Workflow
1. Save baseline: `python tools/kernel_diff/kernel_diff.py --save v1.json`
2. Make optimization changes
3. Compare: `python tools/kernel_diff/kernel_diff.py --baseline v1.json`
4. If improvement, save new version

### Documentation
- Full docs: `tools/kernel_diff/README.md`
- Quick ref: `tools/kernel_diff/quickstart.md`

---

## Optimization Loop

**Status**: Completed | **Folder**: `tools/optimization_loop/`

Meta-tool that automates the profile->analyze->transform->validate optimization loop. Orchestrates all analysis tools to provide a comprehensive view.

### Quick Usage
```bash
# Full optimization loop
python tools/optimization_loop/optimize.py

# Quick analysis (skip validation)
python tools/optimization_loop/optimize.py --suggest

# Profile only
python tools/optimization_loop/optimize.py --profile

# JSON output
python tools/optimization_loop/optimize.py --json
```

### Features
- Automated profiling (runs slot_analyzer, dependency_graph, cycle_profiler)
- Bottleneck detection (slot utilization, dependencies, memory/hash bound)
- Transform suggestions (packing, pipelining, vectorization)
- Regression checking (validates correctness, measures cycles)
- Rich colored output
- JSON output for scripting
- Dry-run mode (preview without running tools)

### Bottleneck Types Detected
| Type | Description |
|------|-------------|
| `slot_utilization` | Low overall slot usage |
| `dependency_chain` | Long critical path |
| `memory_bound` | Load/store operations dominate |
| `hash_bound` | Hash computation dominates |
| `engine_imbalance` | Some engines saturated, others idle |

### Key Output
```
BOTTLENECKS DETECTED (2)
1. [HIGH] dependency_chain
   Long dependency chains. 94.7% of cycles could be eliminated.

SUGGESTED TRANSFORMS (3)
1. P1 [HARD] Break dependency chains
   Potential: 18.86x theoretical speedup

2. P2 [EASY] Pack more instructions per cycle
   Potential: Up to 55% cycle reduction
```

### Typical Workflow
```bash
# 1. Get suggestions
python tools/optimization_loop/optimize.py --suggest

# 2. Apply top priority, easiest transform

# 3. Verify improvement
python tools/optimization_loop/optimize.py
```

### Documentation
- Full docs: `tools/optimization_loop/README.md`
- Quick ref: `tools/optimization_loop/quickstart.md`

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
