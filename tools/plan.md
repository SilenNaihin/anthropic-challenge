# Tool Suggestions for VLIW SIMD Optimization Challenge

## Methodology & Reasoning

### Why Build Tools for This Challenge?

The challenge is fundamentally about **optimization in a constrained search space**. Claude Code can reason about code, but lacks:

1. **Rapid feedback loops** - Running `python tests/submission_tests.py` takes time and provides only a single number (cycles). Tools can provide richer, faster feedback.

2. **Visibility into bottlenecks** - The current output is just cycle count. We need to see WHERE cycles are being wasted.

3. **Automated mechanical transformations** - Some optimizations (VLIW packing, loop unrolling) are tedious and error-prone. Tools can handle the bookkeeping.

4. **Constraint validation** - The architecture has strict limits (12 ALU slots, 6 VALU, etc.). Tools can catch violations before runtime.

### Core Insight: The Optimization Funnel

```
Understanding → Analysis → Transformation → Validation
     ↓              ↓            ↓              ↓
  Read ISA      Profile      Pack/Unroll    Test cycles
```

Each stage needs different tools. Most LLM failures happen because:
- They skip analysis and jump to transformation
- They can't validate intermediate steps
- They lose track of dependencies during complex refactors

### Tool Design Principles

1. **Single responsibility** - Each tool does one thing well
2. **JSON/structured output** - LLMs parse structured data better than prose
3. **Incremental** - Support partial analysis, not just full-program
4. **Explanatory** - Include "why" not just "what"

---

## Tool Suggestions

### 1. Cycle Profiler (`profile_cycles.py`)

**Purpose**: Break down where cycles are spent.

**Reasoning**: The baseline is 147,734 cycles. Knowing "60% of cycles are in the hash function" vs "90% are in memory loads" completely changes strategy.

**Output**:
```json
{
  "total_cycles": 147734,
  "breakdown": {
    "hash_computation": {"cycles": 98000, "pct": 66.3},
    "memory_loads": {"cycles": 30000, "pct": 20.3},
    "index_computation": {"cycles": 15000, "pct": 10.1},
    "stores": {"cycles": 4734, "pct": 3.2}
  },
  "per_round_avg": 9233,
  "per_batch_element_avg": 36
}
```

**Implementation approach**: Instrument the Machine class to tag instructions by "phase" and accumulate.

---

### 2. Slot Utilization Analyzer (`slot_utilization.py`)

**Purpose**: Show how many engine slots are used per cycle.

**Reasoning**: VLIW means we have 12 ALU + 6 VALU + 2 load + 2 store + 1 flow = 23 potential slots per cycle. The naive code uses ~1. This tool shows the waste.

**Output**:
```json
{
  "avg_slots_per_cycle": 1.02,
  "max_possible": 23,
  "utilization_pct": 4.4,
  "by_engine": {
    "alu": {"avg": 0.8, "max": 12},
    "valu": {"avg": 0.0, "max": 6},
    "load": {"avg": 0.15, "max": 2},
    "store": {"avg": 0.05, "max": 2},
    "flow": {"avg": 0.02, "max": 1}
  },
  "histogram": {
    "1_slot": 95000,
    "2_slots": 2000,
    "3+_slots": 0
  }
}
```

**Why this matters**: If utilization is 4%, we have ~23x theoretical speedup available. This motivates aggressive packing.

---

### 3. Dependency Graph Builder (`dependency_graph.py`)

**Purpose**: Build a DAG of instruction dependencies (read-after-write, write-after-read).

**Reasoning**: You can only pack instructions that don't depend on each other. The naive code puts everything sequentially even when independent. This tool identifies what CAN be parallel.

**Output**:
```json
{
  "instructions": [
    {"id": 0, "op": "load idx", "deps": [], "can_parallel_with": [1, 2, 3]},
    {"id": 1, "op": "load val", "deps": [], "can_parallel_with": [0, 2, 3]},
    {"id": 2, "op": "xor", "deps": [0, 1], "can_parallel_with": []},
    ...
  ],
  "critical_path_length": 42,
  "parallelism_potential": 8.2
}
```

**Key insight**: The hash function has 6 stages, each with 3 ops. But within a stage, 2 ops are independent. Across batch elements, everything is independent until stores.

---

### 4. VLIW Auto-Packer (`vliw_packer.py`)

**Purpose**: Given a list of instructions and their dependencies, automatically pack into VLIW bundles respecting slot limits.

**Reasoning**: Manual packing is tedious and error-prone. This is a well-defined constraint satisfaction problem.

**Algorithm**:
1. Build dependency graph
2. Topological sort with levels (ASAP scheduling)
3. Bin-pack each level respecting slot limits
4. Output packed instruction bundles

**Why automated**: The search space is huge. 256 batch elements × 16 rounds × ~30 ops per element = ~122,000 operations. Manually scheduling is impractical.

---

### 5. Vectorization Opportunity Finder (`vectorize_finder.py`)

**Purpose**: Identify loops/patterns that can use VALU instead of ALU.

**Reasoning**: VALU processes 8 elements per slot. The batch has 256 elements = 32 vector operations. This is a 8x multiplier on throughput.

**Output**:
```json
{
  "opportunities": [
    {
      "location": "batch_loop",
      "current": "256 scalar iterations",
      "vectorized": "32 vector iterations",
      "speedup_potential": "8x",
      "blockers": ["scattered memory access at idx lookup"],
      "suggestion": "gather indices first, then batch vector ops"
    }
  ]
}
```

**Key challenge**: The tree lookup `mem[forest_values_p + idx]` has non-contiguous addresses. Need gather/scatter or restructuring.

---

### 6. Memory Access Pattern Analyzer (`memory_analyzer.py`)

**Purpose**: Analyze memory access patterns - stride, locality, conflicts.

**Reasoning**: Only 2 load slots per cycle. Memory is often the bottleneck. Understanding access patterns reveals optimization opportunities.

**Output**:
```json
{
  "load_patterns": [
    {"type": "sequential", "count": 512, "can_vectorize": true},
    {"type": "indirect", "count": 4096, "indices_from": "inp_indices"},
    {"type": "constant", "count": 100, "suggestion": "hoist to scratch"}
  ],
  "store_patterns": [...],
  "bandwidth_utilization": 0.15,
  "suggestions": [
    "Batch sequential loads into vload",
    "Prefetch tree nodes based on index pattern"
  ]
}
```

---

### 7. Hash Pipeline Analyzer (`hash_pipeline.py`)

**Purpose**: Specifically analyze the 6-stage hash function for pipelining opportunities.

**Reasoning**: The hash function is the hottest code (called 256×16 = 4096 times). Each stage has internal parallelism. Across elements, stages are independent.

**Analysis**:
```
Stage i for element j depends on:
  - Stage i-1 for element j (data dependency)
  - Nothing else!

So: Stage 0 for elements 0-7 can run in parallel
    Then Stage 1 for elements 0-7 while Stage 0 for elements 8-15
    → Software pipeline
```

**Output**: Pipeline schedule showing which stages for which elements run each cycle.

---

### 8. Constraint Validator (`validate_kernel.py`)

**Purpose**: Check a kernel for constraint violations BEFORE running.

**Reasoning**: Runtime errors are slow feedback. Catch issues statically:
- Slot limit exceeded
- Scratch space overflow
- Invalid instruction encoding
- Dependency violations in same cycle

**Output**:
```json
{
  "valid": false,
  "errors": [
    {"cycle": 42, "error": "13 ALU ops exceed limit of 12"},
    {"cycle": 100, "error": "Write to addr 5 depends on read from addr 5 in same cycle"}
  ]
}
```

---

### 9. Transformation Library (`transforms.py`)

**Purpose**: Common code transformations as functions.

**Transforms**:
- `unroll_loop(kernel, factor)` - Unroll inner loop
- `vectorize_batch(kernel, vlen)` - Convert scalar batch loop to vector
- `pipeline_stages(kernel, depth)` - Software pipeline
- `hoist_invariants(kernel)` - Move loop-invariant loads out
- `pack_instructions(kernel)` - Apply VLIW packing

**Reasoning**: LLMs make mistakes in mechanical transformations. Codify them once, correctly.

---

### 10. Diff-Based Regression Checker (`diff_check.py`)

**Purpose**: When making changes, verify correctness by comparing memory snapshots.

**Reasoning**: The challenge explicitly warns that LLMs cheat by modifying tests. Instead, we compare against the reference kernel's output directly.

**Usage**:
```bash
python diff_check.py --before baseline.py --after optimized.py
# Output: "Memory identical. Cycles: 147734 → 50000 (2.95x speedup)"
```

---

## Implementation Priority

Based on impact vs effort:

### High Priority (Build First)
1. **Slot Utilization Analyzer** - Quick to build, immediately shows waste
2. **Cycle Profiler** - Essential for knowing where to focus
3. **Constraint Validator** - Prevents wasted debug time

### Medium Priority
4. **Dependency Graph Builder** - Enables other tools
5. **VLIW Auto-Packer** - High impact once dependencies known
6. **Memory Access Pattern Analyzer** - Reveals vectorization blockers

### Lower Priority (Build if Needed)
7. **Hash Pipeline Analyzer** - Specialized but important
8. **Vectorization Finder** - Useful but can reason manually
9. **Transformation Library** - Nice-to-have
10. **Diff Checker** - Safety net

---

## Meta-Tool: Optimization Loop Runner

A meta-tool that runs the full optimization loop:

```
while cycles > target:
    profile = profiler.run(kernel)
    bottleneck = profile.worst_section()

    if bottleneck.is_packable():
        kernel = packer.pack(kernel, bottleneck)
    elif bottleneck.is_vectorizable():
        kernel = vectorizer.vectorize(kernel, bottleneck)
    elif bottleneck.is_memory_bound():
        kernel = memory_optimizer.optimize(kernel, bottleneck)

    validator.check(kernel)
    cycles = runner.measure(kernel)
```

This automates the search process itself.

---

## Final Thoughts

The challenge is about **systematic optimization, not heroic insight**. The path from 147,734 to 1,487 cycles (99x speedup) requires:

1. Vectorization: ~8x (use VALU)
2. VLIW packing: ~5-10x (use all slots)
3. Pipelining: ~2-3x (hide latencies)
4. Memory optimization: ~2x (batch loads/stores)

8 × 7 × 2 × 2 = 224x theoretical. Reality is less due to dependencies, but 100x is achievable.

Tools make this systematic rather than ad-hoc.

---

## What The Tests Already Cover (Don't Duplicate)

Looking at `tests/submission_tests.py` and `tests/frozen_problem.py`:

| Already Handled | Our Tools Should NOT |
|-----------------|---------------------|
| Correctness verification (output vs reference) | Build another correctness checker |
| Cycle counting | Just re-count cycles |
| Speed threshold checks | Duplicate threshold logic |
| Frozen simulator copy | Modify the simulator |

**Our tools should focus on**: Analysis, profiling breakdown, optimization assistance, development workflow - things that help UNDERSTAND and TRANSFORM, not verify.

---

## Useful Libraries

### Graph & Dependency Analysis

**NetworkX** (`pip install networkx`)
- Build dependency DAGs
- Topological sort for scheduling
- Critical path analysis
- Find strongly connected components (cycles = bad)

```python
import networkx as nx
G = nx.DiGraph()
G.add_edge("load_idx", "xor", weight=1)
G.add_edge("load_val", "xor", weight=1)
critical_path = nx.dag_longest_path(G)
```

**graphviz** (`pip install graphviz`)
- Visualize instruction dependencies
- Debug why things can't parallelize

---

### Constraint Solving & Optimization

**Google OR-Tools** (`pip install ortools`)
- Optimal instruction scheduling as constraint satisfaction
- Bin-packing for VLIW slots
- Integer linear programming for resource allocation

```python
from ortools.sat.python import cp_model
model = cp_model.CpModel()
# Schedule instruction i at cycle t
schedule = {i: model.NewIntVar(0, max_cycles, f'instr_{i}') for i in instructions}
# Add dependency constraints
for (i, j) in dependencies:
    model.Add(schedule[j] > schedule[i])
```

**Z3** (`pip install z3-solver`)
- SMT solver for complex constraints
- Prove properties about schedules
- Find counterexamples

---

### Performance & Profiling

**Rich** (`pip install rich`)
- Beautiful terminal tables for profiling output
- Progress bars for long optimization runs
- Syntax highlighting for instruction dumps

```python
from rich.table import Table
from rich.console import Console
table = Table(title="Slot Utilization")
table.add_column("Engine", style="cyan")
table.add_column("Used", justify="right")
table.add_column("Max", justify="right")
```

**line_profiler** (`pip install line_profiler`)
- Profile the Python simulator itself (if too slow)

---

### Data Processing

**NumPy** (`pip install numpy`)
- Fast array ops for trace analysis
- Histogram computations
- Statistical analysis of patterns

**Pandas** (`pip install pandas`)
- Tabular analysis of instruction streams
- Groupby for per-engine statistics

---

### Testing & Verification

**Hypothesis** (`pip install hypothesis`)
- Property-based testing for transformations
- Generate random kernels, verify transforms preserve semantics

```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers(0, 100)))
def test_packing_preserves_semantics(instructions):
    packed = pack(instructions)
    assert execute(packed) == execute(instructions)
```

---

## Outside-The-Box Approaches

### 1. DSL + Compiler Approach

Instead of manually writing instructions, define a higher-level DSL:

```python
# High-level
for batch in vectorized(256, 8):
    idx = load(indices[batch])
    val = load(values[batch])
    node = gather(tree, idx)  # Handles scatter/gather
    val = hash_pipeline(val ^ node)
    store(values[batch], val)

# Compiles to optimal VLIW
```

**Why**: Separate "what" from "how". Let the compiler handle scheduling.

**Library**: Could use `lark` or `ply` for parsing, then custom backend.

---

### 2. Genetic Algorithm / Evolutionary Search

Treat instruction ordering as a genome. Evolve toward lower cycles.

```python
# Genome = permutation of instruction order + packing decisions
population = [random_schedule() for _ in range(100)]
for generation in range(1000):
    fitness = [measure_cycles(p) for p in population]
    parents = select_best(population, fitness)
    population = crossover_and_mutate(parents)
```

**Library**: `DEAP` (`pip install deap`)

**Why**: Search space is huge. Evolution explores without getting stuck.

---

### 3. Monte Carlo Tree Search (MCTS)

Used by AlphaGo. Apply to instruction scheduling:

- State = partially scheduled program
- Actions = "schedule instruction X next" or "pack instruction X with current"
- Reward = -cycles (lower is better)

**Library**: Custom or `mcts` package

**Why**: Balances exploration vs exploitation in the search.

---

### 4. LLM-in-the-Loop Optimization

Use Claude API as an optimization oracle:

```python
while cycles > target:
    profile = analyze(kernel)
    prompt = f"""
    Current kernel has {cycles} cycles.
    Profile: {profile}
    Suggest ONE specific optimization.
    """
    suggestion = claude.complete(prompt)
    kernel = apply_suggestion(kernel, suggestion)
```

**Why**: LLMs are good at pattern recognition. Use them for strategy, tools for execution.

**Library**: `anthropic` SDK

---

### 5. Superoptimization (Brute Force)

For small code sequences, enumerate ALL possible instruction orderings:

```python
def superoptimize(instructions):
    best = None
    for perm in itertools.permutations(instructions):
        for packing in all_valid_packings(perm):
            cycles = simulate(packing)
            if best is None or cycles < best[0]:
                best = (cycles, packing)
    return best
```

**Why**: Optimal for small sequences (<10 instructions). Use for hot inner loops.

**Limitation**: Exponential. Only for micro-optimization.

---

### 6. Pattern Database

Catalog known optimization patterns:

```yaml
patterns:
  - name: "parallel_loads"
    match: "load A; load B where no_dependency(A, B)"
    transform: "pack(load A, load B)"

  - name: "vector_batch"
    match: "for i in range(N): scalar_op(arr[i])"
    transform: "for i in range(N/VLEN): vector_op(arr[i*VLEN:(i+1)*VLEN])"
```

Then pattern-match and apply:

```python
for pattern in patterns:
    while match := pattern.find(kernel):
        kernel = pattern.apply(kernel, match)
```

**Why**: Encode human expertise. Reusable across problems.

---

### 7. Incremental / Differential Compilation

Don't recompile everything on each change:

```python
class IncrementalKernel:
    def __init__(self):
        self.cache = {}  # hash(code_section) -> compiled

    def compile(self, section):
        h = hash(section)
        if h not in self.cache:
            self.cache[h] = full_compile(section)
        return self.cache[h]
```

**Why**: Faster iteration. Change one part, only recompile that part.

---

### 8. Symbolic Execution

Track values symbolically to find optimization opportunities:

```python
# Instead of concrete values, track symbolic expressions
idx = Symbol("idx")
val = Symbol("val")
result = hash(val ^ tree[idx])
# Analyze: which operations commute? which are loop-invariant?
```

**Library**: `sympy` for symbolic math

**Why**: Proves properties like "these ops are independent" without running.

---

### 9. Profile-Guided Optimization

Run once to collect a trace, then optimize based on actual behavior:

```python
# Phase 1: Profile
trace = run_with_tracing(kernel)
hot_paths = find_hot_paths(trace)
memory_patterns = analyze_access_patterns(trace)

# Phase 2: Optimize based on profile
for path in hot_paths:
    kernel = optimize_path(kernel, path, memory_patterns)
```

**Why**: Real data beats speculation. Optimize what actually matters.

---

### 10. Constraint Propagation + Arc Consistency

Model the problem as CSP, use arc consistency to prune search:

```python
# Variables: instruction -> cycle assignment
# Constraints: dependencies, slot limits
# Propagate: if instr A must be at cycle 3, and B depends on A, B >= 4

def propagate(assignment, constraints):
    changed = True
    while changed:
        changed = False
        for constraint in constraints:
            if constraint.reduces_domain(assignment):
                changed = True
```

**Why**: Dramatically prunes search space before enumeration.

---

## Hybrid Approach (Recommended)

Combine multiple techniques:

1. **Static analysis** (dependency graph) to find parallelism
2. **Constraint solver** (OR-Tools) for initial schedule
3. **Pattern matching** for known optimizations
4. **Local search** (genetic/MCTS) to refine
5. **LLM** for high-level strategy when stuck

```
┌─────────────────┐
│ High-level DSL  │ ← Human writes this
└────────┬────────┘
         ▼
┌─────────────────┐
│ Pattern Matcher │ ← Apply known transforms
└────────┬────────┘
         ▼
┌─────────────────┐
│ Constraint Solver│ ← Find valid schedule
└────────┬────────┘
         ▼
┌─────────────────┐
│ Local Search    │ ← Refine toward optimal
└────────┬────────┘
         ▼
┌─────────────────┐
│ Validator       │ ← Check constraints
└────────┬────────┘
         ▼
     Run tests
```

---

## Quick Wins (Low Effort, High Value)

1. **Wrap tests with timing**: `time python tests/submission_tests.py` is too coarse. Add per-iteration timing.

2. **Instruction counter by type**: Simple dict counting ALU/VALU/load/store per cycle.

3. **Scratch usage visualizer**: Show which scratch addresses are hot.

4. **Git hook for cycle regression**: Fail commit if cycles increase.

5. **Makefile/script for common workflows**:
```bash
make profile   # Run profiler
make pack      # Auto-pack instructions
make test      # Run submission tests
make trace     # Generate Perfetto trace
```
