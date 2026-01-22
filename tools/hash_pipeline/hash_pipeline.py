#!/usr/bin/env python3
"""
Hash Pipeline Analyzer for VLIW SIMD Optimization

Analyzes the 6-stage hash function dependency structure and generates
optimal pipeline schedules for maximum throughput.

The hash function runs 4096 times (256 batch x 16 rounds) - this is THE
hottest code path. Each stage has internal parallelism that can be exploited.

IMPORTANT DISCLAIMERS:
- This is an ILP analysis tool, NOT a cycle-accurate simulator
- Cycle counts shown are THEORETICAL LOWER BOUNDS, not achievable targets
- Real implementations will have additional overhead from:
  * vbroadcast operations for constants
  * Memory loads/stores for input/output
  * Loop control and flow operations
  * Register pressure and potential spills
- Always validate against slot_analyzer.py on actual instruction streams

Usage:
    python hash_pipeline.py                    # Full analysis with defaults
    python hash_pipeline.py --elements 8       # Analyze pipeline for 8 elements
    python hash_pipeline.py --visualize        # Show cycle-by-cycle schedule
    python hash_pipeline.py --codegen          # Generate VLIW code hints
    python hash_pipeline.py --compare          # Compare pipelined vs sequential
    python hash_pipeline.py --realistic        # Show realistic estimates with overhead
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Architecture constants (from problem.py)
SLOT_LIMITS = {
    "alu": 12,
    "valu": 6,
    "load": 2,
    "store": 2,
    "flow": 1,
}
VLEN = 8

# Hash stages from problem.py
HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),
    ("^", 0xC761C23C, "^", ">>", 19),
    ("+", 0x165667B1, "+", "<<", 5),
    ("+", 0xD3A2646C, "^", "<<", 9),
    ("+", 0xFD7046C5, "+", "<<", 3),
    ("^", 0xB55A4F09, "^", ">>", 16),
]


class OpType(Enum):
    """Operation types for scheduling"""
    TMP1 = "tmp1"           # a op1 const1
    TMP2 = "tmp2"           # a op3 const3 (shift)
    COMBINE = "combine"     # tmp1 op2 tmp2
    VBROADCAST = "vbcast"   # vbroadcast constant (overhead not in ideal model)


@dataclass
class Operation:
    """Single operation in the hash pipeline"""
    stage: int          # Which hash stage (0-5)
    element: int        # Which element being processed
    op_type: OpType     # tmp1, tmp2, or combine
    op: str             # Actual operator (+, ^, <<, >>)
    slot_type: str      # "alu", "valu", or "const"
    dependencies: list  # List of (stage, element, op_type) this depends on
    cycle: int = -1     # Scheduled cycle (-1 = unscheduled)

    @property
    def id(self) -> tuple:
        return (self.stage, self.element, self.op_type)

    def __repr__(self):
        return f"S{self.stage}E{self.element}:{self.op_type.value}"


@dataclass
class StageAnalysis:
    """Analysis of a single hash stage"""
    stage_num: int
    op1: str        # First operation (a op1 const)
    const1: int     # First constant
    op2: str        # Combine operation
    op3: str        # Shift operation
    shift_amount: int

    @property
    def tmp1_ops(self) -> int:
        """Number of ALU operations for tmp1 (a op1 const)"""
        # Need: load const, then op
        return 1  # Just the op, const can be pre-loaded

    @property
    def tmp2_ops(self) -> int:
        """Number of ALU operations for tmp2 (a op3 shift_amount)"""
        return 1  # Just the shift

    @property
    def combine_ops(self) -> int:
        """Number of ALU operations for combine (tmp1 op2 tmp2)"""
        return 1

    @property
    def can_parallelize_tmp(self) -> bool:
        """Whether tmp1 and tmp2 can run in parallel"""
        # Both only depend on 'a', so always parallelizable
        return True


@dataclass
class PipelineSchedule:
    """Complete pipeline schedule for hash computation"""
    n_elements: int
    n_stages: int = 6
    operations: list = field(default_factory=list)
    cycle_schedule: dict = field(default_factory=dict)  # cycle -> list of operations
    total_cycles: int = 0

    def schedule_operation(self, op: Operation, cycle: int):
        """Schedule an operation at a specific cycle"""
        op.cycle = cycle
        if cycle not in self.cycle_schedule:
            self.cycle_schedule[cycle] = []
        self.cycle_schedule[cycle].append(op)
        self.total_cycles = max(self.total_cycles, cycle + 1)


class HashPipelineAnalyzer:
    """Analyzes and schedules the hash function pipeline"""

    def __init__(self, n_elements: int = 8, use_vector: bool = True):
        self.n_elements = n_elements
        self.use_vector = use_vector and n_elements >= VLEN
        self.stages = self._analyze_stages()

    def _analyze_stages(self) -> list[StageAnalysis]:
        """Analyze each hash stage"""
        return [
            StageAnalysis(
                stage_num=i,
                op1=stage[0],
                const1=stage[1],
                op2=stage[2],
                op3=stage[3],
                shift_amount=stage[4]
            )
            for i, stage in enumerate(HASH_STAGES)
        ]

    def print_stage_analysis(self):
        """Print detailed analysis of each stage"""
        print("\n" + "=" * 70)
        print("HASH STAGE DEPENDENCY ANALYSIS")
        print("=" * 70)

        for s in self.stages:
            print(f"\nStage {s.stage_num}:")
            print(f"  Formula: a = (a {s.op1} 0x{s.const1:08X}) {s.op2} (a {s.op3} {s.shift_amount})")
            print(f"  tmp1: a {s.op1} const     (1 ALU op)")
            print(f"  tmp2: a {s.op3} {s.shift_amount:<2}        (1 ALU op)")
            print(f"  combine: tmp1 {s.op2} tmp2 (1 ALU op)")
            print(f"  Parallelism: tmp1 || tmp2  (2-way ILP within stage)")

        print("\n" + "-" * 70)
        print("KEY INSIGHT: Within each stage, tmp1 and tmp2 are INDEPENDENT")
        print("             They both only read 'a', don't write until combine")
        print("-" * 70)

    def build_dependency_graph(self) -> list[Operation]:
        """Build all operations with their dependencies"""
        operations = []

        for elem in range(self.n_elements):
            for stage_idx, stage in enumerate(self.stages):
                # tmp1: depends on previous stage's combine (or input for stage 0)
                tmp1_deps = []
                if stage_idx > 0:
                    tmp1_deps.append((stage_idx - 1, elem, OpType.COMBINE))

                tmp1 = Operation(
                    stage=stage_idx,
                    element=elem,
                    op_type=OpType.TMP1,
                    op=stage.op1,
                    slot_type="valu" if self.use_vector else "alu",
                    dependencies=tmp1_deps
                )

                # tmp2: same dependencies as tmp1 (both read 'a')
                tmp2_deps = list(tmp1_deps)  # Same deps
                tmp2 = Operation(
                    stage=stage_idx,
                    element=elem,
                    op_type=OpType.TMP2,
                    op=stage.op3,
                    slot_type="valu" if self.use_vector else "alu",
                    dependencies=tmp2_deps
                )

                # combine: depends on this stage's tmp1 and tmp2
                combine = Operation(
                    stage=stage_idx,
                    element=elem,
                    op_type=OpType.COMBINE,
                    op=stage.op2,
                    slot_type="valu" if self.use_vector else "alu",
                    dependencies=[
                        (stage_idx, elem, OpType.TMP1),
                        (stage_idx, elem, OpType.TMP2)
                    ]
                )

                operations.extend([tmp1, tmp2, combine])

        return operations

    def schedule_sequential(self) -> PipelineSchedule:
        """Generate a naive sequential schedule (baseline)"""
        schedule = PipelineSchedule(n_elements=self.n_elements)
        operations = self.build_dependency_graph()
        schedule.operations = operations

        cycle = 0
        for elem in range(self.n_elements):
            for stage_idx in range(6):
                # Find the 3 ops for this element/stage
                elem_stage_ops = [op for op in operations
                                   if op.element == elem and op.stage == stage_idx]

                # tmp1 and tmp2 could be parallel but we're being naive
                for op in sorted(elem_stage_ops, key=lambda o: o.op_type.value):
                    schedule.schedule_operation(op, cycle)
                    cycle += 1

        return schedule

    def schedule_within_stage_parallel(self) -> PipelineSchedule:
        """Schedule with intra-stage parallelism (tmp1 || tmp2)"""
        schedule = PipelineSchedule(n_elements=self.n_elements)
        operations = self.build_dependency_graph()
        schedule.operations = operations

        cycle = 0
        for elem in range(self.n_elements):
            for stage_idx in range(6):
                # Find ops for this element/stage
                elem_stage_ops = {
                    op.op_type: op for op in operations
                    if op.element == elem and op.stage == stage_idx
                }

                # Cycle N: tmp1 and tmp2 in parallel
                schedule.schedule_operation(elem_stage_ops[OpType.TMP1], cycle)
                schedule.schedule_operation(elem_stage_ops[OpType.TMP2], cycle)
                cycle += 1

                # Cycle N+1: combine
                schedule.schedule_operation(elem_stage_ops[OpType.COMBINE], cycle)
                cycle += 1

        return schedule

    def schedule_software_pipelined(self) -> PipelineSchedule:
        """
        Generate optimal software-pipelined schedule.

        Key insight: Stage N for element B doesn't depend on stage N for element A.
        We can interleave elements to fill pipeline bubbles.

        Optimal pattern (for 2+ elements):
        Cycle 0: [S0.E0: tmp1,tmp2]
        Cycle 1: [S0.E0: combine] [S0.E1: tmp1,tmp2]
        Cycle 2: [S1.E0: tmp1,tmp2] [S0.E1: combine] [S0.E2: tmp1,tmp2]
        ...etc
        """
        schedule = PipelineSchedule(n_elements=self.n_elements)
        operations = self.build_dependency_graph()
        schedule.operations = operations

        # Build lookup for ops
        op_map = {op.id: op for op in operations}

        # Track when each operation can start (based on dependencies)
        ready_time = {}  # op.id -> earliest cycle

        def get_ready_time(op: Operation) -> int:
            """Get earliest cycle this op can execute"""
            if not op.dependencies:
                return 0
            max_dep_finish = 0
            for dep in op.dependencies:
                if dep in ready_time:
                    max_dep_finish = max(max_dep_finish, ready_time[dep] + 1)
            return max_dep_finish

        # Track slot usage per cycle
        slot_usage = {}  # cycle -> count
        max_parallel = SLOT_LIMITS["valu"] if self.use_vector else SLOT_LIMITS["alu"]

        # Schedule in dependency order
        # Priority: earlier stages first, then earlier elements
        unscheduled = sorted(operations, key=lambda o: (o.stage, o.element, o.op_type.value))

        # Group tmp1/tmp2 for same stage/element (they should be parallel)
        scheduled_set = set()

        while unscheduled:
            op = unscheduled.pop(0)
            if op.id in scheduled_set:
                continue

            earliest = get_ready_time(op)

            # If this is tmp1 or tmp2, try to schedule with its pair
            if op.op_type in (OpType.TMP1, OpType.TMP2):
                pair_type = OpType.TMP2 if op.op_type == OpType.TMP1 else OpType.TMP1
                pair_id = (op.stage, op.element, pair_type)
                pair_op = op_map.get(pair_id)

                if pair_op and pair_id not in scheduled_set:
                    pair_earliest = get_ready_time(pair_op)
                    earliest = max(earliest, pair_earliest)

                    # Find cycle with room for both
                    cycle = earliest
                    while slot_usage.get(cycle, 0) + 2 > max_parallel:
                        cycle += 1

                    schedule.schedule_operation(op, cycle)
                    schedule.schedule_operation(pair_op, cycle)
                    scheduled_set.add(op.id)
                    scheduled_set.add(pair_id)
                    ready_time[op.id] = cycle
                    ready_time[pair_id] = cycle
                    slot_usage[cycle] = slot_usage.get(cycle, 0) + 2
                    continue

            # Schedule single op
            cycle = earliest
            while slot_usage.get(cycle, 0) + 1 > max_parallel:
                cycle += 1

            schedule.schedule_operation(op, cycle)
            scheduled_set.add(op.id)
            ready_time[op.id] = cycle
            slot_usage[cycle] = slot_usage.get(cycle, 0) + 1

        return schedule

    def schedule_maximally_pipelined(self) -> PipelineSchedule:
        """
        Most aggressive pipelining - pack as many ops as architecture allows.

        With 6 VALU slots (or 12 ALU slots), we can potentially execute
        multiple stages' operations in a single cycle.
        """
        schedule = PipelineSchedule(n_elements=self.n_elements)
        operations = self.build_dependency_graph()
        schedule.operations = operations

        op_map = {op.id: op for op in operations}
        ready_time = {}
        scheduled = set()

        max_slots = SLOT_LIMITS["valu"] if self.use_vector else SLOT_LIMITS["alu"]

        def get_ready_time(op: Operation) -> int:
            if not op.dependencies:
                return 0
            return max(ready_time.get(dep, -1) + 1 for dep in op.dependencies)

        cycle = 0
        remaining = set(op.id for op in operations)

        while remaining:
            # Find all ops ready to execute this cycle
            ready_ops = []
            for op_id in remaining:
                op = op_map[op_id]
                if get_ready_time(op) <= cycle:
                    ready_ops.append(op)

            # Sort by priority: combine ops first (to release dependencies faster),
            # then by stage (lower first), then by element
            ready_ops.sort(key=lambda o: (
                0 if o.op_type == OpType.COMBINE else 1,
                o.stage,
                o.element
            ))

            # Schedule as many as fit in slot limit
            scheduled_this_cycle = 0
            for op in ready_ops:
                if scheduled_this_cycle >= max_slots:
                    break

                # For tmp1/tmp2, try to schedule together
                if op.op_type in (OpType.TMP1, OpType.TMP2) and op.id in remaining:
                    pair_type = OpType.TMP2 if op.op_type == OpType.TMP1 else OpType.TMP1
                    pair_id = (op.stage, op.element, pair_type)

                    if pair_id in remaining and scheduled_this_cycle + 2 <= max_slots:
                        pair_op = op_map[pair_id]
                        if get_ready_time(pair_op) <= cycle:
                            schedule.schedule_operation(op, cycle)
                            schedule.schedule_operation(pair_op, cycle)
                            ready_time[op.id] = cycle
                            ready_time[pair_id] = cycle
                            remaining.remove(op.id)
                            remaining.remove(pair_id)
                            scheduled_this_cycle += 2
                            continue

                # Schedule single op
                if op.id in remaining:
                    schedule.schedule_operation(op, cycle)
                    ready_time[op.id] = cycle
                    remaining.remove(op.id)
                    scheduled_this_cycle += 1

            cycle += 1

        return schedule

    def visualize_schedule(self, schedule: PipelineSchedule, max_cycles: int = 30):
        """Print visual representation of the schedule"""
        print("\n" + "=" * 80)
        print("PIPELINE SCHEDULE VISUALIZATION")
        print(f"Elements: {schedule.n_elements}, Total Cycles: {schedule.total_cycles}")
        print("=" * 80)

        show_cycles = min(schedule.total_cycles, max_cycles)

        for cycle in range(show_cycles):
            ops = schedule.cycle_schedule.get(cycle, [])
            if not ops:
                continue

            # Group by stage
            by_stage = {}
            for op in ops:
                key = f"S{op.stage}"
                if key not in by_stage:
                    by_stage[key] = []
                by_stage[key].append(op)

            line_parts = []
            for stage_key in sorted(by_stage.keys()):
                stage_ops = by_stage[stage_key]
                op_strs = []
                for op in stage_ops:
                    if op.op_type == OpType.COMBINE:
                        op_strs.append(f"E{op.element}:comb")
                    else:
                        op_strs.append(f"E{op.element}:{op.op_type.value}")
                line_parts.append(f"[{stage_key}: {', '.join(op_strs)}]")

            print(f"Cycle {cycle:3d}: {' '.join(line_parts)}")

        if schedule.total_cycles > max_cycles:
            print(f"  ... ({schedule.total_cycles - max_cycles} more cycles)")

        print()

    def calculate_theoretical_minimum(self) -> dict:
        """Calculate theoretical minimum cycles for different scenarios"""
        n = self.n_elements

        # Each element needs 6 stages x 3 ops = 18 ops
        total_ops = n * 6 * 3

        # But tmp1 || tmp2, so effectively 6 stages x 2 cycles = 12 cycles per element
        # if we had unlimited parallelism across elements

        # Critical path for ONE element: 6 stages, each needs 2 cycles (tmp||tmp, then combine)
        critical_path_one = 6 * 2  # 12 cycles

        # With pipelining, after filling the pipeline:
        # - Pipeline depth is 12 cycles (6 stages * 2 cycles each)
        # - Steady state: 1 element completes every 2 cycles (after each combine)
        # Actually better: can interleave so 1 element starts per cycle in steady state

        # Theoretical best with n elements:
        # Start all n pipelines, they progress through 12-cycle pipeline
        # Best case: critical_path + (n-1) * cycles_between_elements

        # With 6 VALU slots: can do 3 pairs of (tmp1,tmp2) or 6 combines per cycle
        max_slots = SLOT_LIMITS["valu"]

        # Minimum ops per cycle: max_slots
        # But dependency limited: can only do combines after tmp1/tmp2

        # Realistic minimum:
        # - Critical path = 12 cycles (sequential dependency within one element)
        # - Parallelism across elements can fill bubbles
        # - Best: ~12 + ceil(n/slots_per_stage) additional

        results = {
            "elements": n,
            "total_operations": total_ops,
            "ops_per_element": 18,
            "critical_path_one_element": critical_path_one,
            "sequential_cycles": n * critical_path_one,
            "max_valu_slots": max_slots,
        }

        # With perfect pipelining:
        # After 12 cycles, pipeline is full
        # Then each additional element adds ~0 cycles if enough slots
        # But limited by combine ops (bottleneck)
        # Each cycle can do 6 VALUs = 3 combines, so throughput ~3 elements per 2 cycles

        # Pipeline math:
        # Fill time = 12 cycles (critical path)
        # Drain time = 12 cycles
        # Work = n elements * 12 effective ops
        # Throughput = 6 VALU/cycle = 3 op-pairs/cycle
        # So n elements * 12 ops / 6 per cycle = 2*n cycles minimum (no overlap)

        # With pipeline overlap:
        # theoretical_min = critical_path + (n-1) * throughput_rate
        # where throughput_rate = time to start next element's first op

        # Actually: with 6 VALU, 3 (tmp1,tmp2) pairs per cycle
        # or 6 individual ops per cycle
        # So minimum = ceil(total_ops / 6) = ceil(18n / 6) = 3n cycles

        results["theoretical_min_cycles"] = max(critical_path_one, (total_ops + max_slots - 1) // max_slots)
        results["throughput_limited_min"] = (total_ops + max_slots - 1) // max_slots
        results["latency_limited_min"] = critical_path_one

        return results

    def calculate_realistic_estimate(self) -> dict:
        """
        Calculate more realistic cycle estimates including overhead.

        This accounts for operations the idealized model ignores:
        - vbroadcast for constants (2 per stage if not pre-loaded)
        - Memory operations (load input, store output)
        - Loop control overhead
        - Register pressure effects
        """
        n = self.n_elements
        n_stages = 6

        # Base hash operations (idealized)
        ideal = self.calculate_theoretical_minimum()

        # Additional overhead per hash call:
        # 1. vbroadcast: 2 per stage (const and shift) if done inline
        #    With pre-loading: 0 (constants loaded once before loop)
        #    Conservative: assume 1 cycle to load from pre-broadcasted constant
        vbroadcast_overhead_inline = n_stages * 2  # 12 vbroadcasts per hash
        vbroadcast_overhead_preload = 0  # if constants pre-loaded

        # 2. Memory operations per batch of VLEN elements:
        #    - 1 vload for input values
        #    - 1 vload for node values (tree lookup)
        #    - 1 vstore for output values
        #    - 1 vload/vstore for indices
        #    These compete with 2 load/store slots
        memory_ops_per_batch = 4  # vload input, vload node, vstore output, vload/store idx

        # 3. Loop control: jump, counter update, etc.
        #    ~2-3 cycles per batch iteration (with good packing)
        loop_overhead_per_batch = 2

        # 4. XOR with node value before hash (1 VALU op)
        xor_overhead = 1

        # 5. Index calculation after hash (branch decision, multiply, add)
        index_calc_overhead = 3

        # Calculate per-batch cycles (batch = VLEN elements)
        batches = (n + VLEN - 1) // VLEN

        # Hash cycles per batch (using vectorized ops)
        # 6 stages * 3 ops = 18 ops, but tmp1||tmp2 so 12 effective cycles
        # With 6 VALU slots, we need ceil(12/6) = 2 cycles minimum per stage
        # But dependency limited: 2 cycles per stage (tmp||tmp, then combine)
        hash_cycles_per_batch = n_stages * 2  # 12 cycles

        # Add overhead
        total_per_batch_conservative = (
            hash_cycles_per_batch +
            vbroadcast_overhead_inline +
            xor_overhead +
            index_calc_overhead +
            loop_overhead_per_batch
        )

        total_per_batch_optimized = (
            hash_cycles_per_batch +
            vbroadcast_overhead_preload +
            xor_overhead +
            index_calc_overhead +
            loop_overhead_per_batch
        )

        # Memory ops can often be overlapped with ALU
        # But they're limited to 2 slots/cycle
        memory_cycles = (memory_ops_per_batch + 1) // 2

        # Full kernel: 256 batch * 16 rounds = 4096 hash calls
        # But with VLEN=8, that's 512 batch iterations (256/8 * 16 rounds... wait)
        # Actually: 256 elements * 16 rounds = 4096 element-iterations
        # With VLEN=8: 4096/8 = 512 vector iterations
        total_hash_iterations = 4096
        vector_iterations = total_hash_iterations // VLEN

        return {
            "elements_analyzed": n,
            "ideal_cycles_per_element": ideal["theoretical_min_cycles"] / n if n > 0 else 0,

            "per_batch_breakdown": {
                "hash_computation": hash_cycles_per_batch,
                "vbroadcast_inline": vbroadcast_overhead_inline,
                "vbroadcast_preload": vbroadcast_overhead_preload,
                "xor_with_node": xor_overhead,
                "index_calculation": index_calc_overhead,
                "loop_overhead": loop_overhead_per_batch,
                "memory_ops": memory_cycles,
            },

            "per_batch_total_conservative": total_per_batch_conservative + memory_cycles,
            "per_batch_total_optimized": total_per_batch_optimized + memory_cycles,

            "full_kernel": {
                "total_hash_calls": total_hash_iterations,
                "vector_iterations": vector_iterations,
                "cycles_conservative": vector_iterations * (total_per_batch_conservative + memory_cycles),
                "cycles_optimized": vector_iterations * (total_per_batch_optimized + memory_cycles),
                "cycles_ideal_unreachable": vector_iterations * hash_cycles_per_batch,
            },

            "notes": [
                "Conservative: inline vbroadcast each stage",
                "Optimized: pre-broadcast constants before loop",
                "Ideal: pure hash ops only, no overhead (UNREACHABLE)",
                "Memory ops assumed to partially overlap with compute",
                "Does not account for register spills or scratch limits",
            ]
        }

    def generate_code_hints(self) -> list[str]:
        """Generate VLIW assembly hints for optimal implementation"""
        hints = []
        hints.append("=" * 70)
        hints.append("VLIW CODE GENERATION HINTS")
        hints.append("=" * 70)

        # Constant preloading
        hints.append("\n1. PRELOAD CONSTANTS (once before loop):")
        for i, stage in enumerate(self.stages):
            hints.append(f"   const_{i}: 0x{stage.const1:08X}  ; Stage {i} constant")
            hints.append(f"   shift_{i}: {stage.shift_amount}            ; Stage {i} shift amount")

        # Register allocation suggestion
        hints.append("\n2. REGISTER ALLOCATION (for VLEN=8 vectorized):")
        hints.append("   a_vec[0:7]:    Current hash values (8 elements)")
        hints.append("   tmp1_vec[0:7]: First partial result")
        hints.append("   tmp2_vec[0:7]: Second partial result (shift)")
        hints.append("   const_vec[0:7]: Broadcast constant (reuse)")
        hints.append("   shift_vec[0:7]: Broadcast shift amount (reuse)")

        # Optimal loop structure
        hints.append("\n3. OPTIMAL LOOP STRUCTURE (per stage):")
        hints.append("   Cycle N:")
        hints.append("     valu: vbroadcast const_vec, const_i")
        hints.append("     valu: vbroadcast shift_vec, shift_i")
        hints.append("   Cycle N+1:")
        hints.append("     valu: tmp1 = a_vec OP1 const_vec  ; e.g., + or ^")
        hints.append("     valu: tmp2 = a_vec OP3 shift_vec  ; << or >>")
        hints.append("   Cycle N+2:")
        hints.append("     valu: a_vec = tmp1 OP2 tmp2       ; combine")

        # Software pipelining hint
        hints.append("\n4. SOFTWARE PIPELINING (advanced):")
        hints.append("   Process multiple batches of 8 elements simultaneously.")
        hints.append("   While batch A is at stage 3, batch B can be at stage 2.")
        hints.append("   This hides latency and keeps VALU units busy.")
        hints.append("")
        hints.append("   Example with 2 batches (16 elements total):")
        hints.append("   Cycle 0: [S0.A: broadcast const0]")
        hints.append("   Cycle 1: [S0.A: tmp1,tmp2] [S0.B: broadcast const0]")
        hints.append("   Cycle 2: [S0.A: combine] [S0.B: tmp1,tmp2]")
        hints.append("   Cycle 3: [S1.A: broadcast const1] [S0.B: combine]")
        hints.append("   ... etc")

        # Optimization opportunities
        hints.append("\n5. ADVANCED OPTIMIZATIONS:")
        hints.append("   a) Pre-compute all constants at kernel start")
        hints.append("   b) Use multiply_add if available for + operations")
        hints.append("   c) Consider lookup tables for certain hash patterns")
        hints.append("   d) Loop unrolling: unroll 2-4 stages together")
        hints.append("   e) Register renaming to avoid false dependencies")

        return hints

    def print_comparison(self):
        """Compare different scheduling strategies"""
        print("\n" + "=" * 70)
        print("SCHEDULING STRATEGY COMPARISON")
        print("=" * 70)

        seq = self.schedule_sequential()
        intra = self.schedule_within_stage_parallel()
        pipelined = self.schedule_software_pipelined()
        max_pipe = self.schedule_maximally_pipelined()

        theory = self.calculate_theoretical_minimum()

        print(f"\nElements: {self.n_elements}")
        print(f"Total operations: {theory['total_operations']}")
        print()
        print(f"{'Strategy':<35} {'Cycles':>10} {'Speedup':>10}")
        print("-" * 60)
        print(f"{'Sequential (baseline)':<35} {seq.total_cycles:>10} {1.0:>10.2f}x")
        print(f"{'Intra-stage parallel (tmp1||tmp2)':<35} {intra.total_cycles:>10} {seq.total_cycles/intra.total_cycles:>10.2f}x")
        print(f"{'Software pipelined':<35} {pipelined.total_cycles:>10} {seq.total_cycles/pipelined.total_cycles:>10.2f}x")
        print(f"{'Maximally pipelined':<35} {max_pipe.total_cycles:>10} {seq.total_cycles/max_pipe.total_cycles:>10.2f}x")
        print("-" * 60)
        print(f"{'Theoretical minimum':<35} {theory['theoretical_min_cycles']:>10}")
        print(f"  (limited by: {'throughput' if theory['throughput_limited_min'] >= theory['latency_limited_min'] else 'latency'})")
        print()

        return {
            "sequential": seq.total_cycles,
            "intra_parallel": intra.total_cycles,
            "pipelined": pipelined.total_cycles,
            "max_pipelined": max_pipe.total_cycles,
            "theoretical_min": theory["theoretical_min_cycles"]
        }

    def analyze_batch_sizes(self) -> dict:
        """Analyze different batch sizes for optimal pipelining"""
        print("\n" + "=" * 70)
        print("BATCH SIZE ANALYSIS")
        print("=" * 70)
        print("\nHow many elements to process together for best efficiency?\n")

        results = []
        print(f"{'Elements':>10} {'Max Pipe':>12} {'Cycles/Elem':>14} {'Efficiency':>12}")
        print("-" * 52)

        for n_elem in [1, 2, 4, 8, 16, 32, 64]:
            analyzer = HashPipelineAnalyzer(n_elements=n_elem, use_vector=self.use_vector)
            schedule = analyzer.schedule_maximally_pipelined()
            theory = analyzer.calculate_theoretical_minimum()

            cycles_per_elem = schedule.total_cycles / n_elem
            efficiency = theory["theoretical_min_cycles"] / schedule.total_cycles * 100

            results.append({
                "elements": n_elem,
                "cycles": schedule.total_cycles,
                "cycles_per_element": cycles_per_elem,
                "theoretical_min": theory["theoretical_min_cycles"],
                "efficiency": efficiency
            })

            print(f"{n_elem:>10} {schedule.total_cycles:>12} {cycles_per_elem:>14.2f} {efficiency:>11.1f}%")

        print("\nRecommendation: Process in batches of 8 (VLEN) or 16 for best efficiency")
        return results

    def print_realistic_estimate(self):
        """Print realistic cycle estimates with overhead breakdown"""
        est = self.calculate_realistic_estimate()

        print("\n" + "=" * 70)
        print("REALISTIC CYCLE ESTIMATES (with overhead)")
        print("=" * 70)
        print("\n*** DISCLAIMER: These are estimates, not guarantees ***")
        print("*** Always validate with slot_analyzer on real code  ***\n")

        print("Per-Batch Breakdown (1 batch = 8 elements via VLEN):")
        print("-" * 50)
        bd = est["per_batch_breakdown"]
        print(f"  Hash computation (6 stages x 2):  {bd['hash_computation']:>4} cycles")
        print(f"  vbroadcast (inline, 6 x 2):       {bd['vbroadcast_inline']:>4} cycles")
        print(f"  vbroadcast (pre-loaded):          {bd['vbroadcast_preload']:>4} cycles")
        print(f"  XOR with node value:              {bd['xor_with_node']:>4} cycles")
        print(f"  Index calculation:                {bd['index_calculation']:>4} cycles")
        print(f"  Loop overhead:                    {bd['loop_overhead']:>4} cycles")
        print(f"  Memory ops (load/store):          {bd['memory_ops']:>4} cycles")
        print("-" * 50)
        print(f"  TOTAL (conservative):             {est['per_batch_total_conservative']:>4} cycles/batch")
        print(f"  TOTAL (optimized):                {est['per_batch_total_optimized']:>4} cycles/batch")

        print("\nFull Kernel Estimate (256 batch x 16 rounds = 4096 hashes):")
        print("-" * 50)
        fk = est["full_kernel"]
        print(f"  Vector iterations (4096/8):       {fk['vector_iterations']:>6}")
        print(f"  Cycles (conservative):            {fk['cycles_conservative']:>6}")
        print(f"  Cycles (optimized):               {fk['cycles_optimized']:>6}")
        print(f"  Cycles (ideal, UNREACHABLE):      {fk['cycles_ideal_unreachable']:>6}")

        print("\nContext:")
        print(f"  Current baseline:                 ~8,500 cycles")
        print(f"  Target (Opus 4.5 best):            1,487 cycles")
        print(f"  Our optimized estimate:           {fk['cycles_optimized']:>6} cycles")

        gap = fk['cycles_optimized'] - 1487
        if gap > 0:
            print(f"\n  Gap to target:                    {gap:>6} cycles")
            print("  (Need additional optimizations beyond hash pipelining)")
        else:
            print(f"\n  Under target by:                  {-gap:>6} cycles")

        print("\nNotes:")
        for note in est["notes"]:
            print(f"  - {note}")

        return est

    def full_analysis(self, visualize: bool = False, codegen: bool = False, realistic: bool = False):
        """Run complete analysis"""
        self.print_stage_analysis()

        # Dependency graph stats
        ops = self.build_dependency_graph()
        print("\n" + "=" * 70)
        print("DEPENDENCY GRAPH STATISTICS")
        print("=" * 70)
        print(f"Total operations: {len(ops)}")
        print(f"Operations per element: {len(ops) // self.n_elements}")
        print(f"Operations per stage: 3 (tmp1, tmp2, combine)")
        print(f"Independent op pairs per stage: 1 (tmp1 || tmp2)")

        # Theoretical minimums
        theory = self.calculate_theoretical_minimum()
        print("\n" + "=" * 70)
        print("THEORETICAL ANALYSIS (idealized, lower bound only)")
        print("=" * 70)
        print(f"Elements: {theory['elements']}")
        print(f"Total operations: {theory['total_operations']}")
        print(f"Critical path (1 element): {theory['critical_path_one_element']} cycles")
        print(f"Sequential (no parallelism): {theory['sequential_cycles']} cycles")
        print(f"Throughput limit: {theory['throughput_limited_min']} cycles (VALU limited)")
        print(f"Latency limit: {theory['latency_limited_min']} cycles (dependency limited)")
        print(f"Theoretical minimum: {theory['theoretical_min_cycles']} cycles")
        print("\n*** NOTE: These are LOWER BOUNDS - real implementation will be higher ***")

        # Compare strategies
        comparison = self.print_comparison()

        # Batch analysis
        self.analyze_batch_sizes()

        # Realistic estimates (always show in full analysis)
        self.print_realistic_estimate()

        # Visualization
        if visualize:
            max_schedule = self.schedule_maximally_pipelined()
            self.visualize_schedule(max_schedule)

        # Code generation hints
        if codegen:
            hints = self.generate_code_hints()
            print("\n".join(hints))

        return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Hash Pipeline Analyzer for VLIW SIMD Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hash_pipeline.py                    # Full analysis with 8 elements
  python hash_pipeline.py --elements 16     # Analyze 16-element pipeline
  python hash_pipeline.py --visualize       # Show cycle-by-cycle schedule
  python hash_pipeline.py --codegen         # Show VLIW code generation hints
  python hash_pipeline.py --json            # Output as JSON
        """
    )

    parser.add_argument("--elements", "-e", type=int, default=8,
                        help="Number of elements to analyze (default: 8)")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Show cycle-by-cycle schedule visualization")
    parser.add_argument("--codegen", "-c", action="store_true",
                        help="Show VLIW code generation hints")
    parser.add_argument("--compare", action="store_true",
                        help="Show comparison of scheduling strategies")
    parser.add_argument("--batch-analysis", "-b", action="store_true",
                        help="Analyze different batch sizes")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--scalar", "-s", action="store_true",
                        help="Use scalar ALU instead of vector VALU")
    parser.add_argument("--realistic", "-r", action="store_true",
                        help="Show realistic estimates with overhead (included in full analysis)")

    args = parser.parse_args()

    analyzer = HashPipelineAnalyzer(
        n_elements=args.elements,
        use_vector=not args.scalar
    )

    if args.json:
        result = {
            "elements": args.elements,
            "use_vector": not args.scalar,
            "comparison": analyzer.print_comparison(),
            "theory": analyzer.calculate_theoretical_minimum()
        }
        print(json.dumps(result, indent=2))
    else:
        analyzer.full_analysis(
            visualize=args.visualize,
            codegen=args.codegen
        )


if __name__ == "__main__":
    main()
