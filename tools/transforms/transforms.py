#!/usr/bin/env python3
"""
Transformation Library for VLIW SIMD Optimization

Codified transformations to reduce manual errors in mechanical transforms.

Transforms available:
1. Loop unrolling - Unroll a loop by factor N
2. Vectorize batch - Convert scalar ops to vector ops (VLEN=8)
3. Software pipelining helpers - Overlap iteration N+1 prep with N execution
4. Hoist loop invariants - Move loop-invariant code outside the loop

Usage:
    python tools/transforms/transforms.py [--json]
    python tools/transforms/transforms.py --demo

    # From Python:
    from tools.transforms.transforms import (
        unroll_loop, vectorize_batch, software_pipeline, hoist_invariants
    )
"""

import sys
import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from collections import defaultdict
from copy import deepcopy
from enum import Enum

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import SLOT_LIMITS, VLEN, HASH_STAGES

# Try to import Rich for better formatting (optional)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ============== Data Types ==============

@dataclass
class TransformResult:
    """Result of applying a transformation."""
    success: bool
    original_cycles: int
    transformed_cycles: int
    cycles_saved: int
    speedup: float
    description: str
    warnings: List[str] = field(default_factory=list)
    transformed_code: Optional[List[dict]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "original_cycles": self.original_cycles,
            "transformed_cycles": self.transformed_cycles,
            "cycles_saved": self.cycles_saved,
            "speedup": round(self.speedup, 3),
            "description": self.description,
            "warnings": self.warnings,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class LoopStructure:
    """Represents an identified loop in instruction stream."""
    start_cycle: int
    end_cycle: int
    body_cycles: int
    iteration_count: int
    induction_var: Optional[int]  # scratch address of loop counter
    loop_invariants: Set[int]  # scratch addresses that don't change
    body_instructions: List[dict] = field(default_factory=list)


@dataclass
class ScalarOp:
    """A scalar operation that could be vectorized."""
    engine: str
    op: str
    dest: int
    sources: List[int]
    cycle: int


@dataclass
class PipelineStage:
    """A stage in a software pipeline."""
    name: str
    instructions: List[dict]
    cycles: int
    produces: Set[int]  # scratch addresses produced
    consumes: Set[int]  # scratch addresses consumed
    can_overlap_with: List[str] = field(default_factory=list)


# ============== Utility Functions ==============

def extract_reads_writes(slot: tuple, engine: str) -> Tuple[Set[int], Set[int]]:
    """Extract scratch addresses read and written by an instruction slot."""
    reads = set()
    writes = set()

    if not slot or len(slot) == 0:
        return reads, writes

    op = slot[0]

    if engine == "alu":
        if len(slot) >= 4:
            writes.add(slot[1])
            reads.add(slot[2])
            reads.add(slot[3])

    elif engine == "valu":
        if op == "vbroadcast":
            if len(slot) >= 3:
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                reads.add(slot[2])
        elif op == "multiply_add":
            if len(slot) >= 5:
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)
                    reads.add(slot[4] + i)
        else:
            if len(slot) >= 4:
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)

    elif engine == "load":
        if op == "load":
            if len(slot) >= 3:
                writes.add(slot[1])
                reads.add(slot[2])
        elif op == "vload":
            if len(slot) >= 3:
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                reads.add(slot[2])
        elif op == "const":
            if len(slot) >= 2:
                writes.add(slot[1])

    elif engine == "store":
        if op == "store":
            if len(slot) >= 3:
                reads.add(slot[1])
                reads.add(slot[2])
        elif op == "vstore":
            if len(slot) >= 3:
                reads.add(slot[1])
                for i in range(VLEN):
                    reads.add(slot[2] + i)

    elif engine == "flow":
        if op == "select":
            if len(slot) >= 5:
                writes.add(slot[1])
                reads.add(slot[2])
                reads.add(slot[3])
                reads.add(slot[4])
        elif op == "vselect":
            if len(slot) >= 5:
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)
                    reads.add(slot[4] + i)
        elif op == "add_imm":
            if len(slot) >= 3:
                writes.add(slot[1])
                reads.add(slot[2])

    return reads, writes


def count_cycles(instructions: List[dict]) -> int:
    """Count total cycles in instruction list."""
    return len(instructions)


def get_all_reads_writes(instructions: List[dict]) -> Tuple[Set[int], Set[int]]:
    """Get all reads and writes across all instructions."""
    all_reads = set()
    all_writes = set()
    for instr in instructions:
        for engine, slots in instr.items():
            if engine == "debug" or not slots:
                continue
            for slot in slots:
                reads, writes = extract_reads_writes(slot, engine)
                all_reads.update(reads)
                all_writes.update(writes)
    return all_reads, all_writes


# ============== Loop Unrolling ==============

def unroll_loop(
    body: List[dict],
    unroll_factor: int,
    induction_var: Optional[int] = None,
    induction_increment: int = 1,
    rename_map_fn: Optional[Callable[[int, int], int]] = None
) -> TransformResult:
    """
    Unroll a loop body by a given factor.

    Args:
        body: List of instruction bundles forming the loop body
        unroll_factor: How many times to replicate the body (2, 4, 8, etc.)
        induction_var: Scratch address of loop counter (if any)
        induction_increment: How much induction var increases per iteration
        rename_map_fn: Function(scratch_addr, iteration) -> new_scratch_addr
                      for register renaming between unrolled iterations

    Returns:
        TransformResult with unrolled code

    Example:
        # Unroll 4x with automatic register renaming
        result = unroll_loop(body, 4, rename_map_fn=lambda addr, i: addr + i*100)

    Notes:
        - Register renaming prevents WAW/WAR hazards between iterations
        - Without renaming, iterations must be serialized
        - Unrolling increases code size but reduces loop overhead
    """
    if unroll_factor < 2:
        return TransformResult(
            success=False,
            original_cycles=len(body),
            transformed_cycles=len(body),
            cycles_saved=0,
            speedup=1.0,
            description="Unroll factor must be >= 2",
            warnings=["Invalid unroll factor"]
        )

    original_cycles = len(body)
    unrolled = []
    warnings = []

    # Default rename function: offset by iteration * large_offset
    if rename_map_fn is None:
        max_addr = 0
        for instr in body:
            for engine, slots in instr.items():
                if engine == "debug" or not slots:
                    continue
                for slot in slots:
                    reads, writes = extract_reads_writes(slot, engine)
                    max_addr = max(max_addr, max(reads | writes, default=0))

        # Use offset larger than any used address
        offset = max_addr + 100
        rename_map_fn = lambda addr, iter_num: addr if iter_num == 0 else addr + iter_num * offset
        warnings.append(f"Using default register renaming with offset={offset}")

    # Unroll the loop
    for iter_num in range(unroll_factor):
        for orig_instr in body:
            new_instr = {}
            for engine, slots in orig_instr.items():
                if engine == "debug":
                    new_instr[engine] = slots
                    continue
                if not slots:
                    continue

                new_slots = []
                for slot in slots:
                    if not slot:
                        new_slots.append(slot)
                        continue

                    # Rename registers for this iteration
                    new_slot = list(slot)
                    op = slot[0]

                    # Rename destination and sources based on engine/op
                    if engine == "alu" and len(slot) >= 4:
                        new_slot[1] = rename_map_fn(slot[1], iter_num)
                        new_slot[2] = rename_map_fn(slot[2], iter_num)
                        new_slot[3] = rename_map_fn(slot[3], iter_num)
                    elif engine == "valu":
                        if op == "vbroadcast" and len(slot) >= 3:
                            new_slot[1] = rename_map_fn(slot[1], iter_num)
                            new_slot[2] = rename_map_fn(slot[2], iter_num)
                        elif len(slot) >= 4:
                            new_slot[1] = rename_map_fn(slot[1], iter_num)
                            new_slot[2] = rename_map_fn(slot[2], iter_num)
                            new_slot[3] = rename_map_fn(slot[3], iter_num)
                    elif engine == "load":
                        if op == "load" and len(slot) >= 3:
                            new_slot[1] = rename_map_fn(slot[1], iter_num)
                            new_slot[2] = rename_map_fn(slot[2], iter_num)
                        elif op == "vload" and len(slot) >= 3:
                            new_slot[1] = rename_map_fn(slot[1], iter_num)
                            new_slot[2] = rename_map_fn(slot[2], iter_num)
                        elif op == "const" and len(slot) >= 2:
                            new_slot[1] = rename_map_fn(slot[1], iter_num)
                    elif engine == "store":
                        if op in ("store", "vstore") and len(slot) >= 3:
                            new_slot[1] = rename_map_fn(slot[1], iter_num)
                            new_slot[2] = rename_map_fn(slot[2], iter_num)
                    elif engine == "flow":
                        if op in ("select", "vselect") and len(slot) >= 5:
                            new_slot[1] = rename_map_fn(slot[1], iter_num)
                            new_slot[2] = rename_map_fn(slot[2], iter_num)
                            new_slot[3] = rename_map_fn(slot[3], iter_num)
                            new_slot[4] = rename_map_fn(slot[4], iter_num)

                    new_slots.append(tuple(new_slot))

                new_instr[engine] = new_slots
            unrolled.append(new_instr)

    # Adjust induction variable updates if present
    if induction_var is not None:
        # The induction variable should increment by unroll_factor * original_increment
        # This would need to be handled by the caller adjusting the loop bounds
        warnings.append(f"Loop bounds need adjustment: now covers {unroll_factor}x iterations")

    transformed_cycles = len(unrolled)
    # Theoretical cycles assuming independent iterations can be packed
    # In practice, packing will reduce this further
    cycles_saved = 0  # Before packing, no savings yet
    speedup = 1.0

    return TransformResult(
        success=True,
        original_cycles=original_cycles * unroll_factor,  # Original cost for same work
        transformed_cycles=transformed_cycles,
        cycles_saved=cycles_saved,
        speedup=speedup,
        description=f"Unrolled loop {unroll_factor}x with register renaming",
        warnings=warnings,
        transformed_code=unrolled,
        metadata={
            "unroll_factor": unroll_factor,
            "body_cycles": original_cycles,
            "unrolled_cycles": transformed_cycles,
            "register_offset": "auto"
        }
    )


def generate_unroll_code_hint(
    op_template: str,
    unroll_factor: int,
    vector_base: int,
    scalar_inputs: List[int]
) -> str:
    """
    Generate code hint for unrolled loop implementation.

    Args:
        op_template: Operation template like "valu: ('+', dest, src1, src2)"
        unroll_factor: How many iterations to unroll
        vector_base: Base address for vector registers
        scalar_inputs: List of scalar input addresses

    Returns:
        String with Python code hint for implementing unrolled loop
    """
    lines = [
        f"# Unrolled {unroll_factor}x loop",
        f"# Vector registers: {vector_base} to {vector_base + unroll_factor * VLEN}",
        ""
    ]

    for i in range(unroll_factor):
        offset = i * VLEN
        lines.append(f"# Iteration {i}:")
        lines.append(f"#   dest = {vector_base + offset}")
        lines.append(f"#   sources = scalar_inputs mapped for iteration {i}")

    return "\n".join(lines)


# ============== Vectorize Batch ==============

def vectorize_batch(
    scalar_ops: List[ScalarOp],
    batch_size: int = VLEN
) -> TransformResult:
    """
    Convert scalar operations to vector operations.

    Args:
        scalar_ops: List of scalar operations to vectorize
        batch_size: How many scalar ops to combine (default VLEN=8)

    Returns:
        TransformResult with vectorized code

    Example:
        ops = [
            ScalarOp("alu", "+", dest=10, sources=[20, 30], cycle=0),
            ScalarOp("alu", "+", dest=11, sources=[21, 31], cycle=1),
            # ... 6 more similar ops
        ]
        result = vectorize_batch(ops)  # Produces single valu instruction

    Notes:
        - All ops must have same operation type
        - Destinations must be consecutive (for efficient vstore)
        - Sources should have regular stride for vload
    """
    if len(scalar_ops) < 2:
        return TransformResult(
            success=False,
            original_cycles=len(scalar_ops),
            transformed_cycles=len(scalar_ops),
            cycles_saved=0,
            speedup=1.0,
            description="Need at least 2 scalar ops to vectorize",
            warnings=["Insufficient operations for vectorization"]
        )

    # Group by operation type
    op_groups = defaultdict(list)
    for op in scalar_ops:
        op_groups[(op.engine, op.op)].append(op)

    warnings = []
    vectorized = []

    for (engine, op_type), ops in op_groups.items():
        if len(ops) < batch_size:
            warnings.append(f"Only {len(ops)} ops of type {engine}:{op_type}, need {batch_size} for full vectorization")

        # Check if destinations are consecutive
        dests = sorted(op.dest for op in ops[:batch_size])
        is_consecutive_dest = all(dests[i+1] - dests[i] == 1 for i in range(len(dests)-1))

        if not is_consecutive_dest:
            warnings.append(f"Non-consecutive destinations for {engine}:{op_type}: {dests}")

        # For ALU -> VALU conversion
        if engine == "alu" and len(ops) >= batch_size:
            # Find the minimum destination as vector base
            base_dest = min(op.dest for op in ops[:batch_size])

            # Build source vectors
            src1_base = min(op.sources[0] for op in ops[:batch_size]) if ops[0].sources else 0
            src2_base = min(op.sources[1] for op in ops[:batch_size]) if len(ops[0].sources) > 1 else 0

            # Generate vector instruction
            vector_instr = {
                "valu": [(op_type, base_dest, src1_base, src2_base)]
            }
            vectorized.append(vector_instr)

    original_cycles = len(scalar_ops)
    transformed_cycles = len(vectorized)
    cycles_saved = original_cycles - transformed_cycles
    speedup = original_cycles / transformed_cycles if transformed_cycles > 0 else float('inf')

    return TransformResult(
        success=len(vectorized) > 0,
        original_cycles=original_cycles,
        transformed_cycles=transformed_cycles,
        cycles_saved=cycles_saved,
        speedup=speedup,
        description=f"Vectorized {len(scalar_ops)} scalar ops into {len(vectorized)} vector ops",
        warnings=warnings,
        transformed_code=vectorized,
        metadata={
            "batch_size": batch_size,
            "scalar_ops": len(scalar_ops),
            "vector_ops": len(vectorized),
            "groups": len(op_groups)
        }
    )


def analyze_vectorization_opportunity(instructions: List[dict]) -> Dict[str, Any]:
    """
    Analyze instruction stream for vectorization opportunities.

    Returns dict with:
        - scalar_alu_count: Number of scalar ALU ops
        - vectorizable_sequences: List of consecutive scalar ops
        - potential_savings: Estimated cycle savings
        - blockers: Reasons preventing vectorization
    """
    scalar_alu_count = 0
    vectorizable_sequences = []
    current_sequence = []
    blockers = []

    for cycle, instr in enumerate(instructions):
        alu_slots = instr.get("alu", [])
        if not alu_slots:
            # Sequence break
            if len(current_sequence) >= VLEN:
                vectorizable_sequences.append(current_sequence)
            current_sequence = []
            continue

        for slot in alu_slots:
            if not slot:
                continue
            scalar_alu_count += 1
            op = slot[0]
            dest = slot[1] if len(slot) > 1 else None
            sources = slot[2:] if len(slot) > 2 else []

            current_sequence.append(ScalarOp(
                engine="alu",
                op=op,
                dest=dest,
                sources=list(sources),
                cycle=cycle
            ))

    # Final sequence
    if len(current_sequence) >= VLEN:
        vectorizable_sequences.append(current_sequence)

    # Calculate potential savings
    total_vectorizable = sum(len(seq) for seq in vectorizable_sequences)
    potential_vector_ops = total_vectorizable // VLEN
    potential_savings = total_vectorizable - potential_vector_ops

    return {
        "scalar_alu_count": scalar_alu_count,
        "vectorizable_sequences": len(vectorizable_sequences),
        "total_vectorizable_ops": total_vectorizable,
        "potential_vector_ops": potential_vector_ops,
        "potential_savings": potential_savings,
        "blockers": blockers
    }


def generate_vectorized_hash_stage(
    stage_idx: int,
    val_base: int,
    tmp1_base: int,
    tmp2_base: int,
    const1_vec: int,
    const3_vec: int
) -> List[dict]:
    """
    Generate vectorized hash stage instructions.

    This is a code generation helper for the 6-stage hash function.
    Each stage has 3 operations that can use 2-way ILP (tmp1 || tmp2).

    Args:
        stage_idx: Which hash stage (0-5)
        val_base: Base address of value vector
        tmp1_base: Base address of tmp1 vector
        tmp2_base: Base address of tmp2 vector
        const1_vec: Address of first constant vector (pre-broadcast)
        const3_vec: Address of second constant vector (pre-broadcast)

    Returns:
        List of instruction bundles for this stage
    """
    op1, val1, op2, op3, val3 = HASH_STAGES[stage_idx]

    # Cycle 1: op1 to tmp1 || op3 to tmp2 (can run in parallel)
    cycle1 = {
        "valu": [
            (op1, tmp1_base, val_base, const1_vec),
            (op3, tmp2_base, val_base, const3_vec),
        ]
    }

    # Cycle 2: op2 combining tmp1 and tmp2
    cycle2 = {
        "valu": [
            (op2, val_base, tmp1_base, tmp2_base),
        ]
    }

    return [cycle1, cycle2]


# ============== Software Pipelining ==============

def software_pipeline(
    prologue: List[dict],
    body: List[dict],
    epilogue: List[dict],
    overlap_cycles: int = 0
) -> TransformResult:
    """
    Apply software pipelining to overlap iterations.

    Software pipelining overlaps execution of multiple loop iterations
    to hide latency and improve throughput.

    Args:
        prologue: Setup code before first iteration
        body: Main loop body (one iteration)
        epilogue: Cleanup code after last iteration
        overlap_cycles: How many cycles of iteration N+1 can overlap with N

    Returns:
        TransformResult with pipelined code

    Example:
        # Hash computation: overlap prep of batch N+1 with hash of batch N
        # During 16 hash cycles, 12 can overlap with next batch setup
        result = software_pipeline(setup, hash_loop, finish, overlap_cycles=12)

    Notes:
        - Requires careful register allocation to avoid conflicts
        - Need double-buffering or renaming between overlapped iterations
        - Analysis helps identify which operations can overlap
    """
    original_total = len(prologue) + len(body) + len(epilogue)

    # Simple model: assume body repeats multiple times
    # Pipelining saves (overlap_cycles) per iteration after first
    estimated_iterations = 10  # Example
    original_for_n_iters = len(prologue) + len(body) * estimated_iterations + len(epilogue)
    pipelined_for_n_iters = len(prologue) + len(body) + (len(body) - overlap_cycles) * (estimated_iterations - 1) + len(epilogue)

    cycles_saved = original_for_n_iters - pipelined_for_n_iters
    speedup = original_for_n_iters / pipelined_for_n_iters if pipelined_for_n_iters > 0 else 1.0

    # For code generation, we'd need to interleave instructions
    # This is complex and typically done manually with guidance
    pipelined = []
    warnings = []

    if overlap_cycles > len(body):
        warnings.append(f"Overlap ({overlap_cycles}) exceeds body length ({len(body)})")
        overlap_cycles = len(body)

    if overlap_cycles > 0:
        warnings.append("Code generation for software pipelining is complex - use generate_pipeline_schedule() for guidance")

    return TransformResult(
        success=True,
        original_cycles=original_for_n_iters,
        transformed_cycles=pipelined_for_n_iters,
        cycles_saved=cycles_saved,
        speedup=speedup,
        description=f"Software pipeline with {overlap_cycles} cycles overlap",
        warnings=warnings,
        transformed_code=None,  # Complex - provide hints instead
        metadata={
            "prologue_cycles": len(prologue),
            "body_cycles": len(body),
            "epilogue_cycles": len(epilogue),
            "overlap_cycles": overlap_cycles,
            "estimated_iterations": estimated_iterations,
            "savings_per_iteration": overlap_cycles
        }
    )


def generate_pipeline_schedule(
    stages: List[PipelineStage],
    iterations: int
) -> Dict[str, Any]:
    """
    Generate a visual schedule for software pipelining.

    Shows which stages run in each cycle across iterations.

    Args:
        stages: List of pipeline stages with their dependencies
        iterations: Number of iterations to schedule

    Returns:
        Dict with schedule visualization and overlap analysis
    """
    schedule = []
    stage_names = [s.name for s in stages]
    total_stage_cycles = sum(s.cycles for s in stages)

    # Build dependency graph between stages
    stage_deps = {}
    for stage in stages:
        deps = []
        for other in stages:
            if stage.consumes & other.produces:
                deps.append(other.name)
        stage_deps[stage.name] = deps

    # Simple schedule: identify overlap potential
    overlap_potential = []
    for i, stage in enumerate(stages):
        for j, other in enumerate(stages):
            if i != j and not (stage.consumes & other.produces) and not (other.consumes & stage.produces):
                overlap_potential.append((stage.name, other.name))

    # Generate textual schedule
    schedule_text = []
    schedule_text.append("Pipeline Schedule:")
    schedule_text.append("=" * 60)

    for iter_num in range(min(iterations, 3)):
        schedule_text.append(f"\nIteration {iter_num}:")
        cycle = iter_num * (total_stage_cycles // 2) if iter_num > 0 else 0  # Simplified overlap
        for stage in stages:
            schedule_text.append(f"  Cycle {cycle:3d}-{cycle+stage.cycles-1:3d}: {stage.name}")
            cycle += stage.cycles

    return {
        "stages": stage_names,
        "total_stage_cycles": total_stage_cycles,
        "stage_dependencies": stage_deps,
        "overlap_potential": overlap_potential,
        "schedule_visualization": "\n".join(schedule_text),
        "suggested_overlap": len(overlap_potential) > 0
    }


def analyze_pipeline_opportunity(
    body: List[dict]
) -> Dict[str, Any]:
    """
    Analyze loop body for software pipelining opportunities.

    Identifies:
    - Natural stages (setup, compute, store)
    - Dependencies between stages
    - Potential overlap windows
    """
    # Categorize instructions by type
    setup_cycles = []  # Address computation, loads
    compute_cycles = []  # ALU, VALU operations
    store_cycles = []  # Stores

    for i, instr in enumerate(body):
        has_load = "load" in instr and instr["load"]
        has_store = "store" in instr and instr["store"]
        has_alu = "alu" in instr and instr["alu"]
        has_valu = "valu" in instr and instr["valu"]

        if has_load and not has_alu and not has_valu:
            setup_cycles.append(i)
        elif has_store:
            store_cycles.append(i)
        elif has_alu or has_valu:
            compute_cycles.append(i)
        else:
            # Mixed or other
            compute_cycles.append(i)

    # Compute overlap potential
    # Setup of N+1 can overlap with compute of N if they use different resources
    setup_uses_load = len(setup_cycles) > 0
    compute_uses_valu = any("valu" in body[i] and body[i]["valu"] for i in compute_cycles)

    overlap_opportunity = setup_uses_load and compute_uses_valu

    return {
        "setup_cycles": len(setup_cycles),
        "compute_cycles": len(compute_cycles),
        "store_cycles": len(store_cycles),
        "total_cycles": len(body),
        "overlap_opportunity": overlap_opportunity,
        "recommended_overlap": len(setup_cycles) if overlap_opportunity else 0,
        "description": "Setup (LOAD) can overlap with compute (VALU)" if overlap_opportunity else "Limited overlap potential"
    }


# ============== Hoist Loop Invariants ==============

def hoist_invariants(
    loop_body: List[dict],
    iterations: int,
    known_invariants: Optional[Set[int]] = None
) -> TransformResult:
    """
    Identify and hoist loop-invariant code outside the loop.

    Args:
        loop_body: Instructions in loop body
        iterations: Number of loop iterations
        known_invariants: Scratch addresses known to be invariant

    Returns:
        TransformResult with hoisted code

    Example:
        # Constants computed inside loop can be hoisted
        result = hoist_invariants(loop_body, 16, known_invariants={100, 101, 102})

    Notes:
        - Loop invariant = value doesn't change across iterations
        - Constants, broadcast values, and pre-computed addresses are candidates
        - Hoisting trades space for time (may increase register pressure)
    """
    if known_invariants is None:
        known_invariants = set()

    # Analyze each instruction
    hoistable = []
    remaining_body = []
    warnings = []

    # Track what gets written in the loop
    loop_writes = set()
    for instr in loop_body:
        for engine, slots in instr.items():
            if engine == "debug" or not slots:
                continue
            for slot in slots:
                _, writes = extract_reads_writes(slot, engine)
                loop_writes.update(writes)

    # Find instructions that only read invariants
    for i, instr in enumerate(loop_body):
        can_hoist = True
        instr_reads = set()
        instr_writes = set()

        for engine, slots in instr.items():
            if engine == "debug" or not slots:
                continue
            for slot in slots:
                reads, writes = extract_reads_writes(slot, engine)
                instr_reads.update(reads)
                instr_writes.update(writes)

                # Check if this is a const load (always hoistable)
                if engine == "load" and slot and slot[0] == "const":
                    continue

                # Check if reads any loop-variant values
                if reads - known_invariants - instr_writes:
                    # Reads something that might vary
                    can_hoist = False

        # Check for flow control (can't hoist)
        if "flow" in instr and instr["flow"]:
            for slot in instr["flow"]:
                if slot and slot[0] in ("pause", "halt", "jump", "cond_jump"):
                    can_hoist = False

        if can_hoist and instr_writes:
            hoistable.append(instr)
            # Add written addresses to invariants for next pass
            known_invariants.update(instr_writes)
        else:
            remaining_body.append(instr)

    # Calculate savings
    original_cycles = len(loop_body) * iterations
    hoisted_cycles = len(hoistable)  # Run once
    body_cycles = len(remaining_body) * iterations  # Run each iteration
    transformed_cycles = hoisted_cycles + body_cycles

    cycles_saved = original_cycles - transformed_cycles
    speedup = original_cycles / transformed_cycles if transformed_cycles > 0 else 1.0

    return TransformResult(
        success=len(hoistable) > 0,
        original_cycles=original_cycles,
        transformed_cycles=transformed_cycles,
        cycles_saved=cycles_saved,
        speedup=speedup,
        description=f"Hoisted {len(hoistable)} instructions ({cycles_saved} cycles saved over {iterations} iterations)",
        warnings=warnings,
        transformed_code=hoistable + remaining_body,  # Simplified - hoisted first, then body
        metadata={
            "hoisted_count": len(hoistable),
            "remaining_body_count": len(remaining_body),
            "iterations": iterations,
            "invariants_found": len(known_invariants)
        }
    )


def identify_invariants(
    loop_body: List[dict],
    entry_values: Set[int]
) -> Set[int]:
    """
    Identify loop-invariant scratch addresses.

    Args:
        loop_body: Instructions in loop body
        entry_values: Addresses that are set before loop entry

    Returns:
        Set of scratch addresses that are loop-invariant
    """
    # Start with entry values
    invariants = set(entry_values)

    # Find all writes in loop body
    writes_in_loop = set()
    for instr in loop_body:
        for engine, slots in instr.items():
            if engine == "debug" or not slots:
                continue
            for slot in slots:
                _, writes = extract_reads_writes(slot, engine)
                writes_in_loop.update(writes)

    # Entry values that aren't written in loop are invariant
    invariants = entry_values - writes_in_loop

    # Also, const loads are invariant
    for instr in loop_body:
        load_slots = instr.get("load", [])
        if not load_slots:
            continue
        for slot in load_slots:
            if slot and slot[0] == "const":
                invariants.add(slot[1])

    return invariants


# ============== Combined Analysis ==============

def analyze_transform_opportunities(instructions: List[dict]) -> Dict[str, Any]:
    """
    Analyze instruction stream for all transformation opportunities.

    Returns comprehensive analysis with recommendations.
    """
    total_cycles = len(instructions)

    # Vectorization analysis
    vec_analysis = analyze_vectorization_opportunity(instructions)

    # Simple loop detection (look for repeated patterns)
    # This is a simplified heuristic
    loop_candidates = []
    for window_size in [10, 20, 50]:
        if total_cycles > window_size * 2:
            # Check if pattern repeats
            pass  # Simplified

    # Pipeline analysis (assume main body is hash-like)
    hash_cycles = 0
    for instr in instructions:
        if "valu" in instr and instr["valu"]:
            hash_cycles += 1

    pipeline_potential = hash_cycles > total_cycles * 0.3

    return {
        "total_cycles": total_cycles,
        "vectorization": vec_analysis,
        "loop_unroll": {
            "recommended": total_cycles > 100,
            "suggested_factor": 4 if total_cycles > 200 else 2
        },
        "software_pipeline": {
            "recommended": pipeline_potential,
            "hash_dominated": hash_cycles / total_cycles if total_cycles > 0 else 0
        },
        "hoist_invariants": {
            "recommended": True,
            "note": "Always beneficial to hoist constants and broadcasts"
        },
        "summary": {
            "primary_bottleneck": "hash" if hash_cycles > total_cycles * 0.5 else "memory" if vec_analysis["potential_savings"] > 10 else "unknown",
            "recommended_transforms": []
        }
    }


# ============== Output Formatting ==============

class PlainPrinter:
    """Plain text output without Rich."""

    def print_header(self, text: str):
        print("=" * 70)
        print(text)
        print("=" * 70)

    def print_result(self, result: TransformResult):
        print()
        self.print_header("TRANSFORMATION RESULT")
        print()
        print(f"Success:             {result.success}")
        print(f"Description:         {result.description}")
        print(f"Original cycles:     {result.original_cycles:,}")
        print(f"Transformed cycles:  {result.transformed_cycles:,}")
        print(f"Cycles saved:        {result.cycles_saved:,}")
        print(f"Speedup:             {result.speedup:.2f}x")
        print()

        if result.warnings:
            print("Warnings:")
            for w in result.warnings:
                print(f"  - {w}")
            print()

        if result.metadata:
            print("Metadata:")
            for k, v in result.metadata.items():
                print(f"  {k}: {v}")

    def print_analysis(self, analysis: Dict[str, Any]):
        print()
        self.print_header("TRANSFORMATION OPPORTUNITY ANALYSIS")
        print()

        print(f"Total Cycles: {analysis['total_cycles']:,}")
        print()

        print("Vectorization:")
        vec = analysis['vectorization']
        print(f"  Scalar ALU ops:     {vec['scalar_alu_count']}")
        print(f"  Vectorizable:       {vec['total_vectorizable_ops']}")
        print(f"  Potential savings:  {vec['potential_savings']} cycles")
        print()

        print("Loop Unrolling:")
        unroll = analysis['loop_unroll']
        print(f"  Recommended:        {unroll['recommended']}")
        print(f"  Suggested factor:   {unroll['suggested_factor']}x")
        print()

        print("Software Pipelining:")
        pipe = analysis['software_pipeline']
        print(f"  Recommended:        {pipe['recommended']}")
        print(f"  Hash-dominated:     {pipe['hash_dominated']:.1%}")
        print()

        print(f"Primary Bottleneck: {analysis['summary']['primary_bottleneck']}")


class RichPrinter:
    """Rich-enabled colorful output."""

    def __init__(self):
        self.console = Console()

    def print_header(self, text: str):
        self.console.print(Panel(text, style="bold cyan", box=box.DOUBLE))

    def print_result(self, result: TransformResult):
        self.print_header("TRANSFORMATION RESULT")

        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        success_style = "green" if result.success else "red"
        table.add_row("Success", f"[{success_style}]{result.success}[/{success_style}]")
        table.add_row("Description", result.description)
        table.add_row("Original cycles", f"{result.original_cycles:,}")
        table.add_row("Transformed cycles", f"{result.transformed_cycles:,}")

        savings_style = "green" if result.cycles_saved > 0 else "yellow"
        table.add_row("Cycles saved", f"[{savings_style}]{result.cycles_saved:,}[/{savings_style}]")

        speedup_style = "green" if result.speedup > 1.1 else "yellow"
        table.add_row("Speedup", f"[{speedup_style}]{result.speedup:.2f}x[/{speedup_style}]")

        self.console.print(table)

        if result.warnings:
            self.console.print("\n[yellow]Warnings:[/yellow]")
            for w in result.warnings:
                self.console.print(f"  [yellow]-[/yellow] {w}")

        if result.metadata:
            self.console.print("\n[dim]Metadata:[/dim]")
            for k, v in result.metadata.items():
                self.console.print(f"  [dim]{k}:[/dim] {v}")

    def print_analysis(self, analysis: Dict[str, Any]):
        self.print_header("TRANSFORMATION OPPORTUNITY ANALYSIS")

        self.console.print(f"\n[bold]Total Cycles:[/bold] {analysis['total_cycles']:,}")

        # Vectorization
        self.console.print("\n[bold yellow]Vectorization[/bold yellow]")
        vec_table = Table(box=box.ROUNDED)
        vec_table.add_column("Metric")
        vec_table.add_column("Value", justify="right")

        vec = analysis['vectorization']
        vec_table.add_row("Scalar ALU ops", str(vec['scalar_alu_count']))
        vec_table.add_row("Vectorizable", str(vec['total_vectorizable_ops']))
        vec_table.add_row("Potential savings", f"{vec['potential_savings']} cycles")
        self.console.print(vec_table)

        # Unroll
        self.console.print("\n[bold yellow]Loop Unrolling[/bold yellow]")
        unroll = analysis['loop_unroll']
        rec_style = "green" if unroll['recommended'] else "red"
        self.console.print(f"  Recommended: [{rec_style}]{unroll['recommended']}[/{rec_style}]")
        self.console.print(f"  Suggested factor: {unroll['suggested_factor']}x")

        # Pipeline
        self.console.print("\n[bold yellow]Software Pipelining[/bold yellow]")
        pipe = analysis['software_pipeline']
        rec_style = "green" if pipe['recommended'] else "red"
        self.console.print(f"  Recommended: [{rec_style}]{pipe['recommended']}[/{rec_style}]")
        self.console.print(f"  Hash-dominated: {pipe['hash_dominated']:.1%}")

        # Summary
        bottleneck = analysis['summary']['primary_bottleneck']
        self.console.print(f"\n[bold]Primary Bottleneck:[/bold] [magenta]{bottleneck}[/magenta]")


def get_printer():
    """Get appropriate printer based on Rich availability."""
    if RICH_AVAILABLE:
        return RichPrinter()
    return PlainPrinter()


# ============== Demo and Main ==============

def run_demo():
    """Run demonstration of transformation capabilities."""
    printer = get_printer()

    print("=" * 70)
    print("TRANSFORMATION LIBRARY DEMO")
    print("=" * 70)
    print()

    # Demo 1: Unroll
    print("\n1. LOOP UNROLLING")
    print("-" * 40)

    demo_body = [
        {"alu": [("+", 10, 20, 30)]},
        {"alu": [("*", 11, 10, 31)]},
        {"store": [("store", 40, 11)]},
    ]

    result = unroll_loop(demo_body, unroll_factor=4)
    printer.print_result(result)

    # Demo 2: Vectorization analysis
    print("\n2. VECTORIZATION ANALYSIS")
    print("-" * 40)

    demo_scalar = [
        {"alu": [("+", i, i+100, i+200)]} for i in range(16)
    ]
    analysis = analyze_vectorization_opportunity(demo_scalar)
    print(f"Scalar ops: {analysis['scalar_alu_count']}")
    print(f"Vectorizable: {analysis['total_vectorizable_ops']}")
    print(f"Potential savings: {analysis['potential_savings']} cycles")

    # Demo 3: Software pipelining
    print("\n3. SOFTWARE PIPELINING")
    print("-" * 40)

    stages = [
        PipelineStage("setup", [], 4, produces={10, 11}, consumes={0}),
        PipelineStage("hash", [], 16, produces={20}, consumes={10}),
        PipelineStage("finish", [], 6, produces={30}, consumes={20}),
    ]
    schedule = generate_pipeline_schedule(stages, iterations=3)
    print(schedule['schedule_visualization'])

    # Demo 4: Hash stage generation
    print("\n4. VECTORIZED HASH STAGE")
    print("-" * 40)

    hash_instrs = generate_vectorized_hash_stage(
        stage_idx=0,
        val_base=100,
        tmp1_base=200,
        tmp2_base=300,
        const1_vec=400,
        const3_vec=500
    )
    print(f"Generated {len(hash_instrs)} cycles for hash stage 0:")
    for i, instr in enumerate(hash_instrs):
        print(f"  Cycle {i}: {instr}")


def analyze_kernel():
    """Analyze the current kernel from perf_takehome.py"""
    from perf_takehome import KernelBuilder

    # Standard test params
    forest_height = 10
    n_nodes = 2 ** (forest_height + 1) - 1
    batch_size = 256
    rounds = 16

    kb = KernelBuilder()
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)

    return kb.instrs


def main():
    parser = argparse.ArgumentParser(
        description="Transformation Library for VLIW SIMD Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/transforms/transforms.py                 # Analyze current kernel
    python tools/transforms/transforms.py --demo          # Run demo
    python tools/transforms/transforms.py --json          # JSON output
        """
    )
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    printer = PlainPrinter() if args.no_color else get_printer()

    print("Loading kernel...", file=sys.stderr)
    instructions = analyze_kernel()

    print(f"Analyzing {len(instructions)} cycles...", file=sys.stderr)
    analysis = analyze_transform_opportunities(instructions)

    if args.json:
        print(json.dumps(analysis, indent=2))
    else:
        printer.print_analysis(analysis)


if __name__ == "__main__":
    main()
