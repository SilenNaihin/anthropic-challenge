#!/usr/bin/env python3
"""
VLIW Auto-Packer - Automatic instruction packing for VLIW SIMD processors

Takes unpacked instruction streams and automatically packs independent instructions
into VLIW bundles while respecting:
- Data dependencies (RAW, WAW, WAR hazards)
- Slot limits per engine (12 alu, 6 valu, 2 load, 2 store, 1 flow)
- Program semantics (flow control ordering)

Usage:
    python tools/vliw_packer/vliw_packer.py [--stats] [--verbose] [--output FILE]

    # From Python:
    from tools.vliw_packer.vliw_packer import VLIWPacker
    packer = VLIWPacker()
    packed = packer.pack(instructions)
"""

import sys
import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Tuple, Any
from collections import defaultdict
from enum import Enum

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import SLOT_LIMITS, VLEN

# Slot limits per engine (excluding debug)
ENGINES = {k: v for k, v in SLOT_LIMITS.items() if k != "debug"}


class HazardType(Enum):
    RAW = "Read-After-Write"   # True dependency - must preserve order
    WAW = "Write-After-Write"  # Output dependency - must preserve order
    WAR = "Write-After-Read"   # Anti-dependency - can sometimes reorder with renaming


@dataclass
class Instruction:
    """Represents a single instruction (one slot in one engine)."""
    id: int
    engine: str
    slot: tuple
    reads: Set[int] = field(default_factory=set)   # Scratch addresses read
    writes: Set[int] = field(default_factory=set)  # Scratch addresses written
    is_flow_control: bool = False  # Jump, pause, halt - affects ordering
    original_cycle: int = 0  # Original position in unpacked stream

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class DependencyEdge:
    """An edge in the dependency graph."""
    from_instr: Instruction
    to_instr: Instruction
    hazard_type: HazardType
    register: int  # The scratch address causing the dependency


@dataclass
class PackingStats:
    """Statistics about the packing results."""
    original_cycles: int
    packed_cycles: int
    cycles_saved: int
    speedup: float
    original_utilization: float
    packed_utilization: float
    utilization_improvement: float
    per_engine_stats: Dict[str, Dict[str, Any]]

    def to_dict(self) -> dict:
        return {
            "original_cycles": self.original_cycles,
            "packed_cycles": self.packed_cycles,
            "cycles_saved": self.cycles_saved,
            "speedup": round(self.speedup, 3),
            "original_utilization_pct": round(self.original_utilization, 2),
            "packed_utilization_pct": round(self.packed_utilization, 2),
            "utilization_improvement_pct": round(self.utilization_improvement, 2),
            "per_engine": self.per_engine_stats
        }


def extract_reads_writes(slot: tuple, engine: str) -> Tuple[Set[int], Set[int]]:
    """
    Extract scratch addresses read and written by an instruction slot.
    Returns (reads, writes) sets.

    This handles all instruction types in the VLIW ISA.
    """
    reads = set()
    writes = set()

    if not slot or len(slot) == 0:
        return reads, writes

    op = slot[0]

    if engine == "alu":
        # ALU: (op, dest, src1, src2)
        if len(slot) >= 4:
            writes.add(slot[1])
            reads.add(slot[2])
            reads.add(slot[3])

    elif engine == "valu":
        # VALU operations work on vectors of VLEN elements
        if op == "vbroadcast":
            # (vbroadcast, dest, src) - dest is vector, src is scalar
            if len(slot) >= 3:
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                reads.add(slot[2])
        elif op == "multiply_add":
            # (multiply_add, dest, a, b, c)
            if len(slot) >= 5:
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)
                    reads.add(slot[4] + i)
        else:
            # Standard valu: (op, dest, src1, src2)
            if len(slot) >= 4:
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)

    elif engine == "load":
        if op == "load":
            # (load, dest, addr_reg)
            if len(slot) >= 3:
                writes.add(slot[1])
                reads.add(slot[2])
        elif op == "load_offset":
            # (load_offset, dest, addr, offset)
            if len(slot) >= 4:
                writes.add(slot[1] + slot[3])
                reads.add(slot[2] + slot[3])
        elif op == "vload":
            # (vload, dest, addr) - loads VLEN elements
            if len(slot) >= 3:
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                reads.add(slot[2])
        elif op == "const":
            # (const, dest, value) - no reads
            if len(slot) >= 2:
                writes.add(slot[1])

    elif engine == "store":
        if op == "store":
            # (store, addr_reg, src)
            if len(slot) >= 3:
                reads.add(slot[1])
                reads.add(slot[2])
        elif op == "vstore":
            # (vstore, addr, src) - stores VLEN elements
            if len(slot) >= 3:
                reads.add(slot[1])
                for i in range(VLEN):
                    reads.add(slot[2] + i)

    elif engine == "flow":
        if op == "select":
            # (select, dest, cond, a, b)
            if len(slot) >= 5:
                writes.add(slot[1])
                reads.add(slot[2])
                reads.add(slot[3])
                reads.add(slot[4])
        elif op == "vselect":
            # (vselect, dest, cond, a, b) - vector select
            if len(slot) >= 5:
                for i in range(VLEN):
                    writes.add(slot[1] + i)
                    reads.add(slot[2] + i)
                    reads.add(slot[3] + i)
                    reads.add(slot[4] + i)
        elif op == "add_imm":
            # (add_imm, dest, src, imm)
            if len(slot) >= 3:
                writes.add(slot[1])
                reads.add(slot[2])
        elif op in ("cond_jump", "cond_jump_rel"):
            if len(slot) >= 2:
                reads.add(slot[1])
        elif op == "jump_indirect":
            if len(slot) >= 2:
                reads.add(slot[1])
        elif op == "trace_write":
            if len(slot) >= 2:
                reads.add(slot[1])
        elif op == "coreid":
            if len(slot) >= 2:
                writes.add(slot[1])
        # pause, halt, jump don't read/write scratch

    return reads, writes


def is_flow_control(slot: tuple, engine: str) -> bool:
    """Check if instruction affects control flow and must preserve order."""
    if engine != "flow":
        return False
    if not slot:
        return False
    op = slot[0]
    return op in ("pause", "halt", "jump", "cond_jump", "cond_jump_rel", "jump_indirect")


class VLIWPacker:
    """
    VLIW instruction packer using list scheduling with dependency analysis.

    The algorithm:
    1. Flatten all instruction bundles into individual instructions
    2. Build dependency graph (RAW, WAW, WAR hazards)
    3. Use list scheduling to pack instructions:
       - Maintain ready queue of instructions with satisfied dependencies
       - Pack as many ready instructions as slot limits allow
       - Mark packed instructions as complete, update ready queue
    4. Output packed bundles

    Key optimizations:
    - Prioritizes instructions on critical path
    - Respects program semantics for flow control
    - Minimizes register pressure by scheduling writes early
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.instructions: List[Instruction] = []
        self.dependencies: List[DependencyEdge] = []
        self.successors: Dict[int, List[Instruction]] = defaultdict(list)
        self.predecessors: Dict[int, List[Instruction]] = defaultdict(list)
        self.in_degree: Dict[int, int] = {}

    def _flatten_instructions(self, bundles: List[dict]) -> List[Instruction]:
        """Convert instruction bundles into flat list of individual instructions."""
        instructions = []
        instr_id = 0

        for cycle_num, bundle in enumerate(bundles):
            for engine, slots in bundle.items():
                if engine == "debug":
                    continue
                if not slots:
                    continue
                for slot in slots:
                    reads, writes = extract_reads_writes(slot, engine)
                    flow_ctrl = is_flow_control(slot, engine)

                    instr = Instruction(
                        id=instr_id,
                        engine=engine,
                        slot=slot,
                        reads=reads,
                        writes=writes,
                        is_flow_control=flow_ctrl,
                        original_cycle=cycle_num
                    )
                    instructions.append(instr)
                    instr_id += 1

        return instructions

    def _build_dependency_graph(self, instructions: List[Instruction]):
        """
        Build dependency graph identifying all hazards.

        We track:
        - RAW (Read-After-Write): Instruction B reads what A writes
        - WAW (Write-After-Write): Both write same location
        - WAR (Write-After-Read): B writes what A reads

        For flow control instructions, we add edges to preserve program order.
        """
        self.dependencies = []
        self.successors = defaultdict(list)
        self.predecessors = defaultdict(list)
        self.in_degree = {instr.id: 0 for instr in instructions}

        # Track last writer and readers for each scratch address
        last_writer: Dict[int, Instruction] = {}
        last_readers: Dict[int, List[Instruction]] = defaultdict(list)

        # Also track flow control for ordering
        last_flow_control: Optional[Instruction] = None

        for instr in instructions:
            # RAW: This instruction reads something that was written before
            for addr in instr.reads:
                if addr in last_writer:
                    writer = last_writer[addr]
                    self._add_edge(writer, instr, HazardType.RAW, addr)

            # WAW: This instruction writes something that was written before
            for addr in instr.writes:
                if addr in last_writer:
                    writer = last_writer[addr]
                    self._add_edge(writer, instr, HazardType.WAW, addr)

            # WAR: This instruction writes something that was read before
            # (anti-dependency - could be eliminated with register renaming)
            for addr in instr.writes:
                for reader in last_readers.get(addr, []):
                    if reader.id != instr.id:
                        self._add_edge(reader, instr, HazardType.WAR, addr)

            # Flow control ordering: flow control instructions must stay in order
            if instr.is_flow_control:
                if last_flow_control is not None:
                    self._add_edge(last_flow_control, instr, HazardType.RAW, -1)
                last_flow_control = instr
            elif last_flow_control is not None:
                # Non-flow instructions after a flow control must wait
                # This is conservative but safe for jumps/pauses
                self._add_edge(last_flow_control, instr, HazardType.RAW, -1)

            # Update tracking
            for addr in instr.writes:
                last_writer[addr] = instr
                last_readers[addr] = []  # Clear readers after write

            for addr in instr.reads:
                if instr not in last_readers[addr]:
                    last_readers[addr].append(instr)

    def _add_edge(self, from_instr: Instruction, to_instr: Instruction,
                  hazard_type: HazardType, register: int):
        """Add a dependency edge, avoiding duplicates."""
        # Check for existing edge
        for succ in self.successors[from_instr.id]:
            if succ.id == to_instr.id:
                return  # Edge already exists

        edge = DependencyEdge(from_instr, to_instr, hazard_type, register)
        self.dependencies.append(edge)
        self.successors[from_instr.id].append(to_instr)
        self.predecessors[to_instr.id].append(from_instr)
        self.in_degree[to_instr.id] += 1

    def _calculate_priorities(self, instructions: List[Instruction]) -> Dict[int, int]:
        """
        Calculate scheduling priority for each instruction.

        Uses critical path length from this instruction to end.
        Higher priority = longer path = should schedule earlier.
        """
        # Build reverse graph and do topological traversal from outputs
        priorities = {instr.id: 1 for instr in instructions}

        # Process in reverse topological order
        # Instructions with no successors have priority 1
        # Others have max(successor priorities) + 1

        # Find all instructions with no successors (outputs)
        worklist = [instr for instr in instructions
                   if not self.successors[instr.id]]

        visited = set()
        while worklist:
            instr = worklist.pop()
            if instr.id in visited:
                continue
            visited.add(instr.id)

            # Calculate priority based on successors
            succ_priorities = [priorities[s.id] for s in self.successors[instr.id]]
            if succ_priorities:
                priorities[instr.id] = max(succ_priorities) + 1

            # Add predecessors to worklist
            for pred in self.predecessors[instr.id]:
                if pred.id not in visited:
                    worklist.append(pred)

        return priorities

    def _can_pack_into_bundle(self, bundle: Dict[str, List[tuple]],
                               instr: Instruction) -> bool:
        """Check if instruction can be added to bundle within slot limits."""
        engine = instr.engine
        current_count = len(bundle.get(engine, []))
        limit = ENGINES.get(engine, 0)
        return current_count < limit

    def _add_to_bundle(self, bundle: Dict[str, List[tuple]], instr: Instruction):
        """Add instruction to bundle."""
        if instr.engine not in bundle:
            bundle[instr.engine] = []
        bundle[instr.engine].append(instr.slot)

    def pack(self, bundles: List[dict]) -> Tuple[List[dict], PackingStats]:
        """
        Pack instruction bundles using list scheduling.

        Args:
            bundles: List of instruction bundles (dict of engine -> list of slots)

        Returns:
            Tuple of (packed_bundles, statistics)
        """
        # Step 1: Flatten to individual instructions
        self.instructions = self._flatten_instructions(bundles)

        if not self.instructions:
            stats = PackingStats(
                original_cycles=len(bundles),
                packed_cycles=0,
                cycles_saved=len(bundles),
                speedup=float('inf') if len(bundles) > 0 else 1.0,
                original_utilization=0.0,
                packed_utilization=0.0,
                utilization_improvement=0.0,
                per_engine_stats={}
            )
            return [], stats

        # Step 2: Build dependency graph
        self._build_dependency_graph(self.instructions)

        # Step 3: Calculate priorities (critical path length)
        priorities = self._calculate_priorities(self.instructions)

        # Step 4: List scheduling
        packed_bundles = []
        scheduled = set()
        remaining_in_degree = dict(self.in_degree)

        # Ready queue: instructions with no unscheduled predecessors
        ready = [instr for instr in self.instructions
                if remaining_in_degree[instr.id] == 0]

        # Sort by priority (higher = schedule first)
        ready.sort(key=lambda i: -priorities[i.id])

        while ready or len(scheduled) < len(self.instructions):
            if not ready:
                # Circular dependency or bug - should not happen
                unscheduled = [i for i in self.instructions if i.id not in scheduled]
                if self.verbose:
                    print(f"Warning: {len(unscheduled)} instructions could not be scheduled")
                    for instr in unscheduled[:5]:
                        print(f"  {instr.id}: {instr.engine} {instr.slot}")
                break

            # Create new bundle
            bundle: Dict[str, List[tuple]] = {}
            scheduled_this_cycle = []

            # Try to pack as many ready instructions as possible
            remaining_ready = []
            for instr in ready:
                if self._can_pack_into_bundle(bundle, instr):
                    self._add_to_bundle(bundle, instr)
                    scheduled_this_cycle.append(instr)
                    scheduled.add(instr.id)
                else:
                    remaining_ready.append(instr)

            # Update ready queue
            ready = remaining_ready

            # Add newly ready instructions
            for instr in scheduled_this_cycle:
                for succ in self.successors[instr.id]:
                    remaining_in_degree[succ.id] -= 1
                    if remaining_in_degree[succ.id] == 0 and succ.id not in scheduled:
                        ready.append(succ)

            # Re-sort by priority
            ready.sort(key=lambda i: -priorities[i.id])

            if bundle:
                packed_bundles.append(bundle)

        # Step 5: Calculate statistics
        stats = self._calculate_stats(bundles, packed_bundles)

        return packed_bundles, stats

    def _calculate_stats(self, original: List[dict], packed: List[dict]) -> PackingStats:
        """Calculate packing statistics."""
        max_slots = sum(ENGINES.values())

        # Original stats
        orig_cycles = len(original)
        orig_slots = sum(
            sum(len(slots) for engine, slots in bundle.items() if engine != "debug" and slots)
            for bundle in original
        )
        orig_util = 100.0 * orig_slots / (orig_cycles * max_slots) if orig_cycles > 0 else 0

        # Packed stats
        pack_cycles = len(packed)
        pack_slots = sum(
            sum(len(slots) for engine, slots in bundle.items() if engine != "debug" and slots)
            for bundle in packed
        )
        pack_util = 100.0 * pack_slots / (pack_cycles * max_slots) if pack_cycles > 0 else 0

        # Per-engine breakdown
        per_engine = {}
        for engine, limit in ENGINES.items():
            orig_count = sum(
                len(bundle.get(engine, []))
                for bundle in original
            )
            pack_count = sum(
                len(bundle.get(engine, []))
                for bundle in packed
            )
            per_engine[engine] = {
                "instructions": orig_count,
                "original_utilization_pct": round(100.0 * orig_count / (orig_cycles * limit), 2) if orig_cycles > 0 else 0,
                "packed_utilization_pct": round(100.0 * pack_count / (pack_cycles * limit), 2) if pack_cycles > 0 else 0
            }

        return PackingStats(
            original_cycles=orig_cycles,
            packed_cycles=pack_cycles,
            cycles_saved=orig_cycles - pack_cycles,
            speedup=orig_cycles / pack_cycles if pack_cycles > 0 else float('inf'),
            original_utilization=orig_util,
            packed_utilization=pack_util,
            utilization_improvement=pack_util - orig_util,
            per_engine_stats=per_engine
        )

    def get_dependency_info(self) -> dict:
        """Get information about dependencies for debugging."""
        raw_count = sum(1 for d in self.dependencies if d.hazard_type == HazardType.RAW)
        waw_count = sum(1 for d in self.dependencies if d.hazard_type == HazardType.WAW)
        war_count = sum(1 for d in self.dependencies if d.hazard_type == HazardType.WAR)

        return {
            "total_instructions": len(self.instructions),
            "total_dependencies": len(self.dependencies),
            "raw_dependencies": raw_count,
            "waw_dependencies": waw_count,
            "war_dependencies": war_count,
            "avg_in_degree": sum(self.in_degree.values()) / len(self.in_degree) if self.in_degree else 0
        }


class AggressiveVLIWPacker(VLIWPacker):
    """
    More aggressive packer that tries to minimize cycles at all costs.

    Additional optimizations:
    - Reorders independent instructions more aggressively
    - Packs based on engine slot availability
    - Uses bin-packing heuristics
    """

    def _calculate_engine_demand(self, instructions: List[Instruction]) -> Dict[str, int]:
        """Calculate total demand per engine."""
        demand = defaultdict(int)
        for instr in instructions:
            demand[instr.engine] += 1
        return dict(demand)

    def _select_instruction(self, ready: List[Instruction],
                           bundle: Dict[str, List[tuple]],
                           priorities: Dict[int, int]) -> Optional[Instruction]:
        """
        Select best instruction to add to bundle.

        Considers:
        1. Priority (critical path length)
        2. Engine slot availability
        3. Opportunity cost (scarce resources)
        """
        # Filter to only instructions that can fit
        candidates = [i for i in ready if self._can_pack_into_bundle(bundle, i)]

        if not candidates:
            return None

        # Score each candidate
        def score(instr: Instruction) -> tuple:
            # Higher priority is better
            priority_score = priorities[instr.id]

            # Prefer filling scarce slots
            engine = instr.engine
            limit = ENGINES.get(engine, 1)
            scarcity = 1.0 / limit  # Lower limit = higher scarcity

            # Prefer instructions that don't block others
            blocking = len(self.successors[instr.id])

            return (priority_score, scarcity, -blocking)

        candidates.sort(key=score, reverse=True)
        return candidates[0]

    def pack(self, bundles: List[dict]) -> Tuple[List[dict], PackingStats]:
        """
        Pack using more aggressive selection heuristics.
        """
        # Step 1: Flatten to individual instructions
        self.instructions = self._flatten_instructions(bundles)

        if not self.instructions:
            stats = PackingStats(
                original_cycles=len(bundles),
                packed_cycles=0,
                cycles_saved=len(bundles),
                speedup=float('inf') if len(bundles) > 0 else 1.0,
                original_utilization=0.0,
                packed_utilization=0.0,
                utilization_improvement=0.0,
                per_engine_stats={}
            )
            return [], stats

        # Step 2: Build dependency graph
        self._build_dependency_graph(self.instructions)

        # Step 3: Calculate priorities
        priorities = self._calculate_priorities(self.instructions)

        # Step 4: Aggressive list scheduling
        packed_bundles = []
        scheduled = set()
        remaining_in_degree = dict(self.in_degree)

        ready = [instr for instr in self.instructions
                if remaining_in_degree[instr.id] == 0]

        while ready or len(scheduled) < len(self.instructions):
            if not ready:
                break

            bundle: Dict[str, List[tuple]] = {}
            scheduled_this_cycle = []

            # Iteratively select best instruction until no more fit
            while True:
                best = self._select_instruction(ready, bundle, priorities)
                if best is None:
                    break

                self._add_to_bundle(bundle, best)
                scheduled_this_cycle.append(best)
                scheduled.add(best.id)
                ready.remove(best)

            # Update ready queue
            for instr in scheduled_this_cycle:
                for succ in self.successors[instr.id]:
                    remaining_in_degree[succ.id] -= 1
                    if remaining_in_degree[succ.id] == 0 and succ.id not in scheduled:
                        ready.append(succ)

            if bundle:
                packed_bundles.append(bundle)

        stats = self._calculate_stats(bundles, packed_bundles)
        return packed_bundles, stats


def pack_kernel(bundles: List[dict], aggressive: bool = False,
                verbose: bool = False) -> Tuple[List[dict], PackingStats]:
    """
    Convenience function to pack a kernel.

    Args:
        bundles: List of instruction bundles
        aggressive: Use aggressive packing heuristics
        verbose: Print debug information

    Returns:
        Tuple of (packed_bundles, statistics)
    """
    if aggressive:
        packer = AggressiveVLIWPacker(verbose=verbose)
    else:
        packer = VLIWPacker(verbose=verbose)

    return packer.pack(bundles)


def print_stats(stats: PackingStats, verbose: bool = False):
    """Print packing statistics."""
    print("=" * 60)
    print("VLIW PACKING RESULTS")
    print("=" * 60)
    print()
    print(f"Original cycles:    {stats.original_cycles:,}")
    print(f"Packed cycles:      {stats.packed_cycles:,}")
    print(f"Cycles saved:       {stats.cycles_saved:,}")
    print(f"Speedup:            {stats.speedup:.2f}x")
    print()
    print(f"Original utilization: {stats.original_utilization:.1f}%")
    print(f"Packed utilization:   {stats.packed_utilization:.1f}%")
    print(f"Improvement:          +{stats.utilization_improvement:.1f}%")

    if verbose:
        print()
        print("-" * 60)
        print("PER-ENGINE BREAKDOWN")
        print("-" * 60)
        print(f"{'Engine':<10} {'Instrs':>10} {'Orig Util':>12} {'Pack Util':>12}")
        print("-" * 60)
        for engine, data in stats.per_engine_stats.items():
            print(f"{engine:<10} {data['instructions']:>10} "
                  f"{data['original_utilization_pct']:>11.1f}% "
                  f"{data['packed_utilization_pct']:>11.1f}%")
    print()


def analyze_current_kernel():
    """Analyze and pack the current kernel from perf_takehome.py"""
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
    parser = argparse.ArgumentParser(description="VLIW Auto-Packer")
    parser.add_argument("--aggressive", "-a", action="store_true",
                       help="Use aggressive packing heuristics")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print detailed statistics")
    parser.add_argument("--json", action="store_true",
                       help="Output statistics as JSON")
    parser.add_argument("--output", "-o", metavar="FILE",
                       help="Save packed kernel to JSON file")
    parser.add_argument("--deps", "-d", action="store_true",
                       help="Print dependency analysis")
    args = parser.parse_args()

    print("Loading kernel...", file=sys.stderr)
    instructions = analyze_current_kernel()

    print(f"Packing {len(instructions)} instruction bundles...", file=sys.stderr)

    if args.aggressive:
        packer = AggressiveVLIWPacker(verbose=args.verbose)
    else:
        packer = VLIWPacker(verbose=args.verbose)

    packed, stats = packer.pack(instructions)

    if args.deps:
        print()
        print("=" * 60)
        print("DEPENDENCY ANALYSIS")
        print("=" * 60)
        dep_info = packer.get_dependency_info()
        for key, value in dep_info.items():
            print(f"  {key}: {value}")
        print()

    if args.json:
        print(json.dumps(stats.to_dict(), indent=2))
    else:
        print_stats(stats, args.verbose)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(packed, f, indent=2)
        print(f"Packed kernel saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
