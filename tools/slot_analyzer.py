#!/usr/bin/env python3
"""
Slot Utilization Analyzer for VLIW SIMD Kernel

Analyzes instruction streams to show:
1. Per-cycle slot utilization (how many slots used vs available)
2. Engine-by-engine breakdown
3. Waste identification (cycles with low utilization)
4. Utilization histogram
5. Critical path estimation
6. Dependency analysis (RAW hazards)
7. Optimization recommendations
8. Kernel comparison (before/after diff)

Usage:
    python tools/slot_analyzer.py [--verbose] [--top N] [--histogram]
    python tools/slot_analyzer.py --compare kernel1.json kernel2.json
"""

import sys
import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Tuple, Any
from collections import defaultdict
from enum import Enum

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem import SLOT_LIMITS, VLEN

# Try to import Rich for better formatting
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Engine slot limits (excluding debug)
ENGINES = {k: v for k, v in SLOT_LIMITS.items() if k != "debug"}
MAX_SLOTS_PER_CYCLE = sum(ENGINES.values())  # 12 + 6 + 2 + 2 + 1 = 23


class HazardType(Enum):
    RAW = "Read-After-Write"  # True dependency
    WAW = "Write-After-Write"  # Output dependency
    WAR = "Write-After-Read"  # Anti-dependency


@dataclass
class Dependency:
    """Represents a data dependency between instructions."""
    from_cycle: int
    to_cycle: int
    register: int  # scratch address
    hazard_type: HazardType
    from_op: str
    to_op: str


@dataclass
class CycleStats:
    cycle_num: int
    slots_used: dict[str, int]
    instructions: list[tuple]
    writes: set = field(default_factory=set)  # scratch addresses written
    reads: set = field(default_factory=set)   # scratch addresses read

    @property
    def total_used(self) -> int:
        return sum(self.slots_used.values())

    @property
    def utilization_pct(self) -> float:
        return 100.0 * self.total_used / MAX_SLOTS_PER_CYCLE

    def is_low_utilization(self, threshold: float = 50.0) -> bool:
        return self.utilization_pct < threshold


@dataclass
class DependencyAnalysis:
    """Results of dependency analysis."""
    dependencies: List[Dependency]
    critical_path_length: int
    critical_path: List[int]  # cycle numbers on critical path
    raw_hazard_count: int
    blocking_pairs: List[Tuple[int, int]]  # (cycle1, cycle2) pairs that can't be packed due to RAW


@dataclass
class AnalysisResult:
    total_cycles: int
    total_slots_used: int
    per_engine: dict[str, dict]
    utilization_pct: float
    low_util_cycles: list[CycleStats]
    histogram: dict[int, int]
    per_cycle: list[CycleStats]
    dependency_analysis: Optional[DependencyAnalysis] = None
    packing_opportunities: Optional[dict] = None
    recommendations: Optional[List[dict]] = None

    def to_dict(self) -> dict:
        result = {
            "total_cycles": self.total_cycles,
            "total_slots_used": self.total_slots_used,
            "max_possible_slots": self.total_cycles * MAX_SLOTS_PER_CYCLE,
            "utilization_pct": round(self.utilization_pct, 2),
            "slots_per_cycle_avg": round(self.total_slots_used / max(1, self.total_cycles), 2),
            "per_engine": self.per_engine,
            "histogram": self.histogram,
            "low_utilization_cycles_count": len(self.low_util_cycles),
            "theoretical_speedup_if_full": round(MAX_SLOTS_PER_CYCLE / max(0.01, self.total_slots_used / max(1, self.total_cycles)), 2)
        }
        if self.dependency_analysis:
            result["dependency_analysis"] = {
                "critical_path_length": self.dependency_analysis.critical_path_length,
                "raw_hazard_count": self.dependency_analysis.raw_hazard_count,
                "blocking_pairs_count": len(self.dependency_analysis.blocking_pairs)
            }
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def extract_reads_writes(slot: tuple, engine: str) -> Tuple[Set[int], Set[int]]:
    """
    Extract scratch addresses read and written by an instruction slot.
    Returns (reads, writes) sets.
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

    return reads, writes


def analyze_dependencies(instructions: list[dict]) -> DependencyAnalysis:
    """
    Analyze data dependencies between instructions to find:
    1. RAW (Read-After-Write) hazards that prevent packing
    2. Critical path through the dependency graph
    """
    dependencies = []
    blocking_pairs = []

    # Build per-cycle read/write sets
    cycle_info = []
    for cycle_num, instr in enumerate(instructions):
        cycle_reads = set()
        cycle_writes = set()
        cycle_ops = []

        for engine, slots in instr.items():
            if engine == "debug":
                continue
            for slot in (slots or []):
                reads, writes = extract_reads_writes(slot, engine)
                cycle_reads.update(reads)
                cycle_writes.update(writes)
                cycle_ops.append((engine, slot[0] if slot else ""))

        cycle_info.append({
            "reads": cycle_reads,
            "writes": cycle_writes,
            "ops": cycle_ops
        })

    # Find RAW dependencies between adjacent cycles
    for i in range(len(cycle_info) - 1):
        curr = cycle_info[i]
        next_cycle = cycle_info[i + 1]

        # RAW: next reads what current writes
        raw_regs = curr["writes"] & next_cycle["reads"]
        if raw_regs:
            for reg in raw_regs:
                dependencies.append(Dependency(
                    from_cycle=i,
                    to_cycle=i + 1,
                    register=reg,
                    hazard_type=HazardType.RAW,
                    from_op=str(curr["ops"]),
                    to_op=str(next_cycle["ops"])
                ))
            blocking_pairs.append((i, i + 1))

    # Build dependency graph for critical path analysis
    # Each node is a cycle, edges are dependencies
    # Use longest path algorithm (topological order since it's a DAG)

    n = len(instructions)
    if n == 0:
        return DependencyAnalysis(
            dependencies=dependencies,
            critical_path_length=0,
            critical_path=[],
            raw_hazard_count=len([d for d in dependencies if d.hazard_type == HazardType.RAW]),
            blocking_pairs=blocking_pairs
        )

    # Build adjacency list from dependencies
    # Also consider implicit sequential dependencies for instructions
    # that can't be reordered due to program semantics
    adj = defaultdict(list)
    for dep in dependencies:
        adj[dep.from_cycle].append(dep.to_cycle)

    # For critical path, we need to find the longest path
    # Since cycles are numbered sequentially, we can use DP
    dist = [1] * n  # Each instruction takes at least 1 cycle
    pred = [-1] * n

    # Process all RAW dependencies
    for dep in dependencies:
        if dep.hazard_type == HazardType.RAW:
            if dist[dep.from_cycle] + 1 > dist[dep.to_cycle]:
                dist[dep.to_cycle] = dist[dep.from_cycle] + 1
                pred[dep.to_cycle] = dep.from_cycle

    # Find the critical path length
    critical_path_length = max(dist) if dist else 0

    # Reconstruct critical path
    end_cycle = dist.index(max(dist)) if dist else -1
    critical_path = []
    curr = end_cycle
    while curr != -1:
        critical_path.append(curr)
        curr = pred[curr]
    critical_path.reverse()

    return DependencyAnalysis(
        dependencies=dependencies,
        critical_path_length=critical_path_length,
        critical_path=critical_path,
        raw_hazard_count=len([d for d in dependencies if d.hazard_type == HazardType.RAW]),
        blocking_pairs=blocking_pairs
    )


def analyze_instructions(instructions: list[dict], include_deps: bool = True) -> AnalysisResult:
    """
    Analyze a list of instruction bundles for slot utilization.

    Each instruction is a dict like:
        {"alu": [(...), (...)], "load": [(...)] }
    """
    per_cycle_stats = []
    engine_totals = defaultdict(int)
    histogram = defaultdict(int)

    for cycle_num, instr in enumerate(instructions):
        slots_used = {}
        cycle_instrs = []
        cycle_reads = set()
        cycle_writes = set()

        for engine, slots in instr.items():
            if engine == "debug":
                continue
            count = len(slots) if slots else 0
            slots_used[engine] = count
            engine_totals[engine] += count
            if slots:
                cycle_instrs.extend(slots)
                for slot in slots:
                    reads, writes = extract_reads_writes(slot, engine)
                    cycle_reads.update(reads)
                    cycle_writes.update(writes)

        # Fill in zeros for unused engines
        for engine in ENGINES:
            if engine not in slots_used:
                slots_used[engine] = 0

        stats = CycleStats(
            cycle_num=cycle_num,
            slots_used=slots_used,
            instructions=cycle_instrs,
            reads=cycle_reads,
            writes=cycle_writes
        )
        per_cycle_stats.append(stats)
        histogram[stats.total_used] += 1

    total_cycles = len(per_cycle_stats)
    total_slots = sum(s.total_used for s in per_cycle_stats)

    # Per-engine stats
    per_engine = {}
    for engine, limit in ENGINES.items():
        used = engine_totals[engine]
        per_engine[engine] = {
            "total_used": used,
            "max_per_cycle": limit,
            "max_possible": limit * total_cycles,
            "utilization_pct": round(100.0 * used / max(1, limit * total_cycles), 2),
            "avg_per_cycle": round(used / max(1, total_cycles), 2)
        }

    # Find low utilization cycles
    low_util = [s for s in per_cycle_stats if s.is_low_utilization(50.0)]

    utilization_pct = 100.0 * total_slots / max(1, total_cycles * MAX_SLOTS_PER_CYCLE)

    # Dependency analysis
    dep_analysis = None
    if include_deps:
        dep_analysis = analyze_dependencies(instructions)

    return AnalysisResult(
        total_cycles=total_cycles,
        total_slots_used=total_slots,
        per_engine=per_engine,
        utilization_pct=utilization_pct,
        low_util_cycles=low_util,
        histogram=dict(sorted(histogram.items())),
        per_cycle=per_cycle_stats,
        dependency_analysis=dep_analysis
    )


def analyze_packing_opportunities(instructions: list[dict], dep_analysis: Optional[DependencyAnalysis] = None) -> dict:
    """
    Analyze consecutive instructions to find packing opportunities.
    Now considers RAW hazards that prevent packing.
    """
    opportunities = []
    blocked_by_deps = []
    blocking_pairs_set = set(dep_analysis.blocking_pairs) if dep_analysis else set()

    i = 0
    while i < len(instructions) - 1:
        curr = instructions[i]
        next_instr = instructions[i + 1]

        # Check if they could be combined based on slot limits
        can_pack_slots = True
        combined_slots = defaultdict(int)

        for engine, slots in curr.items():
            if engine == "debug":
                continue
            combined_slots[engine] += len(slots) if slots else 0

        for engine, slots in next_instr.items():
            if engine == "debug":
                continue
            combined_slots[engine] += len(slots) if slots else 0

        # Check slot limits
        for engine, count in combined_slots.items():
            if count > ENGINES.get(engine, 0):
                can_pack_slots = False
                break

        # Check for RAW hazards
        has_raw_hazard = (i, i + 1) in blocking_pairs_set

        if can_pack_slots:
            if has_raw_hazard:
                blocked_by_deps.append({
                    "cycle": i,
                    "reason": "RAW hazard",
                    "combined_slots": dict(combined_slots)
                })
            else:
                opportunities.append({
                    "cycle": i,
                    "curr_engines": list(curr.keys()),
                    "next_engines": list(next_instr.keys()),
                    "combined_slots": dict(combined_slots)
                })

        i += 1

    return {
        "total_opportunities": len(opportunities),
        "blocked_by_dependencies": len(blocked_by_deps),
        "potential_cycle_savings": len(opportunities),
        "sample_opportunities": opportunities[:20],
        "sample_blocked": blocked_by_deps[:10]
    }


def analyze_packing_by_type(instructions: list[dict]) -> dict:
    """Categorize packing opportunities by engine combination."""
    combos = defaultdict(int)

    for i in range(len(instructions) - 1):
        curr = instructions[i]
        next_instr = instructions[i + 1]

        curr_eng = frozenset(e for e in curr.keys() if e != "debug" and curr[e])
        next_eng = frozenset(e for e in next_instr.keys() if e != "debug" and next_instr[e])

        if curr_eng and next_eng:
            # Check if packable
            combined_slots = defaultdict(int)
            for engine, slots in curr.items():
                if engine != "debug":
                    combined_slots[engine] += len(slots) if slots else 0
            for engine, slots in next_instr.items():
                if engine != "debug":
                    combined_slots[engine] += len(slots) if slots else 0

            can_pack = all(combined_slots[e] <= ENGINES.get(e, 0) for e in combined_slots)
            if can_pack:
                combo_key = (tuple(sorted(curr_eng)), tuple(sorted(next_eng)))
                combos[combo_key] += 1

    return dict(combos)


def generate_recommendations(result: AnalysisResult, packing_opps: dict) -> List[dict]:
    """
    Generate prioritized optimization recommendations based on analysis.
    """
    recommendations = []

    # 1. Critical path analysis
    if result.dependency_analysis:
        dep = result.dependency_analysis
        if dep.critical_path_length > 0:
            theoretical_min = dep.critical_path_length
            current = result.total_cycles
            if current > theoretical_min * 1.5:
                recommendations.append({
                    "priority": 1,
                    "category": "Critical Path",
                    "title": "Long dependency chains detected",
                    "description": f"Critical path is {dep.critical_path_length} cycles, but kernel runs {current} cycles. "
                                   f"Consider breaking long dependency chains or using instruction-level parallelism.",
                    "potential_savings": f"Up to {current - theoretical_min} cycles",
                    "difficulty": "Medium"
                })

    # 2. Low utilization cycles
    low_util_count = len(result.low_util_cycles)
    if low_util_count > result.total_cycles * 0.3:
        recommendations.append({
            "priority": 1,
            "category": "Slot Utilization",
            "title": "Many low-utilization cycles",
            "description": f"{low_util_count} cycles ({100*low_util_count/result.total_cycles:.1f}%) have <50% slot utilization. "
                           "Consider packing more operations per cycle.",
            "potential_savings": f"Could potentially reduce cycles significantly",
            "difficulty": "Medium"
        })

    # 3. Packing opportunities
    if packing_opps:
        packable = packing_opps.get("total_opportunities", 0)
        blocked = packing_opps.get("blocked_by_dependencies", 0)

        if packable > 10:
            recommendations.append({
                "priority": 2,
                "category": "Instruction Packing",
                "title": f"{packable} adjacent instruction pairs can be packed",
                "description": "These instruction pairs fit within slot limits and have no RAW hazards. "
                               "Combining them could save one cycle each.",
                "potential_savings": f"Up to {packable} cycles",
                "difficulty": "Easy"
            })

        if blocked > 10:
            recommendations.append({
                "priority": 3,
                "category": "Dependency Breaking",
                "title": f"{blocked} packings blocked by RAW hazards",
                "description": "These pairs would fit in slot limits but have read-after-write dependencies. "
                               "Consider reordering or computing intermediate values differently.",
                "potential_savings": f"Up to {blocked} cycles if dependencies can be removed",
                "difficulty": "Hard"
            })

    # 4. Per-engine analysis
    for engine, stats in result.per_engine.items():
        util = stats["utilization_pct"]
        if util < 10 and stats["total_used"] > 0:
            recommendations.append({
                "priority": 3,
                "category": "Engine Balance",
                "title": f"{engine} engine severely underutilized ({util:.1f}%)",
                "description": f"The {engine} engine has {stats['max_per_cycle']} slots/cycle but averages only "
                               f"{stats['avg_per_cycle']:.2f}. Consider moving work to this engine or packing more {engine} ops.",
                "potential_savings": "Varies",
                "difficulty": "Medium"
            })

    # 5. Load/Store bottleneck detection
    load_util = result.per_engine.get("load", {}).get("utilization_pct", 0)
    store_util = result.per_engine.get("store", {}).get("utilization_pct", 0)
    alu_util = result.per_engine.get("alu", {}).get("utilization_pct", 0)
    valu_util = result.per_engine.get("valu", {}).get("utilization_pct", 0)

    if (load_util > 50 or store_util > 50) and (alu_util < 20 and valu_util < 20):
        recommendations.append({
            "priority": 2,
            "category": "Memory Bound",
            "title": "Kernel appears memory-bound",
            "description": "Load/Store utilization is high while ALU/VALU utilization is low. "
                           "Consider prefetching, vectorization, or computing more values in registers.",
            "potential_savings": "Significant if memory access patterns can be optimized",
            "difficulty": "Hard"
        })

    # 6. Vector utilization
    if valu_util < alu_util * 0.5 and alu_util > 20:
        recommendations.append({
            "priority": 2,
            "category": "Vectorization",
            "title": "Consider more vectorization",
            "description": f"ALU utilization ({alu_util:.1f}%) is much higher than VALU ({valu_util:.1f}%). "
                           f"VALU processes {VLEN} elements per slot. Vectorizing scalar ops could help.",
            "potential_savings": f"Up to {VLEN}x for vectorizable operations",
            "difficulty": "Medium"
        })

    # Sort by priority
    recommendations.sort(key=lambda x: x["priority"])

    return recommendations


def compare_kernels(result1: AnalysisResult, result2: AnalysisResult, name1: str = "Before", name2: str = "After") -> dict:
    """
    Compare two kernel analyses and return a comparison summary.
    """
    comparison = {
        "names": (name1, name2),
        "cycles": {
            name1: result1.total_cycles,
            name2: result2.total_cycles,
            "delta": result2.total_cycles - result1.total_cycles,
            "speedup": result1.total_cycles / max(1, result2.total_cycles)
        },
        "utilization": {
            name1: result1.utilization_pct,
            name2: result2.utilization_pct,
            "delta": result2.utilization_pct - result1.utilization_pct
        },
        "slots_used": {
            name1: result1.total_slots_used,
            name2: result2.total_slots_used,
            "delta": result2.total_slots_used - result1.total_slots_used
        },
        "per_engine_delta": {}
    }

    for engine in ENGINES:
        util1 = result1.per_engine.get(engine, {}).get("utilization_pct", 0)
        util2 = result2.per_engine.get(engine, {}).get("utilization_pct", 0)
        comparison["per_engine_delta"][engine] = {
            name1: util1,
            name2: util2,
            "delta": util2 - util1
        }

    # Compare low utilization cycles
    comparison["low_util_cycles"] = {
        name1: len(result1.low_util_cycles),
        name2: len(result2.low_util_cycles),
        "delta": len(result2.low_util_cycles) - len(result1.low_util_cycles)
    }

    return comparison


# ============== Output Formatting ==============

class PlainPrinter:
    """Plain text output without Rich."""

    def __init__(self):
        pass

    def print_header(self, text: str):
        print("=" * 60)
        print(text)
        print("=" * 60)

    def print_subheader(self, text: str):
        print("-" * 60)
        print(text)
        print("-" * 60)

    def print_summary(self, result: AnalysisResult):
        self.print_header("SLOT UTILIZATION ANALYSIS")
        print()
        print(f"Total Cycles: {result.total_cycles:,}")
        print(f"Total Slots Used: {result.total_slots_used:,}")
        print(f"Max Possible Slots: {result.total_cycles * MAX_SLOTS_PER_CYCLE:,}")
        print(f"Overall Utilization: {result.utilization_pct:.1f}%")
        print(f"Avg Slots/Cycle: {result.total_slots_used / max(1, result.total_cycles):.2f} / {MAX_SLOTS_PER_CYCLE}")
        print()

        # Theoretical speedup
        avg_slots = result.total_slots_used / max(1, result.total_cycles)
        theoretical_speedup = MAX_SLOTS_PER_CYCLE / max(0.01, avg_slots)
        print(f">>> Theoretical speedup if 100% utilized: {theoretical_speedup:.1f}x")
        print(f">>> That would be: {result.total_cycles / theoretical_speedup:.0f} cycles")
        print()

        # Per-engine breakdown
        self.print_subheader("PER-ENGINE BREAKDOWN")
        print(f"{'Engine':<8} {'Used':>10} {'Max':>10} {'Util%':>8} {'Avg/Cyc':>10}")
        print("-" * 60)
        for engine, stats in result.per_engine.items():
            print(f"{engine:<8} {stats['total_used']:>10,} {stats['max_possible']:>10,} {stats['utilization_pct']:>7.1f}% {stats['avg_per_cycle']:>10.2f}")
        print()

        # Histogram
        self.print_subheader("SLOTS-PER-CYCLE HISTOGRAM")
        max_count = max(result.histogram.values()) if result.histogram else 1
        for slots, count in sorted(result.histogram.items()):
            bar_len = int(40 * count / max_count)
            bar = "#" * bar_len
            pct = 100.0 * count / result.total_cycles
            print(f"{slots:>2} slots: {count:>6,} ({pct:>5.1f}%) {bar}")
        print()

        # Low utilization warning
        if result.low_util_cycles:
            self.print_subheader(f"WARNING: {len(result.low_util_cycles):,} cycles with <50% utilization")

    def print_dependency_analysis(self, dep: DependencyAnalysis, total_cycles: int):
        self.print_header("DEPENDENCY ANALYSIS")
        print()
        print(f"RAW Hazards Detected: {dep.raw_hazard_count:,}")
        print(f"Blocking Pairs (prevent packing): {len(dep.blocking_pairs):,}")
        print(f"Critical Path Length: {dep.critical_path_length} cycles")
        print(f"Current Cycles: {total_cycles}")

        if dep.critical_path_length > 0:
            theoretical_min = dep.critical_path_length
            overhead = total_cycles - theoretical_min
            print(f"Overhead vs Critical Path: {overhead} cycles ({100*overhead/total_cycles:.1f}%)")
        print()

    def print_worst_cycles(self, result: AnalysisResult, n: int = 10):
        print()
        self.print_subheader(f"TOP {n} WORST-UTILIZED CYCLES")
        print(f"{'Cycle':>8} {'Slots':>6} {'Util%':>7}  Breakdown")
        print("-" * 60)

        sorted_cycles = sorted(result.per_cycle, key=lambda c: c.total_used)
        for stats in sorted_cycles[:n]:
            breakdown = " ".join(f"{e}:{stats.slots_used.get(e, 0)}" for e in ENGINES)
            print(f"{stats.cycle_num:>8} {stats.total_used:>6} {stats.utilization_pct:>6.1f}%  {breakdown}")

    def print_best_cycles(self, result: AnalysisResult, n: int = 10):
        print()
        self.print_subheader(f"TOP {n} BEST-UTILIZED CYCLES")
        print(f"{'Cycle':>8} {'Slots':>6} {'Util%':>7}  Breakdown")
        print("-" * 60)

        sorted_cycles = sorted(result.per_cycle, key=lambda c: -c.total_used)
        for stats in sorted_cycles[:n]:
            breakdown = " ".join(f"{e}:{stats.slots_used.get(e, 0)}" for e in ENGINES)
            print(f"{stats.cycle_num:>8} {stats.total_used:>6} {stats.utilization_pct:>6.1f}%  {breakdown}")

    def print_packing_opportunities(self, instructions: list[dict], dep_analysis: Optional[DependencyAnalysis] = None):
        opps = analyze_packing_opportunities(instructions, dep_analysis)

        print()
        self.print_header("PACKING OPPORTUNITY ANALYSIS")
        print(f"Adjacent instruction pairs that CAN be combined: {opps['total_opportunities']:,}")
        print(f"Pairs blocked by RAW dependencies: {opps['blocked_by_dependencies']:,}")
        print(f"Potential cycle savings: {opps['potential_cycle_savings']:,}")
        print()

        # Analyze by type
        combos = analyze_packing_by_type(instructions)
        sorted_combos = sorted(combos.items(), key=lambda x: -x[1])

        print("Top packing opportunities by engine combination:")
        print("-" * 60)
        print(f"{'Combination':<40} {'Count':>10} {'% of Total':>12}")
        print("-" * 60)
        total = sum(combos.values())
        for (curr, next_), count in sorted_combos[:15]:
            combo_str = f"{'+'.join(curr)} | {'+'.join(next_)}"
            pct = 100.0 * count / total if total > 0 else 0
            print(f"{combo_str:<40} {count:>10,} {pct:>11.1f}%")

        print()
        if opps['sample_opportunities']:
            print("Sample packable opportunities (first 10):")
            print("-" * 60)
            for opp in opps['sample_opportunities'][:10]:
                curr_eng = [e for e in opp['curr_engines'] if e != 'debug']
                next_eng = [e for e in opp['next_engines'] if e != 'debug']
                combined = opp['combined_slots']
                print(f"  Cycle {opp['cycle']:>5}: {curr_eng} + {next_eng} -> {dict(combined)}")

        if opps['sample_blocked']:
            print()
            print("Sample blocked by dependencies (first 5):")
            print("-" * 60)
            for blocked in opps['sample_blocked'][:5]:
                print(f"  Cycle {blocked['cycle']:>5}: blocked by {blocked['reason']}")

        return opps

    def print_recommendations(self, recommendations: List[dict]):
        print()
        self.print_header("OPTIMIZATION RECOMMENDATIONS")
        print()

        if not recommendations:
            print("No specific recommendations at this time.")
            return

        for i, rec in enumerate(recommendations, 1):
            priority_str = ["HIGH", "MEDIUM", "LOW"][min(rec["priority"]-1, 2)]
            print(f"[{priority_str}] {i}. {rec['title']}")
            print(f"    Category: {rec['category']}")
            print(f"    {rec['description']}")
            print(f"    Potential: {rec['potential_savings']}")
            print(f"    Difficulty: {rec['difficulty']}")
            print()

    def print_comparison(self, comparison: dict):
        name1, name2 = comparison["names"]

        print()
        self.print_header(f"KERNEL COMPARISON: {name1} vs {name2}")
        print()

        cycles = comparison["cycles"]
        delta = cycles["delta"]
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        speedup = cycles["speedup"]

        print(f"Cycles: {cycles[name1]:,} -> {cycles[name2]:,} ({delta_str})")
        if speedup > 1:
            print(f"  IMPROVEMENT: {speedup:.2f}x speedup!")
        elif speedup < 1:
            print(f"  REGRESSION: {1/speedup:.2f}x slower")
        else:
            print(f"  No change")
        print()

        util = comparison["utilization"]
        util_delta = util["delta"]
        util_str = f"+{util_delta:.1f}%" if util_delta > 0 else f"{util_delta:.1f}%"
        print(f"Utilization: {util[name1]:.1f}% -> {util[name2]:.1f}% ({util_str})")
        print()

        print("Per-Engine Utilization Changes:")
        print("-" * 50)
        print(f"{'Engine':<10} {name1:>12} {name2:>12} {'Delta':>10}")
        print("-" * 50)
        for engine, data in comparison["per_engine_delta"].items():
            delta = data["delta"]
            delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
            print(f"{engine:<10} {data[name1]:>11.1f}% {data[name2]:>11.1f}% {delta_str:>10}")
        print()


class RichPrinter:
    """Rich-enabled colorful output."""

    def __init__(self):
        self.console = Console()

    def print_header(self, text: str):
        self.console.print(Panel(text, style="bold cyan", box=box.DOUBLE))

    def print_subheader(self, text: str):
        self.console.print(f"\n[bold yellow]{text}[/bold yellow]")
        self.console.print("-" * 60)

    def print_summary(self, result: AnalysisResult):
        self.print_header("SLOT UTILIZATION ANALYSIS")

        # Summary stats
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Cycles", f"{result.total_cycles:,}")
        table.add_row("Total Slots Used", f"{result.total_slots_used:,}")
        table.add_row("Max Possible Slots", f"{result.total_cycles * MAX_SLOTS_PER_CYCLE:,}")

        util_color = "green" if result.utilization_pct > 50 else "yellow" if result.utilization_pct > 25 else "red"
        table.add_row("Overall Utilization", f"[{util_color}]{result.utilization_pct:.1f}%[/{util_color}]")
        table.add_row("Avg Slots/Cycle", f"{result.total_slots_used / max(1, result.total_cycles):.2f} / {MAX_SLOTS_PER_CYCLE}")

        self.console.print(table)

        # Theoretical speedup
        avg_slots = result.total_slots_used / max(1, result.total_cycles)
        theoretical_speedup = MAX_SLOTS_PER_CYCLE / max(0.01, avg_slots)
        self.console.print(f"\n[bold magenta]>>> Theoretical speedup if 100% utilized: {theoretical_speedup:.1f}x[/bold magenta]")
        self.console.print(f"[bold magenta]>>> That would be: {result.total_cycles / theoretical_speedup:.0f} cycles[/bold magenta]")

        # Per-engine breakdown
        self.print_subheader("PER-ENGINE BREAKDOWN")

        eng_table = Table(box=box.ROUNDED)
        eng_table.add_column("Engine", style="cyan")
        eng_table.add_column("Used", justify="right")
        eng_table.add_column("Max", justify="right")
        eng_table.add_column("Util%", justify="right")
        eng_table.add_column("Avg/Cyc", justify="right")

        for engine, stats in result.per_engine.items():
            util = stats['utilization_pct']
            util_color = "green" if util > 50 else "yellow" if util > 25 else "red"
            eng_table.add_row(
                engine,
                f"{stats['total_used']:,}",
                f"{stats['max_possible']:,}",
                f"[{util_color}]{util:.1f}%[/{util_color}]",
                f"{stats['avg_per_cycle']:.2f}"
            )

        self.console.print(eng_table)

        # Histogram
        self.print_subheader("SLOTS-PER-CYCLE HISTOGRAM")
        max_count = max(result.histogram.values()) if result.histogram else 1
        for slots, count in sorted(result.histogram.items()):
            bar_len = int(40 * count / max_count)
            pct = 100.0 * count / result.total_cycles

            # Color based on slot count
            if slots < MAX_SLOTS_PER_CYCLE * 0.25:
                color = "red"
            elif slots < MAX_SLOTS_PER_CYCLE * 0.5:
                color = "yellow"
            else:
                color = "green"

            bar = "[" + color + "]" + "#" * bar_len + "[/" + color + "]"
            self.console.print(f"{slots:>2} slots: {count:>6,} ({pct:>5.1f}%) {bar}")

        # Low utilization warning
        if result.low_util_cycles:
            self.console.print(f"\n[bold red]WARNING: {len(result.low_util_cycles):,} cycles with <50% utilization[/bold red]")

    def print_dependency_analysis(self, dep: DependencyAnalysis, total_cycles: int):
        self.print_header("DEPENDENCY ANALYSIS")

        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("RAW Hazards Detected", f"{dep.raw_hazard_count:,}")
        table.add_row("Blocking Pairs", f"{len(dep.blocking_pairs):,}")
        table.add_row("Critical Path Length", f"{dep.critical_path_length} cycles")
        table.add_row("Current Cycles", f"{total_cycles}")

        if dep.critical_path_length > 0:
            theoretical_min = dep.critical_path_length
            overhead = total_cycles - theoretical_min
            overhead_pct = 100*overhead/total_cycles
            overhead_color = "green" if overhead_pct < 20 else "yellow" if overhead_pct < 50 else "red"
            table.add_row("Overhead vs Critical Path", f"[{overhead_color}]{overhead} cycles ({overhead_pct:.1f}%)[/{overhead_color}]")

        self.console.print(table)

    def print_worst_cycles(self, result: AnalysisResult, n: int = 10):
        self.print_subheader(f"TOP {n} WORST-UTILIZED CYCLES")

        table = Table(box=box.ROUNDED)
        table.add_column("Cycle", justify="right")
        table.add_column("Slots", justify="right")
        table.add_column("Util%", justify="right")
        table.add_column("Breakdown")

        sorted_cycles = sorted(result.per_cycle, key=lambda c: c.total_used)
        for stats in sorted_cycles[:n]:
            breakdown = " ".join(f"{e}:{stats.slots_used.get(e, 0)}" for e in ENGINES)
            util = stats.utilization_pct
            util_color = "green" if util > 50 else "yellow" if util > 25 else "red"
            table.add_row(
                str(stats.cycle_num),
                str(stats.total_used),
                f"[{util_color}]{util:.1f}%[/{util_color}]",
                breakdown
            )

        self.console.print(table)

    def print_best_cycles(self, result: AnalysisResult, n: int = 10):
        self.print_subheader(f"TOP {n} BEST-UTILIZED CYCLES")

        table = Table(box=box.ROUNDED)
        table.add_column("Cycle", justify="right")
        table.add_column("Slots", justify="right")
        table.add_column("Util%", justify="right")
        table.add_column("Breakdown")

        sorted_cycles = sorted(result.per_cycle, key=lambda c: -c.total_used)
        for stats in sorted_cycles[:n]:
            breakdown = " ".join(f"{e}:{stats.slots_used.get(e, 0)}" for e in ENGINES)
            util = stats.utilization_pct
            util_color = "green" if util > 50 else "yellow" if util > 25 else "red"
            table.add_row(
                str(stats.cycle_num),
                str(stats.total_used),
                f"[{util_color}]{util:.1f}%[/{util_color}]",
                breakdown
            )

        self.console.print(table)

    def print_packing_opportunities(self, instructions: list[dict], dep_analysis: Optional[DependencyAnalysis] = None):
        opps = analyze_packing_opportunities(instructions, dep_analysis)

        self.print_header("PACKING OPPORTUNITY ANALYSIS")

        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Pairs that CAN be combined", f"[green]{opps['total_opportunities']:,}[/green]")
        table.add_row("Pairs blocked by RAW deps", f"[red]{opps['blocked_by_dependencies']:,}[/red]")
        table.add_row("Potential cycle savings", f"[bold green]{opps['potential_cycle_savings']:,}[/bold green]")

        self.console.print(table)

        # Analyze by type
        combos = analyze_packing_by_type(instructions)
        sorted_combos = sorted(combos.items(), key=lambda x: -x[1])

        self.print_subheader("Top packing opportunities by engine combination")

        combo_table = Table(box=box.ROUNDED)
        combo_table.add_column("Combination", style="cyan")
        combo_table.add_column("Count", justify="right")
        combo_table.add_column("% of Total", justify="right")

        total = sum(combos.values())
        for (curr, next_), count in sorted_combos[:15]:
            combo_str = f"{'+'.join(curr)} | {'+'.join(next_)}"
            pct = 100.0 * count / total if total > 0 else 0
            combo_table.add_row(combo_str, f"{count:,}", f"{pct:.1f}%")

        self.console.print(combo_table)

        if opps['sample_opportunities']:
            self.print_subheader("Sample packable opportunities (first 10)")
            for opp in opps['sample_opportunities'][:10]:
                curr_eng = [e for e in opp['curr_engines'] if e != 'debug']
                next_eng = [e for e in opp['next_engines'] if e != 'debug']
                combined = opp['combined_slots']
                self.console.print(f"  Cycle [cyan]{opp['cycle']:>5}[/cyan]: {curr_eng} + {next_eng} -> {dict(combined)}")

        if opps['sample_blocked']:
            self.print_subheader("Sample blocked by dependencies (first 5)")
            for blocked in opps['sample_blocked'][:5]:
                self.console.print(f"  Cycle [red]{blocked['cycle']:>5}[/red]: blocked by [red]{blocked['reason']}[/red]")

        return opps

    def print_recommendations(self, recommendations: List[dict]):
        self.print_header("OPTIMIZATION RECOMMENDATIONS")

        if not recommendations:
            self.console.print("[dim]No specific recommendations at this time.[/dim]")
            return

        for i, rec in enumerate(recommendations, 1):
            priority = rec["priority"]
            if priority == 1:
                priority_style = "bold red"
                priority_str = "HIGH"
            elif priority == 2:
                priority_style = "bold yellow"
                priority_str = "MEDIUM"
            else:
                priority_style = "bold blue"
                priority_str = "LOW"

            panel_content = Text()
            panel_content.append(f"Category: ", style="dim")
            panel_content.append(f"{rec['category']}\n", style="cyan")
            panel_content.append(f"{rec['description']}\n\n", style="white")
            panel_content.append(f"Potential: ", style="dim")
            panel_content.append(f"{rec['potential_savings']}\n", style="green")
            panel_content.append(f"Difficulty: ", style="dim")

            diff = rec['difficulty']
            diff_style = "green" if diff == "Easy" else "yellow" if diff == "Medium" else "red"
            panel_content.append(f"{diff}", style=diff_style)

            self.console.print(Panel(
                panel_content,
                title=f"[{priority_style}][{priority_str}][/{priority_style}] {i}. {rec['title']}",
                border_style=priority_style
            ))

    def print_comparison(self, comparison: dict):
        name1, name2 = comparison["names"]

        self.print_header(f"KERNEL COMPARISON: {name1} vs {name2}")

        cycles = comparison["cycles"]
        delta = cycles["delta"]
        speedup = cycles["speedup"]

        # Summary panel
        if speedup > 1:
            result_text = f"[bold green]IMPROVEMENT: {speedup:.2f}x speedup![/bold green]"
        elif speedup < 1:
            result_text = f"[bold red]REGRESSION: {1/speedup:.2f}x slower[/bold red]"
        else:
            result_text = "[bold yellow]No change[/bold yellow]"

        delta_str = f"+{delta}" if delta > 0 else str(delta)
        self.console.print(f"\nCycles: [cyan]{cycles[name1]:,}[/cyan] -> [cyan]{cycles[name2]:,}[/cyan] ({delta_str})")
        self.console.print(result_text)

        util = comparison["utilization"]
        util_delta = util["delta"]
        util_str = f"+{util_delta:.1f}%" if util_delta > 0 else f"{util_delta:.1f}%"
        util_color = "green" if util_delta > 0 else "red" if util_delta < 0 else "yellow"
        self.console.print(f"\nUtilization: {util[name1]:.1f}% -> {util[name2]:.1f}% ([{util_color}]{util_str}[/{util_color}])")

        self.print_subheader("Per-Engine Utilization Changes")

        table = Table(box=box.ROUNDED)
        table.add_column("Engine", style="cyan")
        table.add_column(name1, justify="right")
        table.add_column(name2, justify="right")
        table.add_column("Delta", justify="right")

        for engine, data in comparison["per_engine_delta"].items():
            delta = data["delta"]
            delta_color = "green" if delta > 0 else "red" if delta < 0 else "white"
            delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
            table.add_row(
                engine,
                f"{data[name1]:.1f}%",
                f"{data[name2]:.1f}%",
                f"[{delta_color}]{delta_str}[/{delta_color}]"
            )

        self.console.print(table)


def get_printer():
    """Get the appropriate printer based on Rich availability."""
    if RICH_AVAILABLE:
        return RichPrinter()
    return PlainPrinter()


def analyze_kernel(kernel_builder=None):
    """Analyze the current kernel from perf_takehome.py"""
    if kernel_builder is None:
        from perf_takehome import KernelBuilder

        # Standard test params
        forest_height = 10
        n_nodes = 2 ** (forest_height + 1) - 1
        batch_size = 256
        rounds = 16

        kb = KernelBuilder()
        kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
        return analyze_instructions(kb.instrs), kb.instrs
    else:
        return analyze_instructions(kernel_builder.instrs), kernel_builder.instrs


def save_kernel_json(instructions: list[dict], filepath: str):
    """Save kernel instructions to JSON for later comparison."""
    with open(filepath, 'w') as f:
        json.dump(instructions, f)


def load_kernel_json(filepath: str) -> list[dict]:
    """Load kernel instructions from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze VLIW slot utilization")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-cycle details")
    parser.add_argument("--top", "-n", type=int, default=10, help="Number of worst/best cycles to show")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable")
    parser.add_argument("--histogram", action="store_true", help="Show detailed histogram")
    parser.add_argument("--packing", "-p", action="store_true", help="Show packing opportunities")
    parser.add_argument("--deps", "-d", action="store_true", help="Show dependency analysis")
    parser.add_argument("--recommendations", "-r", action="store_true", help="Show optimization recommendations")
    parser.add_argument("--compare", nargs=2, metavar=('KERNEL1', 'KERNEL2'), help="Compare two kernel JSON files")
    parser.add_argument("--save", metavar='FILE', help="Save current kernel to JSON file")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    args = parser.parse_args()

    printer = PlainPrinter() if args.no_color else get_printer()

    if args.compare:
        # Compare two kernels
        print("Loading kernels for comparison...", file=sys.stderr)
        instrs1 = load_kernel_json(args.compare[0])
        instrs2 = load_kernel_json(args.compare[1])

        result1 = analyze_instructions(instrs1)
        result2 = analyze_instructions(instrs2)

        comparison = compare_kernels(result1, result2,
                                      os.path.basename(args.compare[0]),
                                      os.path.basename(args.compare[1]))
        printer.print_comparison(comparison)
        return

    print("Analyzing kernel...", file=sys.stderr)
    result, instructions = analyze_kernel()

    if args.save:
        save_kernel_json(instructions, args.save)
        print(f"Saved kernel to {args.save}", file=sys.stderr)

    if args.json:
        print(result.to_json())
    else:
        printer.print_summary(result)

        if args.deps or args.recommendations:
            if result.dependency_analysis:
                printer.print_dependency_analysis(result.dependency_analysis, result.total_cycles)

        printer.print_worst_cycles(result, args.top)
        printer.print_best_cycles(result, args.top)

        packing_opps = None
        if args.packing or args.recommendations:
            packing_opps = printer.print_packing_opportunities(instructions, result.dependency_analysis)

        if args.recommendations:
            if packing_opps is None:
                packing_opps = analyze_packing_opportunities(instructions, result.dependency_analysis)
            recommendations = generate_recommendations(result, packing_opps)
            result.recommendations = recommendations
            printer.print_recommendations(recommendations)

        if args.verbose:
            print()
            print("-" * 60)
            print("ALL CYCLES")
            print("-" * 60)
            for stats in result.per_cycle[:100]:  # Limit to first 100
                breakdown = " ".join(f"{e}:{stats.slots_used.get(e, 0)}" for e in ENGINES)
                print(f"Cycle {stats.cycle_num:>5}: {stats.total_used:>2} slots ({stats.utilization_pct:>5.1f}%) - {breakdown}")
            if len(result.per_cycle) > 100:
                print(f"... and {len(result.per_cycle) - 100} more cycles")


if __name__ == "__main__":
    main()
