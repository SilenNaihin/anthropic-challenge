#!/usr/bin/env python3
"""
Cycle Profiler for VLIW SIMD Kernel

Breaks down cycles by code section (hash, memory, index calc) to understand
WHERE time is spent, not just how many cycles total.

Key Features:
- Phase tagging: Automatically categorize instructions into phases
  (init, hash, memory, index_calc, flow_control, bounds_check, store)
- Per-round breakdown: See how cycles distribute across rounds
- Hotspot identification: Find which phases dominate execution time
- Rich output with color coding (falls back to plain text)

Usage:
    python tools/cycle_profiler/cycle_profiler.py
    python tools/cycle_profiler/cycle_profiler.py --json
    python tools/cycle_profiler/cycle_profiler.py --detailed
    python tools/cycle_profiler/cycle_profiler.py --per-round
"""

import sys
import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from enum import Enum

# Add parent dirs to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import SLOT_LIMITS, VLEN, HASH_STAGES

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


class Phase(Enum):
    """Code phase categories for profiling."""
    INIT = "init"                    # Initial setup, constant loading
    HASH = "hash"                    # Hash computation (6-stage mixing)
    MEMORY_LOAD = "memory_load"      # Load operations (vload, load)
    MEMORY_STORE = "memory_store"    # Store operations (vstore, store)
    INDEX_CALC = "index_calc"        # Index/address computation
    BOUNDS_CHECK = "bounds_check"    # Bounds checking (<, >, vselect)
    FLOW_CONTROL = "flow"            # Control flow (pause, select)
    XOR_MIX = "xor_mix"             # XOR with node values
    BROADCAST = "broadcast"          # vbroadcast operations
    UNKNOWN = "unknown"              # Unclassified operations


# Hash operations for phase detection
HASH_OPS = {"+", "^", "<<", ">>"}

# Phase colors for Rich output
PHASE_COLORS = {
    Phase.INIT: "dim",
    Phase.HASH: "red bold",
    Phase.MEMORY_LOAD: "blue",
    Phase.MEMORY_STORE: "cyan",
    Phase.INDEX_CALC: "yellow",
    Phase.BOUNDS_CHECK: "magenta",
    Phase.FLOW_CONTROL: "green",
    Phase.XOR_MIX: "orange3",
    Phase.BROADCAST: "purple",
    Phase.UNKNOWN: "white",
}


@dataclass
class PhaseStats:
    """Statistics for a single phase."""
    phase: Phase
    total_cycles: int = 0
    total_slots: int = 0
    occurrences: int = 0  # Number of cycles this phase appears
    exclusive_cycles: int = 0  # Cycles where ONLY this phase runs
    slot_breakdown: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def avg_slots_per_occurrence(self) -> float:
        return self.total_slots / max(1, self.occurrences)

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.value,
            "total_cycles": self.total_cycles,
            "total_slots": self.total_slots,
            "occurrences": self.occurrences,
            "exclusive_cycles": self.exclusive_cycles,
            "avg_slots_per_occurrence": round(self.avg_slots_per_occurrence, 2),
            "slot_breakdown": dict(self.slot_breakdown)
        }


@dataclass
class CycleProfile:
    """Profile of a single cycle."""
    cycle_num: int
    phases: Set[Phase] = field(default_factory=set)
    phase_slots: Dict[Phase, int] = field(default_factory=lambda: defaultdict(int))
    engine_slots: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    instructions: List[Tuple[str, tuple]] = field(default_factory=list)

    @property
    def total_slots(self) -> int:
        return sum(self.engine_slots.values())

    @property
    def dominant_phase(self) -> Phase:
        """Return the phase using the most slots this cycle."""
        if not self.phase_slots:
            return Phase.UNKNOWN
        return max(self.phase_slots.keys(), key=lambda p: self.phase_slots[p])


@dataclass
class RoundProfile:
    """Profile of a single round iteration."""
    round_num: int
    start_cycle: int
    end_cycle: int
    phase_cycles: Dict[Phase, int] = field(default_factory=lambda: defaultdict(int))
    phase_slots: Dict[Phase, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def total_cycles(self) -> int:
        return self.end_cycle - self.start_cycle

    def to_dict(self) -> dict:
        return {
            "round_num": self.round_num,
            "start_cycle": self.start_cycle,
            "end_cycle": self.end_cycle,
            "total_cycles": self.total_cycles,
            "phase_cycles": {p.value: c for p, c in self.phase_cycles.items()},
            "phase_slots": {p.value: s for p, s in self.phase_slots.items()}
        }


@dataclass
class ProfileResult:
    """Complete profiling results."""
    total_cycles: int
    total_slots: int
    phase_stats: Dict[Phase, PhaseStats]
    cycle_profiles: List[CycleProfile]
    round_profiles: List[RoundProfile]
    hotspots: List[Tuple[Phase, float]]  # (phase, percentage)
    init_cycles: int
    main_loop_cycles: int

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_cycles": self.total_cycles,
                "total_slots": self.total_slots,
                "init_cycles": self.init_cycles,
                "main_loop_cycles": self.main_loop_cycles,
            },
            "phase_breakdown": {
                p.value: stats.to_dict()
                for p, stats in sorted(self.phase_stats.items(),
                                       key=lambda x: -x[1].total_cycles)
            },
            "hotspots": [
                {"phase": p.value, "percentage": round(pct, 1)}
                for p, pct in self.hotspots
            ],
            "round_profiles": [rp.to_dict() for rp in self.round_profiles[:5]],
            "round_count": len(self.round_profiles),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def classify_instruction(engine: str, slot: tuple) -> Phase:
    """
    Classify a single instruction slot into a phase.

    Uses heuristics based on the operation type and engine.
    """
    if not slot or len(slot) == 0:
        return Phase.UNKNOWN

    op = slot[0]

    # Load engine
    if engine == "load":
        if op == "const":
            return Phase.INIT
        elif op == "vload":
            return Phase.MEMORY_LOAD
        elif op == "load":
            return Phase.MEMORY_LOAD
        elif op == "load_offset":
            return Phase.MEMORY_LOAD
        return Phase.MEMORY_LOAD

    # Store engine
    elif engine == "store":
        return Phase.MEMORY_STORE

    # Flow engine
    elif engine == "flow":
        if op == "vselect":
            return Phase.BOUNDS_CHECK
        elif op == "select":
            return Phase.BOUNDS_CHECK
        elif op == "pause":
            return Phase.FLOW_CONTROL
        return Phase.FLOW_CONTROL

    # VALU engine - most interesting for profiling
    elif engine == "valu":
        if op == "vbroadcast":
            return Phase.BROADCAST

        # Check for hash operations patterns
        # Hash stages use: +, ^, <<, >> in specific patterns
        if op in HASH_OPS:
            # Could be hash or index calc - check context
            # Hash typically has 3-operand forms with constants
            return Phase.HASH

        # Comparison for bounds checking
        if op == "<" or op == ">":
            return Phase.BOUNDS_CHECK

        # multiply_add for index calculation
        if op == "multiply_add":
            return Phase.INDEX_CALC

        # XOR could be hash or xor_mix
        if op == "^":
            return Phase.HASH  # Default to hash, will refine below

        # Addition could be index calc or hash
        if op == "+":
            return Phase.HASH  # Default, refined by context

        # Bitwise ops typically in hash
        if op == "&":
            return Phase.INDEX_CALC  # & 1 for branch decision

        return Phase.HASH  # Default for VALU

    # ALU engine
    elif engine == "alu":
        if op == "+":
            return Phase.INDEX_CALC  # Address arithmetic
        if op in HASH_OPS:
            return Phase.HASH
        return Phase.INDEX_CALC

    return Phase.UNKNOWN


def classify_cycle_context(cycle_num: int, instructions: List[dict],
                           current_instr: dict, total_cycles: int) -> Dict[str, Any]:
    """
    Add context-aware classification refinements.

    Some instructions need context to classify correctly:
    - XOR right after load is xor_mix (XOR with node value)
    - Addition after hash might be index calc
    """
    context = {
        "is_init_phase": cycle_num < 50,  # First ~50 cycles are usually init
        "prev_had_load": False,
        "in_hash_region": False,
    }

    # Check if previous cycles had loads (suggests xor_mix coming)
    if cycle_num > 0 and cycle_num < len(instructions):
        prev = instructions[cycle_num - 1]
        if "load" in prev:
            context["prev_had_load"] = True

    return context


def refine_phase_classification(profile: CycleProfile, context: Dict[str, Any]) -> None:
    """
    Refine phase classifications based on context.

    This allows more accurate categorization by looking at surrounding instructions.
    """
    # If we just loaded and now have XOR on valu, it's xor_mix not hash
    if context.get("prev_had_load"):
        for instr in profile.instructions:
            engine, slot = instr
            if engine == "valu" and slot and slot[0] == "^":
                # Reclassify as xor_mix
                if Phase.HASH in profile.phases:
                    profile.phases.remove(Phase.HASH)
                profile.phases.add(Phase.XOR_MIX)
                # Adjust slot counts
                if Phase.HASH in profile.phase_slots:
                    hash_slots = profile.phase_slots[Phase.HASH]
                    del profile.phase_slots[Phase.HASH]
                    profile.phase_slots[Phase.XOR_MIX] = hash_slots


def profile_instructions(instructions: List[dict]) -> ProfileResult:
    """
    Main profiling function - analyze instruction stream and categorize by phase.

    Returns comprehensive profiling results.
    """
    n_cycles = len(instructions)
    cycle_profiles: List[CycleProfile] = []
    phase_stats: Dict[Phase, PhaseStats] = {p: PhaseStats(phase=p) for p in Phase}

    # First pass - classify each cycle
    for cycle_num, instr in enumerate(instructions):
        profile = CycleProfile(cycle_num=cycle_num)
        context = classify_cycle_context(cycle_num, instructions, instr, n_cycles)

        for engine, slots in instr.items():
            if engine == "debug":
                continue

            slot_count = len(slots) if slots else 0
            profile.engine_slots[engine] = slot_count

            for slot in (slots or []):
                phase = classify_instruction(engine, slot)
                profile.phases.add(phase)
                profile.phase_slots[phase] = profile.phase_slots.get(phase, 0) + 1
                profile.instructions.append((engine, slot))

        # Refine classifications based on context
        refine_phase_classification(profile, context)

        cycle_profiles.append(profile)

    # Second pass - compute phase statistics
    for profile in cycle_profiles:
        for phase in profile.phases:
            stats = phase_stats[phase]
            stats.occurrences += 1
            stats.total_slots += profile.phase_slots.get(phase, 0)

            # Exclusive cycle = only this phase present
            if len(profile.phases) == 1:
                stats.exclusive_cycles += 1

            # Track engine breakdown
            for engine, count in profile.engine_slots.items():
                if profile.dominant_phase == phase:
                    stats.slot_breakdown[engine] += count

        # Every cycle adds to total_cycles for phases present
        for phase in profile.phases:
            phase_stats[phase].total_cycles += 1

    # Detect init vs main loop boundary
    # Init usually ends at first "pause" instruction or when we see hash pattern
    init_cycles = 0
    for i, profile in enumerate(cycle_profiles):
        if Phase.FLOW_CONTROL in profile.phases or Phase.HASH in profile.phases:
            # Found first pause or hash - init ends
            init_cycles = i
            break

    # Detect round boundaries (look for repeating patterns)
    round_profiles = detect_rounds(cycle_profiles, init_cycles)

    # Compute hotspots
    total_phase_cycles = sum(s.total_cycles for s in phase_stats.values())
    hotspots = [
        (phase, 100.0 * stats.total_cycles / max(1, total_phase_cycles))
        for phase, stats in phase_stats.items()
        if stats.total_cycles > 0
    ]
    hotspots.sort(key=lambda x: -x[1])

    total_slots = sum(p.total_slots for p in cycle_profiles)

    return ProfileResult(
        total_cycles=n_cycles,
        total_slots=total_slots,
        phase_stats=phase_stats,
        cycle_profiles=cycle_profiles,
        round_profiles=round_profiles,
        hotspots=hotspots,
        init_cycles=init_cycles,
        main_loop_cycles=n_cycles - init_cycles,
    )


def detect_rounds(cycle_profiles: List[CycleProfile], init_cycles: int) -> List[RoundProfile]:
    """
    Detect round boundaries in the instruction stream.

    Uses heuristics to identify where rounds start/end:
    - Look for repeated patterns of phases
    - Look for FLOW_CONTROL instructions that mark boundaries
    """
    rounds: List[RoundProfile] = []

    if len(cycle_profiles) <= init_cycles:
        return rounds

    # Skip init phase
    main_profiles = cycle_profiles[init_cycles:]

    # Simple heuristic: estimate round size from kernel parameters
    # Standard kernel: 16 rounds, each processes batch_size/VLEN iterations
    # Each iteration has: load + hash + store pattern

    # Look for store patterns that might indicate round boundaries
    store_cycles = []
    for i, profile in enumerate(main_profiles):
        if Phase.MEMORY_STORE in profile.phases:
            store_cycles.append(i)

    # If we have multiple stores, try to detect periodicity
    if len(store_cycles) >= 4:
        # Find common gap between store operations
        gaps = [store_cycles[i+1] - store_cycles[i] for i in range(len(store_cycles)-1)]

        if gaps:
            # Use median gap as estimated round size
            sorted_gaps = sorted(gaps)
            median_gap = sorted_gaps[len(sorted_gaps)//2]

            # Create round profiles based on estimated boundaries
            round_size = max(median_gap, 10)  # Minimum 10 cycles per "round"
            n_rounds = len(main_profiles) // round_size

            for r in range(min(n_rounds, 20)):  # Cap at 20 rounds for output
                start = init_cycles + r * round_size
                end = min(init_cycles + (r + 1) * round_size, len(cycle_profiles))

                rp = RoundProfile(
                    round_num=r,
                    start_cycle=start,
                    end_cycle=end
                )

                # Aggregate phase stats for this round
                for profile in cycle_profiles[start:end]:
                    for phase, slots in profile.phase_slots.items():
                        rp.phase_slots[phase] += slots
                    for phase in profile.phases:
                        rp.phase_cycles[phase] += 1

                rounds.append(rp)

    return rounds


# ============== Output Formatting ==============

class PlainPrinter:
    """Plain text output without Rich."""

    def print_header(self, text: str):
        print("=" * 70)
        print(text)
        print("=" * 70)

    def print_subheader(self, text: str):
        print()
        print("-" * 70)
        print(text)
        print("-" * 70)

    def print_summary(self, result: ProfileResult):
        self.print_header("CYCLE PROFILER - WHERE IS TIME SPENT?")
        print()
        print(f"Total Cycles:        {result.total_cycles:,}")
        print(f"Total Slots Used:    {result.total_slots:,}")
        print(f"Init Cycles:         {result.init_cycles:,}")
        print(f"Main Loop Cycles:    {result.main_loop_cycles:,}")
        print()

        # Hotspot analysis
        self.print_subheader("HOTSPOTS (phases by cycle count)")
        print()
        print(f"{'Phase':<20} {'Cycles':>10} {'% of Total':>12} {'Exclusive':>10}")
        print("-" * 70)

        for phase, pct in result.hotspots:
            stats = result.phase_stats[phase]
            if stats.total_cycles == 0:
                continue
            excl_pct = 100.0 * stats.exclusive_cycles / max(1, stats.total_cycles)
            print(f"{phase.value:<20} {stats.total_cycles:>10,} {pct:>11.1f}% {excl_pct:>9.1f}%")

        print()
        print("Exclusive = cycles where ONLY this phase runs")

    def print_phase_breakdown(self, result: ProfileResult):
        self.print_subheader("DETAILED PHASE BREAKDOWN")

        for phase, pct in result.hotspots:
            stats = result.phase_stats[phase]
            if stats.total_cycles == 0:
                continue

            print(f"\n{phase.value.upper()}:")
            print(f"  Cycles:      {stats.total_cycles:,}")
            print(f"  Slots:       {stats.total_slots:,}")
            print(f"  Occurrences: {stats.occurrences:,}")
            print(f"  Avg slots:   {stats.avg_slots_per_occurrence:.2f}")

            if stats.slot_breakdown:
                breakdown = ", ".join(f"{e}:{c}" for e, c in stats.slot_breakdown.items())
                print(f"  Engines:     {breakdown}")

    def print_per_round(self, result: ProfileResult):
        if not result.round_profiles:
            print("\nNo round boundaries detected.")
            return

        self.print_subheader("PER-ROUND BREAKDOWN")
        print()

        # Header
        phases = [Phase.HASH, Phase.MEMORY_LOAD, Phase.MEMORY_STORE,
                  Phase.INDEX_CALC, Phase.XOR_MIX]
        phase_names = [p.value[:8] for p in phases]

        print(f"{'Round':>6} {'Cycles':>8} " + " ".join(f"{n:>10}" for n in phase_names))
        print("-" * 70)

        for rp in result.round_profiles[:10]:  # Show first 10 rounds
            row = f"{rp.round_num:>6} {rp.total_cycles:>8} "
            for phase in phases:
                cycles = rp.phase_cycles.get(phase, 0)
                row += f"{cycles:>10} "
            print(row)

        if len(result.round_profiles) > 10:
            print(f"  ... and {len(result.round_profiles) - 10} more rounds")

    def print_hotspot_bar(self, result: ProfileResult):
        """Print ASCII bar chart of hotspots."""
        self.print_subheader("HOTSPOT VISUALIZATION")
        print()

        max_width = 50
        for phase, pct in result.hotspots[:8]:  # Top 8 phases
            if pct < 1:
                continue
            bar_len = int(pct / 100 * max_width)
            bar = "#" * bar_len
            print(f"{phase.value:<15} [{bar:<{max_width}}] {pct:>5.1f}%")

    def print_recommendations(self, result: ProfileResult):
        self.print_subheader("OPTIMIZATION RECOMMENDATIONS")
        print()

        # Analyze hotspots and give recommendations
        recommendations = []

        hash_pct = next((pct for p, pct in result.hotspots if p == Phase.HASH), 0)
        load_pct = next((pct for p, pct in result.hotspots if p == Phase.MEMORY_LOAD), 0)
        store_pct = next((pct for p, pct in result.hotspots if p == Phase.MEMORY_STORE), 0)
        index_pct = next((pct for p, pct in result.hotspots if p == Phase.INDEX_CALC), 0)

        if hash_pct > 40:
            recommendations.append({
                "issue": f"Hash computation dominates ({hash_pct:.1f}% of cycles)",
                "suggestion": "Focus on hash pipelining - exploit tmp1||tmp2 independence",
                "tool": "See tools/hash_pipeline/hash_pipeline.py for analysis"
            })

        if load_pct > 30:
            recommendations.append({
                "issue": f"Memory loads are significant ({load_pct:.1f}% of cycles)",
                "suggestion": "Consider prefetching or overlapping loads with computation",
                "tool": "Use software pipelining to hide load latency"
            })

        if index_pct > 20:
            recommendations.append({
                "issue": f"Index calculation overhead ({index_pct:.1f}% of cycles)",
                "suggestion": "Pre-compute addresses or use strength reduction",
                "tool": "Look for repeated address patterns"
            })

        if result.init_cycles > result.total_cycles * 0.1:
            recommendations.append({
                "issue": f"Init phase is {result.init_cycles} cycles ({100*result.init_cycles/result.total_cycles:.1f}%)",
                "suggestion": "Consider hoisting constants or reducing init overhead",
                "tool": "Check if constants can be pre-loaded once"
            })

        if not recommendations:
            recommendations.append({
                "issue": "Balanced profile",
                "suggestion": "No single phase dominates - focus on overall utilization",
                "tool": "Use slot_analyzer.py for utilization analysis"
            })

        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['issue']}")
            print(f"   -> {rec['suggestion']}")
            print(f"   -> {rec['tool']}")
            print()


class RichPrinter:
    """Rich-enabled colorful output."""

    def __init__(self):
        self.console = Console()

    def print_header(self, text: str):
        self.console.print(Panel(text, style="bold cyan", box=box.DOUBLE))

    def print_subheader(self, text: str):
        self.console.print(f"\n[bold yellow]{text}[/bold yellow]")
        self.console.print("-" * 70)

    def print_summary(self, result: ProfileResult):
        self.print_header("CYCLE PROFILER - WHERE IS TIME SPENT?")

        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Cycles", f"{result.total_cycles:,}")
        table.add_row("Total Slots Used", f"{result.total_slots:,}")
        table.add_row("Init Cycles", f"{result.init_cycles:,}")
        table.add_row("Main Loop Cycles", f"{result.main_loop_cycles:,}")

        self.console.print(table)

        # Hotspot analysis
        self.print_subheader("HOTSPOTS (phases by cycle count)")

        hot_table = Table(box=box.ROUNDED)
        hot_table.add_column("Phase", style="cyan")
        hot_table.add_column("Cycles", justify="right")
        hot_table.add_column("% of Total", justify="right")
        hot_table.add_column("Exclusive", justify="right")
        hot_table.add_column("Visual", style="dim")

        max_cycles = max((result.phase_stats[p].total_cycles
                          for p, _ in result.hotspots), default=1)

        for phase, pct in result.hotspots:
            stats = result.phase_stats[phase]
            if stats.total_cycles == 0:
                continue

            excl_pct = 100.0 * stats.exclusive_cycles / max(1, stats.total_cycles)
            bar_len = int(20 * stats.total_cycles / max_cycles)
            bar = "#" * bar_len

            color = PHASE_COLORS.get(phase, "white")
            hot_table.add_row(
                f"[{color}]{phase.value}[/{color}]",
                f"{stats.total_cycles:,}",
                f"{pct:.1f}%",
                f"{excl_pct:.1f}%",
                bar
            )

        self.console.print(hot_table)
        self.console.print("\n[dim]Exclusive = cycles where ONLY this phase runs[/dim]")

    def print_phase_breakdown(self, result: ProfileResult):
        self.print_subheader("DETAILED PHASE BREAKDOWN")

        for phase, pct in result.hotspots:
            stats = result.phase_stats[phase]
            if stats.total_cycles == 0:
                continue

            color = PHASE_COLORS.get(phase, "white")

            panel_content = Text()
            panel_content.append(f"Cycles:      {stats.total_cycles:,}\n")
            panel_content.append(f"Slots:       {stats.total_slots:,}\n")
            panel_content.append(f"Occurrences: {stats.occurrences:,}\n")
            panel_content.append(f"Avg slots:   {stats.avg_slots_per_occurrence:.2f}\n")

            if stats.slot_breakdown:
                breakdown = ", ".join(f"{e}:{c}" for e, c in stats.slot_breakdown.items())
                panel_content.append(f"Engines:     {breakdown}")

            self.console.print(Panel(
                panel_content,
                title=f"[{color}]{phase.value.upper()}[/{color}] ({pct:.1f}%)",
                border_style=color
            ))

    def print_per_round(self, result: ProfileResult):
        if not result.round_profiles:
            self.console.print("\n[dim]No round boundaries detected.[/dim]")
            return

        self.print_subheader("PER-ROUND BREAKDOWN")

        table = Table(box=box.ROUNDED)
        table.add_column("Round", justify="right")
        table.add_column("Cycles", justify="right")
        table.add_column("Hash", justify="right", style="red")
        table.add_column("Load", justify="right", style="blue")
        table.add_column("Store", justify="right", style="cyan")
        table.add_column("Index", justify="right", style="yellow")
        table.add_column("XOR", justify="right", style="orange3")

        phases = [Phase.HASH, Phase.MEMORY_LOAD, Phase.MEMORY_STORE,
                  Phase.INDEX_CALC, Phase.XOR_MIX]

        for rp in result.round_profiles[:10]:
            row = [str(rp.round_num), str(rp.total_cycles)]
            for phase in phases:
                cycles = rp.phase_cycles.get(phase, 0)
                row.append(str(cycles))
            table.add_row(*row)

        self.console.print(table)

        if len(result.round_profiles) > 10:
            self.console.print(f"[dim]... and {len(result.round_profiles) - 10} more rounds[/dim]")

    def print_hotspot_bar(self, result: ProfileResult):
        """Print colored bar chart of hotspots."""
        self.print_subheader("HOTSPOT VISUALIZATION")

        max_width = 50
        for phase, pct in result.hotspots[:8]:
            if pct < 1:
                continue
            bar_len = int(pct / 100 * max_width)
            color = PHASE_COLORS.get(phase, "white")
            bar = "[" + color + "]" + "#" * bar_len + "[/" + color + "]"
            self.console.print(f"{phase.value:<15} [{bar:<{max_width + 20}}] {pct:>5.1f}%")

    def print_recommendations(self, result: ProfileResult):
        self.print_subheader("OPTIMIZATION RECOMMENDATIONS")

        recommendations = []

        hash_pct = next((pct for p, pct in result.hotspots if p == Phase.HASH), 0)
        load_pct = next((pct for p, pct in result.hotspots if p == Phase.MEMORY_LOAD), 0)
        index_pct = next((pct for p, pct in result.hotspots if p == Phase.INDEX_CALC), 0)

        if hash_pct > 40:
            recommendations.append({
                "priority": "HIGH",
                "issue": f"Hash computation dominates ({hash_pct:.1f}% of cycles)",
                "suggestion": "Focus on hash pipelining - exploit tmp1||tmp2 independence",
                "tool": "See tools/hash_pipeline/hash_pipeline.py for analysis"
            })

        if load_pct > 30:
            recommendations.append({
                "priority": "HIGH",
                "issue": f"Memory loads are significant ({load_pct:.1f}% of cycles)",
                "suggestion": "Consider prefetching or overlapping loads with computation",
                "tool": "Use software pipelining to hide load latency"
            })

        if index_pct > 20:
            recommendations.append({
                "priority": "MEDIUM",
                "issue": f"Index calculation overhead ({index_pct:.1f}% of cycles)",
                "suggestion": "Pre-compute addresses or use strength reduction",
                "tool": "Look for repeated address patterns"
            })

        if result.init_cycles > result.total_cycles * 0.1:
            recommendations.append({
                "priority": "LOW",
                "issue": f"Init phase is {result.init_cycles} cycles",
                "suggestion": "Consider hoisting constants or reducing init overhead",
                "tool": "Check if constants can be pre-loaded once"
            })

        if not recommendations:
            recommendations.append({
                "priority": "INFO",
                "issue": "Balanced profile",
                "suggestion": "No single phase dominates - focus on overall utilization",
                "tool": "Use slot_analyzer.py for utilization analysis"
            })

        for rec in recommendations:
            priority = rec["priority"]
            if priority == "HIGH":
                style = "bold red"
            elif priority == "MEDIUM":
                style = "bold yellow"
            else:
                style = "bold blue"

            panel = Panel(
                f"[white]{rec['suggestion']}[/white]\n\n"
                f"[dim]{rec['tool']}[/dim]",
                title=f"[{style}][{priority}][/{style}] {rec['issue']}",
                border_style=style
            )
            self.console.print(panel)


def get_printer(use_rich: bool = True):
    """Get appropriate printer based on Rich availability."""
    if use_rich and RICH_AVAILABLE:
        return RichPrinter()
    return PlainPrinter()


def analyze_kernel(kernel_builder=None):
    """Analyze the current kernel from perf_takehome.py."""
    if kernel_builder is None:
        from perf_takehome import KernelBuilder

        # Standard test params
        forest_height = 10
        n_nodes = 2 ** (forest_height + 1) - 1
        batch_size = 256
        rounds = 16

        kb = KernelBuilder()
        kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
        return kb.instrs
    else:
        return kernel_builder.instrs


def main():
    parser = argparse.ArgumentParser(
        description="Cycle Profiler - Break down cycles by code section",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/cycle_profiler/cycle_profiler.py           # Basic profiling
    python tools/cycle_profiler/cycle_profiler.py --detailed # Show phase details
    python tools/cycle_profiler/cycle_profiler.py --per-round # Per-round breakdown
    python tools/cycle_profiler/cycle_profiler.py --json     # JSON output
        """
    )
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output JSON instead of human-readable")
    parser.add_argument("--detailed", "-d", action="store_true",
                        help="Show detailed phase breakdown")
    parser.add_argument("--per-round", "-r", action="store_true",
                        help="Show per-round breakdown")
    parser.add_argument("--recommendations", action="store_true",
                        help="Show optimization recommendations")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Show all analyses")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")
    args = parser.parse_args()

    print("Loading kernel...", file=sys.stderr)
    instructions = analyze_kernel()

    print(f"Profiling {len(instructions)} cycles...", file=sys.stderr)
    result = profile_instructions(instructions)

    if args.json:
        print(result.to_json())
    else:
        printer = get_printer(not args.no_color)

        printer.print_summary(result)
        printer.print_hotspot_bar(result)

        if args.detailed or args.all:
            printer.print_phase_breakdown(result)

        if args.per_round or args.all:
            printer.print_per_round(result)

        if args.recommendations or args.all:
            printer.print_recommendations(result)


if __name__ == "__main__":
    main()
