#!/usr/bin/env python3
"""
Kernel Diff Tool for VLIW SIMD Optimization

Compares two kernel versions to track optimization impact:
1. Cycle comparison (before/after cycles, speedup)
2. Utilization diff (slot usage changes per engine)
3. Instruction diff (what ops changed, added, removed)
4. Side-by-side comparison view

Usage:
    python tools/kernel_diff/kernel_diff.py kernel1.json kernel2.json
    python tools/kernel_diff/kernel_diff.py --before kernel1.json --after kernel2.json
    python tools/kernel_diff/kernel_diff.py --json kernel1.json kernel2.json

Output Formats:
    - Rich colored output (default, if rich library available)
    - Plain text (--no-color)
    - JSON (--json)
"""

import sys
import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from enum import Enum

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import SLOT_LIMITS, VLEN

# Try to import Rich for better formatting
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Engine slot limits (excluding debug)
ENGINES = {k: v for k, v in SLOT_LIMITS.items() if k != "debug"}
MAX_SLOTS_PER_CYCLE = sum(ENGINES.values())  # 12 + 6 + 2 + 2 + 1 = 23


class ChangeType(Enum):
    """Types of instruction changes between versions."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class CycleStats:
    """Statistics for a single cycle."""
    cycle_num: int
    slots_used: Dict[str, int]
    instructions: List[tuple]

    @property
    def total_used(self) -> int:
        return sum(self.slots_used.values())

    @property
    def utilization_pct(self) -> float:
        return 100.0 * self.total_used / MAX_SLOTS_PER_CYCLE


@dataclass
class KernelStats:
    """Aggregated statistics for a kernel."""
    total_cycles: int
    total_slots_used: int
    per_engine: Dict[str, Dict[str, Any]]
    utilization_pct: float
    per_cycle: List[CycleStats]
    op_counts: Dict[str, int]  # Operation counts by opcode
    histogram: Dict[int, int]  # Slots per cycle histogram

    def to_dict(self) -> dict:
        return {
            "total_cycles": self.total_cycles,
            "total_slots_used": self.total_slots_used,
            "max_possible_slots": self.total_cycles * MAX_SLOTS_PER_CYCLE,
            "utilization_pct": round(self.utilization_pct, 2),
            "slots_per_cycle_avg": round(self.total_slots_used / max(1, self.total_cycles), 2),
            "per_engine": self.per_engine,
            "op_counts": dict(sorted(self.op_counts.items(), key=lambda x: -x[1])[:20]),
            "histogram": self.histogram
        }


@dataclass
class InstructionChange:
    """Represents a change in instructions between versions."""
    change_type: ChangeType
    engine: str
    opcode: str
    count_before: int
    count_after: int


@dataclass
class DiffResult:
    """Complete diff results between two kernels."""
    # Names
    name_before: str
    name_after: str

    # Stats for each kernel
    stats_before: KernelStats
    stats_after: KernelStats

    # Cycle comparison
    cycle_delta: int
    speedup: float

    # Utilization comparison
    utilization_delta: float
    per_engine_delta: Dict[str, Dict[str, float]]

    # Instruction changes
    op_changes: List[InstructionChange]
    ops_added: List[str]
    ops_removed: List[str]

    # Per-cycle alignment
    cycle_alignment: List[Tuple[int, Optional[int], Optional[int]]]  # (idx, before_slots, after_slots)

    def to_dict(self) -> dict:
        return {
            "names": {
                "before": self.name_before,
                "after": self.name_after
            },
            "cycles": {
                "before": self.stats_before.total_cycles,
                "after": self.stats_after.total_cycles,
                "delta": self.cycle_delta,
                "speedup": round(self.speedup, 3),
                "speedup_pct": round((self.speedup - 1) * 100, 1) if self.speedup >= 1 else round((1 - 1/self.speedup) * -100, 1)
            },
            "slots_used": {
                "before": self.stats_before.total_slots_used,
                "after": self.stats_after.total_slots_used,
                "delta": self.stats_after.total_slots_used - self.stats_before.total_slots_used
            },
            "utilization": {
                "before": round(self.stats_before.utilization_pct, 2),
                "after": round(self.stats_after.utilization_pct, 2),
                "delta": round(self.utilization_delta, 2)
            },
            "per_engine": self.per_engine_delta,
            "operation_changes": {
                "added": self.ops_added,
                "removed": self.ops_removed,
                "changes": [
                    {
                        "engine": c.engine,
                        "opcode": c.opcode,
                        "before": c.count_before,
                        "after": c.count_after,
                        "delta": c.count_after - c.count_before,
                        "type": c.change_type.value
                    }
                    for c in self.op_changes[:30]  # Limit output
                ]
            },
            "summary": self._summary_text()
        }

    def _summary_text(self) -> str:
        """Generate a brief summary."""
        if self.speedup > 1.0:
            return f"IMPROVEMENT: {self.speedup:.2f}x faster ({self.cycle_delta:+d} cycles)"
        elif self.speedup < 1.0:
            slowdown = 1.0 / self.speedup
            return f"REGRESSION: {slowdown:.2f}x slower ({self.cycle_delta:+d} cycles)"
        else:
            return "NO CHANGE in cycle count"

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def extract_op_from_slot(slot: tuple) -> Optional[str]:
    """Extract operation code from instruction slot."""
    if slot and len(slot) > 0:
        return str(slot[0])
    return None


def analyze_kernel(instructions: List[dict]) -> KernelStats:
    """Analyze a kernel and compute statistics."""
    per_cycle_stats = []
    engine_totals = defaultdict(int)
    histogram = defaultdict(int)
    op_counts = Counter()

    for cycle_num, instr in enumerate(instructions):
        slots_used = {}
        cycle_instrs = []

        for engine, slots in instr.items():
            if engine == "debug":
                continue
            count = len(slots) if slots else 0
            slots_used[engine] = count
            engine_totals[engine] += count

            if slots:
                for slot in slots:
                    cycle_instrs.append(slot)
                    op = extract_op_from_slot(slot)
                    if op:
                        op_counts[f"{engine}:{op}"] += 1

        # Fill in zeros for unused engines
        for engine in ENGINES:
            if engine not in slots_used:
                slots_used[engine] = 0

        stats = CycleStats(
            cycle_num=cycle_num,
            slots_used=slots_used,
            instructions=cycle_instrs
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

    utilization_pct = 100.0 * total_slots / max(1, total_cycles * MAX_SLOTS_PER_CYCLE)

    return KernelStats(
        total_cycles=total_cycles,
        total_slots_used=total_slots,
        per_engine=per_engine,
        utilization_pct=utilization_pct,
        per_cycle=per_cycle_stats,
        op_counts=dict(op_counts),
        histogram=dict(sorted(histogram.items()))
    )


def compute_diff(
    instrs_before: List[dict],
    instrs_after: List[dict],
    name_before: str = "Before",
    name_after: str = "After"
) -> DiffResult:
    """Compute complete diff between two kernel versions."""

    stats_before = analyze_kernel(instrs_before)
    stats_after = analyze_kernel(instrs_after)

    # Cycle comparison
    cycle_delta = stats_after.total_cycles - stats_before.total_cycles
    speedup = stats_before.total_cycles / max(1, stats_after.total_cycles)

    # Utilization comparison
    utilization_delta = stats_after.utilization_pct - stats_before.utilization_pct

    # Per-engine delta
    per_engine_delta = {}
    for engine in ENGINES:
        before = stats_before.per_engine.get(engine, {})
        after = stats_after.per_engine.get(engine, {})
        per_engine_delta[engine] = {
            "before_util": before.get("utilization_pct", 0),
            "after_util": after.get("utilization_pct", 0),
            "delta_util": after.get("utilization_pct", 0) - before.get("utilization_pct", 0),
            "before_total": before.get("total_used", 0),
            "after_total": after.get("total_used", 0),
            "delta_total": after.get("total_used", 0) - before.get("total_used", 0)
        }

    # Operation changes
    all_ops = set(stats_before.op_counts.keys()) | set(stats_after.op_counts.keys())
    ops_added = []
    ops_removed = []
    op_changes = []

    for op in sorted(all_ops):
        before_count = stats_before.op_counts.get(op, 0)
        after_count = stats_after.op_counts.get(op, 0)

        # Parse engine:opcode
        parts = op.split(":", 1)
        engine = parts[0] if parts else "unknown"
        opcode = parts[1] if len(parts) > 1 else op

        if before_count == 0:
            change_type = ChangeType.ADDED
            ops_added.append(op)
        elif after_count == 0:
            change_type = ChangeType.REMOVED
            ops_removed.append(op)
        elif before_count != after_count:
            change_type = ChangeType.MODIFIED
        else:
            change_type = ChangeType.UNCHANGED

        if change_type != ChangeType.UNCHANGED:
            op_changes.append(InstructionChange(
                change_type=change_type,
                engine=engine,
                opcode=opcode,
                count_before=before_count,
                count_after=after_count
            ))

    # Sort by magnitude of change
    op_changes.sort(key=lambda c: abs(c.count_after - c.count_before), reverse=True)

    # Per-cycle alignment (for side-by-side)
    max_cycles = max(stats_before.total_cycles, stats_after.total_cycles)
    cycle_alignment = []
    for i in range(max_cycles):
        before_slots = stats_before.per_cycle[i].total_used if i < stats_before.total_cycles else None
        after_slots = stats_after.per_cycle[i].total_used if i < stats_after.total_cycles else None
        cycle_alignment.append((i, before_slots, after_slots))

    return DiffResult(
        name_before=name_before,
        name_after=name_after,
        stats_before=stats_before,
        stats_after=stats_after,
        cycle_delta=cycle_delta,
        speedup=speedup,
        utilization_delta=utilization_delta,
        per_engine_delta=per_engine_delta,
        op_changes=op_changes,
        ops_added=ops_added,
        ops_removed=ops_removed,
        cycle_alignment=cycle_alignment
    )


def load_kernel_json(filepath: str) -> List[dict]:
    """Load kernel instructions from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_kernel_json(instructions: List[dict], filepath: str) -> None:
    """Save kernel instructions to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(instructions, f)


# ============== Output Formatting ==============

class PlainPrinter:
    """Plain text output without Rich."""

    def print_header(self, text: str):
        print("=" * 70)
        print(text)
        print("=" * 70)

    def print_subheader(self, text: str):
        print("-" * 70)
        print(text)
        print("-" * 70)

    def print_diff(self, diff: DiffResult, verbose: bool = False):
        self.print_header(f"KERNEL DIFF: {diff.name_before} vs {diff.name_after}")
        print()

        # Summary
        self.print_subheader("SUMMARY")
        print(diff._summary_text())
        print()

        # Cycle comparison
        self.print_subheader("CYCLE COMPARISON")
        delta_str = f"+{diff.cycle_delta}" if diff.cycle_delta > 0 else str(diff.cycle_delta)
        print(f"  Before:  {diff.stats_before.total_cycles:,} cycles")
        print(f"  After:   {diff.stats_after.total_cycles:,} cycles")
        print(f"  Delta:   {delta_str} cycles")

        if diff.speedup > 1.0:
            print(f"  Speedup: {diff.speedup:.2f}x FASTER")
        elif diff.speedup < 1.0:
            print(f"  Speedup: {1/diff.speedup:.2f}x SLOWER")
        else:
            print(f"  Speedup: No change")
        print()

        # Utilization comparison
        self.print_subheader("UTILIZATION COMPARISON")
        util_delta_str = f"+{diff.utilization_delta:.1f}%" if diff.utilization_delta > 0 else f"{diff.utilization_delta:.1f}%"
        print(f"  Before:  {diff.stats_before.utilization_pct:.1f}%")
        print(f"  After:   {diff.stats_after.utilization_pct:.1f}%")
        print(f"  Delta:   {util_delta_str}")
        print()

        # Per-engine breakdown
        self.print_subheader("PER-ENGINE CHANGES")
        print(f"{'Engine':<10} {'Before':>12} {'After':>12} {'Delta':>12} {'Util Delta':>12}")
        print("-" * 70)

        for engine, data in diff.per_engine_delta.items():
            delta_total = data["delta_total"]
            delta_util = data["delta_util"]
            delta_str = f"+{delta_total}" if delta_total > 0 else str(delta_total)
            util_str = f"+{delta_util:.1f}%" if delta_util > 0 else f"{delta_util:.1f}%"
            print(f"{engine:<10} {data['before_total']:>12,} {data['after_total']:>12,} {delta_str:>12} {util_str:>12}")
        print()

        # Operation changes
        if diff.op_changes:
            self.print_subheader("SIGNIFICANT OPERATION CHANGES")
            print(f"{'Operation':<25} {'Before':>10} {'After':>10} {'Delta':>10} {'Type':>12}")
            print("-" * 70)

            for change in diff.op_changes[:20]:
                delta = change.count_after - change.count_before
                delta_str = f"+{delta}" if delta > 0 else str(delta)
                op_name = f"{change.engine}:{change.opcode}"
                print(f"{op_name:<25} {change.count_before:>10,} {change.count_after:>10,} {delta_str:>10} {change.change_type.value:>12}")

            if len(diff.op_changes) > 20:
                print(f"  ... and {len(diff.op_changes) - 20} more changes")
        print()

        # Added/Removed operations
        if diff.ops_added:
            print(f"NEW OPERATIONS ({len(diff.ops_added)}): {', '.join(diff.ops_added[:10])}")
            if len(diff.ops_added) > 10:
                print(f"  ... and {len(diff.ops_added) - 10} more")

        if diff.ops_removed:
            print(f"REMOVED OPERATIONS ({len(diff.ops_removed)}): {', '.join(diff.ops_removed[:10])}")
            if len(diff.ops_removed) > 10:
                print(f"  ... and {len(diff.ops_removed) - 10} more")

        print()

        # Side-by-side (verbose)
        if verbose:
            self.print_side_by_side(diff)

    def print_side_by_side(self, diff: DiffResult, limit: int = 50):
        """Print side-by-side cycle comparison."""
        self.print_subheader(f"SIDE-BY-SIDE COMPARISON (first {limit} cycles)")
        print(f"{'Cycle':>8} | {'Before':>8} | {'After':>8} | {'Delta':>8}")
        print("-" * 42)

        for i, before_slots, after_slots in diff.cycle_alignment[:limit]:
            before_str = str(before_slots) if before_slots is not None else "-"
            after_str = str(after_slots) if after_slots is not None else "-"

            if before_slots is not None and after_slots is not None:
                delta = after_slots - before_slots
                delta_str = f"+{delta}" if delta > 0 else str(delta)
            else:
                delta_str = "-"

            print(f"{i:>8} | {before_str:>8} | {after_str:>8} | {delta_str:>8}")

        if len(diff.cycle_alignment) > limit:
            print(f"  ... {len(diff.cycle_alignment) - limit} more cycles")
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

    def print_diff(self, diff: DiffResult, verbose: bool = False):
        self.print_header(f"KERNEL DIFF: {diff.name_before} vs {diff.name_after}")

        # Summary panel
        summary = diff._summary_text()
        if diff.speedup > 1.0:
            style = "bold green"
        elif diff.speedup < 1.0:
            style = "bold red"
        else:
            style = "bold yellow"

        self.console.print(Panel(f"[{style}]{summary}[/{style}]", title="Summary", border_style=style))

        # Cycle comparison
        self.print_subheader("CYCLE COMPARISON")

        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Before", f"{diff.stats_before.total_cycles:,} cycles")
        table.add_row("After", f"{diff.stats_after.total_cycles:,} cycles")

        delta = diff.cycle_delta
        delta_color = "red" if delta > 0 else "green" if delta < 0 else "white"
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        table.add_row("Delta", f"[{delta_color}]{delta_str} cycles[/{delta_color}]")

        if diff.speedup > 1.0:
            table.add_row("Speedup", f"[bold green]{diff.speedup:.2f}x FASTER[/bold green]")
        elif diff.speedup < 1.0:
            table.add_row("Speedup", f"[bold red]{1/diff.speedup:.2f}x SLOWER[/bold red]")
        else:
            table.add_row("Speedup", "[yellow]No change[/yellow]")

        self.console.print(table)

        # Utilization comparison
        self.print_subheader("UTILIZATION COMPARISON")

        util_table = Table(show_header=False, box=box.SIMPLE)
        util_table.add_column("Metric", style="cyan")
        util_table.add_column("Value", style="white")

        util_table.add_row("Before", f"{diff.stats_before.utilization_pct:.1f}%")
        util_table.add_row("After", f"{diff.stats_after.utilization_pct:.1f}%")

        util_delta = diff.utilization_delta
        util_color = "green" if util_delta > 0 else "red" if util_delta < 0 else "white"
        util_str = f"+{util_delta:.1f}%" if util_delta > 0 else f"{util_delta:.1f}%"
        util_table.add_row("Delta", f"[{util_color}]{util_str}[/{util_color}]")

        self.console.print(util_table)

        # Per-engine breakdown
        self.print_subheader("PER-ENGINE CHANGES")

        eng_table = Table(box=box.ROUNDED)
        eng_table.add_column("Engine", style="cyan")
        eng_table.add_column("Before", justify="right")
        eng_table.add_column("After", justify="right")
        eng_table.add_column("Delta", justify="right")
        eng_table.add_column("Util Delta", justify="right")

        for engine, data in diff.per_engine_delta.items():
            delta_total = data["delta_total"]
            delta_util = data["delta_util"]

            delta_color = "green" if delta_total < 0 else "red" if delta_total > 0 else "white"
            util_color = "green" if delta_util > 0 else "red" if delta_util < 0 else "white"

            delta_str = f"+{delta_total}" if delta_total > 0 else str(delta_total)
            util_str = f"+{delta_util:.1f}%" if delta_util > 0 else f"{delta_util:.1f}%"

            eng_table.add_row(
                engine,
                f"{data['before_total']:,}",
                f"{data['after_total']:,}",
                f"[{delta_color}]{delta_str}[/{delta_color}]",
                f"[{util_color}]{util_str}[/{util_color}]"
            )

        self.console.print(eng_table)

        # Operation changes
        if diff.op_changes:
            self.print_subheader("SIGNIFICANT OPERATION CHANGES")

            op_table = Table(box=box.ROUNDED)
            op_table.add_column("Operation", style="cyan")
            op_table.add_column("Before", justify="right")
            op_table.add_column("After", justify="right")
            op_table.add_column("Delta", justify="right")
            op_table.add_column("Type", justify="center")

            for change in diff.op_changes[:20]:
                delta = change.count_after - change.count_before
                delta_str = f"+{delta}" if delta > 0 else str(delta)

                if change.change_type == ChangeType.ADDED:
                    type_style = "green"
                    delta_style = "green"
                elif change.change_type == ChangeType.REMOVED:
                    type_style = "red"
                    delta_style = "red"
                else:
                    type_style = "yellow"
                    delta_style = "green" if delta < 0 else "red" if delta > 0 else "white"

                op_name = f"{change.engine}:{change.opcode}"
                op_table.add_row(
                    op_name,
                    f"{change.count_before:,}",
                    f"{change.count_after:,}",
                    f"[{delta_style}]{delta_str}[/{delta_style}]",
                    f"[{type_style}]{change.change_type.value}[/{type_style}]"
                )

            self.console.print(op_table)

            if len(diff.op_changes) > 20:
                self.console.print(f"[dim]  ... and {len(diff.op_changes) - 20} more changes[/dim]")

        # Added/Removed operations
        if diff.ops_added:
            self.console.print(f"\n[green]NEW OPERATIONS ({len(diff.ops_added)}):[/green] {', '.join(diff.ops_added[:10])}")
            if len(diff.ops_added) > 10:
                self.console.print(f"[dim]  ... and {len(diff.ops_added) - 10} more[/dim]")

        if diff.ops_removed:
            self.console.print(f"\n[red]REMOVED OPERATIONS ({len(diff.ops_removed)}):[/red] {', '.join(diff.ops_removed[:10])}")
            if len(diff.ops_removed) > 10:
                self.console.print(f"[dim]  ... and {len(diff.ops_removed) - 10} more[/dim]")

        # Side-by-side (verbose)
        if verbose:
            self.print_side_by_side(diff)

    def print_side_by_side(self, diff: DiffResult, limit: int = 50):
        """Print side-by-side cycle comparison."""
        self.print_subheader(f"SIDE-BY-SIDE COMPARISON (first {limit} cycles)")

        table = Table(box=box.SIMPLE)
        table.add_column("Cycle", justify="right", style="dim")
        table.add_column("Before", justify="right")
        table.add_column("After", justify="right")
        table.add_column("Delta", justify="right")

        for i, before_slots, after_slots in diff.cycle_alignment[:limit]:
            before_str = str(before_slots) if before_slots is not None else "[dim]-[/dim]"
            after_str = str(after_slots) if after_slots is not None else "[dim]-[/dim]"

            if before_slots is not None and after_slots is not None:
                delta = after_slots - before_slots
                delta_color = "green" if delta > 0 else "red" if delta < 0 else "dim"
                delta_str = f"+{delta}" if delta > 0 else str(delta)
                delta_str = f"[{delta_color}]{delta_str}[/{delta_color}]"
            else:
                delta_str = "[dim]-[/dim]"

            table.add_row(str(i), before_str, after_str, delta_str)

        self.console.print(table)

        if len(diff.cycle_alignment) > limit:
            self.console.print(f"[dim]  ... {len(diff.cycle_alignment) - limit} more cycles[/dim]")


def get_printer(use_color: bool = True):
    """Get the appropriate printer based on Rich availability."""
    if use_color and RICH_AVAILABLE:
        return RichPrinter()
    return PlainPrinter()


def get_current_kernel():
    """Load the current kernel from perf_takehome.py."""
    from perf_takehome import KernelBuilder

    forest_height = 10
    n_nodes = 2 ** (forest_height + 1) - 1
    batch_size = 256
    rounds = 16

    kb = KernelBuilder()
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
    return kb.instrs


def main():
    parser = argparse.ArgumentParser(
        description="Compare two VLIW SIMD kernel versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare two saved kernels
    python tools/kernel_diff/kernel_diff.py kernel1.json kernel2.json

    # Named comparison
    python tools/kernel_diff/kernel_diff.py --before v1.json --after v2.json

    # Compare current kernel with saved baseline
    python tools/kernel_diff/kernel_diff.py --baseline baseline.json

    # JSON output
    python tools/kernel_diff/kernel_diff.py --json kernel1.json kernel2.json

    # Verbose with side-by-side
    python tools/kernel_diff/kernel_diff.py -v kernel1.json kernel2.json

    # Save current kernel
    python tools/kernel_diff/kernel_diff.py --save current.json
        """
    )

    parser.add_argument("files", nargs="*", help="Kernel JSON files to compare (before after)")
    parser.add_argument("--before", "-b", metavar="FILE", help="Before kernel JSON file")
    parser.add_argument("--after", "-a", metavar="FILE", help="After kernel JSON file")
    parser.add_argument("--baseline", metavar="FILE", help="Compare current kernel against baseline")
    parser.add_argument("--save", metavar="FILE", help="Save current kernel to JSON file")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show side-by-side cycle comparison")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    args = parser.parse_args()

    # Handle --save
    if args.save:
        print("Loading current kernel...", file=sys.stderr)
        current = get_current_kernel()
        save_kernel_json(current, args.save)
        print(f"Saved current kernel to {args.save}", file=sys.stderr)
        if not args.files and not args.before and not args.baseline:
            return

    # Determine which files to compare
    instrs_before = None
    instrs_after = None
    name_before = "Before"
    name_after = "After"

    if args.baseline:
        # Compare current vs baseline
        print("Loading baseline...", file=sys.stderr)
        instrs_before = load_kernel_json(args.baseline)
        name_before = os.path.basename(args.baseline)

        print("Loading current kernel...", file=sys.stderr)
        instrs_after = get_current_kernel()
        name_after = "Current"

    elif args.before and args.after:
        # Explicit --before/--after
        print("Loading kernels...", file=sys.stderr)
        instrs_before = load_kernel_json(args.before)
        instrs_after = load_kernel_json(args.after)
        name_before = os.path.basename(args.before)
        name_after = os.path.basename(args.after)

    elif len(args.files) >= 2:
        # Positional arguments
        print("Loading kernels...", file=sys.stderr)
        instrs_before = load_kernel_json(args.files[0])
        instrs_after = load_kernel_json(args.files[1])
        name_before = os.path.basename(args.files[0])
        name_after = os.path.basename(args.files[1])

    else:
        if not args.save:
            parser.error("Must provide two kernel files to compare, or use --baseline")
        return

    # Compute diff
    print(f"Comparing {len(instrs_before)} vs {len(instrs_after)} cycles...", file=sys.stderr)
    diff = compute_diff(instrs_before, instrs_after, name_before, name_after)

    # Output
    if args.json:
        print(diff.to_json())
    else:
        printer = get_printer(not args.no_color)
        printer.print_diff(diff, verbose=args.verbose)


if __name__ == "__main__":
    main()
