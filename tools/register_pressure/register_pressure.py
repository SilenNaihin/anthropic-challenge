#!/usr/bin/env python3
"""
Register Pressure Analyzer for VLIW SIMD Kernel

Analyzes scratch memory (register file) usage patterns to understand
register pressure throughout kernel execution.

Features:
1. Live range tracking - when each address is written and last read
2. Pressure computation - simultaneous live values per cycle
3. Peak detection - cycles where pressure is highest
4. Dead register detection - values that could be reused earlier
5. Renaming suggestions - reduce pressure via register renaming
6. Limit warnings - alert when approaching 1536 scratch limit
7. Pressure visualization - text-based chart over time
8. Rich output with plain text fallback
9. JSON output for scripting

Usage:
    python register_pressure.py                    # Full analysis
    python register_pressure.py --peaks            # Focus on pressure peaks
    python register_pressure.py --reuse            # Find reuse opportunities
    python register_pressure.py --visualize        # Show pressure chart
    python register_pressure.py --json             # Output as JSON
"""

import sys
import os
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import SLOT_LIMITS, VLEN, SCRATCH_SIZE

# Try to import Rich for better formatting
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from rich.progress import Progress, BarColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class LiveRange:
    """Tracks the live range of a scratch address."""
    address: int
    def_cycle: int                # Cycle where value is written (defined)
    last_use_cycle: int           # Cycle where value is last read (used)
    def_engine: str               # Engine that wrote the value
    def_slot: tuple               # Slot that wrote the value
    use_count: int = 0            # Number of times read
    is_vector: bool = False       # Is this part of a vector allocation

    @property
    def live_length(self) -> int:
        """Number of cycles this value is live."""
        return max(0, self.last_use_cycle - self.def_cycle)

    @property
    def is_dead_after_def(self) -> bool:
        """True if value is never read after being written."""
        return self.use_count == 0

    def to_dict(self) -> dict:
        return {
            "address": self.address,
            "def_cycle": self.def_cycle,
            "last_use_cycle": self.last_use_cycle,
            "live_length": self.live_length,
            "def_engine": self.def_engine,
            "use_count": self.use_count,
            "is_dead": self.is_dead_after_def,
            "is_vector": self.is_vector
        }


@dataclass
class PressurePeak:
    """A cycle where register pressure is notably high."""
    cycle: int
    live_count: int
    percentage: float  # of SCRATCH_SIZE
    contributing_ranges: List[int]  # addresses contributing to pressure

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "live_count": self.live_count,
            "percentage": round(self.percentage, 2),
            "sample_addresses": self.contributing_ranges[:10]  # First 10
        }


@dataclass
class ReuseOpportunity:
    """Suggestion for register reuse."""
    dead_address: int
    dead_after_cycle: int
    potential_reuse_cycle: int
    saves_addresses: int

    def to_dict(self) -> dict:
        return {
            "dead_address": self.dead_address,
            "dead_after_cycle": self.dead_after_cycle,
            "potential_reuse_cycle": self.potential_reuse_cycle,
            "saves_addresses": self.saves_addresses
        }


@dataclass
class AnalysisResult:
    """Complete register pressure analysis results."""
    total_cycles: int = 0
    total_addresses_used: int = 0
    max_simultaneous_live: int = 0
    max_live_cycle: int = 0
    avg_live_per_cycle: float = 0.0
    scratch_limit: int = SCRATCH_SIZE

    live_ranges: Dict[int, LiveRange] = field(default_factory=dict)
    pressure_per_cycle: List[int] = field(default_factory=list)
    peaks: List[PressurePeak] = field(default_factory=list)
    reuse_opportunities: List[ReuseOpportunity] = field(default_factory=list)

    # Breakdown stats
    vector_addresses: int = 0
    scalar_addresses: int = 0
    dead_after_def: int = 0  # Addresses never read
    long_lived: int = 0       # Live > 50 cycles
    constants: int = 0        # Constant values (live entire program)

    @property
    def utilization_pct(self) -> float:
        """Percentage of scratch space used."""
        return 100.0 * self.total_addresses_used / self.scratch_limit

    @property
    def peak_pressure_pct(self) -> float:
        """Peak pressure as percentage of limit."""
        return 100.0 * self.max_simultaneous_live / self.scratch_limit

    @property
    def headroom(self) -> int:
        """Addresses available before hitting limit."""
        return self.scratch_limit - self.total_addresses_used

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_cycles": self.total_cycles,
                "total_addresses_used": self.total_addresses_used,
                "utilization_pct": round(self.utilization_pct, 2),
                "max_simultaneous_live": self.max_simultaneous_live,
                "max_live_cycle": self.max_live_cycle,
                "peak_pressure_pct": round(self.peak_pressure_pct, 2),
                "avg_live_per_cycle": round(self.avg_live_per_cycle, 2),
                "scratch_limit": self.scratch_limit,
                "headroom": self.headroom,
            },
            "breakdown": {
                "vector_addresses": self.vector_addresses,
                "scalar_addresses": self.scalar_addresses,
                "dead_after_def": self.dead_after_def,
                "long_lived": self.long_lived,
                "constants": self.constants,
            },
            "peaks": [p.to_dict() for p in self.peaks[:10]],
            "reuse_opportunities": [r.to_dict() for r in self.reuse_opportunities[:20]],
            "pressure_histogram": self._pressure_histogram(),
        }

    def _pressure_histogram(self) -> dict:
        """Create histogram of pressure values."""
        if not self.pressure_per_cycle:
            return {}

        buckets = {
            "0-10%": 0,
            "10-25%": 0,
            "25-50%": 0,
            "50-75%": 0,
            "75-90%": 0,
            "90-100%": 0,
        }

        for pressure in self.pressure_per_cycle:
            pct = 100.0 * pressure / self.scratch_limit
            if pct < 10:
                buckets["0-10%"] += 1
            elif pct < 25:
                buckets["10-25%"] += 1
            elif pct < 50:
                buckets["25-50%"] += 1
            elif pct < 75:
                buckets["50-75%"] += 1
            elif pct < 90:
                buckets["75-90%"] += 1
            else:
                buckets["90-100%"] += 1

        return buckets

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class RegisterPressureAnalyzer:
    """Analyzes register pressure in VLIW SIMD kernels."""

    def __init__(self, instructions: List[dict], scratch_size: int = SCRATCH_SIZE):
        self.instructions = instructions
        self.scratch_size = scratch_size
        self.result = AnalysisResult()
        self.result.total_cycles = len(instructions)
        self.result.scratch_limit = scratch_size

        # Tracking state
        self.def_cycles: Dict[int, int] = {}       # addr -> cycle where defined
        self.def_info: Dict[int, Tuple[str, tuple]] = {}  # addr -> (engine, slot)
        self.last_use: Dict[int, int] = {}          # addr -> last cycle read
        self.use_count: Dict[int, int] = defaultdict(int)  # addr -> count of reads
        self.is_vector: Dict[int, bool] = {}        # addr -> is part of vector
        self.is_constant: Dict[int, bool] = {}      # addr -> is constant load

    def analyze(self) -> AnalysisResult:
        """Run complete register pressure analysis."""
        self._collect_def_use_info()
        self._build_live_ranges()
        self._compute_pressure_per_cycle()
        self._find_peaks()
        self._find_reuse_opportunities()
        self._compute_statistics()
        return self.result

    def _collect_def_use_info(self):
        """First pass: collect definition and use information."""
        for cycle_num, instr in enumerate(self.instructions):
            for engine, slots in instr.items():
                if engine == "debug":
                    continue
                for slot in (slots or []):
                    reads, writes, is_vec, is_const = self._extract_reads_writes(slot, engine)

                    # Record definitions
                    for addr in writes:
                        # Only record first definition (later defs create new live ranges)
                        if addr not in self.def_cycles:
                            self.def_cycles[addr] = cycle_num
                            self.def_info[addr] = (engine, slot)
                            self.is_vector[addr] = is_vec
                            self.is_constant[addr] = is_const

                    # Record uses
                    for addr in reads:
                        self.last_use[addr] = cycle_num
                        self.use_count[addr] += 1

    def _build_live_ranges(self):
        """Build live range for each defined address."""
        for addr in self.def_cycles:
            def_cycle = self.def_cycles[addr]
            # If never read, live range ends at definition
            last_use_cycle = self.last_use.get(addr, def_cycle)
            engine, slot = self.def_info[addr]

            live_range = LiveRange(
                address=addr,
                def_cycle=def_cycle,
                last_use_cycle=last_use_cycle,
                def_engine=engine,
                def_slot=slot,
                use_count=self.use_count[addr],
                is_vector=self.is_vector.get(addr, False)
            )
            self.result.live_ranges[addr] = live_range

    def _compute_pressure_per_cycle(self):
        """Compute how many values are live at each cycle."""
        pressure = []

        for cycle in range(self.result.total_cycles):
            live_at_cycle = 0
            for addr, lr in self.result.live_ranges.items():
                # A value is live from def_cycle to last_use_cycle (inclusive)
                if lr.def_cycle <= cycle <= lr.last_use_cycle:
                    live_at_cycle += 1
            pressure.append(live_at_cycle)

        self.result.pressure_per_cycle = pressure

        if pressure:
            self.result.max_simultaneous_live = max(pressure)
            self.result.max_live_cycle = pressure.index(self.result.max_simultaneous_live)
            self.result.avg_live_per_cycle = sum(pressure) / len(pressure)

    def _find_peaks(self):
        """Find cycles with notably high register pressure."""
        if not self.result.pressure_per_cycle:
            return

        # Consider top 5% of pressure as peaks, or anything above 75%
        threshold = max(
            sorted(self.result.pressure_per_cycle, reverse=True)[:max(1, len(self.result.pressure_per_cycle) // 20)][-1],
            int(0.75 * self.scratch_size)
        )

        peaks = []
        for cycle, pressure in enumerate(self.result.pressure_per_cycle):
            if pressure >= threshold:
                # Find which addresses are live at this cycle
                contributing = [
                    addr for addr, lr in self.result.live_ranges.items()
                    if lr.def_cycle <= cycle <= lr.last_use_cycle
                ]
                peaks.append(PressurePeak(
                    cycle=cycle,
                    live_count=pressure,
                    percentage=100.0 * pressure / self.scratch_size,
                    contributing_ranges=contributing
                ))

        # Sort by pressure descending, keep top 20
        peaks.sort(key=lambda p: p.live_count, reverse=True)
        self.result.peaks = peaks[:20]

    def _find_reuse_opportunities(self):
        """Find addresses that become dead and could be reused earlier."""
        opportunities = []

        # Build list of (death_cycle, address) pairs
        death_events = []
        for addr, lr in self.result.live_ranges.items():
            if lr.last_use_cycle < self.result.total_cycles - 1:
                death_events.append((lr.last_use_cycle, addr))

        death_events.sort()

        # For each death, see if there's a later allocation that could reuse it
        new_allocations = sorted([
            (lr.def_cycle, addr) for addr, lr in self.result.live_ranges.items()
        ])

        for death_cycle, dead_addr in death_events[:100]:  # Limit analysis
            # Find first allocation after this death
            for alloc_cycle, alloc_addr in new_allocations:
                if alloc_cycle > death_cycle and alloc_addr != dead_addr:
                    opportunities.append(ReuseOpportunity(
                        dead_address=dead_addr,
                        dead_after_cycle=death_cycle,
                        potential_reuse_cycle=alloc_cycle,
                        saves_addresses=1
                    ))
                    break

        # Sort by how much earlier reuse could happen
        opportunities.sort(key=lambda o: o.potential_reuse_cycle - o.dead_after_cycle, reverse=True)
        self.result.reuse_opportunities = opportunities[:50]

    def _compute_statistics(self):
        """Compute summary statistics."""
        self.result.total_addresses_used = len(self.result.live_ranges)

        for addr, lr in self.result.live_ranges.items():
            if lr.is_vector:
                self.result.vector_addresses += 1
            else:
                self.result.scalar_addresses += 1

            if lr.is_dead_after_def:
                self.result.dead_after_def += 1

            if lr.live_length > 50:
                self.result.long_lived += 1

            if self.is_constant.get(addr, False):
                self.result.constants += 1

    def _extract_reads_writes(self, slot: tuple, engine: str) -> Tuple[Set[int], Set[int], bool, bool]:
        """
        Extract scratch addresses read and written by an instruction slot.
        Returns (reads, writes, is_vector, is_constant).
        """
        reads = set()
        writes = set()
        is_vector = False
        is_constant = False

        if not slot or len(slot) == 0:
            return reads, writes, is_vector, is_constant

        op = slot[0]

        if engine == "alu":
            # ALU: (op, dest, src1, src2)
            if len(slot) >= 4:
                writes.add(slot[1])
                reads.add(slot[2])
                reads.add(slot[3])

        elif engine == "valu":
            is_vector = True
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
                is_vector = True
                # (vload, dest, addr) - loads VLEN elements
                if len(slot) >= 3:
                    for i in range(VLEN):
                        writes.add(slot[1] + i)
                    reads.add(slot[2])
            elif op == "const":
                is_constant = True
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
                is_vector = True
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
                is_vector = True
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

        return reads, writes, is_vector, is_constant


# ============== Output Formatting ==============

def generate_ascii_chart(pressure: List[int], width: int = 60, height: int = 15,
                         scratch_limit: int = SCRATCH_SIZE) -> str:
    """Generate ASCII art pressure chart."""
    if not pressure:
        return "No pressure data"

    lines = []
    max_pressure = max(pressure)

    # Create header
    lines.append(f"Register Pressure Over Time (max: {max_pressure}, limit: {scratch_limit})")
    lines.append("=" * width)

    # Bucket cycles into width bins
    cycles_per_bucket = max(1, len(pressure) // width)
    buckets = []
    for i in range(0, len(pressure), cycles_per_bucket):
        chunk = pressure[i:i + cycles_per_bucket]
        buckets.append(max(chunk) if chunk else 0)

    # Scale to height
    scale = max_pressure / height if max_pressure > 0 else 1

    # Generate chart (top to bottom)
    for row in range(height, 0, -1):
        threshold = row * scale
        line = ""
        for bucket_val in buckets[:width]:
            if bucket_val >= threshold:
                # Color code based on pressure
                pct = 100.0 * bucket_val / scratch_limit
                if pct >= 90:
                    line += "#"  # Critical
                elif pct >= 75:
                    line += "*"  # High
                elif pct >= 50:
                    line += "+"  # Medium
                else:
                    line += "."  # Low
            else:
                line += " "

        # Add y-axis label
        if row == height:
            lines.append(f"{max_pressure:5d} |{line}|")
        elif row == height // 2:
            lines.append(f"{int(max_pressure/2):5d} |{line}|")
        elif row == 1:
            lines.append(f"    0 |{line}|")
        else:
            lines.append(f"      |{line}|")

    # X-axis
    lines.append("      +" + "-" * len(buckets[:width]) + "+")
    lines.append(f"      0{' ' * (len(buckets[:width]) - 6)}cycles: {len(pressure)}")

    # Legend
    lines.append("")
    lines.append("Legend: # = >90%, * = 75-90%, + = 50-75%, . = <50%")

    return "\n".join(lines)


class PlainPrinter:
    """Plain text output without Rich."""

    def print_result(self, result: AnalysisResult, show_peaks: bool = False,
                     show_reuse: bool = False, show_chart: bool = False):
        print("=" * 70)
        print("REGISTER PRESSURE ANALYSIS")
        print("=" * 70)
        print()

        # Summary
        print("-" * 70)
        print("SUMMARY")
        print("-" * 70)
        print(f"Total Cycles:            {result.total_cycles:,}")
        print(f"Total Addresses Used:    {result.total_addresses_used:,}")
        print(f"Scratch Limit:           {result.scratch_limit:,}")
        print(f"Utilization:             {result.utilization_pct:.1f}%")
        print(f"Headroom:                {result.headroom:,} addresses")
        print()
        print(f"Max Simultaneous Live:   {result.max_simultaneous_live:,}")
        print(f"Peak Cycle:              {result.max_live_cycle:,}")
        print(f"Peak Pressure:           {result.peak_pressure_pct:.1f}%")
        print(f"Avg Live per Cycle:      {result.avg_live_per_cycle:.1f}")
        print()

        # Warnings
        if result.peak_pressure_pct >= 90:
            print("[CRITICAL] Peak pressure exceeds 90% of limit!")
            print("           Further unrolling/pipelining may be blocked.")
        elif result.peak_pressure_pct >= 75:
            print("[WARNING] Peak pressure exceeds 75% of limit.")
            print("          Consider register optimization before adding more unrolling.")
        elif result.utilization_pct < 25:
            print("[INFO] Low register pressure - room for aggressive optimization.")
        print()

        # Breakdown
        print("-" * 70)
        print("ADDRESS BREAKDOWN")
        print("-" * 70)
        print(f"Scalar Addresses:        {result.scalar_addresses:,}")
        print(f"Vector Addresses:        {result.vector_addresses:,}")
        print(f"Constants:               {result.constants:,}")
        print(f"Dead After Definition:   {result.dead_after_def:,}")
        print(f"Long-Lived (>50 cycles): {result.long_lived:,}")
        print()

        # Pressure histogram
        histogram = result._pressure_histogram()
        if histogram:
            print("-" * 70)
            print("PRESSURE DISTRIBUTION")
            print("-" * 70)
            for bucket, count in histogram.items():
                bar = "*" * (count * 40 // result.total_cycles) if result.total_cycles else ""
                print(f"{bucket:>10}: {count:6,} cycles {bar}")
            print()

        # Chart
        if show_chart and result.pressure_per_cycle:
            print("-" * 70)
            print("PRESSURE OVER TIME")
            print("-" * 70)
            print(generate_ascii_chart(result.pressure_per_cycle))
            print()

        # Peaks
        if show_peaks and result.peaks:
            print("-" * 70)
            print(f"TOP PRESSURE PEAKS ({len(result.peaks)})")
            print("-" * 70)
            for i, peak in enumerate(result.peaks[:10], 1):
                print(f"{i:2}. Cycle {peak.cycle:5}: {peak.live_count:4} live ({peak.percentage:.1f}%)")
            print()

        # Reuse opportunities
        if show_reuse and result.reuse_opportunities:
            print("-" * 70)
            print(f"REUSE OPPORTUNITIES ({len(result.reuse_opportunities)})")
            print("-" * 70)
            print("Address    Dead After    Could Reuse At    Gap")
            for opp in result.reuse_opportunities[:10]:
                gap = opp.potential_reuse_cycle - opp.dead_after_cycle
                print(f"  {opp.dead_address:5}    cycle {opp.dead_after_cycle:5}     cycle {opp.potential_reuse_cycle:5}      {gap:3} cycles")
            if len(result.reuse_opportunities) > 10:
                print(f"  ... and {len(result.reuse_opportunities) - 10} more opportunities")
            print()


class RichPrinter:
    """Rich colored output."""

    def __init__(self):
        self.console = Console()

    def print_result(self, result: AnalysisResult, show_peaks: bool = False,
                     show_reuse: bool = False, show_chart: bool = False):
        # Header
        self.console.print(Panel("REGISTER PRESSURE ANALYSIS", style="bold cyan", box=box.DOUBLE))

        # Status panel
        if result.peak_pressure_pct >= 90:
            status_panel = Panel(
                f"[bold red]CRITICAL[/bold red] - Peak pressure at {result.peak_pressure_pct:.1f}%\n"
                "Further unrolling/pipelining may be blocked!",
                style="red", box=box.ROUNDED
            )
        elif result.peak_pressure_pct >= 75:
            status_panel = Panel(
                f"[bold yellow]WARNING[/bold yellow] - Peak pressure at {result.peak_pressure_pct:.1f}%\n"
                "Consider register optimization before adding more unrolling.",
                style="yellow", box=box.ROUNDED
            )
        else:
            status_panel = Panel(
                f"[bold green]HEALTHY[/bold green] - Peak pressure at {result.peak_pressure_pct:.1f}%\n"
                f"Headroom: {result.headroom:,} addresses available",
                style="green", box=box.ROUNDED
            )
        self.console.print(status_panel)

        # Summary table
        summary_table = Table(title="Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right", style="white")
        summary_table.add_column("Status", style="white")

        summary_table.add_row("Total Cycles", f"{result.total_cycles:,}", "")
        summary_table.add_row("Addresses Used", f"{result.total_addresses_used:,}",
                            f"{result.utilization_pct:.1f}% of limit")
        summary_table.add_row("Scratch Limit", f"{result.scratch_limit:,}", "")
        summary_table.add_row("Headroom", f"{result.headroom:,}",
                            "[green]plenty[/green]" if result.headroom > 500 else "[yellow]limited[/yellow]")

        pres_style = "red" if result.peak_pressure_pct >= 90 else "yellow" if result.peak_pressure_pct >= 75 else "green"
        summary_table.add_row("Max Simultaneous Live", f"{result.max_simultaneous_live:,}",
                            f"[{pres_style}]{result.peak_pressure_pct:.1f}%[/{pres_style}]")
        summary_table.add_row("Peak at Cycle", f"{result.max_live_cycle:,}", "")
        summary_table.add_row("Avg Live/Cycle", f"{result.avg_live_per_cycle:.1f}", "")

        self.console.print(summary_table)

        # Breakdown table
        breakdown_table = Table(title="Address Breakdown", box=box.ROUNDED)
        breakdown_table.add_column("Category", style="cyan")
        breakdown_table.add_column("Count", justify="right", style="white")
        breakdown_table.add_column("Notes", style="dim")

        breakdown_table.add_row("Scalar", f"{result.scalar_addresses:,}", "Single values")
        breakdown_table.add_row("Vector", f"{result.vector_addresses:,}", f"VLEN={VLEN} groups")
        breakdown_table.add_row("Constants", f"{result.constants:,}", "Live entire program")
        breakdown_table.add_row("Dead After Def", f"{result.dead_after_def:,}", "Never read")
        breakdown_table.add_row("Long-Lived (>50)", f"{result.long_lived:,}", "May block reuse")

        self.console.print(breakdown_table)

        # Pressure histogram
        histogram = result._pressure_histogram()
        if histogram:
            hist_table = Table(title="Pressure Distribution", box=box.ROUNDED)
            hist_table.add_column("Range", style="cyan")
            hist_table.add_column("Cycles", justify="right")
            hist_table.add_column("", style="white")

            colors = {"0-10%": "green", "10-25%": "green", "25-50%": "yellow",
                     "50-75%": "yellow", "75-90%": "red", "90-100%": "red bold"}

            for bucket, count in histogram.items():
                bar_len = int(count * 30 / max(1, result.total_cycles))
                bar = "[" + colors.get(bucket, "white") + "]" + "*" * bar_len + "[/]"
                hist_table.add_row(bucket, f"{count:,}", bar)

            self.console.print(hist_table)

        # Chart
        if show_chart and result.pressure_per_cycle:
            self.console.print("\n[bold cyan]PRESSURE OVER TIME[/bold cyan]")
            chart = generate_ascii_chart(result.pressure_per_cycle)
            self.console.print(Panel(chart, box=box.ROUNDED))

        # Peaks
        if show_peaks and result.peaks:
            peaks_table = Table(title=f"Top Pressure Peaks ({len(result.peaks)})", box=box.ROUNDED)
            peaks_table.add_column("#", justify="right", style="dim")
            peaks_table.add_column("Cycle", justify="right", style="cyan")
            peaks_table.add_column("Live", justify="right", style="white")
            peaks_table.add_column("% of Limit", justify="right")

            for i, peak in enumerate(result.peaks[:10], 1):
                pct_style = "red" if peak.percentage >= 90 else "yellow" if peak.percentage >= 75 else "green"
                peaks_table.add_row(
                    str(i),
                    f"{peak.cycle:,}",
                    f"{peak.live_count:,}",
                    f"[{pct_style}]{peak.percentage:.1f}%[/{pct_style}]"
                )

            self.console.print(peaks_table)

        # Reuse opportunities
        if show_reuse and result.reuse_opportunities:
            reuse_table = Table(title=f"Reuse Opportunities ({len(result.reuse_opportunities)})", box=box.ROUNDED)
            reuse_table.add_column("Address", justify="right", style="cyan")
            reuse_table.add_column("Dead After", justify="right")
            reuse_table.add_column("Reuse At", justify="right")
            reuse_table.add_column("Gap", justify="right", style="green")

            for opp in result.reuse_opportunities[:10]:
                gap = opp.potential_reuse_cycle - opp.dead_after_cycle
                reuse_table.add_row(
                    str(opp.dead_address),
                    f"cycle {opp.dead_after_cycle}",
                    f"cycle {opp.potential_reuse_cycle}",
                    f"{gap} cycles"
                )

            if len(result.reuse_opportunities) > 10:
                self.console.print(f"  [dim]... and {len(result.reuse_opportunities) - 10} more[/dim]")

            self.console.print(reuse_table)


def get_printer():
    """Get the appropriate printer based on Rich availability."""
    if RICH_AVAILABLE:
        return RichPrinter()
    return PlainPrinter()


# ============== Main Entry Points ==============

def load_kernel_from_file(filepath: str) -> List[dict]:
    """Load kernel instructions from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_kernel_from_builder():
    """Load kernel from perf_takehome.py KernelBuilder."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from perf_takehome import KernelBuilder

    # Standard test params
    forest_height = 10
    n_nodes = 2 ** (forest_height + 1) - 1
    batch_size = 256
    rounds = 16

    kb = KernelBuilder()
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
    return kb.instrs


def analyze_register_pressure(instructions: List[dict],
                              scratch_size: int = SCRATCH_SIZE) -> AnalysisResult:
    """
    Main analysis function - can be imported and used by other tools.

    Args:
        instructions: List of instruction bundles
        scratch_size: Maximum scratch memory size

    Returns:
        AnalysisResult with complete pressure analysis
    """
    analyzer = RegisterPressureAnalyzer(instructions, scratch_size)
    return analyzer.analyze()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze register pressure in VLIW SIMD kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python register_pressure.py                    # Full analysis
    python register_pressure.py --peaks            # Show pressure peaks
    python register_pressure.py --reuse            # Show reuse opportunities
    python register_pressure.py --visualize        # Show pressure chart
    python register_pressure.py --all              # Show everything
    python register_pressure.py --json             # Output as JSON
        """
    )
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--peaks", "-p", action="store_true",
                        help="Show pressure peaks")
    parser.add_argument("--reuse", "-r", action="store_true",
                        help="Show reuse opportunities")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Show pressure chart over time")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Show all analysis (peaks, reuse, chart)")
    parser.add_argument("--kernel", "-k", metavar="FILE",
                        help="Load kernel from JSON file")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")

    args = parser.parse_args()

    # Load kernel
    print("Loading kernel...", file=sys.stderr)
    if args.kernel:
        instructions = load_kernel_from_file(args.kernel)
    else:
        instructions = load_kernel_from_builder()

    print(f"Analyzing {len(instructions)} cycles...", file=sys.stderr)

    # Analyze
    result = analyze_register_pressure(instructions)

    # Output
    if args.json:
        print(result.to_json())
    else:
        show_peaks = args.peaks or args.all
        show_reuse = args.reuse or args.all
        show_chart = args.visualize or args.all

        printer = PlainPrinter() if args.no_color else get_printer()
        printer.print_result(result, show_peaks=show_peaks, show_reuse=show_reuse,
                           show_chart=show_chart)

    # Exit code based on pressure
    if result.peak_pressure_pct >= 95:
        sys.exit(2)  # Critical
    elif result.peak_pressure_pct >= 90:
        sys.exit(1)  # Warning
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
