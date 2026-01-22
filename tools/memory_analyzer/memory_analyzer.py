#!/usr/bin/env python3
"""
Memory Access Pattern Analyzer for VLIW SIMD Kernel

Analyzes load/store patterns to identify vectorization opportunities and blockers.
Reveals WHY vload/vstore can't be used in certain places.

Key Features:
1. Access pattern detection (sequential, strided, random/scattered)
2. Stride analysis (identify consistent strides for potential vectorization)
3. Vectorization blocker identification (why vload/vstore isn't possible)
4. Address source tracking (where do load/store addresses come from)
5. Memory region analysis (which regions are accessed how)

Usage:
    python tools/memory_analyzer/memory_analyzer.py [--json] [--verbose] [--top N]
    python tools/memory_analyzer/memory_analyzer.py --help

Example Output:
    Sequential Loads: 45 (vectorizable with vload)
    Strided Loads: 12 (stride=2, could use gather)
    Scattered Loads: 156 (not vectorizable - tree lookups)
"""

import sys
import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from enum import Enum

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import SLOT_LIMITS, VLEN

# Try to import Rich for better formatting (optional)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class AccessPattern(Enum):
    """Classification of memory access patterns."""
    SEQUENTIAL = "sequential"       # Consecutive addresses (addr, addr+1, addr+2...)
    STRIDED = "strided"             # Regular stride (addr, addr+s, addr+2s...)
    SCATTERED = "scattered"         # Irregular/computed addresses (tree lookups)
    BROADCAST = "broadcast"         # Same address loaded multiple times
    UNKNOWN = "unknown"             # Cannot determine pattern


class AddressSource(Enum):
    """Source of address computation."""
    CONSTANT = "constant"           # Fixed address (header loads)
    LINEAR = "linear"               # Base + offset * i (array traversal)
    COMPUTED = "computed"           # Result of ALU operations (tree index)
    INDIRECT = "indirect"           # Loaded from another address
    UNKNOWN = "unknown"             # Cannot determine source


@dataclass
class MemoryAccess:
    """Single memory access (load or store)."""
    cycle: int
    op_type: str                    # "load", "vload", "store", "vstore"
    dest_or_src: int                # Scratch address (dest for load, src for store)
    addr_reg: int                   # Scratch address holding memory address
    is_vector: bool                 # True for vload/vstore
    width: int                      # Number of elements (1 for scalar, VLEN for vector)

    @property
    def is_load(self) -> bool:
        return self.op_type in ("load", "vload")


@dataclass
class AccessGroup:
    """Group of related memory accesses."""
    pattern: AccessPattern
    accesses: List[MemoryAccess]
    addr_source: AddressSource
    stride: Optional[int] = None    # For strided patterns
    base_addr_reg: Optional[int] = None
    vectorization_blocker: Optional[str] = None

    @property
    def count(self) -> int:
        return len(self.accesses)

    @property
    def total_elements(self) -> int:
        return sum(a.width for a in self.accesses)


@dataclass
class VectorizationBlocker:
    """Reason why a group of accesses can't be vectorized."""
    reason: str
    description: str
    affected_cycles: List[int]
    potential_fix: Optional[str] = None


@dataclass
class MemoryRegion:
    """Analysis of a memory region."""
    name: str
    base_addr_hint: Optional[str]   # e.g., "inp_values_p"
    load_count: int = 0
    store_count: int = 0
    vectorized_load_count: int = 0
    vectorized_store_count: int = 0
    patterns: Dict[AccessPattern, int] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete memory access pattern analysis."""
    total_loads: int
    total_stores: int
    total_vloads: int
    total_vstores: int

    load_patterns: Dict[AccessPattern, int]
    store_patterns: Dict[AccessPattern, int]

    access_groups: List[AccessGroup]
    vectorization_blockers: List[VectorizationBlocker]

    # Summary metrics
    vectorization_rate: float       # % of accesses that use vector ops
    sequential_opportunity: int     # Scalar sequential loads that could be vloads
    scattered_loads: int            # Loads with computed addresses (tree lookups)

    # Detailed breakdowns
    addr_source_breakdown: Dict[AddressSource, int]
    stride_histogram: Dict[int, int]

    def to_dict(self) -> dict:
        return {
            "total_loads": self.total_loads,
            "total_stores": self.total_stores,
            "total_vloads": self.total_vloads,
            "total_vstores": self.total_vstores,
            "load_patterns": {k.value: v for k, v in self.load_patterns.items()},
            "store_patterns": {k.value: v for k, v in self.store_patterns.items()},
            "vectorization_rate": round(self.vectorization_rate, 2),
            "sequential_opportunity": self.sequential_opportunity,
            "scattered_loads": self.scattered_loads,
            "addr_source_breakdown": {k.value: v for k, v in self.addr_source_breakdown.items()},
            "stride_histogram": self.stride_histogram,
            "vectorization_blockers": [
                {
                    "reason": b.reason,
                    "description": b.description,
                    "affected_cycles": len(b.affected_cycles),
                    "potential_fix": b.potential_fix
                }
                for b in self.vectorization_blockers
            ]
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def extract_memory_accesses(instructions: List[dict]) -> List[MemoryAccess]:
    """Extract all memory access operations from instruction stream."""
    accesses = []

    for cycle_num, instr in enumerate(instructions):
        # Process load operations
        if "load" in instr:
            for slot in instr["load"]:
                if not slot:
                    continue
                op = slot[0]
                if op == "load":
                    # (load, dest, addr_reg)
                    accesses.append(MemoryAccess(
                        cycle=cycle_num,
                        op_type="load",
                        dest_or_src=slot[1],
                        addr_reg=slot[2],
                        is_vector=False,
                        width=1
                    ))
                elif op == "vload":
                    # (vload, dest, addr_reg)
                    accesses.append(MemoryAccess(
                        cycle=cycle_num,
                        op_type="vload",
                        dest_or_src=slot[1],
                        addr_reg=slot[2],
                        is_vector=True,
                        width=VLEN
                    ))
                elif op == "load_offset":
                    # (load_offset, dest, addr, offset)
                    accesses.append(MemoryAccess(
                        cycle=cycle_num,
                        op_type="load",
                        dest_or_src=slot[1] + slot[3],
                        addr_reg=slot[2] + slot[3],
                        is_vector=False,
                        width=1
                    ))

        # Process store operations
        if "store" in instr:
            for slot in instr["store"]:
                if not slot:
                    continue
                op = slot[0]
                if op == "store":
                    # (store, addr_reg, src)
                    accesses.append(MemoryAccess(
                        cycle=cycle_num,
                        op_type="store",
                        dest_or_src=slot[2],
                        addr_reg=slot[1],
                        is_vector=False,
                        width=1
                    ))
                elif op == "vstore":
                    # (vstore, addr_reg, src)
                    accesses.append(MemoryAccess(
                        cycle=cycle_num,
                        op_type="vstore",
                        dest_or_src=slot[2],
                        addr_reg=slot[1],
                        is_vector=True,
                        width=VLEN
                    ))

    return accesses


def analyze_address_source(
    instructions: List[dict],
    addr_reg: int,
    access_cycle: int,
    debug_info: Optional[Any] = None
) -> Tuple[AddressSource, Optional[str]]:
    """
    Determine how an address register got its value.

    Traces back through instructions to find the origin of the address.
    Returns (source_type, description).
    """
    # Look backwards to find what wrote to addr_reg
    for cycle in range(access_cycle - 1, -1, -1):
        instr = instructions[cycle]

        # Check load operations (const sets addresses directly)
        if "load" in instr:
            for slot in instr["load"]:
                if not slot:
                    continue
                if slot[0] == "const" and slot[1] == addr_reg:
                    return AddressSource.CONSTANT, f"const={slot[2]}"
                if slot[0] == "load" and slot[1] == addr_reg:
                    return AddressSource.INDIRECT, f"loaded from scratch[{slot[2]}]"
                if slot[0] == "vload":
                    # Check if any element of vector includes our addr_reg
                    if slot[1] <= addr_reg < slot[1] + VLEN:
                        return AddressSource.INDIRECT, f"vload element"

        # Check ALU operations
        if "alu" in instr:
            for slot in instr["alu"]:
                if not slot:
                    continue
                if len(slot) >= 4 and slot[1] == addr_reg:
                    op = slot[0]
                    if op == "+":
                        return AddressSource.COMPUTED, f"computed: [{slot[2]}] + [{slot[3]}]"
                    else:
                        return AddressSource.COMPUTED, f"computed: {op}"

        # Check VALU operations
        if "valu" in instr:
            for slot in instr["valu"]:
                if not slot:
                    continue
                if slot[0] == "vbroadcast" and slot[1] <= addr_reg < slot[1] + VLEN:
                    return AddressSource.LINEAR, f"broadcast from [{slot[2]}]"
                if len(slot) >= 4 and slot[1] <= addr_reg < slot[1] + VLEN:
                    return AddressSource.COMPUTED, f"vector computed"

        # Check flow operations
        if "flow" in instr:
            for slot in instr["flow"]:
                if not slot:
                    continue
                if slot[0] == "select" and slot[1] == addr_reg:
                    return AddressSource.COMPUTED, "conditional select"
                if slot[0] == "vselect" and slot[1] <= addr_reg < slot[1] + VLEN:
                    return AddressSource.COMPUTED, "vector conditional select"

    return AddressSource.UNKNOWN, None


def find_consecutive_accesses(accesses: List[MemoryAccess]) -> List[AccessGroup]:
    """
    Find groups of consecutive accesses that share patterns.

    Groups accesses by:
    1. Consecutive cycles using same address register
    2. Scalar loads that could potentially be vectorized
    """
    groups = []

    # Group scalar loads by address register (detect sequential patterns)
    loads = [a for a in accesses if a.is_load and not a.is_vector]

    # Sort by cycle
    loads_by_cycle = sorted(loads, key=lambda a: a.cycle)

    # Find consecutive scalar loads with related address registers
    i = 0
    while i < len(loads_by_cycle):
        group_loads = [loads_by_cycle[i]]
        j = i + 1

        # Look for loads in nearby cycles with consecutive dest registers
        base_dest = loads_by_cycle[i].dest_or_src
        while j < len(loads_by_cycle) and j - i < VLEN:
            curr = loads_by_cycle[j]
            expected_dest = base_dest + (j - i)

            # Check if destination registers are consecutive (pattern for loop unrolling)
            if curr.dest_or_src == expected_dest:
                group_loads.append(curr)
            elif curr.cycle - loads_by_cycle[j-1].cycle > 3:
                # Too far apart, break group
                break
            j += 1

        if len(group_loads) >= 2:
            # Determine if pattern is sequential or scattered
            dest_regs = [a.dest_or_src for a in group_loads]
            diffs = [dest_regs[k+1] - dest_regs[k] for k in range(len(dest_regs)-1)]

            if all(d == 1 for d in diffs):
                # Consecutive destination registers - might be vectorizable
                groups.append(AccessGroup(
                    pattern=AccessPattern.SEQUENTIAL,
                    accesses=group_loads,
                    addr_source=AddressSource.UNKNOWN,
                    vectorization_blocker="Scattered source addresses"
                ))
            elif len(set(diffs)) == 1:
                # Strided pattern
                groups.append(AccessGroup(
                    pattern=AccessPattern.STRIDED,
                    accesses=group_loads,
                    addr_source=AddressSource.UNKNOWN,
                    stride=diffs[0]
                ))

        i = max(j, i + 1)

    return groups


def identify_vectorization_blockers(
    instructions: List[dict],
    accesses: List[MemoryAccess],
    debug_info: Optional[Any] = None
) -> List[VectorizationBlocker]:
    """
    Identify specific reasons why vectorization isn't being used.

    Common blockers:
    1. Non-contiguous addresses (tree lookups with computed indices)
    2. Address not known at compile time
    3. Insufficient consecutive elements
    4. Data dependencies between accesses
    """
    blockers = []

    # Find scalar loads that could be vloads if addresses were sequential
    scalar_loads = [a for a in accesses if a.is_load and not a.is_vector]

    # Group by destination register ranges
    # Detect: loads to consecutive scratch addresses from scattered memory
    dest_groups = defaultdict(list)
    for load in scalar_loads:
        # Round to nearest VLEN boundary
        base = (load.dest_or_src // VLEN) * VLEN
        dest_groups[base].append(load)

    scattered_count = 0
    scattered_cycles = []

    for base, loads in dest_groups.items():
        if len(loads) >= VLEN // 2:
            # Multiple loads to same vector region
            # Check if source addresses come from computation (tree lookups)
            computed_addrs = 0
            for load in loads:
                source, _ = analyze_address_source(
                    instructions, load.addr_reg, load.cycle, debug_info
                )
                if source in (AddressSource.COMPUTED, AddressSource.INDIRECT):
                    computed_addrs += 1
                    scattered_cycles.append(load.cycle)

            if computed_addrs >= len(loads) // 2:
                scattered_count += len(loads)

    if scattered_count > 0:
        blockers.append(VectorizationBlocker(
            reason="Scattered tree lookups",
            description=f"{scattered_count} scalar loads use computed addresses from tree index calculation. "
                        "These are inherently non-contiguous (random access into tree values array).",
            affected_cycles=scattered_cycles[:20],
            potential_fix="Consider: 1) Prefetching tree nodes, 2) Batch tree walks, "
                          "3) Software managed cache for hot nodes"
        ))

    # Check for stores that could be vectorized
    scalar_stores = [a for a in accesses if not a.is_load and not a.is_vector]
    if len(scalar_stores) > len([a for a in accesses if not a.is_load and a.is_vector]) * VLEN:
        blockers.append(VectorizationBlocker(
            reason="Scalar stores with vectorized computation",
            description="Some store operations are scalar while corresponding computations are vectorized.",
            affected_cycles=[a.cycle for a in scalar_stores[:20]],
            potential_fix="Ensure output arrays are properly aligned and use vstore for contiguous outputs"
        ))

    return blockers


def analyze_stride_patterns(
    instructions: List[dict],
    accesses: List[MemoryAccess]
) -> Dict[int, int]:
    """
    Analyze stride patterns in memory accesses.

    Returns histogram of detected strides.
    """
    stride_histogram = defaultdict(int)

    # Sort loads by cycle
    loads = sorted([a for a in accesses if a.is_load], key=lambda a: a.cycle)

    # For sequential scalar loads, check destination register strides
    for i in range(len(loads) - 1):
        curr = loads[i]
        next_load = loads[i + 1]

        if not curr.is_vector and not next_load.is_vector:
            # Check cycle gap
            if next_load.cycle - curr.cycle <= 2:
                # Check destination stride
                dest_stride = next_load.dest_or_src - curr.dest_or_src
                if 1 <= dest_stride <= 16:
                    stride_histogram[dest_stride] += 1

    return dict(stride_histogram)


def analyze_memory_patterns(
    instructions: List[dict],
    debug_info: Optional[Any] = None
) -> AnalysisResult:
    """
    Main entry point: Analyze all memory access patterns in the instruction stream.
    """
    accesses = extract_memory_accesses(instructions)

    # Count totals
    total_loads = sum(1 for a in accesses if a.is_load and not a.is_vector)
    total_vloads = sum(1 for a in accesses if a.is_load and a.is_vector)
    total_stores = sum(1 for a in accesses if not a.is_load and not a.is_vector)
    total_vstores = sum(1 for a in accesses if not a.is_load and a.is_vector)

    # Analyze address sources
    addr_source_breakdown = defaultdict(int)
    scattered_loads = 0

    for access in accesses:
        source, _ = analyze_address_source(
            instructions, access.addr_reg, access.cycle, debug_info
        )
        addr_source_breakdown[source] += 1
        if source in (AddressSource.COMPUTED, AddressSource.INDIRECT) and access.is_load:
            scattered_loads += 1

    # Find access groups
    access_groups = find_consecutive_accesses(accesses)

    # Categorize patterns
    load_patterns = defaultdict(int)
    store_patterns = defaultdict(int)

    for access in accesses:
        source, _ = analyze_address_source(
            instructions, access.addr_reg, access.cycle, debug_info
        )

        if source == AddressSource.CONSTANT:
            pattern = AccessPattern.SEQUENTIAL
        elif source == AddressSource.LINEAR:
            pattern = AccessPattern.SEQUENTIAL
        elif source in (AddressSource.COMPUTED, AddressSource.INDIRECT):
            pattern = AccessPattern.SCATTERED
        else:
            pattern = AccessPattern.UNKNOWN

        if access.is_vector:
            pattern = AccessPattern.SEQUENTIAL  # vload/vstore are always sequential

        if access.is_load:
            load_patterns[pattern] += 1
        else:
            store_patterns[pattern] += 1

    # Identify vectorization blockers
    blockers = identify_vectorization_blockers(instructions, accesses, debug_info)

    # Analyze strides
    stride_histogram = analyze_stride_patterns(instructions, accesses)

    # Calculate metrics
    total_accesses = total_loads + total_vloads + total_stores + total_vstores
    vector_elements = (total_vloads + total_vstores) * VLEN
    scalar_elements = total_loads + total_stores
    all_elements = vector_elements + scalar_elements
    vectorization_rate = 100.0 * vector_elements / max(1, all_elements)

    # Sequential opportunity: scalar loads with linear/constant addresses
    sequential_opportunity = (
        load_patterns.get(AccessPattern.SEQUENTIAL, 0) +
        addr_source_breakdown.get(AddressSource.LINEAR, 0) +
        addr_source_breakdown.get(AddressSource.CONSTANT, 0)
    ) // 2  # Conservative estimate

    return AnalysisResult(
        total_loads=total_loads,
        total_stores=total_stores,
        total_vloads=total_vloads,
        total_vstores=total_vstores,
        load_patterns=dict(load_patterns),
        store_patterns=dict(store_patterns),
        access_groups=access_groups,
        vectorization_blockers=blockers,
        vectorization_rate=vectorization_rate,
        sequential_opportunity=sequential_opportunity,
        scattered_loads=scattered_loads,
        addr_source_breakdown=dict(addr_source_breakdown),
        stride_histogram=stride_histogram
    )


# ============== Output Formatting ==============

class PlainPrinter:
    """Plain text output without Rich."""

    def __init__(self):
        pass

    def print_header(self, text: str):
        print("=" * 70)
        print(text)
        print("=" * 70)

    def print_subheader(self, text: str):
        print()
        print("-" * 70)
        print(text)
        print("-" * 70)

    def print_results(self, result: AnalysisResult, verbose: bool = False):
        self.print_header("MEMORY ACCESS PATTERN ANALYSIS")
        print()

        # Summary table
        print(f"{'Metric':<35} {'Count':>10} {'Elements':>12}")
        print("-" * 60)
        print(f"{'Scalar Loads':<35} {result.total_loads:>10} {result.total_loads:>12}")
        print(f"{'Vector Loads (vload)':<35} {result.total_vloads:>10} {result.total_vloads * VLEN:>12}")
        print(f"{'Scalar Stores':<35} {result.total_stores:>10} {result.total_stores:>12}")
        print(f"{'Vector Stores (vstore)':<35} {result.total_vstores:>10} {result.total_vstores * VLEN:>12}")
        print("-" * 60)

        total_ops = result.total_loads + result.total_vloads + result.total_stores + result.total_vstores
        total_elements = (result.total_loads + result.total_stores +
                         (result.total_vloads + result.total_vstores) * VLEN)
        print(f"{'TOTAL':<35} {total_ops:>10} {total_elements:>12}")
        print()

        print(f"Vectorization Rate: {result.vectorization_rate:.1f}% (of elements)")
        print(f"Scattered Loads (tree lookups): {result.scattered_loads}")
        print(f"Sequential Opportunity: ~{result.sequential_opportunity} potential vloads")

        # Load patterns
        self.print_subheader("LOAD ACCESS PATTERNS")
        print(f"{'Pattern':<25} {'Count':>10} {'Description':<35}")
        print("-" * 70)
        for pattern, count in sorted(result.load_patterns.items(), key=lambda x: -x[1]):
            desc = {
                AccessPattern.SEQUENTIAL: "Contiguous addresses (vectorizable)",
                AccessPattern.STRIDED: "Regular stride (gather possible)",
                AccessPattern.SCATTERED: "Computed addresses (not vectorizable)",
                AccessPattern.BROADCAST: "Same address (use vbroadcast)",
                AccessPattern.UNKNOWN: "Unknown pattern",
            }.get(pattern, "")
            print(f"{pattern.value:<25} {count:>10} {desc:<35}")

        # Store patterns
        self.print_subheader("STORE ACCESS PATTERNS")
        print(f"{'Pattern':<25} {'Count':>10}")
        print("-" * 40)
        for pattern, count in sorted(result.store_patterns.items(), key=lambda x: -x[1]):
            print(f"{pattern.value:<25} {count:>10}")

        # Address source breakdown
        self.print_subheader("ADDRESS SOURCE BREAKDOWN")
        print(f"{'Source':<25} {'Count':>10} {'Implication':<35}")
        print("-" * 70)
        for source, count in sorted(result.addr_source_breakdown.items(), key=lambda x: -x[1]):
            impl = {
                AddressSource.CONSTANT: "Fixed offset (header/init)",
                AddressSource.LINEAR: "Array traversal (vectorizable)",
                AddressSource.COMPUTED: "ALU result (often tree index)",
                AddressSource.INDIRECT: "Pointer chase (not vectorizable)",
                AddressSource.UNKNOWN: "Cannot determine",
            }.get(source, "")
            print(f"{source.value:<25} {count:>10} {impl:<35}")

        # Stride histogram
        if result.stride_histogram:
            self.print_subheader("STRIDE HISTOGRAM (consecutive loads)")
            print(f"{'Stride':<15} {'Count':>10} {'Visualization':<45}")
            print("-" * 70)
            max_count = max(result.stride_histogram.values()) if result.stride_histogram else 1
            for stride, count in sorted(result.stride_histogram.items()):
                bar_len = int(40 * count / max_count)
                bar = "#" * bar_len
                vectorizable = "vectorizable!" if stride == 1 else ""
                print(f"stride={stride:<8} {count:>10} {bar} {vectorizable}")

        # Vectorization blockers
        if result.vectorization_blockers:
            self.print_subheader("VECTORIZATION BLOCKERS")
            for i, blocker in enumerate(result.vectorization_blockers, 1):
                print(f"\n{i}. {blocker.reason}")
                print(f"   {blocker.description}")
                print(f"   Affected: {len(blocker.affected_cycles)} cycles")
                if blocker.potential_fix:
                    print(f"   Potential fix: {blocker.potential_fix}")

        # Recommendations
        self.print_subheader("RECOMMENDATIONS")
        self._print_recommendations(result)

    def _print_recommendations(self, result: AnalysisResult):
        recs = []

        # Check for scattered loads
        if result.scattered_loads > 50:
            recs.append({
                "priority": 1,
                "title": "High scattered load count",
                "description": f"{result.scattered_loads} loads use computed addresses (tree lookups). "
                               "This is inherent to tree traversal and cannot be directly vectorized.",
                "suggestion": "Consider: software prefetching, batched tree walks, or caching hot nodes"
            })

        # Check vectorization rate
        if result.vectorization_rate < 50:
            recs.append({
                "priority": 2,
                "title": "Low vectorization rate",
                "description": f"Only {result.vectorization_rate:.1f}% of memory accesses use vector operations.",
                "suggestion": "Look for opportunities to use vload/vstore for array operations"
            })

        # Check for sequential opportunity
        if result.sequential_opportunity > 10:
            recs.append({
                "priority": 2,
                "title": "Sequential access opportunity",
                "description": f"~{result.sequential_opportunity} scalar loads could potentially be vloads "
                               "if data layout allows.",
                "suggestion": "Ensure input/output arrays are aligned and use vector loads where possible"
            })

        # Check for stride=1 pattern
        stride_1_count = result.stride_histogram.get(1, 0)
        if stride_1_count > 20:
            recs.append({
                "priority": 3,
                "title": "Unvectorized sequential loads",
                "description": f"{stride_1_count} consecutive load pairs have stride=1 (sequential) "
                               "but use scalar loads.",
                "suggestion": "These may be loop-unrolled tree lookups. Consider restructuring to use vload."
            })

        if not recs:
            print("No specific recommendations. Memory access patterns look reasonable.")
            return

        for rec in sorted(recs, key=lambda x: x["priority"]):
            priority_str = ["HIGH", "MEDIUM", "LOW"][min(rec["priority"]-1, 2)]
            print(f"\n[{priority_str}] {rec['title']}")
            print(f"    {rec['description']}")
            print(f"    Suggestion: {rec['suggestion']}")


class RichPrinter:
    """Rich-enabled colorful output."""

    def __init__(self):
        self.console = Console()

    def print_header(self, text: str):
        self.console.print(Panel(text, style="bold cyan", box=box.DOUBLE))

    def print_subheader(self, text: str):
        self.console.print(f"\n[bold yellow]{text}[/bold yellow]")
        self.console.print("-" * 60)

    def print_results(self, result: AnalysisResult, verbose: bool = False):
        self.print_header("MEMORY ACCESS PATTERN ANALYSIS")

        # Summary table
        table = Table(box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Elements", justify="right")

        table.add_row("Scalar Loads", str(result.total_loads), str(result.total_loads))
        table.add_row("Vector Loads (vload)", str(result.total_vloads),
                      f"[green]{result.total_vloads * VLEN}[/green]")
        table.add_row("Scalar Stores", str(result.total_stores), str(result.total_stores))
        table.add_row("Vector Stores (vstore)", str(result.total_vstores),
                      f"[green]{result.total_vstores * VLEN}[/green]")

        self.console.print(table)

        # Key metrics
        metrics_table = Table(show_header=False, box=box.SIMPLE)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        vec_color = "green" if result.vectorization_rate > 50 else "yellow" if result.vectorization_rate > 25 else "red"
        metrics_table.add_row("Vectorization Rate", f"[{vec_color}]{result.vectorization_rate:.1f}%[/{vec_color}]")

        scattered_color = "red" if result.scattered_loads > 100 else "yellow" if result.scattered_loads > 50 else "green"
        metrics_table.add_row("Scattered Loads (tree lookups)", f"[{scattered_color}]{result.scattered_loads}[/{scattered_color}]")
        metrics_table.add_row("Sequential Opportunity", f"~{result.sequential_opportunity} potential vloads")

        self.console.print(metrics_table)

        # Load patterns
        self.print_subheader("LOAD ACCESS PATTERNS")

        pattern_table = Table(box=box.ROUNDED)
        pattern_table.add_column("Pattern", style="cyan")
        pattern_table.add_column("Count", justify="right")
        pattern_table.add_column("Description")

        pattern_desc = {
            AccessPattern.SEQUENTIAL: ("green", "Contiguous addresses (vectorizable)"),
            AccessPattern.STRIDED: ("yellow", "Regular stride (gather possible)"),
            AccessPattern.SCATTERED: ("red", "Computed addresses (not vectorizable)"),
            AccessPattern.BROADCAST: ("cyan", "Same address (use vbroadcast)"),
            AccessPattern.UNKNOWN: ("dim", "Unknown pattern"),
        }

        for pattern, count in sorted(result.load_patterns.items(), key=lambda x: -x[1]):
            color, desc = pattern_desc.get(pattern, ("white", ""))
            pattern_table.add_row(f"[{color}]{pattern.value}[/{color}]", str(count), desc)

        self.console.print(pattern_table)

        # Address source breakdown
        self.print_subheader("ADDRESS SOURCE BREAKDOWN")

        source_table = Table(box=box.ROUNDED)
        source_table.add_column("Source", style="cyan")
        source_table.add_column("Count", justify="right")
        source_table.add_column("Implication")

        source_impl = {
            AddressSource.CONSTANT: "Fixed offset (header/init)",
            AddressSource.LINEAR: "Array traversal (vectorizable)",
            AddressSource.COMPUTED: "ALU result (often tree index)",
            AddressSource.INDIRECT: "Pointer chase (not vectorizable)",
            AddressSource.UNKNOWN: "Cannot determine",
        }

        for source, count in sorted(result.addr_source_breakdown.items(), key=lambda x: -x[1]):
            impl = source_impl.get(source, "")
            source_table.add_row(source.value, str(count), impl)

        self.console.print(source_table)

        # Stride histogram
        if result.stride_histogram:
            self.print_subheader("STRIDE HISTOGRAM (consecutive loads)")
            max_count = max(result.stride_histogram.values())
            for stride, count in sorted(result.stride_histogram.items()):
                bar_len = int(40 * count / max_count)
                color = "green" if stride == 1 else "yellow"
                bar = "[" + color + "]" + "#" * bar_len + "[/" + color + "]"
                vectorizable = "[green] vectorizable![/green]" if stride == 1 else ""
                self.console.print(f"stride={stride:<3} {count:>6} {bar} {vectorizable}")

        # Vectorization blockers
        if result.vectorization_blockers:
            self.print_subheader("VECTORIZATION BLOCKERS")
            for i, blocker in enumerate(result.vectorization_blockers, 1):
                self.console.print(Panel(
                    f"{blocker.description}\n\n"
                    f"Affected cycles: {len(blocker.affected_cycles)}\n"
                    f"[dim]Potential fix: {blocker.potential_fix or 'None'}[/dim]",
                    title=f"[bold red]{i}. {blocker.reason}[/bold red]",
                    border_style="red"
                ))

        # Recommendations
        self.print_subheader("RECOMMENDATIONS")
        self._print_recommendations(result)

    def _print_recommendations(self, result: AnalysisResult):
        recs = []

        if result.scattered_loads > 50:
            recs.append({
                "priority": 1,
                "title": "High scattered load count",
                "description": f"{result.scattered_loads} loads use computed addresses (tree lookups).",
                "suggestion": "Consider: software prefetching, batched tree walks, or caching hot nodes"
            })

        if result.vectorization_rate < 50:
            recs.append({
                "priority": 2,
                "title": "Low vectorization rate",
                "description": f"Only {result.vectorization_rate:.1f}% of memory accesses use vector operations.",
                "suggestion": "Look for opportunities to use vload/vstore for array operations"
            })

        if result.sequential_opportunity > 10:
            recs.append({
                "priority": 2,
                "title": "Sequential access opportunity",
                "description": f"~{result.sequential_opportunity} scalar loads could potentially be vloads.",
                "suggestion": "Ensure input/output arrays are aligned and use vector loads where possible"
            })

        stride_1_count = result.stride_histogram.get(1, 0)
        if stride_1_count > 20:
            recs.append({
                "priority": 3,
                "title": "Unvectorized sequential loads",
                "description": f"{stride_1_count} consecutive load pairs have stride=1.",
                "suggestion": "Consider restructuring to use vload for these sequential accesses."
            })

        if not recs:
            self.console.print("[dim]No specific recommendations. Memory access patterns look reasonable.[/dim]")
            return

        for rec in sorted(recs, key=lambda x: x["priority"]):
            priority = rec["priority"]
            if priority == 1:
                style = "bold red"
                priority_str = "HIGH"
            elif priority == 2:
                style = "bold yellow"
                priority_str = "MEDIUM"
            else:
                style = "bold blue"
                priority_str = "LOW"

            self.console.print(f"\n[{style}][{priority_str}][/{style}] {rec['title']}")
            self.console.print(f"    {rec['description']}")
            self.console.print(f"    [dim]Suggestion: {rec['suggestion']}[/dim]")


def get_printer(use_rich: bool = True):
    """Get the appropriate printer based on Rich availability."""
    if use_rich and RICH_AVAILABLE:
        return RichPrinter()
    return PlainPrinter()


# ============== Main ==============

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
        return kb.instrs, kb.debug_info()
    else:
        return kernel_builder.instrs, getattr(kernel_builder, 'debug_info', lambda: None)()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze memory access patterns for VLIW SIMD kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/memory_analyzer/memory_analyzer.py           # Full analysis
    python tools/memory_analyzer/memory_analyzer.py --json    # JSON output
    python tools/memory_analyzer/memory_analyzer.py --verbose # Detailed output
    python tools/memory_analyzer/memory_analyzer.py --no-color # Plain text
        """
    )
    parser.add_argument("--json", "-j", action="store_true", help="Output JSON instead of human-readable")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed access information")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--top", "-n", type=int, default=10, help="Number of items to show in lists")
    args = parser.parse_args()

    print("Loading kernel...", file=sys.stderr)
    instructions, debug_info = analyze_kernel()

    print(f"Analyzing {len(instructions)} cycles...", file=sys.stderr)
    result = analyze_memory_patterns(instructions, debug_info)

    if args.json:
        print(result.to_json())
    else:
        printer = get_printer(use_rich=not args.no_color)
        printer.print_results(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
