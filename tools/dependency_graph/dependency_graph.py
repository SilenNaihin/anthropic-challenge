#!/usr/bin/env python3
"""
Dependency Graph Builder for VLIW SIMD Kernel

Builds a complete dependency DAG (Directed Acyclic Graph) for instruction streams
to enable accurate critical path analysis and parallelism detection.

Unlike slot_analyzer's adjacent-cycle dependency checking, this tool tracks ALL
dependencies across the entire instruction stream to find:
1. TRUE critical path (not just adjacent dependencies)
2. Hot registers that cause the most blocking
3. Parallelism potential (average instructions that could run in parallel)
4. All RAW, WAW, WAR hazards

Algorithm: O(n + edges) using last-writer/last-reader maps instead of O(n^2) pairwise comparison.

Usage:
    python tools/dependency_graph.py [--json] [--dot output.dot] [--top N]
    python tools/dependency_graph.py --help

Example Output:
    Critical Path Length: 42 cycles
    Total Dependencies: 15000
    Hot Registers: [5 (500 deps), 8 (300 deps), ...]
    Parallelism Potential: 8.5x
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

from problem import SLOT_LIMITS, VLEN

# Try to import NetworkX for advanced graph algorithms (optional)
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Try to import Rich for better formatting (optional)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class HazardType(Enum):
    """Types of data hazards in instruction pipelines."""
    RAW = "Read-After-Write"   # True dependency: must wait for write to complete
    WAW = "Write-After-Write"  # Output dependency: must preserve write order
    WAR = "Write-After-Read"   # Anti-dependency: can often be eliminated by renaming


@dataclass
class Dependency:
    """Represents a single dependency edge in the DAG."""
    from_cycle: int
    to_cycle: int
    register: int  # scratch address
    hazard_type: HazardType

    def __hash__(self):
        return hash((self.from_cycle, self.to_cycle, self.register, self.hazard_type))


@dataclass
class CycleInfo:
    """Information about reads/writes in a single cycle."""
    cycle_num: int
    reads: Set[int] = field(default_factory=set)
    writes: Set[int] = field(default_factory=set)
    ops: List[Tuple[str, str]] = field(default_factory=list)  # [(engine, opcode), ...]


@dataclass
class RegisterStats:
    """Statistics for a single register (scratch address)."""
    addr: int
    write_count: int = 0
    read_count: int = 0
    dep_count: int = 0  # Number of dependencies caused
    name: Optional[str] = None  # Human-readable name if available

    @property
    def total_accesses(self) -> int:
        return self.write_count + self.read_count


@dataclass
class DependencyGraphResult:
    """Complete results of dependency graph analysis."""
    total_cycles: int
    total_dependencies: int
    raw_count: int
    waw_count: int
    war_count: int

    # Critical path analysis
    critical_path_length: int
    critical_path_cycles: List[int]

    # Hot registers
    hot_registers: List[RegisterStats]

    # Parallelism metrics
    parallelism_potential: float  # Average parallel instructions possible
    max_parallel_width: int       # Maximum concurrent independent instructions
    dependency_density: float     # deps / (cycles * cycles) - how interconnected

    # For advanced analysis
    strongly_connected_components: int = 0
    longest_chain_registers: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "total_cycles": self.total_cycles,
            "total_dependencies": self.total_dependencies,
            "dependency_breakdown": {
                "RAW": self.raw_count,
                "WAW": self.waw_count,
                "WAR": self.war_count
            },
            "critical_path_length": self.critical_path_length,
            "critical_path_cycles": self.critical_path_cycles[:50],  # Limit output
            "critical_path_cycles_count": len(self.critical_path_cycles),
            "hot_registers": [
                {
                    "addr": r.addr,
                    "deps": r.dep_count,
                    "writes": r.write_count,
                    "reads": r.read_count,
                    "name": r.name
                }
                for r in self.hot_registers[:20]  # Top 20
            ],
            "parallelism_potential": round(self.parallelism_potential, 2),
            "max_parallel_width": self.max_parallel_width,
            "dependency_density": round(self.dependency_density, 6),
            "theoretical_speedup": round(self.total_cycles / max(1, self.critical_path_length), 2),
            "wasted_cycles": self.total_cycles - self.critical_path_length
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def extract_reads_writes(slot: tuple, engine: str) -> Tuple[Set[int], Set[int]]:
    """
    Extract scratch addresses read and written by an instruction slot.
    Returns (reads, writes) sets.

    This is the same logic as slot_analyzer but kept here for independence.
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


def build_cycle_info(instructions: List[dict]) -> List[CycleInfo]:
    """Extract read/write sets for each cycle."""
    cycle_infos = []

    for cycle_num, instr in enumerate(instructions):
        info = CycleInfo(cycle_num=cycle_num)

        for engine, slots in instr.items():
            if engine == "debug":
                continue
            for slot in (slots or []):
                reads, writes = extract_reads_writes(slot, engine)
                info.reads.update(reads)
                info.writes.update(writes)
                if slot:
                    info.ops.append((engine, slot[0]))

        cycle_infos.append(info)

    return cycle_infos


def build_dependency_graph(
    instructions: List[dict],
    include_war: bool = True,
    include_waw: bool = True,
    debug_info: Optional[Any] = None
) -> Tuple[List[Dependency], List[CycleInfo], Dict[int, RegisterStats]]:
    """
    Build complete dependency DAG using last-writer/last-reader maps.

    Algorithm: O(n + edges) where n = number of cycles
    - For each cycle, track last writer and last readers for each register
    - RAW: if we read a register, add edge from its last writer
    - WAW: if we write a register, add edge from its last writer
    - WAR: if we write a register, add edges from all its last readers

    Returns:
        dependencies: List of all dependency edges
        cycle_infos: Per-cycle read/write information
        register_stats: Statistics for each accessed register
    """
    cycle_infos = build_cycle_info(instructions)

    # Maps: register -> last cycle that wrote/read it
    last_writer: Dict[int, int] = {}  # reg -> cycle
    last_readers: Dict[int, Set[int]] = defaultdict(set)  # reg -> {cycles}

    dependencies: List[Dependency] = []
    register_stats: Dict[int, RegisterStats] = defaultdict(lambda: RegisterStats(addr=0))

    # Build scratch_map for human-readable names if available
    scratch_names: Dict[int, str] = {}
    if debug_info and hasattr(debug_info, 'scratch_map'):
        for addr, (name, length) in debug_info.scratch_map.items():
            for i in range(length):
                scratch_names[addr + i] = f"{name}[{i}]" if length > 1 else name

    for cycle_info in cycle_infos:
        cycle = cycle_info.cycle_num

        # Process reads - create RAW dependencies
        for reg in cycle_info.reads:
            # Initialize register stats
            if register_stats[reg].addr == 0:
                register_stats[reg].addr = reg
                register_stats[reg].name = scratch_names.get(reg)
            register_stats[reg].read_count += 1

            # RAW: we read, someone wrote before
            if reg in last_writer:
                writer_cycle = last_writer[reg]
                dep = Dependency(
                    from_cycle=writer_cycle,
                    to_cycle=cycle,
                    register=reg,
                    hazard_type=HazardType.RAW
                )
                dependencies.append(dep)
                register_stats[reg].dep_count += 1

            # Track this read for WAR detection
            last_readers[reg].add(cycle)

        # Process writes
        for reg in cycle_info.writes:
            # Initialize register stats
            if register_stats[reg].addr == 0:
                register_stats[reg].addr = reg
                register_stats[reg].name = scratch_names.get(reg)
            register_stats[reg].write_count += 1

            # WAW: we write, someone wrote before
            if include_waw and reg in last_writer:
                writer_cycle = last_writer[reg]
                if writer_cycle != cycle:  # Not same cycle
                    dep = Dependency(
                        from_cycle=writer_cycle,
                        to_cycle=cycle,
                        register=reg,
                        hazard_type=HazardType.WAW
                    )
                    dependencies.append(dep)
                    register_stats[reg].dep_count += 1

            # WAR: we write, someone read before (but not in same cycle)
            if include_war and reg in last_readers:
                for reader_cycle in last_readers[reg]:
                    if reader_cycle != cycle and reader_cycle < cycle:
                        dep = Dependency(
                            from_cycle=reader_cycle,
                            to_cycle=cycle,
                            register=reg,
                            hazard_type=HazardType.WAR
                        )
                        dependencies.append(dep)
                        register_stats[reg].dep_count += 1

            # Update last writer (clears old readers for this reg)
            last_writer[reg] = cycle
            last_readers[reg] = set()  # Reset readers after a write

    return dependencies, cycle_infos, dict(register_stats)


def compute_critical_path(
    n_cycles: int,
    dependencies: List[Dependency],
    only_raw: bool = True
) -> Tuple[int, List[int]]:
    """
    Compute the critical path (longest path) through the dependency DAG.

    Uses dynamic programming on topologically sorted nodes (cycles are already
    in topological order since dependencies only go forward).

    Args:
        n_cycles: Total number of cycles
        dependencies: List of dependencies
        only_raw: If True, only consider RAW dependencies (true dependencies)
                  WAW and WAR can often be eliminated via register renaming

    Returns:
        (critical_path_length, critical_path_cycles)
    """
    if n_cycles == 0:
        return 0, []

    # Filter dependencies if only considering RAW
    if only_raw:
        deps = [d for d in dependencies if d.hazard_type == HazardType.RAW]
    else:
        deps = dependencies

    # Build adjacency list
    adj: Dict[int, List[int]] = defaultdict(list)
    for dep in deps:
        adj[dep.from_cycle].append(dep.to_cycle)

    # DP: dist[i] = longest path ending at cycle i
    dist = [1] * n_cycles  # Each cycle takes at least 1
    pred = [-1] * n_cycles  # Predecessor for path reconstruction

    # Process in topological order (0 to n-1)
    for dep in deps:
        from_c, to_c = dep.from_cycle, dep.to_cycle
        if dist[from_c] + 1 > dist[to_c]:
            dist[to_c] = dist[from_c] + 1
            pred[to_c] = from_c

    # Find the end of critical path
    critical_path_length = max(dist)
    end_cycle = dist.index(critical_path_length)

    # Reconstruct path
    path = []
    curr = end_cycle
    while curr != -1:
        path.append(curr)
        curr = pred[curr]
    path.reverse()

    return critical_path_length, path


def compute_parallelism_metrics(
    n_cycles: int,
    dependencies: List[Dependency]
) -> Tuple[float, int]:
    """
    Compute parallelism potential.

    Parallelism potential = n_cycles / critical_path_length
    This represents how many instructions could theoretically run in parallel
    on average if we had unlimited execution units.

    Also computes the width of the widest "level" in the DAG.

    Returns:
        (parallelism_potential, max_parallel_width)
    """
    if n_cycles == 0:
        return 0.0, 0

    # Compute level (earliest possible time) for each cycle
    # Using only RAW dependencies as they're the true constraints
    raw_deps = [d for d in dependencies if d.hazard_type == HazardType.RAW]

    level = [0] * n_cycles  # Earliest time each instruction can start

    for dep in raw_deps:
        from_c, to_c = dep.from_cycle, dep.to_cycle
        level[to_c] = max(level[to_c], level[from_c] + 1)

    # Critical path is max level + 1
    critical_path = max(level) + 1 if level else 1

    # Count instructions at each level
    level_counts = defaultdict(int)
    for l in level:
        level_counts[l] += 1

    max_width = max(level_counts.values()) if level_counts else 1
    parallelism = n_cycles / critical_path

    return parallelism, max_width


def find_hot_registers(
    register_stats: Dict[int, RegisterStats],
    top_n: int = 20
) -> List[RegisterStats]:
    """Find registers that cause the most dependencies."""
    sorted_regs = sorted(
        register_stats.values(),
        key=lambda r: r.dep_count,
        reverse=True
    )
    return sorted_regs[:top_n]


def analyze_dependency_graph(
    instructions: List[dict],
    debug_info: Optional[Any] = None,
    include_war: bool = False,  # WAR usually can be eliminated
    include_waw: bool = False,  # WAW usually can be eliminated
) -> DependencyGraphResult:
    """
    Main entry point: build and analyze the complete dependency graph.

    Args:
        instructions: List of instruction bundles
        debug_info: Optional debug info with scratch_map for names
        include_war: Include Write-After-Read hazards (anti-dependencies)
        include_waw: Include Write-After-Write hazards (output dependencies)

    Returns:
        DependencyGraphResult with all analysis
    """
    n_cycles = len(instructions)

    if n_cycles == 0:
        return DependencyGraphResult(
            total_cycles=0,
            total_dependencies=0,
            raw_count=0, waw_count=0, war_count=0,
            critical_path_length=0,
            critical_path_cycles=[],
            hot_registers=[],
            parallelism_potential=0.0,
            max_parallel_width=0,
            dependency_density=0.0
        )

    # Build the graph
    dependencies, cycle_infos, register_stats = build_dependency_graph(
        instructions,
        include_war=include_war,
        include_waw=include_waw,
        debug_info=debug_info
    )

    # Count by type
    raw_count = sum(1 for d in dependencies if d.hazard_type == HazardType.RAW)
    waw_count = sum(1 for d in dependencies if d.hazard_type == HazardType.WAW)
    war_count = sum(1 for d in dependencies if d.hazard_type == HazardType.WAR)

    # Critical path (only RAW - true dependencies)
    critical_path_length, critical_path_cycles = compute_critical_path(
        n_cycles, dependencies, only_raw=True
    )

    # Parallelism metrics
    parallelism_potential, max_parallel_width = compute_parallelism_metrics(
        n_cycles, dependencies
    )

    # Hot registers
    hot_registers = find_hot_registers(register_stats)

    # Dependency density
    max_deps = n_cycles * (n_cycles - 1) // 2  # Maximum possible edges in DAG
    density = len(dependencies) / max_deps if max_deps > 0 else 0.0

    return DependencyGraphResult(
        total_cycles=n_cycles,
        total_dependencies=len(dependencies),
        raw_count=raw_count,
        waw_count=waw_count,
        war_count=war_count,
        critical_path_length=critical_path_length,
        critical_path_cycles=critical_path_cycles,
        hot_registers=hot_registers,
        parallelism_potential=parallelism_potential,
        max_parallel_width=max_parallel_width,
        dependency_density=density
    )


def export_dot(
    instructions: List[dict],
    output_path: str,
    max_nodes: int = 500,
    debug_info: Optional[Any] = None
) -> None:
    """
    Export dependency graph to DOT format for visualization with Graphviz.

    Usage: dot -Tpng output.dot -o graph.png
    Or use online viewer: https://dreampuf.github.io/GraphvizOnline/
    """
    dependencies, cycle_infos, _ = build_dependency_graph(
        instructions,
        include_war=False,
        include_waw=False,
        debug_info=debug_info
    )

    # Only include RAW dependencies for cleaner visualization
    raw_deps = [d for d in dependencies if d.hazard_type == HazardType.RAW]

    # Find cycles on critical path for highlighting
    _, critical_path = compute_critical_path(len(instructions), dependencies)
    critical_set = set(critical_path)

    with open(output_path, 'w') as f:
        f.write("digraph DependencyGraph {\n")
        f.write("  rankdir=TB;\n")
        f.write("  node [shape=box, fontsize=10];\n")
        f.write("  edge [fontsize=8];\n")
        f.write("\n")

        # Only include first max_nodes cycles
        included_cycles = set(range(min(max_nodes, len(instructions))))

        # Add nodes
        for info in cycle_infos[:max_nodes]:
            cycle = info.cycle_num
            ops_str = ", ".join(f"{e}:{o}" for e, o in info.ops[:3])
            if len(info.ops) > 3:
                ops_str += "..."

            # Highlight critical path nodes
            style = 'style="filled", fillcolor="red"' if cycle in critical_set else ""
            f.write(f'  c{cycle} [label="Cycle {cycle}\\n{ops_str}" {style}];\n')

        f.write("\n")

        # Add edges (only between included cycles)
        edge_count = 0
        for dep in raw_deps:
            if dep.from_cycle in included_cycles and dep.to_cycle in included_cycles:
                is_critical = dep.from_cycle in critical_set and dep.to_cycle in critical_set
                color = 'color="red", penwidth=2' if is_critical else ""
                f.write(f'  c{dep.from_cycle} -> c{dep.to_cycle} [label="r{dep.register}" {color}];\n')
                edge_count += 1
                if edge_count > 2000:  # Limit edges for readability
                    break

        f.write("}\n")

    print(f"Exported DOT graph to {output_path} ({min(max_nodes, len(instructions))} nodes, {edge_count} edges)")


def build_networkx_graph(instructions: List[dict]) -> 'nx.DiGraph':
    """
    Build a NetworkX DiGraph for advanced analysis.
    Requires networkx to be installed.
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX is required for this feature. Install with: pip install networkx")

    dependencies, cycle_infos, register_stats = build_dependency_graph(
        instructions, include_war=False, include_waw=False
    )

    G = nx.DiGraph()

    # Add all cycles as nodes
    for info in cycle_infos:
        G.add_node(info.cycle_num,
                   reads=list(info.reads),
                   writes=list(info.writes),
                   ops=info.ops)

    # Add RAW dependency edges
    for dep in dependencies:
        if dep.hazard_type == HazardType.RAW:
            G.add_edge(dep.from_cycle, dep.to_cycle, register=dep.register)

    return G


# ============== Output Formatting ==============

def print_results_plain(result: DependencyGraphResult, top_n: int = 10):
    """Plain text output."""
    print("=" * 70)
    print("DEPENDENCY GRAPH ANALYSIS")
    print("=" * 70)
    print()

    print(f"Total Cycles:           {result.total_cycles:,}")
    print(f"Total Dependencies:     {result.total_dependencies:,}")
    print(f"  - RAW (true deps):    {result.raw_count:,}")
    print(f"  - WAW (output deps):  {result.waw_count:,}")
    print(f"  - WAR (anti deps):    {result.war_count:,}")
    print()

    print("-" * 70)
    print("CRITICAL PATH ANALYSIS")
    print("-" * 70)
    print(f"Critical Path Length:   {result.critical_path_length} cycles")
    print(f"Current Total Cycles:   {result.total_cycles}")

    if result.critical_path_length > 0:
        wasted = result.total_cycles - result.critical_path_length
        speedup = result.total_cycles / result.critical_path_length
        print(f"Wasted Cycles:          {wasted} ({100*wasted/result.total_cycles:.1f}%)")
        print(f"Theoretical Speedup:    {speedup:.2f}x")

    if result.critical_path_cycles:
        path_preview = result.critical_path_cycles[:10]
        path_str = " -> ".join(str(c) for c in path_preview)
        if len(result.critical_path_cycles) > 10:
            path_str += f" -> ... ({len(result.critical_path_cycles)} total)"
        print(f"Critical Path:          {path_str}")
    print()

    print("-" * 70)
    print("PARALLELISM METRICS")
    print("-" * 70)
    print(f"Parallelism Potential:  {result.parallelism_potential:.2f}x")
    print(f"Max Parallel Width:     {result.max_parallel_width}")
    print(f"Dependency Density:     {result.dependency_density:.6f}")
    print()

    print("-" * 70)
    print(f"TOP {top_n} HOT REGISTERS (most dependencies)")
    print("-" * 70)
    print(f"{'Addr':<8} {'Name':<20} {'Deps':>10} {'Writes':>10} {'Reads':>10}")
    print("-" * 70)

    for reg in result.hot_registers[:top_n]:
        name = reg.name or "-"
        print(f"{reg.addr:<8} {name:<20} {reg.dep_count:>10} {reg.write_count:>10} {reg.read_count:>10}")
    print()


def print_results_rich(result: DependencyGraphResult, top_n: int = 10):
    """Rich colored output."""
    console = Console()

    console.print(Panel("DEPENDENCY GRAPH ANALYSIS", style="bold cyan", box=box.DOUBLE))

    # Summary table
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Cycles", f"{result.total_cycles:,}")
    table.add_row("Total Dependencies", f"{result.total_dependencies:,}")
    table.add_row("  RAW (true deps)", f"{result.raw_count:,}")
    table.add_row("  WAW (output deps)", f"{result.waw_count:,}")
    table.add_row("  WAR (anti deps)", f"{result.war_count:,}")

    console.print(table)

    # Critical path
    console.print("\n[bold yellow]CRITICAL PATH ANALYSIS[/bold yellow]")
    console.print("-" * 60)

    cp_table = Table(show_header=False, box=box.SIMPLE)
    cp_table.add_column("Metric", style="cyan")
    cp_table.add_column("Value", style="green")

    cp_table.add_row("Critical Path Length", f"{result.critical_path_length} cycles")
    cp_table.add_row("Current Total Cycles", f"{result.total_cycles}")

    if result.critical_path_length > 0:
        wasted = result.total_cycles - result.critical_path_length
        speedup = result.total_cycles / result.critical_path_length
        wasted_color = "green" if wasted < result.total_cycles * 0.2 else "yellow" if wasted < result.total_cycles * 0.5 else "red"
        cp_table.add_row("Wasted Cycles", f"[{wasted_color}]{wasted} ({100*wasted/result.total_cycles:.1f}%)[/{wasted_color}]")
        cp_table.add_row("Theoretical Speedup", f"[bold magenta]{speedup:.2f}x[/bold magenta]")

    console.print(cp_table)

    # Parallelism
    console.print("\n[bold yellow]PARALLELISM METRICS[/bold yellow]")
    console.print("-" * 60)

    par_table = Table(show_header=False, box=box.SIMPLE)
    par_table.add_column("Metric", style="cyan")
    par_table.add_column("Value", style="green")

    par_table.add_row("Parallelism Potential", f"{result.parallelism_potential:.2f}x")
    par_table.add_row("Max Parallel Width", f"{result.max_parallel_width}")
    par_table.add_row("Dependency Density", f"{result.dependency_density:.6f}")

    console.print(par_table)

    # Hot registers
    console.print(f"\n[bold yellow]TOP {top_n} HOT REGISTERS[/bold yellow]")
    console.print("-" * 60)

    reg_table = Table(box=box.ROUNDED)
    reg_table.add_column("Addr", justify="right")
    reg_table.add_column("Name", style="cyan")
    reg_table.add_column("Deps", justify="right", style="red")
    reg_table.add_column("Writes", justify="right")
    reg_table.add_column("Reads", justify="right")

    for reg in result.hot_registers[:top_n]:
        name = reg.name or "-"
        reg_table.add_row(
            str(reg.addr),
            name,
            f"{reg.dep_count:,}",
            f"{reg.write_count:,}",
            f"{reg.read_count:,}"
        )

    console.print(reg_table)


def print_results(result: DependencyGraphResult, top_n: int = 10, use_rich: bool = True):
    """Print results with appropriate formatter."""
    if use_rich and RICH_AVAILABLE:
        print_results_rich(result, top_n)
    else:
        print_results_plain(result, top_n)


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
        return kb.instrs, kb.debug_info
    else:
        return kernel_builder.instrs, getattr(kernel_builder, 'debug_info', None)


def main():
    parser = argparse.ArgumentParser(
        description="Build and analyze dependency graph for VLIW SIMD kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/dependency_graph.py                     # Basic analysis
    python tools/dependency_graph.py --json              # JSON output
    python tools/dependency_graph.py --dot graph.dot     # Export DOT file
    python tools/dependency_graph.py --top 20            # Show top 20 hot registers
    python tools/dependency_graph.py --all-hazards       # Include WAR/WAW hazards
        """
    )
    parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable")
    parser.add_argument("--dot", metavar="FILE", help="Export graph to DOT file for Graphviz")
    parser.add_argument("--top", "-n", type=int, default=10, help="Number of hot registers to show")
    parser.add_argument("--all-hazards", action="store_true", help="Include WAR and WAW hazards (not just RAW)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    args = parser.parse_args()

    print("Loading kernel...", file=sys.stderr)
    instructions, debug_info = analyze_kernel()

    print(f"Analyzing {len(instructions)} cycles...", file=sys.stderr)
    result = analyze_dependency_graph(
        instructions,
        debug_info=debug_info,
        include_war=args.all_hazards,
        include_waw=args.all_hazards
    )

    if args.dot:
        export_dot(instructions, args.dot, debug_info=debug_info)

    if args.json:
        print(result.to_json())
    else:
        print_results(result, top_n=args.top, use_rich=not args.no_color)


if __name__ == "__main__":
    main()
