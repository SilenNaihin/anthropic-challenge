#!/usr/bin/env python3
"""
ILP Optimal Scheduler for VLIW SIMD Architecture

Uses Integer Linear Programming (via OR-Tools CP-SAT solver) to find the
theoretically optimal schedule for instruction streams, respecting:
- Data dependencies (RAW hazards)
- Slot limits per engine per cycle (12 alu, 6 valu, 2 load, 2 store, 1 flow)

This is a textbook ILP scheduling problem:
- Variables: cycle[i] = cycle at which instruction i is scheduled
- Constraints: cycle[j] > cycle[i] for all dependencies i -> j
- Constraints: sum(instr assigned to cycle c for engine e) <= limit[e]
- Objective: minimize max(cycle[i]) (total cycles)

The optimal schedule provides:
1. Theoretical minimum cycles (lower bound)
2. Gap between current schedule and optimal
3. Proof of what's achievable vs greedy heuristics

Usage:
    python tools/ilp_scheduler/ilp_scheduler.py
    python tools/ilp_scheduler/ilp_scheduler.py --json
    python tools/ilp_scheduler/ilp_scheduler.py --time-limit 60

From Python:
    from tools.ilp_scheduler.ilp_scheduler import solve_optimal_schedule
    result = solve_optimal_schedule(instructions)
"""

import sys
import os
import json
import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from enum import Enum

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import SLOT_LIMITS, VLEN

# Try to import OR-Tools
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

# Try to import Rich
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Engine slot limits (excluding debug)
ENGINES = {k: v for k, v in SLOT_LIMITS.items() if k != "debug"}
MAX_SLOTS_PER_CYCLE = sum(ENGINES.values())  # 23 total


class SolverStatus(Enum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"  # Found a solution but not proven optimal
    INFEASIBLE = "INFEASIBLE"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


@dataclass
class Instruction:
    """Single instruction extracted from bundles."""
    id: int
    engine: str
    slot: tuple
    reads: Set[int] = field(default_factory=set)
    writes: Set[int] = field(default_factory=set)
    original_cycle: int = 0
    is_flow_control: bool = False

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class ScheduleResult:
    """Results from ILP scheduling."""
    status: SolverStatus
    optimal_cycles: int
    current_cycles: int
    gap_cycles: int
    gap_percentage: float
    speedup_potential: float
    solve_time_seconds: float

    # The optimal schedule: instr_id -> cycle
    schedule: Dict[int, int]

    # Per-cycle utilization in optimal schedule
    optimal_utilization: float

    # Additional stats
    total_instructions: int
    total_dependencies: int
    solver_iterations: int

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "optimal_cycles": self.optimal_cycles,
            "current_cycles": self.current_cycles,
            "gap_cycles": self.gap_cycles,
            "gap_percentage": round(self.gap_percentage, 2),
            "speedup_potential": round(self.speedup_potential, 3),
            "solve_time_seconds": round(self.solve_time_seconds, 3),
            "optimal_utilization_pct": round(self.optimal_utilization, 2),
            "total_instructions": self.total_instructions,
            "total_dependencies": self.total_dependencies,
            "solver_iterations": self.solver_iterations,
            "schedule_preview": dict(list(self.schedule.items())[:20])  # First 20 mappings
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


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
        elif op == "load_offset":
            if len(slot) >= 4:
                writes.add(slot[1] + slot[3])
                reads.add(slot[2] + slot[3])
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


def is_flow_control(slot: tuple, engine: str) -> bool:
    """Check if instruction is flow control (affects program order)."""
    if engine != "flow":
        return False
    if not slot:
        return False
    return slot[0] in ("pause", "halt", "jump", "cond_jump", "cond_jump_rel", "jump_indirect")


def flatten_instructions(bundles: List[dict]) -> List[Instruction]:
    """Convert instruction bundles to flat list of individual instructions."""
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
                    original_cycle=cycle_num,
                    is_flow_control=flow_ctrl
                )
                instructions.append(instr)
                instr_id += 1

    return instructions


def build_dependencies(instructions: List[Instruction]) -> List[Tuple[int, int]]:
    """
    Build dependency list: (from_id, to_id) for RAW hazards.

    Algorithm: Track last writer for each register. When an instruction reads
    a register, add edge from last writer to this instruction.
    """
    dependencies = []

    # Track last writer for each scratch address
    last_writer: Dict[int, int] = {}  # addr -> instr_id

    # Track last flow control for ordering
    last_flow_id: Optional[int] = None

    for instr in instructions:
        # RAW: we read what was written before
        for addr in instr.reads:
            if addr in last_writer:
                writer_id = last_writer[addr]
                if writer_id != instr.id:
                    dependencies.append((writer_id, instr.id))

        # Flow control ordering
        if instr.is_flow_control:
            if last_flow_id is not None:
                dependencies.append((last_flow_id, instr.id))
            last_flow_id = instr.id
        elif last_flow_id is not None:
            # Instructions after flow control must wait
            dependencies.append((last_flow_id, instr.id))

        # Update last writer
        for addr in instr.writes:
            last_writer[addr] = instr.id

    # Remove duplicates
    return list(set(dependencies))


def _greedy_schedule_for_hints(
    instructions: List[Instruction],
    dependencies: List[Tuple[int, int]]
) -> Dict[int, int]:
    """
    Quick greedy schedule to provide hints to the ILP solver.
    Returns instr_id -> cycle mapping.
    """
    n = len(instructions)
    if n == 0:
        return {}

    # Build adjacency
    successors: Dict[int, List[int]] = defaultdict(list)
    in_degree: Dict[int, int] = {instr.id: 0 for instr in instructions}

    for from_id, to_id in dependencies:
        successors[from_id].append(to_id)
        in_degree[to_id] += 1

    schedule: Dict[int, int] = {}
    scheduled = set()
    instr_by_id = {instr.id: instr for instr in instructions}

    ready = [instr for instr in instructions if in_degree[instr.id] == 0]
    current_cycle = 0

    while ready or len(scheduled) < n:
        if not ready:
            break

        slots_used: Dict[str, int] = defaultdict(int)
        scheduled_this_cycle = []
        remaining = []

        for instr in ready:
            engine = instr.engine
            limit = ENGINES.get(engine, 1)
            if slots_used[engine] < limit:
                slots_used[engine] += 1
                scheduled_this_cycle.append(instr)
                scheduled.add(instr.id)
                schedule[instr.id] = current_cycle
            else:
                remaining.append(instr)

        ready = remaining

        for instr in scheduled_this_cycle:
            for succ_id in successors[instr.id]:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0 and succ_id not in scheduled:
                    ready.append(instr_by_id[succ_id])

        current_cycle += 1

    return schedule


def solve_with_ortools(
    instructions: List[Instruction],
    dependencies: List[Tuple[int, int]],
    time_limit_seconds: int = 120,
    verbose: bool = False
) -> ScheduleResult:
    """
    Solve the optimal scheduling problem using OR-Tools CP-SAT with Cumulative constraints.

    This uses the efficient cumulative constraint formulation:
    - Each instruction is an interval of duration 1
    - Each engine has a cumulative resource constraint
    - Dependencies are precedence constraints between intervals

    This scales much better than explicit boolean variables per (instruction, cycle) pair.
    """
    n = len(instructions)
    if n == 0:
        return ScheduleResult(
            status=SolverStatus.OPTIMAL,
            optimal_cycles=0,
            current_cycles=0,
            gap_cycles=0,
            gap_percentage=0.0,
            speedup_potential=1.0,
            solve_time_seconds=0.0,
            schedule={},
            optimal_utilization=0.0,
            total_instructions=0,
            total_dependencies=0,
            solver_iterations=0
        )

    model = cp_model.CpModel()

    # Compute a tighter horizon using critical path analysis
    # This significantly reduces the search space
    instr_by_id = {instr.id: instr for instr in instructions}

    # Compute earliest possible cycle for each instruction (forward pass)
    earliest = {instr.id: 0 for instr in instructions}

    # Build successor map for topological order
    dep_dict: Dict[int, List[int]] = defaultdict(list)
    for from_id, to_id in dependencies:
        dep_dict[from_id].append(to_id)

    # Forward pass: compute earliest start times
    for instr in instructions:
        for succ_id in dep_dict[instr.id]:
            earliest[succ_id] = max(earliest[succ_id], earliest[instr.id] + 1)

    # Critical path gives lower bound
    critical_path = max(earliest.values()) + 1 if earliest else 1

    # Get greedy solution first for horizon calculation and hints
    greedy_schedule = _greedy_schedule_for_hints(instructions, dependencies)
    greedy_cycles = max(greedy_schedule.values()) + 1 if greedy_schedule else n

    # Horizon: use greedy solution + small buffer (greedy gives valid upper bound)
    # Also respect critical path as lower bound for search
    max_per_engine = defaultdict(int)
    for instr in instructions:
        max_per_engine[instr.engine] += 1

    # Upper bound is max(critical_path, max_instructions_for_any_engine / limit)
    upper_bounds = []
    for engine, count in max_per_engine.items():
        limit = ENGINES.get(engine, 1)
        upper_bounds.append((count + limit - 1) // limit)  # ceil division

    # Use greedy as the horizon since we know it's achievable
    horizon = greedy_cycles + 1  # +1 buffer for edge cases

    # Variables: start[i] = which cycle instruction i starts (duration = 1)
    start_vars = {}
    end_vars = {}
    interval_vars = {}

    for instr in instructions:
        # Use earliest time as lower bound to tighten search
        lb = earliest[instr.id]
        start = model.NewIntVar(lb, horizon - 1, f"start_{instr.id}")
        end = model.NewIntVar(lb + 1, horizon, f"end_{instr.id}")
        interval = model.NewIntervalVar(start, 1, end, f"interval_{instr.id}")

        start_vars[instr.id] = start
        end_vars[instr.id] = end
        interval_vars[instr.id] = interval

    # Makespan variable
    makespan = model.NewIntVar(critical_path, horizon, "makespan")

    # Constraint: makespan >= all end times
    for instr in instructions:
        model.Add(makespan >= end_vars[instr.id])

    # Dependency constraints: start[j] >= end[i] for dependency (i, j)
    # Since duration is 1, end[i] = start[i] + 1, so: start[j] >= start[i] + 1
    for from_id, to_id in dependencies:
        model.Add(start_vars[to_id] >= end_vars[from_id])

    # Cumulative constraints per engine (much more efficient than explicit booleans)
    # Group instructions by engine
    instrs_by_engine: Dict[str, List[Instruction]] = defaultdict(list)
    for instr in instructions:
        instrs_by_engine[instr.engine].append(instr)

    for engine, instrs in instrs_by_engine.items():
        limit = ENGINES.get(engine, 1)

        # Each instruction has demand 1, capacity is the slot limit
        intervals = [interval_vars[instr.id] for instr in instrs]
        demands = [1] * len(instrs)

        model.AddCumulative(intervals, demands, limit)

    # Objective: minimize makespan
    model.Minimize(makespan)

    # Use greedy solution as hint to warm-start the solver
    for instr in instructions:
        if instr.id in greedy_schedule:
            cycle = greedy_schedule[instr.id]
            # Ensure hint is within bounds
            lb = earliest[instr.id]
            if cycle >= lb and cycle < horizon:
                model.AddHint(start_vars[instr.id], cycle)

    # Solver configuration
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_workers = 8  # Use multiple cores

    # Enable some optimization for scheduling problems
    solver.parameters.linearization_level = 2

    # Focus on finding good solutions quickly
    solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH

    if verbose:
        solver.parameters.log_search_progress = True

    start_time = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - start_time

    # Interpret status
    if status == cp_model.OPTIMAL:
        result_status = SolverStatus.OPTIMAL
    elif status == cp_model.FEASIBLE:
        result_status = SolverStatus.FEASIBLE
    elif status == cp_model.INFEASIBLE:
        result_status = SolverStatus.INFEASIBLE
    else:
        result_status = SolverStatus.TIMEOUT

    # Extract solution
    schedule = {}
    optimal_cycles = 0

    if result_status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
        optimal_cycles = solver.Value(makespan)
        for instr in instructions:
            schedule[instr.id] = solver.Value(start_vars[instr.id])
    if result_status == SolverStatus.TIMEOUT or optimal_cycles == 0:
        # Use greedy solution as fallback when ILP can't find anything
        schedule = greedy_schedule
        optimal_cycles = greedy_cycles
        result_status = SolverStatus.FEASIBLE  # We have a feasible solution from greedy

    # Calculate current cycles (from original schedule)
    current_cycles = max(instr.original_cycle for instr in instructions) + 1

    # Calculate gap
    gap_cycles = current_cycles - optimal_cycles
    gap_pct = 100.0 * gap_cycles / current_cycles if current_cycles > 0 else 0.0
    speedup = current_cycles / optimal_cycles if optimal_cycles > 0 else 1.0

    # Calculate optimal utilization
    opt_util = 100.0 * n / (optimal_cycles * MAX_SLOTS_PER_CYCLE) if optimal_cycles > 0 else 0.0

    return ScheduleResult(
        status=result_status,
        optimal_cycles=optimal_cycles,
        current_cycles=current_cycles,
        gap_cycles=gap_cycles,
        gap_percentage=gap_pct,
        speedup_potential=speedup,
        solve_time_seconds=solve_time,
        schedule=schedule,
        optimal_utilization=opt_util,
        total_instructions=n,
        total_dependencies=len(dependencies),
        solver_iterations=solver.NumBranches() if hasattr(solver, 'NumBranches') else 0
    )


def solve_greedy_lower_bound(
    instructions: List[Instruction],
    dependencies: List[Tuple[int, int]]
) -> ScheduleResult:
    """
    Fallback solver using greedy list scheduling when OR-Tools not available.

    This uses the same algorithm as VLIW packer but tracks the minimum achievable.
    Note: This gives a FEASIBLE solution, not guaranteed optimal.
    """
    n = len(instructions)
    if n == 0:
        return ScheduleResult(
            status=SolverStatus.OPTIMAL,
            optimal_cycles=0,
            current_cycles=0,
            gap_cycles=0,
            gap_percentage=0.0,
            speedup_potential=1.0,
            solve_time_seconds=0.0,
            schedule={},
            optimal_utilization=0.0,
            total_instructions=0,
            total_dependencies=0,
            solver_iterations=0
        )

    start_time = time.time()

    # Build adjacency lists
    successors: Dict[int, List[int]] = defaultdict(list)
    predecessors: Dict[int, List[int]] = defaultdict(list)
    in_degree: Dict[int, int] = {instr.id: 0 for instr in instructions}

    for from_id, to_id in dependencies:
        successors[from_id].append(to_id)
        predecessors[to_id].append(from_id)
        in_degree[to_id] += 1

    # List scheduling
    schedule: Dict[int, int] = {}
    scheduled = set()
    ready = [instr for instr in instructions if in_degree[instr.id] == 0]

    instr_by_id = {instr.id: instr for instr in instructions}
    current_cycle = 0

    while ready or len(scheduled) < n:
        if not ready:
            break

        # Try to pack into current cycle
        slots_used: Dict[str, int] = defaultdict(int)
        scheduled_this_cycle = []
        remaining = []

        for instr in ready:
            engine = instr.engine
            limit = ENGINES.get(engine, 1)
            if slots_used[engine] < limit:
                slots_used[engine] += 1
                scheduled_this_cycle.append(instr)
                scheduled.add(instr.id)
                schedule[instr.id] = current_cycle
            else:
                remaining.append(instr)

        ready = remaining

        # Update ready queue
        for instr in scheduled_this_cycle:
            for succ_id in successors[instr.id]:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0 and succ_id not in scheduled:
                    ready.append(instr_by_id[succ_id])

        current_cycle += 1

    solve_time = time.time() - start_time

    optimal_cycles = current_cycle
    current_cycles = max(instr.original_cycle for instr in instructions) + 1

    gap_cycles = current_cycles - optimal_cycles
    gap_pct = 100.0 * gap_cycles / current_cycles if current_cycles > 0 else 0.0
    speedup = current_cycles / optimal_cycles if optimal_cycles > 0 else 1.0
    opt_util = 100.0 * n / (optimal_cycles * MAX_SLOTS_PER_CYCLE) if optimal_cycles > 0 else 0.0

    return ScheduleResult(
        status=SolverStatus.FEASIBLE,  # Greedy doesn't prove optimality
        optimal_cycles=optimal_cycles,
        current_cycles=current_cycles,
        gap_cycles=gap_cycles,
        gap_percentage=gap_pct,
        speedup_potential=speedup,
        solve_time_seconds=solve_time,
        schedule=schedule,
        optimal_utilization=opt_util,
        total_instructions=n,
        total_dependencies=len(dependencies),
        solver_iterations=0
    )


def solve_optimal_schedule(
    bundles: List[dict],
    time_limit_seconds: int = 120,
    use_greedy_fallback: bool = True,
    verbose: bool = False
) -> ScheduleResult:
    """
    Main entry point: find optimal schedule for instruction bundles.

    Args:
        bundles: List of instruction bundles (dict of engine -> slots)
        time_limit_seconds: Maximum solve time
        use_greedy_fallback: Fall back to greedy if OR-Tools unavailable
        verbose: Print solver progress

    Returns:
        ScheduleResult with optimal schedule and gap analysis
    """
    # Extract instructions
    instructions = flatten_instructions(bundles)

    if not instructions:
        return ScheduleResult(
            status=SolverStatus.OPTIMAL,
            optimal_cycles=0,
            current_cycles=len(bundles),
            gap_cycles=len(bundles),
            gap_percentage=100.0 if bundles else 0.0,
            speedup_potential=float('inf') if bundles else 1.0,
            solve_time_seconds=0.0,
            schedule={},
            optimal_utilization=0.0,
            total_instructions=0,
            total_dependencies=0,
            solver_iterations=0
        )

    # Build dependencies
    dependencies = build_dependencies(instructions)

    # Solve
    if ORTOOLS_AVAILABLE:
        return solve_with_ortools(instructions, dependencies, time_limit_seconds, verbose)
    elif use_greedy_fallback:
        return solve_greedy_lower_bound(instructions, dependencies)
    else:
        return ScheduleResult(
            status=SolverStatus.ERROR,
            optimal_cycles=0,
            current_cycles=len(bundles),
            gap_cycles=0,
            gap_percentage=0.0,
            speedup_potential=1.0,
            solve_time_seconds=0.0,
            schedule={},
            optimal_utilization=0.0,
            total_instructions=len(instructions),
            total_dependencies=len(dependencies),
            solver_iterations=0
        )


# ============== Output Formatting ==============

def print_result_plain(result: ScheduleResult):
    """Plain text output."""
    print("=" * 70)
    print("ILP OPTIMAL SCHEDULER RESULTS")
    print("=" * 70)
    print()

    status_str = result.status.value
    if result.status == SolverStatus.OPTIMAL:
        status_str += " (proven optimal)"
    elif result.status == SolverStatus.FEASIBLE:
        status_str += " (solution found, may not be optimal)"

    print(f"Status:              {status_str}")
    print(f"Solve Time:          {result.solve_time_seconds:.2f} seconds")
    print()

    print("-" * 70)
    print("SCHEDULE COMPARISON")
    print("-" * 70)
    print(f"Current Cycles:      {result.current_cycles:,}")
    print(f"Optimal Cycles:      {result.optimal_cycles:,}")
    print(f"Gap:                 {result.gap_cycles:,} cycles ({result.gap_percentage:.1f}%)")
    print(f"Speedup Potential:   {result.speedup_potential:.2f}x")
    print()

    print("-" * 70)
    print("ANALYSIS")
    print("-" * 70)
    print(f"Total Instructions:  {result.total_instructions:,}")
    print(f"Total Dependencies:  {result.total_dependencies:,}")
    print(f"Optimal Utilization: {result.optimal_utilization:.1f}%")
    print()

    # Interpretation
    print("-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if result.gap_percentage < 5:
        print("EXCELLENT: Current schedule is within 5% of optimal.")
        print("Further gains must come from algorithmic changes, not scheduling.")
    elif result.gap_percentage < 20:
        print("GOOD: Current schedule has some room for improvement.")
        print("Consider more aggressive VLIW packing or reordering.")
    else:
        print("SIGNIFICANT GAP: Current schedule is far from optimal.")
        print("Major scheduling improvements are possible.")
        print("Run VLIW packer with aggressive mode or manual optimization.")

    print()


def print_result_rich(result: ScheduleResult):
    """Rich colored output."""
    console = Console()

    console.print(Panel("ILP OPTIMAL SCHEDULER", style="bold cyan", box=box.DOUBLE))

    # Status
    if result.status == SolverStatus.OPTIMAL:
        status_style = "bold green"
        status_text = "OPTIMAL (proven)"
    elif result.status == SolverStatus.FEASIBLE:
        status_style = "bold yellow"
        status_text = "FEASIBLE (may not be optimal)"
    elif result.status == SolverStatus.TIMEOUT:
        status_style = "bold yellow"
        status_text = "TIMEOUT (best found)"
    else:
        status_style = "bold red"
        status_text = result.status.value

    console.print(f"[{status_style}]Status: {status_text}[/{status_style}]")
    console.print(f"Solve Time: {result.solve_time_seconds:.2f}s")
    console.print()

    # Comparison table
    console.print("[bold yellow]SCHEDULE COMPARISON[/bold yellow]")

    table = Table(box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Current Cycles", f"{result.current_cycles:,}")
    table.add_row("Optimal Cycles", f"[bold green]{result.optimal_cycles:,}[/bold green]")

    gap_style = "green" if result.gap_percentage < 5 else "yellow" if result.gap_percentage < 20 else "red"
    table.add_row("Gap", f"[{gap_style}]{result.gap_cycles:,} cycles ({result.gap_percentage:.1f}%)[/{gap_style}]")

    speedup_style = "bold magenta"
    table.add_row("Speedup Potential", f"[{speedup_style}]{result.speedup_potential:.2f}x[/{speedup_style}]")

    console.print(table)
    console.print()

    # Stats table
    console.print("[bold yellow]STATISTICS[/bold yellow]")

    stats_table = Table(box=box.SIMPLE)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", justify="right")

    stats_table.add_row("Total Instructions", f"{result.total_instructions:,}")
    stats_table.add_row("Total Dependencies", f"{result.total_dependencies:,}")
    stats_table.add_row("Optimal Utilization", f"{result.optimal_utilization:.1f}%")
    stats_table.add_row("Solver Iterations", f"{result.solver_iterations:,}")

    console.print(stats_table)
    console.print()

    # Interpretation
    console.print("[bold yellow]INTERPRETATION[/bold yellow]")

    if result.gap_percentage < 5:
        console.print("[bold green]EXCELLENT:[/bold green] Current schedule is within 5% of optimal.")
        console.print("Further gains must come from [cyan]algorithmic changes[/cyan], not scheduling.")
    elif result.gap_percentage < 20:
        console.print("[bold yellow]GOOD:[/bold yellow] Some room for scheduling improvement.")
        console.print("Try [cyan]aggressive VLIW packing[/cyan] or manual reordering.")
    else:
        console.print("[bold red]SIGNIFICANT GAP:[/bold red] Current schedule is far from optimal!")
        console.print("Major improvements possible via [cyan]VLIW packer[/cyan] or better scheduling.")

    console.print()


def print_result(result: ScheduleResult, use_rich: bool = True):
    """Print result with appropriate formatter."""
    if use_rich and RICH_AVAILABLE:
        print_result_rich(result)
    else:
        print_result_plain(result)


# ============== Main ==============

def load_kernel():
    """Load the current kernel from perf_takehome.py"""
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
        description="ILP Optimal Scheduler for VLIW SIMD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/ilp_scheduler/ilp_scheduler.py                  # Basic analysis
    python tools/ilp_scheduler/ilp_scheduler.py --json           # JSON output
    python tools/ilp_scheduler/ilp_scheduler.py --time-limit 60  # Limit solve time
    python tools/ilp_scheduler/ilp_scheduler.py --verbose        # Show solver progress
        """
    )
    parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable")
    parser.add_argument("--time-limit", "-t", type=int, default=120, help="Solver time limit in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show solver progress")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    args = parser.parse_args()

    # Check OR-Tools
    if not ORTOOLS_AVAILABLE:
        print("Warning: OR-Tools not installed. Using greedy fallback.", file=sys.stderr)
        print("Install with: pip install ortools", file=sys.stderr)
        print(file=sys.stderr)

    print("Loading kernel...", file=sys.stderr)
    bundles = load_kernel()

    print(f"Analyzing {len(bundles)} instruction bundles...", file=sys.stderr)

    if RICH_AVAILABLE and not args.no_color and not args.json:
        console = Console(stderr=True)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Solving ILP...", total=None)
            result = solve_optimal_schedule(
                bundles,
                time_limit_seconds=args.time_limit,
                verbose=args.verbose
            )
            progress.remove_task(task)
    else:
        print("Solving (this may take a while)...", file=sys.stderr)
        result = solve_optimal_schedule(
            bundles,
            time_limit_seconds=args.time_limit,
            verbose=args.verbose
        )

    if args.json:
        print(result.to_json())
    else:
        print_result(result, use_rich=not args.no_color)


if __name__ == "__main__":
    main()
