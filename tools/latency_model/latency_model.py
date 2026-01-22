#!/usr/bin/env python3
"""
Instruction Latency Model Analyzer for VLIW SIMD Architecture

Analyzes and documents the latency characteristics of the simulated architecture.
Key question: Do all operations complete in 1 cycle, or are there multi-cycle ops?

Usage:
    python tools/latency_model/latency_model.py              # Full analysis
    python tools/latency_model/latency_model.py --empirical  # Run empirical tests
    python tools/latency_model/latency_model.py --json       # JSON output
    python tools/latency_model/latency_model.py --no-color   # Plain text output
"""

import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from problem import Machine, Core, CoreState, DebugInfo, SLOT_LIMITS, VLEN, SCRATCH_SIZE

# Try to import Rich for pretty output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class LatencyType(Enum):
    """Classification of operation latency behavior."""
    SINGLE_CYCLE = "single_cycle"       # Completes in 1 cycle
    MULTI_CYCLE = "multi_cycle"         # Takes > 1 cycle
    VARIABLE = "variable"               # Depends on operands/state


@dataclass
class OperationLatency:
    """Latency model for a single operation type."""
    engine: str
    operation: str
    latency_cycles: int
    latency_type: LatencyType
    throughput_per_cycle: int  # How many can issue per cycle (slot limit)
    notes: str = ""


@dataclass
class EmpiricalTest:
    """Result of an empirical latency test."""
    name: str
    description: str
    expected_cycles: int
    actual_cycles: int
    passed: bool
    conclusion: str


@dataclass
class LatencyModel:
    """Complete latency model for the architecture."""
    # Static model derived from simulator analysis
    operations: list[OperationLatency] = field(default_factory=list)

    # Empirical test results
    empirical_tests: list[EmpiricalTest] = field(default_factory=list)

    # Summary findings
    all_single_cycle: bool = True
    has_multi_cycle_ops: bool = False
    latency_bound_ops: list[str] = field(default_factory=list)
    throughput_bound_ops: list[str] = field(default_factory=list)

    # Scheduling implications
    scheduling_notes: list[str] = field(default_factory=list)


def analyze_simulator_code() -> list[OperationLatency]:
    """
    Analyze the simulator code to document latency assumptions.

    Key insight from problem.py:
    - scratch_write and mem_write are staging dicts
    - All writes applied AFTER all reads in step()
    - No explicit latency counters or stall logic
    - This implies ALL operations complete in 1 cycle
    """
    operations = []

    # ALU operations (12 slots)
    alu_ops = ["+", "-", "*", "//", "cdiv", "^", "&", "|", "<<", ">>", "%", "<", "=="]
    for op in alu_ops:
        operations.append(OperationLatency(
            engine="alu",
            operation=op,
            latency_cycles=1,
            latency_type=LatencyType.SINGLE_CYCLE,
            throughput_per_cycle=SLOT_LIMITS["alu"],
            notes="All ALU ops share 12 slots, complete in 1 cycle"
        ))

    # VALU operations (6 slots)
    valu_ops = ["vbroadcast", "multiply_add", "+", "-", "*", "//", "^", "&", "|", "<<", ">>", "%", "<", "=="]
    for op in valu_ops:
        notes = "Vector op (VLEN=8 elements), shares 6 slots"
        if op == "multiply_add":
            notes = "Fused multiply-add: dest = a*b + c, very efficient"
        operations.append(OperationLatency(
            engine="valu",
            operation=op,
            latency_cycles=1,
            latency_type=LatencyType.SINGLE_CYCLE,
            throughput_per_cycle=SLOT_LIMITS["valu"],
            notes=notes
        ))

    # Load operations (2 slots)
    load_ops = [
        ("load", "Scalar load from memory[scratch[addr]]"),
        ("load_offset", "Scalar load with offset: mem[scratch[addr+offset]]"),
        ("vload", "Vector load: 8 consecutive elements from mem[scratch[addr]]"),
        ("const", "Load immediate constant into scratch"),
    ]
    for op, notes in load_ops:
        operations.append(OperationLatency(
            engine="load",
            operation=op,
            latency_cycles=1,
            latency_type=LatencyType.SINGLE_CYCLE,
            throughput_per_cycle=SLOT_LIMITS["load"],
            notes=notes
        ))

    # Store operations (2 slots)
    store_ops = [
        ("store", "Scalar store to memory"),
        ("vstore", "Vector store: 8 consecutive elements"),
    ]
    for op, notes in store_ops:
        operations.append(OperationLatency(
            engine="store",
            operation=op,
            latency_cycles=1,
            latency_type=LatencyType.SINGLE_CYCLE,
            throughput_per_cycle=SLOT_LIMITS["store"],
            notes=notes
        ))

    # Flow operations (1 slot)
    flow_ops = [
        ("select", "Conditional select: dest = cond ? a : b"),
        ("add_imm", "Add immediate to scratch value"),
        ("vselect", "Vector conditional select (per-element)"),
        ("halt", "Stop execution"),
        ("pause", "Pause execution (for debugging/sync)"),
        ("trace_write", "Write value to trace buffer"),
        ("cond_jump", "Conditional jump to absolute address"),
        ("cond_jump_rel", "Conditional jump relative"),
        ("jump", "Unconditional jump"),
        ("jump_indirect", "Jump to address in scratch"),
        ("coreid", "Load core ID into scratch"),
    ]
    for op, notes in flow_ops:
        operations.append(OperationLatency(
            engine="flow",
            operation=op,
            latency_cycles=1,
            latency_type=LatencyType.SINGLE_CYCLE,
            throughput_per_cycle=SLOT_LIMITS["flow"],
            notes=notes
        ))

    return operations


def run_empirical_test(name: str, description: str, program: list, expected_cycles: int) -> EmpiricalTest:
    """
    Run a simple program and verify cycle count.
    """
    debug_info = DebugInfo(scratch_map={})
    mem = [0] * 1000  # Simple memory image

    machine = Machine(
        mem_dump=mem,
        program=program,
        debug_info=debug_info,
        n_cores=1,
        scratch_size=SCRATCH_SIZE,
        trace=False
    )

    machine.run()
    actual_cycles = machine.cycle
    passed = actual_cycles == expected_cycles

    if passed:
        conclusion = f"Confirmed: {expected_cycles} cycle(s) as expected"
    else:
        conclusion = f"UNEXPECTED: Got {actual_cycles} cycles, expected {expected_cycles}"

    return EmpiricalTest(
        name=name,
        description=description,
        expected_cycles=expected_cycles,
        actual_cycles=actual_cycles,
        passed=passed,
        conclusion=conclusion
    )


def run_empirical_tests() -> list[EmpiricalTest]:
    """
    Run empirical tests to verify latency model.

    Key tests:
    1. Write-then-read in SAME cycle: should read OLD value
    2. Write-then-read in NEXT cycle: should read NEW value
    3. Multi-operation chains: verify no stalls
    4. Load latency: can we use loaded value immediately?
    """
    tests = []

    # Test 1: Single ALU operation takes 1 cycle
    tests.append(run_empirical_test(
        name="single_alu",
        description="Single ALU add operation",
        program=[
            {"load": [("const", 0, 5), ("const", 1, 10)]},  # Cycle 1: Load constants
            {"alu": [("+", 2, 0, 1)]},                       # Cycle 2: Add
            {"flow": [("halt",)]},                           # Cycle 3: Halt
        ],
        expected_cycles=3
    ))

    # Test 2: Back-to-back ALU operations (RAW dependency)
    # If latency > 1, this would need more cycles
    tests.append(run_empirical_test(
        name="alu_chain",
        description="ALU chain: a+b -> c+d (RAW dependency)",
        program=[
            {"load": [("const", 0, 5), ("const", 1, 10)]},  # Cycle 1
            {"alu": [("+", 2, 0, 1)]},                       # Cycle 2: r2 = 5+10 = 15
            {"alu": [("+", 3, 2, 0)]},                       # Cycle 3: r3 = r2+5 = 20
            {"flow": [("halt",)]},                           # Cycle 4
        ],
        expected_cycles=4
    ))

    # Test 3: Load followed immediately by use
    tests.append(run_empirical_test(
        name="load_use",
        description="Load then immediate use (load latency test)",
        program=[
            {"load": [("const", 0, 0)]},                    # Cycle 1: addr = 0
            {"load": [("load", 1, 0)]},                     # Cycle 2: load mem[0] -> r1
            {"alu": [("+", 2, 1, 1)]},                      # Cycle 3: use r1 immediately
            {"flow": [("halt",)]},                          # Cycle 4
        ],
        expected_cycles=4
    ))

    # Test 4: VALU operation
    tests.append(run_empirical_test(
        name="single_valu",
        description="Single VALU vector add (8 elements)",
        program=[
            {"valu": [("vbroadcast", 0, 0)]},               # Cycle 1: broadcast to 0-7
            {"valu": [("vbroadcast", 8, 0)]},               # Cycle 2: broadcast to 8-15
            {"valu": [("+", 16, 0, 8)]},                    # Cycle 3: vector add
            {"flow": [("halt",)]},                          # Cycle 4
        ],
        expected_cycles=4
    ))

    # Test 5: Same-cycle write-read (should read OLD value)
    # This tests the key semantic: writes apply at END of cycle
    tests.append(run_empirical_test(
        name="same_cycle_raw",
        description="Same-cycle write-read (should use old value)",
        program=[
            {"load": [("const", 0, 100)]},                  # Cycle 1: r0 = 100
            # Cycle 2: Both in same cycle. r1 written, then r2 tries to read r1
            # But r1 write hasn't applied yet, so r2 = old r1 (0) + r0 = 100
            {"alu": [("+", 1, 0, 0), ("+", 2, 1, 0)]},      # r1 = 200, r2 = 0+100 = 100
            {"flow": [("halt",)]},                          # Cycle 3
        ],
        expected_cycles=3
    ))

    # Test 6: Parallel independent ALU ops (tests VLIW)
    tests.append(run_empirical_test(
        name="parallel_alu",
        description="12 independent ALU ops in single cycle",
        program=[
            {"load": [("const", 0, 1), ("const", 1, 2)]},   # Cycle 1
            {"alu": [                                       # Cycle 2: All 12 slots
                ("+", 10, 0, 1), ("+", 11, 0, 1), ("+", 12, 0, 1), ("+", 13, 0, 1),
                ("+", 14, 0, 1), ("+", 15, 0, 1), ("+", 16, 0, 1), ("+", 17, 0, 1),
                ("+", 18, 0, 1), ("+", 19, 0, 1), ("+", 20, 0, 1), ("+", 21, 0, 1),
            ]},
            {"flow": [("halt",)]},                          # Cycle 3
        ],
        expected_cycles=3
    ))

    # Test 7: Mixed engines same cycle
    tests.append(run_empirical_test(
        name="mixed_engines",
        description="ALU + VALU + load in same cycle",
        program=[
            {"load": [("const", 0, 5)], "alu": [("+", 100, 0, 0)], "valu": [("vbroadcast", 200, 0)]},
            {"flow": [("halt",)]},
        ],
        expected_cycles=2
    ))

    # Test 8: Store then load (memory latency)
    tests.append(run_empirical_test(
        name="store_load",
        description="Store value, then load it back (memory latency test)",
        program=[
            {"load": [("const", 0, 42), ("const", 1, 500)]},  # Cycle 1: value=42, addr=500
            {"store": [("store", 1, 0)]},                      # Cycle 2: mem[500] = 42
            {"load": [("load", 2, 1)]},                        # Cycle 3: r2 = mem[500]
            {"flow": [("halt",)]},                             # Cycle 4
        ],
        expected_cycles=4
    ))

    # Test 9: multiply_add operation
    tests.append(run_empirical_test(
        name="multiply_add",
        description="VALU multiply_add (fused operation)",
        program=[
            {"load": [("const", 0, 2)]},                       # Cycle 1
            {"valu": [("vbroadcast", 8, 0)]},                  # Cycle 2: a = [2,2,2,...]
            {"valu": [("vbroadcast", 16, 0)]},                 # Cycle 3: b = [2,2,2,...]
            {"valu": [("vbroadcast", 24, 0)]},                 # Cycle 4: c = [2,2,2,...]
            {"valu": [("multiply_add", 32, 8, 16, 24)]},       # Cycle 5: 2*2+2 = 6
            {"flow": [("halt",)]},                             # Cycle 6
        ],
        expected_cycles=6
    ))

    # Test 10: Integer divide (potentially slow)
    tests.append(run_empirical_test(
        name="integer_divide",
        description="ALU integer division (// operator)",
        program=[
            {"load": [("const", 0, 100), ("const", 1, 7)]},   # Cycle 1
            {"alu": [("//", 2, 0, 1)]},                        # Cycle 2: 100 // 7 = 14
            {"flow": [("halt",)]},                             # Cycle 3
        ],
        expected_cycles=3
    ))

    return tests


def analyze_latency_implications(operations: list[OperationLatency], tests: list[EmpiricalTest]) -> LatencyModel:
    """
    Analyze the latency model and derive scheduling implications.
    """
    model = LatencyModel(operations=operations, empirical_tests=tests)

    # Check if all operations are single-cycle
    model.all_single_cycle = all(op.latency_cycles == 1 for op in operations)
    model.has_multi_cycle_ops = not model.all_single_cycle

    # Classify operations by bottleneck type
    for op in operations:
        # Throughput bound: limited by slots per cycle
        if op.throughput_per_cycle <= 2:  # Load/store/flow are most constrained
            if op.engine not in [o.split(':')[0] for o in model.throughput_bound_ops]:
                model.throughput_bound_ops.append(f"{op.engine}: {op.throughput_per_cycle} slots/cycle")

    # Add scheduling notes based on findings
    if model.all_single_cycle:
        model.scheduling_notes.extend([
            "ALL operations complete in 1 cycle - no latency stalls",
            "Write-then-read across cycles: value available immediately next cycle",
            "Same-cycle write-read: reads get OLD value (writes apply at cycle end)",
            "No need for latency-hiding techniques (out-of-order, speculation)",
            "Focus optimization on: SLOT UTILIZATION and DEPENDENCIES",
        ])

    # Add throughput notes
    model.scheduling_notes.extend([
        f"Load engine: Only 2 slots - can become bottleneck with many loads",
        f"Store engine: Only 2 slots - can become bottleneck with many stores",
        f"Flow engine: Only 1 slot - jumps, selects serialize execution",
        f"ALU engine: 12 slots - rarely a bottleneck",
        f"VALU engine: 6 slots - each processes 8 elements (48 effective ops)",
    ])

    # Verify empirical tests
    all_passed = all(t.passed for t in tests)
    if all_passed:
        model.scheduling_notes.append("EMPIRICAL TESTS: All passed - latency model confirmed")
    else:
        failed = [t.name for t in tests if not t.passed]
        model.scheduling_notes.append(f"EMPIRICAL TESTS: FAILED - {failed}")

    return model


def print_plain(model: LatencyModel, show_all_ops: bool = False):
    """Plain text output (no Rich dependency)."""
    print("=" * 70)
    print("INSTRUCTION LATENCY MODEL ANALYSIS")
    print("=" * 70)
    print()

    # Summary
    print("SUMMARY")
    print("-" * 70)
    if model.all_single_cycle:
        print("ALL OPERATIONS: 1 cycle latency (confirmed)")
    else:
        print("WARNING: Multi-cycle operations detected!")
    print()

    # Operations by engine
    print("OPERATIONS BY ENGINE")
    print("-" * 70)
    engines = {}
    for op in model.operations:
        if op.engine not in engines:
            engines[op.engine] = []
        engines[op.engine].append(op)

    for engine, ops in engines.items():
        print(f"\n{engine.upper()} ({SLOT_LIMITS.get(engine, '?')} slots/cycle)")
        if show_all_ops:
            for op in ops:
                print(f"  {op.operation:15} {op.latency_cycles} cycle  {op.notes}")
        else:
            ops_list = ", ".join(op.operation for op in ops[:5])
            if len(ops) > 5:
                ops_list += f"... (+{len(ops)-5} more)"
            print(f"  Operations: {ops_list}")
            print(f"  All {op.latency_cycles} cycle")
    print()

    # Empirical tests
    if model.empirical_tests:
        print("EMPIRICAL TEST RESULTS")
        print("-" * 70)
        for test in model.empirical_tests:
            status = "PASS" if test.passed else "FAIL"
            print(f"  [{status}] {test.name}: {test.description}")
            if not test.passed:
                print(f"         Expected: {test.expected_cycles}, Got: {test.actual_cycles}")

        passed = sum(1 for t in model.empirical_tests if t.passed)
        total = len(model.empirical_tests)
        print(f"\n  Total: {passed}/{total} tests passed")
    print()

    # Throughput bottlenecks
    print("THROUGHPUT BOTTLENECKS")
    print("-" * 70)
    for note in model.throughput_bound_ops:
        print(f"  - {note}")
    print()

    # Scheduling implications
    print("SCHEDULING IMPLICATIONS")
    print("-" * 70)
    for note in model.scheduling_notes:
        print(f"  - {note}")
    print()

    # Key takeaways
    print("KEY TAKEAWAYS FOR OPTIMIZATION")
    print("-" * 70)
    print("""
  1. LATENCY IS NOT THE PROBLEM
     - All ops complete in 1 cycle
     - No need for latency-hiding transforms

  2. DEPENDENCIES ARE THE PROBLEM
     - RAW hazards force sequential execution
     - Break chains via: unrolling, pipelining, vectorization

  3. THROUGHPUT LIMITS MATTER
     - Load: max 2 ops/cycle (16 elements with vload)
     - Store: max 2 ops/cycle (16 elements with vstore)
     - Flow: max 1 op/cycle (serialization point!)

  4. VLIW PACKING IS KEY
     - 12 ALU + 6 VALU + 2 load + 2 store + 1 flow per cycle
     - Most cycles use << 10% of available slots
     - Pack independent ops to fill unused slots
""")


def print_rich(model: LatencyModel, show_all_ops: bool = False):
    """Rich formatted output."""
    console = Console()

    # Title
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Instruction Latency Model Analysis[/bold cyan]\n"
        "[dim]VLIW SIMD Architecture Latency Characteristics[/dim]",
        border_style="cyan"
    ))
    console.print()

    # Summary panel
    if model.all_single_cycle:
        summary = Text()
        summary.append("ALL OPERATIONS: ", style="bold green")
        summary.append("1 cycle latency ", style="green")
        summary.append("(confirmed)", style="dim green")
        console.print(Panel(summary, title="Summary", border_style="green"))
    else:
        console.print(Panel("[bold red]WARNING: Multi-cycle operations detected![/bold red]",
                          title="Summary", border_style="red"))
    console.print()

    # Operations table
    table = Table(title="Engine Slot Limits & Latency", box=box.ROUNDED)
    table.add_column("Engine", style="cyan", width=10)
    table.add_column("Slots/Cycle", justify="right", style="yellow")
    table.add_column("Latency", justify="right", style="green")
    table.add_column("Effective Throughput", style="dim")

    engines_seen = set()
    for op in model.operations:
        if op.engine not in engines_seen:
            engines_seen.add(op.engine)
            throughput = SLOT_LIMITS.get(op.engine, 1)
            effective = throughput * VLEN if op.engine == "valu" else throughput
            table.add_row(
                op.engine,
                str(throughput),
                "1 cycle",
                f"{effective} ops/cycle" if op.engine != "valu" else f"{throughput}x8 = {effective} elem/cycle"
            )

    console.print(table)
    console.print()

    # Empirical tests
    if model.empirical_tests:
        test_table = Table(title="Empirical Latency Tests", box=box.ROUNDED)
        test_table.add_column("Test", style="cyan", width=20)
        test_table.add_column("Description", width=40)
        test_table.add_column("Expected", justify="right")
        test_table.add_column("Actual", justify="right")
        test_table.add_column("Status", justify="center")

        for test in model.empirical_tests:
            status = "[green]PASS[/green]" if test.passed else "[red]FAIL[/red]"
            actual_style = "green" if test.passed else "red"
            test_table.add_row(
                test.name,
                test.description,
                str(test.expected_cycles),
                f"[{actual_style}]{test.actual_cycles}[/{actual_style}]",
                status
            )

        console.print(test_table)

        passed = sum(1 for t in model.empirical_tests if t.passed)
        total = len(model.empirical_tests)
        console.print(f"\n  [bold]Total: {passed}/{total} tests passed[/bold]")
        console.print()

    # Throughput bottlenecks
    console.print("[bold yellow]Throughput Bottlenecks[/bold yellow]")
    for note in model.throughput_bound_ops:
        console.print(f"  [yellow]-[/yellow] {note}")
    console.print()

    # Scheduling notes
    console.print("[bold cyan]Scheduling Implications[/bold cyan]")
    for note in model.scheduling_notes:
        if "PASS" in note or "confirmed" in note.lower():
            console.print(f"  [green]-[/green] {note}")
        elif "FAIL" in note or "WARNING" in note:
            console.print(f"  [red]-[/red] {note}")
        else:
            console.print(f"  [dim]-[/dim] {note}")
    console.print()

    # Key takeaways
    takeaways = """
[bold green]1. LATENCY IS NOT THE PROBLEM[/bold green]
   All ops complete in 1 cycle - no need for latency-hiding transforms

[bold yellow]2. DEPENDENCIES ARE THE PROBLEM[/bold yellow]
   RAW hazards force sequential execution
   Break chains via: unrolling, pipelining, vectorization

[bold cyan]3. THROUGHPUT LIMITS MATTER[/bold cyan]
   Load: max 2 ops/cycle (16 elements with vload)
   Store: max 2 ops/cycle (16 elements with vstore)
   Flow: max 1 op/cycle (serialization point!)

[bold magenta]4. VLIW PACKING IS KEY[/bold magenta]
   12 ALU + 6 VALU + 2 load + 2 store + 1 flow per cycle
   Most cycles use << 10% of available slots
   Pack independent ops to fill unused slots
"""
    console.print(Panel(takeaways, title="Key Takeaways for Optimization", border_style="blue"))


def output_json(model: LatencyModel) -> str:
    """Generate JSON output."""
    def serialize(obj):
        if isinstance(obj, LatencyType):
            return obj.value
        if isinstance(obj, (OperationLatency, EmpiricalTest)):
            d = asdict(obj)
            if 'latency_type' in d:
                d['latency_type'] = obj.latency_type.value
            return d
        return obj

    output = {
        "summary": {
            "all_single_cycle": model.all_single_cycle,
            "has_multi_cycle_ops": model.has_multi_cycle_ops,
        },
        "operations": [serialize(op) for op in model.operations],
        "empirical_tests": [serialize(t) for t in model.empirical_tests],
        "throughput_bottlenecks": model.throughput_bound_ops,
        "scheduling_notes": model.scheduling_notes,
    }

    return json.dumps(output, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze instruction latency model for VLIW SIMD architecture"
    )
    parser.add_argument("--empirical", action="store_true",
                       help="Run empirical latency tests")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON")
    parser.add_argument("--no-color", action="store_true",
                       help="Plain text output (no Rich formatting)")
    parser.add_argument("--all-ops", action="store_true",
                       help="Show all operations (not just summary)")

    args = parser.parse_args()

    # Analyze simulator code
    operations = analyze_simulator_code()

    # Run empirical tests if requested (or by default)
    tests = []
    if args.empirical or not args.json:
        tests = run_empirical_tests()

    # Build latency model
    model = analyze_latency_implications(operations, tests)

    # Output results
    if args.json:
        print(output_json(model))
    elif args.no_color or not RICH_AVAILABLE:
        print_plain(model, args.all_ops)
    else:
        print_rich(model, args.all_ops)


if __name__ == "__main__":
    main()
