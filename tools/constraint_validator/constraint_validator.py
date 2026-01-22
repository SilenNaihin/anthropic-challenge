#!/usr/bin/env python3
"""
Constraint Validator for VLIW SIMD Kernel

Static checking of kernel before runtime to catch errors early.

Features:
1. Slot limit validation (12 alu, 6 valu, 2 load, 2 store, 1 flow per cycle)
2. Scratch memory overflow detection
3. Same-cycle hazard detection (RAW within same cycle)
4. Register usage validation
5. Rich output support (with fallback to plain text)
6. JSON output option

Usage:
    python constraint_validator.py                    # Validate current kernel
    python constraint_validator.py --json             # Output as JSON
    python constraint_validator.py --strict           # Fail on warnings too
    python constraint_validator.py --kernel FILE      # Validate saved kernel JSON
"""

import sys
import os
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum
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
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class Severity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Will definitely cause runtime failure
    WARNING = "warning"  # May cause issues or is suboptimal
    INFO = "info"        # Informational message


@dataclass
class ValidationIssue:
    """A single validation issue found during checking."""
    severity: Severity
    category: str
    message: str
    cycle: Optional[int] = None
    engine: Optional[str] = None
    details: Optional[Dict] = None

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "cycle": self.cycle,
            "engine": self.engine,
            "details": self.details
        }


@dataclass
class ValidationResult:
    """Complete validation results."""
    issues: List[ValidationIssue] = field(default_factory=list)
    total_cycles: int = 0
    scratch_high_water: int = 0
    statistics: Dict = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.INFO)

    @property
    def is_valid(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict:
        return {
            "valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "total_cycles": self.total_cycles,
            "scratch_high_water": self.scratch_high_water,
            "scratch_limit": SCRATCH_SIZE,
            "issues": [i.to_dict() for i in self.issues],
            "statistics": self.statistics
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ConstraintValidator:
    """Validates VLIW SIMD kernel constraints."""

    # Engine slot limits (excluding debug)
    SLOT_LIMITS = {k: v for k, v in SLOT_LIMITS.items() if k != "debug"}

    def __init__(self, instructions: List[dict], scratch_size: int = SCRATCH_SIZE):
        self.instructions = instructions
        self.scratch_size = scratch_size
        self.result = ValidationResult()
        self.result.total_cycles = len(instructions)

    def validate_all(self) -> ValidationResult:
        """Run all validation checks."""
        self._validate_slot_limits()
        self._validate_scratch_usage()
        self._validate_same_cycle_hazards()
        self._validate_register_usage()
        self._compute_statistics()
        return self.result

    def _add_issue(self, severity: Severity, category: str, message: str,
                   cycle: Optional[int] = None, engine: Optional[str] = None,
                   details: Optional[Dict] = None):
        """Add a validation issue to results."""
        self.result.issues.append(ValidationIssue(
            severity=severity,
            category=category,
            message=message,
            cycle=cycle,
            engine=engine,
            details=details
        ))

    def _validate_slot_limits(self):
        """Check that no cycle exceeds slot limits for any engine."""
        for cycle_num, instr in enumerate(self.instructions):
            for engine, slots in instr.items():
                if engine == "debug":
                    continue

                slot_count = len(slots) if slots else 0
                limit = self.SLOT_LIMITS.get(engine, 0)

                if limit == 0:
                    self._add_issue(
                        Severity.ERROR,
                        "slot_limit",
                        f"Unknown engine '{engine}'",
                        cycle=cycle_num,
                        engine=engine
                    )
                elif slot_count > limit:
                    self._add_issue(
                        Severity.ERROR,
                        "slot_limit",
                        f"Slot limit exceeded: {slot_count} > {limit} for {engine}",
                        cycle=cycle_num,
                        engine=engine,
                        details={"used": slot_count, "limit": limit}
                    )

    def _validate_scratch_usage(self):
        """Check for scratch memory overflow."""
        max_addr = 0
        accessed_addrs: Set[int] = set()

        for cycle_num, instr in enumerate(self.instructions):
            for engine, slots in instr.items():
                if engine == "debug":
                    continue
                for slot in (slots or []):
                    reads, writes = self._extract_reads_writes(slot, engine)
                    all_addrs = reads | writes
                    accessed_addrs.update(all_addrs)

                    for addr in all_addrs:
                        if addr >= self.scratch_size:
                            self._add_issue(
                                Severity.ERROR,
                                "scratch_overflow",
                                f"Scratch address {addr} exceeds limit {self.scratch_size}",
                                cycle=cycle_num,
                                engine=engine,
                                details={"address": addr, "limit": self.scratch_size}
                            )
                        if addr < 0:
                            self._add_issue(
                                Severity.ERROR,
                                "scratch_overflow",
                                f"Negative scratch address {addr}",
                                cycle=cycle_num,
                                engine=engine,
                                details={"address": addr}
                            )
                        max_addr = max(max_addr, addr)

        self.result.scratch_high_water = max_addr + 1 if accessed_addrs else 0

        # Warning if close to limit
        usage_pct = 100.0 * self.result.scratch_high_water / self.scratch_size
        if usage_pct > 90:
            self._add_issue(
                Severity.WARNING,
                "scratch_overflow",
                f"Scratch usage at {usage_pct:.1f}% ({self.result.scratch_high_water}/{self.scratch_size})",
                details={"usage": self.result.scratch_high_water, "limit": self.scratch_size}
            )
        elif usage_pct > 75:
            self._add_issue(
                Severity.INFO,
                "scratch_overflow",
                f"Scratch usage at {usage_pct:.1f}% ({self.result.scratch_high_water}/{self.scratch_size})",
                details={"usage": self.result.scratch_high_water, "limit": self.scratch_size}
            )

    def _validate_same_cycle_hazards(self):
        """
        Check for same-cycle RAW hazards.

        In VLIW, all slots in a cycle execute "simultaneously" - reads happen
        before writes. However, if one slot writes to address X and another
        slot reads from X in the same cycle, the read sees the OLD value,
        which is often not the intended behavior.
        """
        for cycle_num, instr in enumerate(self.instructions):
            cycle_writes: Dict[int, Tuple[str, tuple]] = {}  # addr -> (engine, slot)
            cycle_reads: Dict[int, List[Tuple[str, tuple]]] = defaultdict(list)  # addr -> [(engine, slot), ...]

            # Collect all reads and writes in this cycle
            for engine, slots in instr.items():
                if engine == "debug":
                    continue
                for slot in (slots or []):
                    reads, writes = self._extract_reads_writes(slot, engine)

                    for addr in writes:
                        cycle_writes[addr] = (engine, slot)

                    for addr in reads:
                        cycle_reads[addr].append((engine, slot))

            # Check for RAW hazards: write and read to same address in same cycle
            for addr, (write_engine, write_slot) in cycle_writes.items():
                if addr in cycle_reads:
                    for read_engine, read_slot in cycle_reads[addr]:
                        # Same slot writing and reading is fine (it's the same instruction)
                        if write_slot == read_slot:
                            continue

                        self._add_issue(
                            Severity.WARNING,
                            "same_cycle_hazard",
                            f"Same-cycle RAW hazard at scratch[{addr}]: "
                            f"write by {write_engine} and read by {read_engine}",
                            cycle=cycle_num,
                            details={
                                "address": addr,
                                "writer": {"engine": write_engine, "slot": str(write_slot)},
                                "reader": {"engine": read_engine, "slot": str(read_slot)}
                            }
                        )

    def _validate_register_usage(self):
        """
        Validate register/scratch usage patterns.

        Checks:
        - Uninitialized reads (reading before any write)
        - Dead writes (writing without subsequent read)
        """
        # Track first write and last read for each address
        first_write: Dict[int, int] = {}  # addr -> cycle
        first_read: Dict[int, int] = {}   # addr -> cycle
        last_read: Dict[int, int] = {}    # addr -> cycle
        last_write: Dict[int, int] = {}   # addr -> cycle

        for cycle_num, instr in enumerate(self.instructions):
            for engine, slots in instr.items():
                if engine == "debug":
                    continue
                for slot in (slots or []):
                    reads, writes = self._extract_reads_writes(slot, engine)

                    for addr in reads:
                        if addr not in first_read:
                            first_read[addr] = cycle_num
                        last_read[addr] = cycle_num

                    for addr in writes:
                        if addr not in first_write:
                            first_write[addr] = cycle_num
                        last_write[addr] = cycle_num

        # Check for reads before writes (uninitialized reads)
        uninitialized_count = 0
        for addr, read_cycle in first_read.items():
            if addr not in first_write or first_write[addr] > read_cycle:
                uninitialized_count += 1
                # Only report first few to avoid noise
                if uninitialized_count <= 10:
                    self._add_issue(
                        Severity.WARNING,
                        "register_usage",
                        f"Read from scratch[{addr}] at cycle {read_cycle} before first write",
                        cycle=read_cycle,
                        details={"address": addr}
                    )

        if uninitialized_count > 10:
            self._add_issue(
                Severity.INFO,
                "register_usage",
                f"... and {uninitialized_count - 10} more uninitialized reads",
                details={"total_uninitialized": uninitialized_count}
            )

        # Dead writes are often intentional (initialization), so just count them
        dead_write_count = 0
        for addr, write_cycle in last_write.items():
            if addr not in last_read or last_read[addr] < write_cycle:
                dead_write_count += 1

        if dead_write_count > 0:
            self._add_issue(
                Severity.INFO,
                "register_usage",
                f"Found {dead_write_count} writes without subsequent reads (may be intentional)",
                details={"dead_write_count": dead_write_count}
            )

    def _extract_reads_writes(self, slot: tuple, engine: str) -> Tuple[Set[int], Set[int]]:
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

    def _compute_statistics(self):
        """Compute summary statistics."""
        engine_usage = defaultdict(int)
        total_slots = 0

        for instr in self.instructions:
            for engine, slots in instr.items():
                if engine == "debug":
                    continue
                count = len(slots) if slots else 0
                engine_usage[engine] += count
                total_slots += count

        max_slots = sum(self.SLOT_LIMITS.values())

        self.result.statistics = {
            "total_cycles": self.result.total_cycles,
            "total_slots_used": total_slots,
            "max_slots_possible": self.result.total_cycles * max_slots,
            "avg_slots_per_cycle": round(total_slots / max(1, self.result.total_cycles), 2),
            "engine_usage": dict(engine_usage),
            "slot_limits": dict(self.SLOT_LIMITS)
        }


# ============== Output Formatting ==============

class PlainPrinter:
    """Plain text output without Rich."""

    def print_result(self, result: ValidationResult, verbose: bool = False):
        print("=" * 70)
        print("CONSTRAINT VALIDATION RESULTS")
        print("=" * 70)
        print()

        # Summary
        status = "PASS" if result.is_valid else "FAIL"
        status_marker = "[OK]" if result.is_valid else "[ERROR]"
        print(f"Status: {status_marker} {status}")
        print(f"Errors: {result.error_count}")
        print(f"Warnings: {result.warning_count}")
        print(f"Info: {result.info_count}")
        print()

        # Statistics
        print("-" * 70)
        print("STATISTICS")
        print("-" * 70)
        stats = result.statistics
        print(f"Total Cycles: {stats.get('total_cycles', 0):,}")
        print(f"Total Slots Used: {stats.get('total_slots_used', 0):,}")
        print(f"Avg Slots/Cycle: {stats.get('avg_slots_per_cycle', 0):.2f}")
        print(f"Scratch High Water: {result.scratch_high_water} / {SCRATCH_SIZE}")
        print()

        # Issues
        if result.issues:
            print("-" * 70)
            print("ISSUES")
            print("-" * 70)

            # Group by severity
            errors = [i for i in result.issues if i.severity == Severity.ERROR]
            warnings = [i for i in result.issues if i.severity == Severity.WARNING]
            infos = [i for i in result.issues if i.severity == Severity.INFO]

            if errors:
                print(f"\n[ERRORS - {len(errors)}]")
                for issue in errors[:20]:  # Limit output
                    loc = f"cycle {issue.cycle}" if issue.cycle is not None else "global"
                    print(f"  [{issue.category}] {issue.message} ({loc})")
                if len(errors) > 20:
                    print(f"  ... and {len(errors) - 20} more errors")

            if warnings:
                print(f"\n[WARNINGS - {len(warnings)}]")
                for issue in warnings[:10]:
                    loc = f"cycle {issue.cycle}" if issue.cycle is not None else "global"
                    print(f"  [{issue.category}] {issue.message} ({loc})")
                if len(warnings) > 10:
                    print(f"  ... and {len(warnings) - 10} more warnings")

            if infos and verbose:
                print(f"\n[INFO - {len(infos)}]")
                for issue in infos[:5]:
                    print(f"  [{issue.category}] {issue.message}")
                if len(infos) > 5:
                    print(f"  ... and {len(infos) - 5} more info messages")

        else:
            print("[OK] No issues found!")

        print()


class RichPrinter:
    """Rich colored output."""

    def __init__(self):
        self.console = Console()

    def print_result(self, result: ValidationResult, verbose: bool = False):
        # Header
        self.console.print(Panel("CONSTRAINT VALIDATION RESULTS", style="bold cyan", box=box.DOUBLE))

        # Status
        if result.is_valid:
            status_panel = Panel(
                "[bold green]PASS[/bold green] - All constraints satisfied",
                style="green",
                box=box.ROUNDED
            )
        else:
            status_panel = Panel(
                f"[bold red]FAIL[/bold red] - {result.error_count} error(s) found",
                style="red",
                box=box.ROUNDED
            )
        self.console.print(status_panel)

        # Summary table
        summary_table = Table(show_header=False, box=box.SIMPLE)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")

        error_style = "red" if result.error_count > 0 else "green"
        warning_style = "yellow" if result.warning_count > 0 else "green"

        summary_table.add_row("Errors", f"[{error_style}]{result.error_count}[/{error_style}]")
        summary_table.add_row("Warnings", f"[{warning_style}]{result.warning_count}[/{warning_style}]")
        summary_table.add_row("Info", f"{result.info_count}")
        summary_table.add_row("Total Cycles", f"{result.total_cycles:,}")

        scratch_pct = 100.0 * result.scratch_high_water / SCRATCH_SIZE
        scratch_style = "red" if scratch_pct > 90 else "yellow" if scratch_pct > 75 else "green"
        summary_table.add_row(
            "Scratch Usage",
            f"[{scratch_style}]{result.scratch_high_water} / {SCRATCH_SIZE} ({scratch_pct:.1f}%)[/{scratch_style}]"
        )

        self.console.print(summary_table)

        # Issues
        if result.issues:
            self.console.print("\n[bold yellow]Issues Found:[/bold yellow]")

            # Group by severity
            errors = [i for i in result.issues if i.severity == Severity.ERROR]
            warnings = [i for i in result.issues if i.severity == Severity.WARNING]
            infos = [i for i in result.issues if i.severity == Severity.INFO]

            if errors:
                self.console.print(f"\n[bold red]ERRORS ({len(errors)})[/bold red]")
                error_table = Table(box=box.ROUNDED)
                error_table.add_column("Cycle", justify="right", style="cyan")
                error_table.add_column("Category", style="yellow")
                error_table.add_column("Message", style="white")

                for issue in errors[:20]:
                    cycle_str = str(issue.cycle) if issue.cycle is not None else "-"
                    error_table.add_row(cycle_str, issue.category, issue.message)

                self.console.print(error_table)
                if len(errors) > 20:
                    self.console.print(f"  [dim]... and {len(errors) - 20} more errors[/dim]")

            if warnings:
                self.console.print(f"\n[bold yellow]WARNINGS ({len(warnings)})[/bold yellow]")
                warning_table = Table(box=box.ROUNDED)
                warning_table.add_column("Cycle", justify="right", style="cyan")
                warning_table.add_column("Category", style="yellow")
                warning_table.add_column("Message", style="white")

                for issue in warnings[:10]:
                    cycle_str = str(issue.cycle) if issue.cycle is not None else "-"
                    warning_table.add_row(cycle_str, issue.category, issue.message)

                self.console.print(warning_table)
                if len(warnings) > 10:
                    self.console.print(f"  [dim]... and {len(warnings) - 10} more warnings[/dim]")

            if infos and verbose:
                self.console.print(f"\n[bold blue]INFO ({len(infos)})[/bold blue]")
                for issue in infos[:5]:
                    self.console.print(f"  [dim]{issue.message}[/dim]")
                if len(infos) > 5:
                    self.console.print(f"  [dim]... and {len(infos) - 5} more[/dim]")

        else:
            self.console.print("\n[bold green]All constraints satisfied![/bold green]")


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
    # Import here to avoid circular dependencies
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


def validate_kernel(instructions: List[dict], scratch_size: int = SCRATCH_SIZE) -> ValidationResult:
    """
    Main validation function - can be imported and used by other tools.

    Args:
        instructions: List of instruction bundles
        scratch_size: Maximum scratch memory size

    Returns:
        ValidationResult with all issues found
    """
    validator = ConstraintValidator(instructions, scratch_size)
    return validator.validate_all()


def main():
    parser = argparse.ArgumentParser(
        description="Validate VLIW SIMD kernel constraints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python constraint_validator.py                    # Validate current kernel
    python constraint_validator.py --json             # Output as JSON
    python constraint_validator.py --strict           # Exit non-zero on warnings
    python constraint_validator.py --kernel FILE      # Validate saved kernel
        """
    )
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--strict", "-s", action="store_true",
                        help="Treat warnings as errors (exit non-zero)")
    parser.add_argument("--kernel", "-k", metavar="FILE",
                        help="Load kernel from JSON file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show all issues including info messages")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")

    args = parser.parse_args()

    # Load kernel
    print("Loading kernel...", file=sys.stderr)
    if args.kernel:
        instructions = load_kernel_from_file(args.kernel)
    else:
        instructions = load_kernel_from_builder()

    print(f"Validating {len(instructions)} cycles...", file=sys.stderr)

    # Validate
    result = validate_kernel(instructions)

    # Output
    if args.json:
        print(result.to_json())
    else:
        printer = PlainPrinter() if args.no_color else get_printer()
        printer.print_result(result, verbose=args.verbose)

    # Exit code
    if result.error_count > 0:
        sys.exit(1)
    elif args.strict and result.warning_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
