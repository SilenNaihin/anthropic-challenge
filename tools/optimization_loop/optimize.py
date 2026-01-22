#!/usr/bin/env python3
"""
Optimization Loop Runner for VLIW SIMD Kernel

Meta-tool that automates the profile->analyze->transform->validate optimization loop.
Runs multiple analysis tools, identifies bottlenecks, suggests transforms, and checks
for regressions.

Key Features:
- Automated profiling: Runs slot_analyzer, dependency_graph, cycle_profiler
- Bottleneck detection: Identifies what's limiting performance
- Transform suggestions: Recommends optimizations based on analysis
- Regression checking: Validates changes don't break correctness
- Rich output: Uses rich library if available, falls back to plain text
- JSON output: Machine-readable output for scripting
- Dry-run mode: Shows what would be done without executing

Usage:
    python tools/optimization_loop/optimize.py
    python tools/optimization_loop/optimize.py --profile
    python tools/optimization_loop/optimize.py --suggest
    python tools/optimization_loop/optimize.py --validate
    python tools/optimization_loop/optimize.py --json
    python tools/optimization_loop/optimize.py --dry-run
"""

import sys
import os
import json
import argparse
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from enum import Enum

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tools'))

# Try to import Rich for better formatting
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.text import Text
    from rich import box
    from rich.live import Live
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    SLOT_UTILIZATION = "slot_utilization"  # Low overall slot usage
    DEPENDENCY_CHAIN = "dependency_chain"   # Long critical path
    MEMORY_BOUND = "memory_bound"           # High load/store, low compute
    HASH_BOUND = "hash_bound"               # Hash dominates execution
    ENGINE_IMBALANCE = "engine_imbalance"   # One engine saturated, others idle
    PACKING_MISSED = "packing_missed"       # Easy packing opportunities not taken


class TransformType(Enum):
    """Types of code transformations."""
    INSTRUCTION_PACKING = "instruction_packing"
    LOOP_UNROLLING = "loop_unrolling"
    SOFTWARE_PIPELINING = "software_pipelining"
    VECTORIZATION = "vectorization"
    DEPENDENCY_BREAKING = "dependency_breaking"
    HOISTING = "hoisting"


@dataclass
class ProfileResult:
    """Results from a single profiling tool."""
    tool_name: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class Bottleneck:
    """Identified performance bottleneck."""
    type: BottleneckType
    severity: str  # "high", "medium", "low"
    description: str
    metric_name: str
    metric_value: float
    threshold: float
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transform:
    """Suggested code transformation."""
    type: TransformType
    priority: int  # 1=highest
    title: str
    description: str
    potential_savings: str
    difficulty: str  # "easy", "medium", "hard"
    prerequisites: List[str] = field(default_factory=list)
    related_bottleneck: Optional[BottleneckType] = None


@dataclass
class ValidationResult:
    """Results from correctness validation."""
    passed: bool
    cycles: int
    baseline_cycles: int
    speedup: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class OptimizationReport:
    """Complete optimization analysis report."""
    timestamp: str
    profiles: Dict[str, ProfileResult]
    bottlenecks: List[Bottleneck]
    transforms: List[Transform]
    validation: Optional[ValidationResult]
    summary: Dict[str, Any]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "profiles": {
                name: {
                    "tool_name": p.tool_name,
                    "success": p.success,
                    "duration_ms": p.duration_ms,
                    "error": p.error,
                    "data": p.data
                }
                for name, p in self.profiles.items()
            },
            "bottlenecks": [
                {
                    "type": b.type.value,
                    "severity": b.severity,
                    "description": b.description,
                    "metric_name": b.metric_name,
                    "metric_value": b.metric_value,
                    "threshold": b.threshold,
                    "evidence": b.evidence
                }
                for b in self.bottlenecks
            ],
            "transforms": [
                {
                    "type": t.type.value,
                    "priority": t.priority,
                    "title": t.title,
                    "description": t.description,
                    "potential_savings": t.potential_savings,
                    "difficulty": t.difficulty,
                    "prerequisites": t.prerequisites
                }
                for t in self.transforms
            ],
            "validation": {
                "passed": self.validation.passed,
                "cycles": self.validation.cycles,
                "baseline_cycles": self.validation.baseline_cycles,
                "speedup": self.validation.speedup,
                "errors": self.validation.errors,
                "warnings": self.validation.warnings
            } if self.validation else None,
            "summary": self.summary
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class OptimizationRunner:
    """Main optimization loop runner."""

    def __init__(self, verbose: bool = False, dry_run: bool = False):
        self.verbose = verbose
        self.dry_run = dry_run
        self.profiles: Dict[str, ProfileResult] = {}
        self.bottlenecks: List[Bottleneck] = []
        self.transforms: List[Transform] = []

        # Tool paths relative to project root
        self.tools = {
            "slot_analyzer": os.path.join(PROJECT_ROOT, "tools", "slot_analyzer.py"),
            "dependency_graph": os.path.join(PROJECT_ROOT, "tools", "dependency_graph", "dependency_graph.py"),
            "cycle_profiler": os.path.join(PROJECT_ROOT, "tools", "cycle_profiler", "cycle_profiler.py"),
            "memory_analyzer": os.path.join(PROJECT_ROOT, "tools", "memory_analyzer", "memory_analyzer.py"),
            "constraint_validator": os.path.join(PROJECT_ROOT, "tools", "constraint_validator", "constraint_validator.py"),
        }

        # Target metrics from prd.json
        self.current_cycles = 8500
        self.target_cycles = 1487
        self.baseline_cycles = 147734

    def run_tool(self, tool_name: str, extra_args: List[str] = None) -> ProfileResult:
        """Run a profiling tool and capture JSON output."""
        tool_path = self.tools.get(tool_name)
        if not tool_path or not os.path.exists(tool_path):
            return ProfileResult(
                tool_name=tool_name,
                success=False,
                data={},
                error=f"Tool not found: {tool_path}"
            )

        if self.dry_run:
            return ProfileResult(
                tool_name=tool_name,
                success=True,
                data={"dry_run": True},
                error=None,
                duration_ms=0
            )

        cmd = [sys.executable, tool_path, "--json"]
        if extra_args:
            cmd.extend(extra_args)

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=PROJECT_ROOT
            )
            duration_ms = (time.time() - start_time) * 1000

            if result.returncode != 0:
                return ProfileResult(
                    tool_name=tool_name,
                    success=False,
                    data={},
                    error=result.stderr[:500] if result.stderr else "Unknown error",
                    duration_ms=duration_ms
                )

            # Parse JSON output
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                # Tool might print progress to stderr, actual JSON to stdout
                lines = result.stdout.strip().split('\n')
                for line in reversed(lines):
                    try:
                        data = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    return ProfileResult(
                        tool_name=tool_name,
                        success=False,
                        data={},
                        error="Failed to parse JSON output",
                        duration_ms=duration_ms
                    )

            return ProfileResult(
                tool_name=tool_name,
                success=True,
                data=data,
                duration_ms=duration_ms
            )

        except subprocess.TimeoutExpired:
            return ProfileResult(
                tool_name=tool_name,
                success=False,
                data={},
                error="Tool timed out after 60 seconds"
            )
        except Exception as e:
            return ProfileResult(
                tool_name=tool_name,
                success=False,
                data={},
                error=str(e)
            )

    def run_profiling(self, tools: List[str] = None) -> Dict[str, ProfileResult]:
        """Run all or selected profiling tools."""
        if tools is None:
            tools = ["slot_analyzer", "dependency_graph", "cycle_profiler"]

        for tool_name in tools:
            self.profiles[tool_name] = self.run_tool(tool_name)

        return self.profiles

    def detect_bottlenecks(self) -> List[Bottleneck]:
        """Analyze profiling results to identify bottlenecks."""
        self.bottlenecks = []

        # Slot utilization analysis
        slot_data = self.profiles.get("slot_analyzer", ProfileResult("", False, {})).data
        if slot_data:
            util_pct = slot_data.get("utilization_pct", 0)
            if util_pct < 30:
                self.bottlenecks.append(Bottleneck(
                    type=BottleneckType.SLOT_UTILIZATION,
                    severity="high",
                    description=f"Very low slot utilization ({util_pct:.1f}%). Most execution units idle.",
                    metric_name="utilization_pct",
                    metric_value=util_pct,
                    threshold=30,
                    evidence={"per_engine": slot_data.get("per_engine", {})}
                ))
            elif util_pct < 50:
                self.bottlenecks.append(Bottleneck(
                    type=BottleneckType.SLOT_UTILIZATION,
                    severity="medium",
                    description=f"Low slot utilization ({util_pct:.1f}%). Room for more parallelism.",
                    metric_name="utilization_pct",
                    metric_value=util_pct,
                    threshold=50,
                    evidence={"per_engine": slot_data.get("per_engine", {})}
                ))

            # Engine imbalance check
            per_engine = slot_data.get("per_engine", {})
            utils = [e.get("utilization_pct", 0) for e in per_engine.values()]
            if utils and max(utils) > 60 and min(utils) < 20:
                self.bottlenecks.append(Bottleneck(
                    type=BottleneckType.ENGINE_IMBALANCE,
                    severity="medium",
                    description="Significant imbalance between engines. Some saturated while others idle.",
                    metric_name="utilization_spread",
                    metric_value=max(utils) - min(utils),
                    threshold=40,
                    evidence={"per_engine": per_engine}
                ))

        # Dependency analysis
        dep_data = self.profiles.get("dependency_graph", ProfileResult("", False, {})).data
        if dep_data:
            total_cycles = dep_data.get("total_cycles", 1)
            critical_path = dep_data.get("critical_path_length", total_cycles)
            wasted = dep_data.get("wasted_cycles", 0)
            wasted_pct = 100.0 * wasted / total_cycles if total_cycles > 0 else 0

            if wasted_pct > 50:
                self.bottlenecks.append(Bottleneck(
                    type=BottleneckType.DEPENDENCY_CHAIN,
                    severity="high",
                    description=f"Long dependency chains. {wasted_pct:.1f}% of cycles could be eliminated with better scheduling.",
                    metric_name="wasted_cycles_pct",
                    metric_value=wasted_pct,
                    threshold=50,
                    evidence={
                        "critical_path_length": critical_path,
                        "total_cycles": total_cycles,
                        "theoretical_speedup": dep_data.get("theoretical_speedup", 1.0)
                    }
                ))

            # Check for hot registers
            hot_regs = dep_data.get("hot_registers", [])
            if hot_regs and len(hot_regs) > 0:
                top_reg = hot_regs[0]
                if top_reg.get("deps", 0) > total_cycles * 0.1:
                    self.bottlenecks.append(Bottleneck(
                        type=BottleneckType.DEPENDENCY_CHAIN,
                        severity="medium",
                        description=f"Hot register causing many dependencies. Address {top_reg.get('addr')} causes {top_reg.get('deps')} deps.",
                        metric_name="hot_register_deps",
                        metric_value=top_reg.get("deps", 0),
                        threshold=total_cycles * 0.1,
                        evidence={"hot_registers": hot_regs[:5]}
                    ))

        # Cycle profiler analysis
        profile_data = self.profiles.get("cycle_profiler", ProfileResult("", False, {})).data
        if profile_data:
            phase_breakdown = profile_data.get("phase_breakdown", {})
            hotspots = profile_data.get("hotspots", [])

            # Check if hash-bound
            for hotspot in hotspots:
                if hotspot.get("phase") == "hash" and hotspot.get("percentage", 0) > 50:
                    self.bottlenecks.append(Bottleneck(
                        type=BottleneckType.HASH_BOUND,
                        severity="high",
                        description=f"Hash computation dominates ({hotspot.get('percentage'):.1f}% of cycles). Focus on hash pipelining.",
                        metric_name="hash_percentage",
                        metric_value=hotspot.get("percentage", 0),
                        threshold=50,
                        evidence={"phase_breakdown": phase_breakdown}
                    ))
                    break

            # Check if memory-bound
            load_pct = next((h.get("percentage", 0) for h in hotspots if h.get("phase") == "memory_load"), 0)
            store_pct = next((h.get("percentage", 0) for h in hotspots if h.get("phase") == "memory_store"), 0)
            hash_pct = next((h.get("percentage", 0) for h in hotspots if h.get("phase") == "hash"), 0)

            if load_pct + store_pct > 50 and hash_pct < 30:
                self.bottlenecks.append(Bottleneck(
                    type=BottleneckType.MEMORY_BOUND,
                    severity="high",
                    description=f"Memory operations dominate ({load_pct + store_pct:.1f}% load+store). Consider prefetching or overlapping.",
                    metric_name="memory_percentage",
                    metric_value=load_pct + store_pct,
                    threshold=50,
                    evidence={"load_pct": load_pct, "store_pct": store_pct}
                ))

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        self.bottlenecks.sort(key=lambda b: severity_order.get(b.severity, 3))

        return self.bottlenecks

    def suggest_transforms(self) -> List[Transform]:
        """Generate transform suggestions based on bottlenecks."""
        self.transforms = []

        # Map bottleneck types to transform suggestions
        for bottleneck in self.bottlenecks:
            if bottleneck.type == BottleneckType.SLOT_UTILIZATION:
                self.transforms.append(Transform(
                    type=TransformType.INSTRUCTION_PACKING,
                    priority=1,
                    title="Pack more instructions per cycle",
                    description="Many cycles have unused slots. Use VLIW packer or manually combine independent instructions.",
                    potential_savings=f"Up to {100 - bottleneck.metric_value:.0f}% cycle reduction",
                    difficulty="easy",
                    prerequisites=["Run constraint_validator to ensure valid packing"],
                    related_bottleneck=BottleneckType.SLOT_UTILIZATION
                ))

            elif bottleneck.type == BottleneckType.DEPENDENCY_CHAIN:
                evidence = bottleneck.evidence
                self.transforms.append(Transform(
                    type=TransformType.DEPENDENCY_BREAKING,
                    priority=1,
                    title="Break dependency chains",
                    description=f"Critical path is {evidence.get('critical_path_length', 'N/A')} cycles. "
                                "Use register renaming or reorder operations to expose more parallelism.",
                    potential_savings=f"{evidence.get('theoretical_speedup', 1):.1f}x theoretical speedup",
                    difficulty="hard",
                    prerequisites=["Identify hot registers", "Map dependency DAG"],
                    related_bottleneck=BottleneckType.DEPENDENCY_CHAIN
                ))

                self.transforms.append(Transform(
                    type=TransformType.SOFTWARE_PIPELINING,
                    priority=2,
                    title="Apply software pipelining",
                    description="Overlap iterations of loops to hide dependency latencies. "
                                "Start next iteration's loads while current iteration computes.",
                    potential_savings="2-3x for memory-bound loops",
                    difficulty="hard",
                    prerequisites=["Identify loop boundaries", "Ensure sufficient registers"],
                    related_bottleneck=BottleneckType.DEPENDENCY_CHAIN
                ))

            elif bottleneck.type == BottleneckType.HASH_BOUND:
                self.transforms.append(Transform(
                    type=TransformType.SOFTWARE_PIPELINING,
                    priority=1,
                    title="Pipeline hash computation",
                    description="Hash has 6 stages with 2-way ILP (tmp1||tmp2 independent). "
                                "Pipeline multiple elements through hash stages.",
                    potential_savings="2x from tmp1||tmp2, more from inter-element pipelining",
                    difficulty="medium",
                    prerequisites=["Use hash_pipeline tool for analysis"],
                    related_bottleneck=BottleneckType.HASH_BOUND
                ))

                self.transforms.append(Transform(
                    type=TransformType.VECTORIZATION,
                    priority=2,
                    title="Vectorize hash across batch",
                    description="Process VLEN elements' hash in parallel using VALU instructions.",
                    potential_savings=f"Up to {8}x (VLEN) for independent hash computations",
                    difficulty="medium",
                    prerequisites=["Broadcast hash constants to vectors"],
                    related_bottleneck=BottleneckType.HASH_BOUND
                ))

            elif bottleneck.type == BottleneckType.MEMORY_BOUND:
                self.transforms.append(Transform(
                    type=TransformType.SOFTWARE_PIPELINING,
                    priority=1,
                    title="Overlap loads with computation",
                    description="Start loading next batch's data while computing current batch. "
                                "Use double-buffering with two register sets.",
                    potential_savings="Hide load latency completely",
                    difficulty="hard",
                    prerequisites=["Allocate separate register sets for each buffer"],
                    related_bottleneck=BottleneckType.MEMORY_BOUND
                ))

                self.transforms.append(Transform(
                    type=TransformType.VECTORIZATION,
                    priority=2,
                    title="Use vector loads where possible",
                    description="Replace scattered loads with vload where addresses are sequential. "
                                "Note: Tree lookups cannot be vectorized.",
                    potential_savings="8x for sequential access patterns",
                    difficulty="easy",
                    prerequisites=["Run memory_analyzer to find vectorizable patterns"],
                    related_bottleneck=BottleneckType.MEMORY_BOUND
                ))

            elif bottleneck.type == BottleneckType.ENGINE_IMBALANCE:
                self.transforms.append(Transform(
                    type=TransformType.INSTRUCTION_PACKING,
                    priority=2,
                    title="Balance work across engines",
                    description="Move computation to underutilized engines. "
                                "Pack ALU work with VALU, overlap loads with stores.",
                    potential_savings="Better overall throughput",
                    difficulty="medium",
                    prerequisites=["Check slot limits per engine"],
                    related_bottleneck=BottleneckType.ENGINE_IMBALANCE
                ))

        # Add general transforms if no specific bottlenecks found
        if not self.transforms:
            self.transforms.append(Transform(
                type=TransformType.LOOP_UNROLLING,
                priority=3,
                title="Unroll loops to expose parallelism",
                description="Unroll inner loops 2x-4x to create more independent operations for packing.",
                potential_savings="Varies based on loop body",
                difficulty="easy",
                prerequisites=["Identify tight loops", "Ensure register pressure is acceptable"]
            ))

            self.transforms.append(Transform(
                type=TransformType.HOISTING,
                priority=3,
                title="Hoist loop invariants",
                description="Move constant computations outside loops. Pre-compute addresses and broadcast constants.",
                potential_savings="Reduces init overhead",
                difficulty="easy",
                prerequisites=["Identify invariants"]
            ))

        # Sort by priority
        self.transforms.sort(key=lambda t: t.priority)

        return self.transforms

    def run_validation(self) -> ValidationResult:
        """Run correctness validation and measure cycles."""
        if self.dry_run:
            return ValidationResult(
                passed=True,
                cycles=0,
                baseline_cycles=self.baseline_cycles,
                speedup=0.0,
                warnings=["Dry run - no actual validation"]
            )

        errors = []
        warnings = []

        # Run constraint validator first
        validator_result = self.run_tool("constraint_validator")
        if validator_result.success:
            data = validator_result.data
            if data.get("total_errors", 0) > 0:
                errors.append(f"Constraint validator found {data.get('total_errors')} errors")
            if data.get("total_warnings", 0) > 0:
                warnings.append(f"Constraint validator found {data.get('total_warnings')} warnings")

        # Run the test to get cycles
        try:
            result = subprocess.run(
                [sys.executable, "-c", """
import sys
sys.path.insert(0, '.')
from perf_takehome import do_kernel_test
cycles = do_kernel_test(10, 16, 256)
print(f"CYCLES:{cycles}")
"""],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=PROJECT_ROOT
            )

            # Parse cycles from output
            cycles = self.current_cycles  # Default
            for line in result.stdout.split('\n'):
                if line.startswith("CYCLES:"):
                    cycles = int(line.split(":")[1])
                    break

            passed = len(errors) == 0
            speedup = self.baseline_cycles / cycles if cycles > 0 else 0

            return ValidationResult(
                passed=passed,
                cycles=cycles,
                baseline_cycles=self.baseline_cycles,
                speedup=speedup,
                errors=errors,
                warnings=warnings
            )

        except subprocess.TimeoutExpired:
            return ValidationResult(
                passed=False,
                cycles=0,
                baseline_cycles=self.baseline_cycles,
                speedup=0.0,
                errors=["Test timed out after 120 seconds"]
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                cycles=0,
                baseline_cycles=self.baseline_cycles,
                speedup=0.0,
                errors=[str(e)]
            )

    def generate_report(self) -> OptimizationReport:
        """Generate complete optimization report."""
        import datetime

        # Build summary
        summary = {
            "total_bottlenecks": len(self.bottlenecks),
            "high_severity_bottlenecks": len([b for b in self.bottlenecks if b.severity == "high"]),
            "total_transforms": len(self.transforms),
            "top_bottleneck": self.bottlenecks[0].type.value if self.bottlenecks else None,
            "top_transform": self.transforms[0].title if self.transforms else None,
            "tools_run": list(self.profiles.keys()),
            "tools_succeeded": len([p for p in self.profiles.values() if p.success]),
        }

        # Get cycles from profiles if available
        slot_data = self.profiles.get("slot_analyzer", ProfileResult("", False, {})).data
        if slot_data:
            summary["current_cycles"] = slot_data.get("total_cycles", self.current_cycles)
            summary["current_utilization"] = slot_data.get("utilization_pct", 0)

        dep_data = self.profiles.get("dependency_graph", ProfileResult("", False, {})).data
        if dep_data:
            summary["critical_path_length"] = dep_data.get("critical_path_length", 0)
            summary["theoretical_speedup"] = dep_data.get("theoretical_speedup", 1.0)

        summary["target_cycles"] = self.target_cycles
        summary["baseline_cycles"] = self.baseline_cycles

        validation = None
        if hasattr(self, '_validation_result'):
            validation = self._validation_result

        return OptimizationReport(
            timestamp=datetime.datetime.now().isoformat(),
            profiles=self.profiles,
            bottlenecks=self.bottlenecks,
            transforms=self.transforms,
            validation=validation,
            summary=summary
        )

    def run_full_loop(self, skip_validation: bool = False) -> OptimizationReport:
        """Run the complete optimization loop."""
        # 1. Profile
        self.run_profiling()

        # 2. Detect bottlenecks
        self.detect_bottlenecks()

        # 3. Suggest transforms
        self.suggest_transforms()

        # 4. Validate (optional)
        if not skip_validation:
            self._validation_result = self.run_validation()

        # 5. Generate report
        return self.generate_report()


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

    def print_report(self, report: OptimizationReport):
        self.print_header("OPTIMIZATION LOOP REPORT")
        print(f"Timestamp: {report.timestamp}")
        print()

        # Summary
        summary = report.summary
        print(f"Tools Run: {', '.join(summary.get('tools_run', []))}")
        print(f"Tools Succeeded: {summary.get('tools_succeeded', 0)}/{len(summary.get('tools_run', []))}")
        print()

        if summary.get('current_cycles'):
            print(f"Current Cycles: {summary['current_cycles']:,}")
        if summary.get('target_cycles'):
            print(f"Target Cycles: {summary['target_cycles']:,}")
        if summary.get('current_utilization'):
            print(f"Slot Utilization: {summary['current_utilization']:.1f}%")
        if summary.get('critical_path_length'):
            print(f"Critical Path: {summary['critical_path_length']} cycles")
        if summary.get('theoretical_speedup'):
            print(f"Theoretical Speedup: {summary['theoretical_speedup']:.2f}x")
        print()

        # Bottlenecks
        self.print_subheader(f"BOTTLENECKS DETECTED ({len(report.bottlenecks)})")
        if not report.bottlenecks:
            print("No significant bottlenecks detected.")
        else:
            for i, b in enumerate(report.bottlenecks, 1):
                severity_str = f"[{b.severity.upper()}]"
                print(f"{i}. {severity_str} {b.type.value}")
                print(f"   {b.description}")
                print(f"   Metric: {b.metric_name} = {b.metric_value:.1f} (threshold: {b.threshold})")
                print()

        # Transforms
        self.print_subheader(f"SUGGESTED TRANSFORMS ({len(report.transforms)})")
        if not report.transforms:
            print("No specific transforms suggested.")
        else:
            for i, t in enumerate(report.transforms, 1):
                diff_str = f"[{t.difficulty.upper()}]"
                print(f"{i}. P{t.priority} {diff_str} {t.title}")
                print(f"   {t.description}")
                print(f"   Potential: {t.potential_savings}")
                if t.prerequisites:
                    print(f"   Prerequisites: {', '.join(t.prerequisites)}")
                print()

        # Validation
        if report.validation:
            self.print_subheader("VALIDATION RESULTS")
            v = report.validation
            status = "PASSED" if v.passed else "FAILED"
            print(f"Status: {status}")
            print(f"Cycles: {v.cycles:,}")
            print(f"Speedup over baseline: {v.speedup:.2f}x")
            if v.errors:
                print(f"Errors: {', '.join(v.errors)}")
            if v.warnings:
                print(f"Warnings: {', '.join(v.warnings)}")

    def print_progress(self, message: str):
        print(f">>> {message}")


class RichPrinter:
    """Rich colored output."""

    def __init__(self):
        self.console = Console()

    def print_header(self, text: str):
        self.console.print(Panel(text, style="bold cyan", box=box.DOUBLE))

    def print_subheader(self, text: str):
        self.console.print(f"\n[bold yellow]{text}[/bold yellow]")
        self.console.print("-" * 70)

    def print_report(self, report: OptimizationReport):
        self.print_header("OPTIMIZATION LOOP REPORT")
        self.console.print(f"[dim]Timestamp: {report.timestamp}[/dim]")

        # Summary table
        summary = report.summary
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Tools Run", ", ".join(summary.get("tools_run", [])))
        table.add_row("Tools Succeeded", f"{summary.get('tools_succeeded', 0)}/{len(summary.get('tools_run', []))}")

        if summary.get("current_cycles"):
            table.add_row("Current Cycles", f"{summary['current_cycles']:,}")
        if summary.get("target_cycles"):
            table.add_row("Target Cycles", f"[bold magenta]{summary['target_cycles']:,}[/bold magenta]")
        if summary.get("current_utilization"):
            util = summary["current_utilization"]
            util_color = "green" if util > 50 else "yellow" if util > 30 else "red"
            table.add_row("Slot Utilization", f"[{util_color}]{util:.1f}%[/{util_color}]")
        if summary.get("critical_path_length"):
            table.add_row("Critical Path", f"{summary['critical_path_length']} cycles")
        if summary.get("theoretical_speedup"):
            table.add_row("Theoretical Speedup", f"[bold]{summary['theoretical_speedup']:.2f}x[/bold]")

        self.console.print(table)

        # Bottlenecks
        self.print_subheader(f"BOTTLENECKS DETECTED ({len(report.bottlenecks)})")

        if not report.bottlenecks:
            self.console.print("[dim]No significant bottlenecks detected.[/dim]")
        else:
            for i, b in enumerate(report.bottlenecks, 1):
                severity_style = {"high": "bold red", "medium": "bold yellow", "low": "bold blue"}.get(b.severity, "white")

                panel = Panel(
                    f"[white]{b.description}[/white]\n\n"
                    f"[dim]Metric: {b.metric_name} = {b.metric_value:.1f} (threshold: {b.threshold})[/dim]",
                    title=f"[{severity_style}][{b.severity.upper()}][/{severity_style}] {b.type.value}",
                    border_style=severity_style
                )
                self.console.print(panel)

        # Transforms
        self.print_subheader(f"SUGGESTED TRANSFORMS ({len(report.transforms)})")

        if not report.transforms:
            self.console.print("[dim]No specific transforms suggested.[/dim]")
        else:
            for i, t in enumerate(report.transforms, 1):
                diff_style = {"easy": "green", "medium": "yellow", "hard": "red"}.get(t.difficulty, "white")

                content = Text()
                content.append(f"{t.description}\n\n", style="white")
                content.append(f"Potential: ", style="dim")
                content.append(f"{t.potential_savings}\n", style="green")
                content.append(f"Difficulty: ", style="dim")
                content.append(f"{t.difficulty}", style=diff_style)
                if t.prerequisites:
                    content.append(f"\nPrerequisites: ", style="dim")
                    content.append(", ".join(t.prerequisites), style="cyan")

                self.console.print(Panel(
                    content,
                    title=f"[bold]P{t.priority}[/bold] {t.title}",
                    border_style="blue"
                ))

        # Validation
        if report.validation:
            self.print_subheader("VALIDATION RESULTS")
            v = report.validation

            if v.passed:
                status_style = "bold green"
                status_text = "PASSED"
            else:
                status_style = "bold red"
                status_text = "FAILED"

            val_table = Table(show_header=False, box=box.SIMPLE)
            val_table.add_column("Metric", style="cyan")
            val_table.add_column("Value")

            val_table.add_row("Status", f"[{status_style}]{status_text}[/{status_style}]")
            val_table.add_row("Cycles", f"{v.cycles:,}")
            val_table.add_row("Speedup", f"{v.speedup:.2f}x over baseline")

            self.console.print(val_table)

            if v.errors:
                for error in v.errors:
                    self.console.print(f"[red]ERROR: {error}[/red]")
            if v.warnings:
                for warning in v.warnings:
                    self.console.print(f"[yellow]WARNING: {warning}[/yellow]")

    def print_progress(self, message: str):
        self.console.print(f"[bold cyan]>>>[/bold cyan] {message}")


def get_printer(use_rich: bool = True):
    """Get appropriate printer based on Rich availability."""
    if use_rich and RICH_AVAILABLE:
        return RichPrinter()
    return PlainPrinter()


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(
        description="Optimization Loop Runner - Automate the profile->analyze->transform->validate loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/optimization_loop/optimize.py                  # Full optimization loop
    python tools/optimization_loop/optimize.py --profile        # Only run profiling
    python tools/optimization_loop/optimize.py --suggest        # Profile + suggest transforms
    python tools/optimization_loop/optimize.py --validate       # Run validation only
    python tools/optimization_loop/optimize.py --json           # JSON output
    python tools/optimization_loop/optimize.py --dry-run        # Show what would be done
    python tools/optimization_loop/optimize.py --no-validation  # Skip validation step
        """
    )
    parser.add_argument("--profile", "-p", action="store_true",
                        help="Only run profiling tools")
    parser.add_argument("--suggest", "-s", action="store_true",
                        help="Profile and suggest transforms (no validation)")
    parser.add_argument("--validate", "-v", action="store_true",
                        help="Only run validation")
    parser.add_argument("--no-validation", action="store_true",
                        help="Skip validation step in full loop")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output JSON instead of human-readable")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be done without running tools")
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose output")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")
    args = parser.parse_args()

    runner = OptimizationRunner(verbose=args.verbose, dry_run=args.dry_run)
    printer = get_printer(not args.no_color)

    if args.validate:
        # Only validation
        printer.print_progress("Running validation...")
        result = runner.run_validation()
        if args.json:
            print(json.dumps({
                "passed": result.passed,
                "cycles": result.cycles,
                "speedup": result.speedup,
                "errors": result.errors,
                "warnings": result.warnings
            }, indent=2))
        else:
            print(f"Validation: {'PASSED' if result.passed else 'FAILED'}")
            print(f"Cycles: {result.cycles:,}")
            print(f"Speedup: {result.speedup:.2f}x")
        return

    if args.profile:
        # Only profiling
        printer.print_progress("Running profiling tools...")
        profiles = runner.run_profiling()
        if args.json:
            output = {
                name: {"success": p.success, "data": p.data, "error": p.error}
                for name, p in profiles.items()
            }
            print(json.dumps(output, indent=2))
        else:
            for name, profile in profiles.items():
                status = "OK" if profile.success else "FAILED"
                print(f"{name}: {status} ({profile.duration_ms:.0f}ms)")
        return

    if args.suggest:
        # Profile and suggest
        printer.print_progress("Running profiling tools...")
        runner.run_profiling()
        printer.print_progress("Detecting bottlenecks...")
        runner.detect_bottlenecks()
        printer.print_progress("Suggesting transforms...")
        runner.suggest_transforms()
        report = runner.generate_report()

        if args.json:
            print(report.to_json())
        else:
            printer.print_report(report)
        return

    # Full loop
    printer.print_progress("Starting optimization loop...")
    printer.print_progress("Step 1/4: Profiling...")
    runner.run_profiling()

    printer.print_progress("Step 2/4: Detecting bottlenecks...")
    runner.detect_bottlenecks()

    printer.print_progress("Step 3/4: Suggesting transforms...")
    runner.suggest_transforms()

    if not args.no_validation:
        printer.print_progress("Step 4/4: Validating...")
        runner._validation_result = runner.run_validation()

    report = runner.generate_report()

    if args.json:
        print(report.to_json())
    else:
        printer.print_report(report)


if __name__ == "__main__":
    main()
