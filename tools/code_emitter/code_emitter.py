#!/usr/bin/env python3
"""
Code Emitter for VLIW SIMD Kernel Optimization

Takes transformation specifications and generates actual Python code
compatible with KernelBuilder that can be copy-pasted into perf_takehome.py.

Features:
1. Generate unrolled loops with register renaming
2. Generate vectorized batches (scalar -> vector conversion)
3. Generate software pipelined stages
4. Automatic register allocation (assigns scratch addresses)
5. Validates generated code with constraint_validator
6. Rich output with fallback to plain text
7. JSON output option

Usage:
    python code_emitter.py --demo              # Run demonstration
    python code_emitter.py --emit-hash         # Emit vectorized hash function
    python code_emitter.py --emit-pipeline     # Emit pipelined batch processing
    python code_emitter.py --json              # Output as JSON
    python code_emitter.py --validate          # Validate generated code

    # From Python:
    from tools.code_emitter.code_emitter import CodeEmitter
    emitter = CodeEmitter()
    code = emitter.emit_unrolled_loop(body, factor=4)
"""

import sys
import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from collections import defaultdict
from copy import deepcopy
from textwrap import dedent, indent

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem import SLOT_LIMITS, VLEN, SCRATCH_SIZE, HASH_STAGES

# Try to import Rich for better formatting
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ============== Data Types ==============

@dataclass
class EmittedCode:
    """Result of code emission."""
    success: bool
    python_code: str
    instruction_list: List[dict]
    cycle_count: int
    scratch_usage: int
    description: str
    warnings: List[str] = field(default_factory=list)
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "python_code": self.python_code,
            "cycle_count": self.cycle_count,
            "scratch_usage": self.scratch_usage,
            "description": self.description,
            "warnings": self.warnings,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "metadata": self.metadata,
            "instruction_list": self.instruction_list
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class RegisterAllocation:
    """Tracks allocated scratch addresses."""
    next_addr: int = 0
    allocations: Dict[str, int] = field(default_factory=dict)
    vector_allocations: Dict[str, int] = field(default_factory=dict)

    def alloc(self, name: str, length: int = 1) -> int:
        """Allocate a scalar or vector register."""
        addr = self.next_addr
        if length == 1:
            self.allocations[name] = addr
        else:
            self.vector_allocations[name] = addr
        self.next_addr += length
        return addr

    def alloc_vector(self, name: str) -> int:
        """Allocate a VLEN-sized vector register."""
        return self.alloc(name, VLEN)

    def get(self, name: str) -> Optional[int]:
        """Get address for a named register."""
        return self.allocations.get(name) or self.vector_allocations.get(name)

    def total_usage(self) -> int:
        return self.next_addr


@dataclass
class TransformSpec:
    """Specification for a transformation to apply."""
    transform_type: str  # "unroll", "vectorize", "pipeline", "hoist"
    params: Dict[str, Any] = field(default_factory=dict)


# ============== Code Generator ==============

class CodeEmitter:
    """
    Generates actual Python code compatible with KernelBuilder.

    The emitted code can be directly copy-pasted into perf_takehome.py
    or used programmatically to build instruction lists.
    """

    def __init__(self, base_scratch_addr: int = 0):
        self.regs = RegisterAllocation(next_addr=base_scratch_addr)
        self.warnings: List[str] = []

    def reset(self, base_addr: int = 0):
        """Reset the emitter state."""
        self.regs = RegisterAllocation(next_addr=base_addr)
        self.warnings = []

    # ============== Low-Level Code Generation ==============

    def emit_const_load(self, dest_name: str, value: int) -> Tuple[str, dict]:
        """Emit code to load a constant."""
        addr = self.regs.alloc(dest_name)
        code = f'{dest_name} = self.alloc_scratch("{dest_name}")\n'
        code += f'self.add("load", ("const", {dest_name}, {value}))\n'
        instr = {"load": [("const", addr, value)]}
        return code, instr

    def emit_vbroadcast(self, dest_name: str, src_name_or_addr: Any) -> Tuple[str, dict]:
        """Emit code to broadcast a scalar to a vector."""
        dest_addr = self.regs.alloc_vector(dest_name)
        code = f'{dest_name} = self.alloc_scratch("{dest_name}", VLEN)\n'
        if isinstance(src_name_or_addr, str):
            code += f'self.instrs.append({{"valu": [("vbroadcast", {dest_name}, {src_name_or_addr})]}})\n'
            src_addr = self.regs.get(src_name_or_addr) or 0
        else:
            code += f'self.instrs.append({{"valu": [("vbroadcast", {dest_name}, {src_name_or_addr})]}})\n'
            src_addr = src_name_or_addr
        instr = {"valu": [("vbroadcast", dest_addr, src_addr)]}
        return code, instr

    def emit_alu_op(self, op: str, dest: str, src1: str, src2: str) -> Tuple[str, dict]:
        """Emit a scalar ALU operation."""
        dest_addr = self.regs.get(dest) or self.regs.alloc(dest)
        src1_addr = self.regs.get(src1) or 0
        src2_addr = self.regs.get(src2) or 0
        code = f'self.add("alu", ("{op}", {dest}, {src1}, {src2}))\n'
        instr = {"alu": [(op, dest_addr, src1_addr, src2_addr)]}
        return code, instr

    def emit_valu_op(self, op: str, dest: str, src1: str, src2: str) -> Tuple[str, dict]:
        """Emit a vector ALU operation."""
        dest_addr = self.regs.get(dest) or self.regs.alloc_vector(dest)
        src1_addr = self.regs.get(src1) or 0
        src2_addr = self.regs.get(src2) or 0
        code = f'self.instrs.append({{"valu": [("{op}", {dest}, {src1}, {src2})]}})\n'
        instr = {"valu": [(op, dest_addr, src1_addr, src2_addr)]}
        return code, instr

    def emit_vload(self, dest: str, addr: str) -> Tuple[str, dict]:
        """Emit a vector load."""
        dest_addr = self.regs.get(dest) or self.regs.alloc_vector(dest)
        addr_val = self.regs.get(addr) or 0
        code = f'self.instrs.append({{"load": [("vload", {dest}, {addr})]}})\n'
        instr = {"load": [("vload", dest_addr, addr_val)]}
        return code, instr

    def emit_vstore(self, addr: str, src: str) -> Tuple[str, dict]:
        """Emit a vector store."""
        addr_val = self.regs.get(addr) or 0
        src_addr = self.regs.get(src) or 0
        code = f'self.instrs.append({{"store": [("vstore", {addr}, {src})]}})\n'
        instr = {"store": [("vstore", addr_val, src_addr)]}
        return code, instr

    def emit_multiply_add(self, dest: str, a: str, b: str, c: str) -> Tuple[str, dict]:
        """Emit a multiply-add operation (dest = a * b + c)."""
        dest_addr = self.regs.get(dest) or self.regs.alloc_vector(dest)
        a_addr = self.regs.get(a) or 0
        b_addr = self.regs.get(b) or 0
        c_addr = self.regs.get(c) or 0
        code = f'self.instrs.append({{"valu": [("multiply_add", {dest}, {a}, {b}, {c})]}})\n'
        instr = {"valu": [("multiply_add", dest_addr, a_addr, b_addr, c_addr)]}
        return code, instr

    # ============== High-Level Transformations ==============

    def emit_unrolled_loop(
        self,
        body_template: List[dict],
        factor: int,
        iteration_offset_fn: Optional[Callable[[int], int]] = None,
        register_prefix: str = "u"
    ) -> EmittedCode:
        """
        Emit code for an unrolled loop.

        Args:
            body_template: Single iteration of loop body (instruction list)
            factor: Unroll factor (2, 4, 8, etc.)
            iteration_offset_fn: Function(iter_num) -> register offset for renaming
            register_prefix: Prefix for renamed registers

        Returns:
            EmittedCode with unrolled loop
        """
        if factor < 2:
            return EmittedCode(
                success=False,
                python_code="",
                instruction_list=[],
                cycle_count=0,
                scratch_usage=0,
                description="Unroll factor must be >= 2",
                warnings=["Invalid unroll factor"]
            )

        code_lines = []
        code_lines.append(f"# Unrolled loop ({factor}x)")
        code_lines.append(f"# Original body: {len(body_template)} cycles")
        code_lines.append(f"# Unrolled: {len(body_template) * factor} cycles (before packing)")
        code_lines.append("")

        # Calculate register offset for each iteration
        if iteration_offset_fn is None:
            # Default: large offset between iterations
            max_addr = self._find_max_address(body_template)
            offset = max_addr + 50
            iteration_offset_fn = lambda i: i * offset

        instructions = []

        for iter_num in range(factor):
            reg_offset = iteration_offset_fn(iter_num)
            code_lines.append(f"# --- Iteration {iter_num} (register offset: {reg_offset}) ---")

            for orig_instr in body_template:
                new_instr = self._rename_registers(orig_instr, reg_offset)
                instructions.append(new_instr)
                code_lines.append(f"self.instrs.append({new_instr})")

        python_code = "\n".join(code_lines)

        return EmittedCode(
            success=True,
            python_code=python_code,
            instruction_list=instructions,
            cycle_count=len(instructions),
            scratch_usage=self.regs.total_usage(),
            description=f"Loop unrolled {factor}x ({len(body_template)} -> {len(instructions)} cycles)",
            metadata={
                "unroll_factor": factor,
                "original_cycles": len(body_template),
                "unrolled_cycles": len(instructions)
            }
        )

    def emit_vectorized_hash(
        self,
        val_base: int,
        tmp1_base: int,
        tmp2_base: int,
        const_vectors: List[Tuple[int, int]]
    ) -> EmittedCode:
        """
        Emit vectorized hash computation code.

        The hash function has 6 stages, each with 3 operations.
        This emits optimized VALU code using 2-way ILP per stage.

        Args:
            val_base: Base address of value vector (modified in place)
            tmp1_base: Base address of tmp1 vector
            tmp2_base: Base address of tmp2 vector
            const_vectors: List of (const1_vec, const3_vec) for each stage

        Returns:
            EmittedCode with vectorized hash
        """
        code_lines = []
        code_lines.append("# Vectorized hash function (6 stages)")
        code_lines.append("# Uses 2-way ILP: op1||op3 in cycle 1, op2 in cycle 2")
        code_lines.append("")

        instructions = []

        for stage_idx, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1_vec, const3_vec = const_vectors[stage_idx]

            code_lines.append(f"# Stage {stage_idx}: {op1}(val, {val1}) {op2} {op3}(val, {val3})")

            # Cycle 1: op1 and op3 in parallel
            instr1 = {"valu": [
                (op1, tmp1_base, val_base, const1_vec),
                (op3, tmp2_base, val_base, const3_vec),
            ]}
            instructions.append(instr1)
            code_lines.append(f'self.instrs.append({{"valu": [')
            code_lines.append(f'    ("{op1}", {tmp1_base}, {val_base}, {const1_vec}),')
            code_lines.append(f'    ("{op3}", {tmp2_base}, {val_base}, {const3_vec}),')
            code_lines.append(f']}})')

            # Cycle 2: combine with op2
            instr2 = {"valu": [(op2, val_base, tmp1_base, tmp2_base)]}
            instructions.append(instr2)
            code_lines.append(f'self.instrs.append({{"valu": [("{op2}", {val_base}, {tmp1_base}, {tmp2_base})]}})')
            code_lines.append("")

        python_code = "\n".join(code_lines)

        return EmittedCode(
            success=True,
            python_code=python_code,
            instruction_list=instructions,
            cycle_count=len(instructions),
            scratch_usage=3 * VLEN + len(HASH_STAGES) * 2 * VLEN,  # val + tmp1 + tmp2 + consts
            description=f"Vectorized hash: 6 stages, {len(instructions)} cycles total",
            metadata={
                "stages": len(HASH_STAGES),
                "cycles_per_stage": 2,
                "total_cycles": len(instructions),
                "ilp_factor": 2
            }
        )

    def emit_dual_hash(
        self,
        val_a: int, tmp1_a: int, tmp2_a: int,
        val_b: int, tmp1_b: int, tmp2_b: int,
        const_vectors: List[Tuple[int, int]]
    ) -> EmittedCode:
        """
        Emit code for processing two hash batches in parallel.

        Uses 4 VALU slots per cycle for op1/op3, 2 slots for op2.

        Args:
            val_a, tmp1_a, tmp2_a: Registers for batch A
            val_b, tmp1_b, tmp2_b: Registers for batch B
            const_vectors: List of (const1_vec, const3_vec) for each stage

        Returns:
            EmittedCode with dual-batch hash
        """
        code_lines = []
        code_lines.append("# Dual-batch vectorized hash (2 batches in parallel)")
        code_lines.append("# Uses 4 VALU slots for op1||op3 phase")
        code_lines.append("")

        instructions = []

        for stage_idx, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1_vec, const3_vec = const_vectors[stage_idx]

            code_lines.append(f"# Stage {stage_idx}")

            # Cycle 1: op1 and op3 for both batches (4 slots)
            instr1 = {"valu": [
                (op1, tmp1_a, val_a, const1_vec),
                (op3, tmp2_a, val_a, const3_vec),
                (op1, tmp1_b, val_b, const1_vec),
                (op3, tmp2_b, val_b, const3_vec),
            ]}
            instructions.append(instr1)
            code_lines.append(f'self.instrs.append({{"valu": [')
            code_lines.append(f'    ("{op1}", {tmp1_a}, {val_a}, {const1_vec}),')
            code_lines.append(f'    ("{op3}", {tmp2_a}, {val_a}, {const3_vec}),')
            code_lines.append(f'    ("{op1}", {tmp1_b}, {val_b}, {const1_vec}),')
            code_lines.append(f'    ("{op3}", {tmp2_b}, {val_b}, {const3_vec}),')
            code_lines.append(f']}})')

            # Cycle 2: op2 for both batches (2 slots)
            instr2 = {"valu": [
                (op2, val_a, tmp1_a, tmp2_a),
                (op2, val_b, tmp1_b, tmp2_b),
            ]}
            instructions.append(instr2)
            code_lines.append(f'self.instrs.append({{"valu": [')
            code_lines.append(f'    ("{op2}", {val_a}, {tmp1_a}, {tmp2_a}),')
            code_lines.append(f'    ("{op2}", {val_b}, {tmp1_b}, {tmp2_b}),')
            code_lines.append(f']}})')
            code_lines.append("")

        python_code = "\n".join(code_lines)

        return EmittedCode(
            success=True,
            python_code=python_code,
            instruction_list=instructions,
            cycle_count=len(instructions),
            scratch_usage=6 * VLEN,
            description=f"Dual-batch hash: 6 stages, {len(instructions)} cycles, 4 VALU slots/cycle",
            metadata={
                "batches": 2,
                "stages": len(HASH_STAGES),
                "cycles_per_stage": 2,
                "max_valu_slots": 4
            }
        )

    def emit_triple_hash(
        self,
        val_a: int, tmp1_a: int, tmp2_a: int,
        val_b: int, tmp1_b: int, tmp2_b: int,
        val_c: int, tmp1_c: int, tmp2_c: int,
        const_vectors: List[Tuple[int, int]]
    ) -> EmittedCode:
        """
        Emit code for processing three hash batches in parallel.

        Uses all 6 VALU slots per cycle.

        Args:
            val_*, tmp1_*, tmp2_*: Registers for batches A, B, C
            const_vectors: List of (const1_vec, const3_vec) for each stage

        Returns:
            EmittedCode with triple-batch hash
        """
        code_lines = []
        code_lines.append("# Triple-batch vectorized hash (3 batches in parallel)")
        code_lines.append("# Uses all 6 VALU slots")
        code_lines.append("")

        instructions = []

        for stage_idx, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1_vec, const3_vec = const_vectors[stage_idx]

            code_lines.append(f"# Stage {stage_idx}")

            # Cycle 1: op1 and op3 for all 3 batches (6 slots)
            instr1 = {"valu": [
                (op1, tmp1_a, val_a, const1_vec), (op3, tmp2_a, val_a, const3_vec),
                (op1, tmp1_b, val_b, const1_vec), (op3, tmp2_b, val_b, const3_vec),
                (op1, tmp1_c, val_c, const1_vec), (op3, tmp2_c, val_c, const3_vec),
            ]}
            instructions.append(instr1)
            code_lines.append(f'self.instrs.append({{"valu": [')
            code_lines.append(f'    ("{op1}", {tmp1_a}, {val_a}, {const1_vec}), ("{op3}", {tmp2_a}, {val_a}, {const3_vec}),')
            code_lines.append(f'    ("{op1}", {tmp1_b}, {val_b}, {const1_vec}), ("{op3}", {tmp2_b}, {val_b}, {const3_vec}),')
            code_lines.append(f'    ("{op1}", {tmp1_c}, {val_c}, {const1_vec}), ("{op3}", {tmp2_c}, {val_c}, {const3_vec}),')
            code_lines.append(f']}})')

            # Cycle 2: op2 for all 3 batches (3 slots)
            instr2 = {"valu": [
                (op2, val_a, tmp1_a, tmp2_a),
                (op2, val_b, tmp1_b, tmp2_b),
                (op2, val_c, tmp1_c, tmp2_c),
            ]}
            instructions.append(instr2)
            code_lines.append(f'self.instrs.append({{"valu": [')
            code_lines.append(f'    ("{op2}", {val_a}, {tmp1_a}, {tmp2_a}),')
            code_lines.append(f'    ("{op2}", {val_b}, {tmp1_b}, {tmp2_b}),')
            code_lines.append(f'    ("{op2}", {val_c}, {tmp1_c}, {tmp2_c}),')
            code_lines.append(f']}})')
            code_lines.append("")

        python_code = "\n".join(code_lines)

        return EmittedCode(
            success=True,
            python_code=python_code,
            instruction_list=instructions,
            cycle_count=len(instructions),
            scratch_usage=9 * VLEN,
            description=f"Triple-batch hash: 6 stages, {len(instructions)} cycles, 6 VALU slots/cycle",
            metadata={
                "batches": 3,
                "stages": len(HASH_STAGES),
                "cycles_per_stage": 2,
                "max_valu_slots": 6
            }
        )

    def emit_pipelined_iteration(
        self,
        hash_regs_cur: Dict[str, int],
        prep_regs_next: Dict[str, int],
        const_vectors: List[Tuple[int, int]],
        inp_indices_p: int,
        inp_values_p: int,
        forest_p: int,
        offset_a: int,
        offset_b: int
    ) -> EmittedCode:
        """
        Emit code for a pipelined iteration: hash current batch while preparing next.

        During 16 hash cycles (6 stages * 2 + overhead), we interleave:
        - Cycle 0: Address computation for next batch (ALU)
        - Cycles 2-3: Vector loads for next batch (LOAD)
        - Cycle 4: Tree address computation (VALU free slots)
        - Cycles 5-6: Extract addresses (ALU)
        - Cycles 7-14: Scattered loads for next batch (LOAD)

        Args:
            hash_regs_cur: Register addresses for current batch hash
            prep_regs_next: Register addresses for next batch preparation
            const_vectors: Hash constant vectors
            inp_indices_p, inp_values_p, forest_p: Memory pointers
            offset_a, offset_b: Batch offsets for next iteration

        Returns:
            EmittedCode with pipelined iteration
        """
        code_lines = []
        code_lines.append("# Pipelined iteration: hash current batch + prepare next batch")
        code_lines.append("# Overlaps ALU/LOAD prep with VALU hash computation")
        code_lines.append("")

        instructions = []

        # Extract register addresses
        val_a = hash_regs_cur['val_a']
        tmp1_a = hash_regs_cur['tmp1_a']
        tmp2_a = hash_regs_cur['tmp2_a']
        val_b = hash_regs_cur['val_b']
        tmp1_b = hash_regs_cur['tmp1_b']
        tmp2_b = hash_regs_cur['tmp2_b']

        # Next batch registers
        n_idx_base_a = prep_regs_next.get('idx_base_a', 0)
        n_val_base_a = prep_regs_next.get('val_base_a', 0)
        n_idx_base_b = prep_regs_next.get('idx_base_b', 0)
        n_val_base_b = prep_regs_next.get('val_base_b', 0)
        n_v_idx_a = prep_regs_next.get('v_idx_a', 0)
        n_v_val_a = prep_regs_next.get('v_val_a', 0)
        n_v_idx_b = prep_regs_next.get('v_idx_b', 0)
        n_v_val_b = prep_regs_next.get('v_val_b', 0)
        n_v_taddr_a = prep_regs_next.get('v_taddr_a', 0)
        n_v_taddr_b = prep_regs_next.get('v_taddr_b', 0)
        n_v_node_a = prep_regs_next.get('v_node_a', 0)
        n_v_node_b = prep_regs_next.get('v_node_b', 0)
        n_addr_a = prep_regs_next.get('addr_a', [0]*VLEN)
        n_addr_b = prep_regs_next.get('addr_b', [0]*VLEN)

        for stage_idx, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1_vec, const3_vec = const_vectors[stage_idx]
            cycle_num = stage_idx * 2

            code_lines.append(f"# Hash stage {stage_idx} (cycles {cycle_num}-{cycle_num+1})")

            # Cycle A: op1 + op3 for dual batch (4 VALU slots)
            instr_a = {"valu": [
                (op1, tmp1_a, val_a, const1_vec),
                (op3, tmp2_a, val_a, const3_vec),
                (op1, tmp1_b, val_b, const1_vec),
                (op3, tmp2_b, val_b, const3_vec),
            ]}

            # Add pipelined prep operations
            if cycle_num == 0:
                # Address computation
                instr_a["alu"] = [
                    ("+", n_idx_base_a, inp_indices_p, offset_a),
                    ("+", n_val_base_a, inp_values_p, offset_a),
                    ("+", n_idx_base_b, inp_indices_p, offset_b),
                    ("+", n_val_base_b, inp_values_p, offset_b),
                ]
                code_lines.append("# + Address computation for next batch (ALU)")
            elif cycle_num == 4:
                # Tree addresses (2 free VALU slots)
                instr_a["valu"].extend([
                    ("+", n_v_taddr_a, forest_p, n_v_idx_a),
                    ("+", n_v_taddr_b, forest_p, n_v_idx_b),
                ])
                code_lines.append("# + Tree address computation (2 extra VALU)")
            elif cycle_num == 6:
                # Finish extract
                instr_a["alu"] = [("+", n_addr_b[i], n_v_taddr_b + i, 0) for i in range(4, VLEN)]
                code_lines.append("# + Extract addresses part 2 (ALU)")
            elif cycle_num == 8:
                instr_a["load"] = [
                    ("load", n_v_node_a + 2, n_addr_a[2]),
                    ("load", n_v_node_a + 3, n_addr_a[3])
                ]
                code_lines.append("# + Scattered loads (LOAD)")
            elif cycle_num == 10:
                instr_a["load"] = [
                    ("load", n_v_node_a + 6, n_addr_a[6]),
                    ("load", n_v_node_a + 7, n_addr_a[7])
                ]
            elif cycle_num == 12:
                instr_a["load"] = [
                    ("load", n_v_node_b + 2, n_addr_b[2]),
                    ("load", n_v_node_b + 3, n_addr_b[3])
                ]
            elif cycle_num == 14:
                instr_a["load"] = [
                    ("load", n_v_node_b + 6, n_addr_b[6]),
                    ("load", n_v_node_b + 7, n_addr_b[7])
                ]

            instructions.append(instr_a)
            code_lines.append(f"self.instrs.append({instr_a})")

            # Cycle B: op2 for dual batch (2 VALU slots)
            instr_b = {"valu": [
                (op2, val_a, tmp1_a, tmp2_a),
                (op2, val_b, tmp1_b, tmp2_b),
            ]}

            # More pipelined operations
            if cycle_num == 2:
                instr_b["load"] = [
                    ("vload", n_v_idx_b, n_idx_base_b),
                    ("vload", n_v_val_b, n_val_base_b)
                ]
                code_lines.append("# + Vector loads for next batch (LOAD)")
            elif cycle_num == 4:
                # Extract addresses part 1
                instr_b["alu"] = [("+", n_addr_a[i], n_v_taddr_a + i, 0) for i in range(VLEN)]
                instr_b["alu"].extend([("+", n_addr_b[i], n_v_taddr_b + i, 0) for i in range(4)])
                code_lines.append("# + Extract addresses (12 ALU ops)")
            elif cycle_num == 6:
                instr_b["load"] = [
                    ("load", n_v_node_b, n_addr_b[0]),
                    ("load", n_v_node_b + 1, n_addr_b[1])
                ]
            elif cycle_num == 8:
                instr_b["load"] = [
                    ("load", n_v_node_a + 4, n_addr_a[4]),
                    ("load", n_v_node_a + 5, n_addr_a[5])
                ]
            elif cycle_num == 10:
                instr_b["load"] = [
                    ("load", n_v_node_b, n_addr_b[0]),
                    ("load", n_v_node_b + 1, n_addr_b[1])
                ]
            elif cycle_num == 12:
                instr_b["load"] = [
                    ("load", n_v_node_b + 4, n_addr_b[4]),
                    ("load", n_v_node_b + 5, n_addr_b[5])
                ]

            # Vector loads on cycle 2-3
            if cycle_num == 2:
                instr_a["load"] = [
                    ("vload", n_v_idx_a, n_idx_base_a),
                    ("vload", n_v_val_a, n_val_base_a)
                ]

            instructions.append(instr_b)
            code_lines.append(f"self.instrs.append({instr_b})")
            code_lines.append("")

        python_code = "\n".join(code_lines)

        return EmittedCode(
            success=True,
            python_code=python_code,
            instruction_list=instructions,
            cycle_count=len(instructions),
            scratch_usage=self.regs.total_usage(),
            description=f"Pipelined iteration: {len(instructions)} cycles (hash + prep overlapped)",
            metadata={
                "hash_cycles": 12,
                "prep_cycles_hidden": 10,
                "total_cycles": len(instructions)
            }
        )

    def emit_batch_setup(
        self,
        regs: Dict[str, int],
        inp_indices_p: int,
        inp_values_p: int,
        forest_p: int,
        offset_a: int,
        offset_b: int,
        zero_const: int
    ) -> EmittedCode:
        """
        Emit setup code for a dual-batch iteration.

        This is the non-pipelined version used for the first iteration.

        Args:
            regs: Register addresses for this batch
            inp_indices_p, inp_values_p, forest_p: Memory pointers
            offset_a, offset_b: Batch offsets
            zero_const: Address of zero constant

        Returns:
            EmittedCode with batch setup
        """
        code_lines = []
        code_lines.append("# Batch setup (non-pipelined, for first iteration)")
        code_lines.append("")

        instructions = []

        # Address computation (1 cycle, 4 ALU)
        instr1 = {"alu": [
            ("+", regs['idx_base_a'], inp_indices_p, offset_a),
            ("+", regs['val_base_a'], inp_values_p, offset_a),
            ("+", regs['idx_base_b'], inp_indices_p, offset_b),
            ("+", regs['val_base_b'], inp_values_p, offset_b),
        ]}
        instructions.append(instr1)
        code_lines.append(f"# Address computation (4 ALU)")
        code_lines.append(f"self.instrs.append({instr1})")

        # Vector loads (2 cycles, 2 LOAD each)
        instr2 = {"load": [
            ("vload", regs['v_idx_a'], regs['idx_base_a']),
            ("vload", regs['v_val_a'], regs['val_base_a'])
        ]}
        instructions.append(instr2)
        code_lines.append(f"self.instrs.append({instr2})")

        instr3 = {"load": [
            ("vload", regs['v_idx_b'], regs['idx_base_b']),
            ("vload", regs['v_val_b'], regs['val_base_b'])
        ]}
        instructions.append(instr3)
        code_lines.append(f"self.instrs.append({instr3})")

        # Tree addresses (1 cycle, 2 VALU)
        instr4 = {"valu": [
            ("+", regs['v_taddr_a'], forest_p, regs['v_idx_a']),
            ("+", regs['v_taddr_b'], forest_p, regs['v_idx_b'])
        ]}
        instructions.append(instr4)
        code_lines.append(f"self.instrs.append({instr4})")

        # Extract addresses (2 cycles, 12+4 ALU)
        addr_a = regs.get('addr_a', [regs.get('v_taddr_a', 0) + 100 + i for i in range(VLEN)])
        addr_b = regs.get('addr_b', [regs.get('v_taddr_b', 0) + 100 + i for i in range(VLEN)])

        instr5 = {"alu": [("+", addr_a[i], regs['v_taddr_a'] + i, zero_const) for i in range(VLEN)]
                        + [("+", addr_b[i], regs['v_taddr_b'] + i, zero_const) for i in range(4)]}
        instructions.append(instr5)
        code_lines.append(f"# Extract addresses part 1 (12 ALU)")

        instr6 = {"alu": [("+", addr_b[i], regs['v_taddr_b'] + i, zero_const) for i in range(4, VLEN)]}
        instructions.append(instr6)
        code_lines.append(f"# Extract addresses part 2 (4 ALU)")

        # Scattered loads (8 cycles, 2 LOAD each)
        for i in range(0, VLEN, 2):
            instr = {"load": [
                ("load", regs['v_node_a'] + i, addr_a[i]),
                ("load", regs['v_node_a'] + i + 1, addr_a[i + 1])
            ]}
            instructions.append(instr)

        for i in range(0, VLEN, 2):
            instr = {"load": [
                ("load", regs['v_node_b'] + i, addr_b[i]),
                ("load", regs['v_node_b'] + i + 1, addr_b[i + 1])
            ]}
            instructions.append(instr)

        code_lines.append(f"# Scattered loads (8 cycles)")

        # XOR (1 cycle, 2 VALU)
        instr_xor = {"valu": [
            ("^", regs['v_val_a'], regs['v_val_a'], regs['v_node_a']),
            ("^", regs['v_val_b'], regs['v_val_b'], regs['v_node_b'])
        ]}
        instructions.append(instr_xor)
        code_lines.append(f"# XOR with tree values (2 VALU)")

        python_code = "\n".join(code_lines)

        return EmittedCode(
            success=True,
            python_code=python_code,
            instruction_list=instructions,
            cycle_count=len(instructions),
            scratch_usage=self.regs.total_usage(),
            description=f"Batch setup: {len(instructions)} cycles",
            metadata={
                "setup_cycles": len(instructions),
                "alu_heavy_cycles": 3,
                "load_heavy_cycles": 10
            }
        )

    def emit_batch_finish(
        self,
        regs: Dict[str, int],
        v_one: int,
        v_two: int,
        v_zero: int,
        v_n_nodes: int,
        has_next: bool = False,
        next_regs: Optional[Dict[str, int]] = None
    ) -> EmittedCode:
        """
        Emit finish code for a batch: index computation, bounds check, stores.

        Uses multiply_add for efficient bounds check.

        Args:
            regs: Register addresses for this batch
            v_one, v_two, v_zero, v_n_nodes: Vector constants
            has_next: Whether to overlap loads for next batch
            next_regs: Registers for next batch (if has_next)

        Returns:
            EmittedCode with batch finish
        """
        code_lines = []
        code_lines.append("# Batch finish: index update, bounds check, stores")
        code_lines.append("")

        instructions = []

        # Cycle 1: bit extract + multiply_add for idx = idx*2+1 (4 VALU)
        instr1 = {"valu": [
            ("&", regs['v_htmp1_a'], regs['v_val_a'], v_one),
            ("&", regs['v_htmp1_b'], regs['v_val_b'], v_one),
            ("multiply_add", regs['v_idx_a'], regs['v_idx_a'], v_two, v_one),
            ("multiply_add", regs['v_idx_b'], regs['v_idx_b'], v_two, v_one),
        ]}
        if has_next and next_regs:
            # Overlap scattered loads
            instr1["load"] = [
                ("load", next_regs['v_node_b'] + 2, next_regs['addr_b'][2]),
                ("load", next_regs['v_node_b'] + 3, next_regs['addr_b'][3])
            ]
        instructions.append(instr1)
        code_lines.append(f"# Bit extract + idx*2+1 (4 VALU)")

        # Cycle 2: idx = idx + bit (2 VALU)
        instr2 = {"valu": [
            ("+", regs['v_idx_a'], regs['v_idx_a'], regs['v_htmp1_a']),
            ("+", regs['v_idx_b'], regs['v_idx_b'], regs['v_htmp1_b'])
        ]}
        if has_next and next_regs:
            instr2["load"] = [
                ("load", next_regs['v_node_b'] + 4, next_regs['addr_b'][4]),
                ("load", next_regs['v_node_b'] + 5, next_regs['addr_b'][5])
            ]
        instructions.append(instr2)

        # Cycle 3: bounds check (2 VALU for <)
        instr3 = {"valu": [
            ("<", regs['v_htmp1_a'], regs['v_idx_a'], v_n_nodes),
            ("<", regs['v_htmp1_b'], regs['v_idx_b'], v_n_nodes)
        ]}
        if has_next and next_regs:
            instr3["load"] = [
                ("load", next_regs['v_node_b'] + 6, next_regs['addr_b'][6]),
                ("load", next_regs['v_node_b'] + 7, next_regs['addr_b'][7])
            ]
        instructions.append(instr3)
        code_lines.append(f"# Bounds check (2 VALU)")

        # Cycle 4: multiply_add for bounds + stores (2 VALU + 2 STORE)
        instr4 = {
            "valu": [
                ("multiply_add", regs['v_idx_a'], regs['v_idx_a'], regs['v_htmp1_a'], v_zero),
                ("multiply_add", regs['v_idx_b'], regs['v_idx_b'], regs['v_htmp1_b'], v_zero),
            ],
            "store": [
                ("vstore", regs['val_base_a'], regs['v_val_a']),
                ("vstore", regs['val_base_b'], regs['v_val_b'])
            ]
        }
        instructions.append(instr4)
        code_lines.append(f"# Bounds via multiply_add + value stores (2 VALU + 2 STORE)")

        # Cycle 5: index stores + XOR for next (2 STORE + 2 VALU)
        instr5 = {"store": [
            ("vstore", regs['idx_base_a'], regs['v_idx_a']),
            ("vstore", regs['idx_base_b'], regs['v_idx_b'])
        ]}
        if has_next and next_regs:
            instr5["valu"] = [
                ("^", next_regs['v_val_a'], next_regs['v_val_a'], next_regs['v_node_a']),
                ("^", next_regs['v_val_b'], next_regs['v_val_b'], next_regs['v_node_b'])
            ]
        instructions.append(instr5)
        code_lines.append(f"# Index stores + XOR for next batch (2 STORE + 2 VALU)")

        python_code = "\n".join(code_lines)

        return EmittedCode(
            success=True,
            python_code=python_code,
            instruction_list=instructions,
            cycle_count=len(instructions),
            scratch_usage=self.regs.total_usage(),
            description=f"Batch finish: {len(instructions)} cycles (with overlapped next prep)" if has_next else f"Batch finish: {len(instructions)} cycles",
            metadata={
                "finish_cycles": len(instructions),
                "valu_slots_used": 10,
                "store_cycles": 2
            }
        )

    # ============== Helper Methods ==============

    def _find_max_address(self, instructions: List[dict]) -> int:
        """Find the maximum scratch address used in instructions."""
        max_addr = 0
        for instr in instructions:
            for engine, slots in instr.items():
                if engine == "debug" or not slots:
                    continue
                for slot in slots:
                    for item in slot[1:]:  # Skip opcode
                        if isinstance(item, int) and item > 0:
                            max_addr = max(max_addr, item + VLEN)  # Account for vectors
        return max_addr

    def _rename_registers(self, instr: dict, offset: int) -> dict:
        """Rename all register addresses in an instruction by adding offset."""
        new_instr = {}
        for engine, slots in instr.items():
            if engine == "debug":
                new_instr[engine] = slots
                continue
            if not slots:
                continue

            new_slots = []
            for slot in slots:
                if not slot:
                    new_slots.append(slot)
                    continue

                new_slot = list(slot)
                op = slot[0]

                # Rename addresses based on engine/op pattern
                if engine == "alu" and len(slot) >= 4:
                    new_slot[1] = slot[1] + offset
                    new_slot[2] = slot[2] + offset
                    new_slot[3] = slot[3] + offset
                elif engine == "valu":
                    if op == "vbroadcast" and len(slot) >= 3:
                        new_slot[1] = slot[1] + offset
                        new_slot[2] = slot[2] + offset
                    elif op == "multiply_add" and len(slot) >= 5:
                        new_slot[1] = slot[1] + offset
                        new_slot[2] = slot[2] + offset
                        new_slot[3] = slot[3] + offset
                        new_slot[4] = slot[4] + offset
                    elif len(slot) >= 4:
                        new_slot[1] = slot[1] + offset
                        new_slot[2] = slot[2] + offset
                        new_slot[3] = slot[3] + offset
                elif engine == "load":
                    if op == "load" and len(slot) >= 3:
                        new_slot[1] = slot[1] + offset
                        new_slot[2] = slot[2] + offset
                    elif op == "vload" and len(slot) >= 3:
                        new_slot[1] = slot[1] + offset
                        new_slot[2] = slot[2] + offset
                    elif op == "const" and len(slot) >= 2:
                        new_slot[1] = slot[1] + offset
                elif engine == "store":
                    if len(slot) >= 3:
                        new_slot[1] = slot[1] + offset
                        new_slot[2] = slot[2] + offset
                elif engine == "flow":
                    if op in ("select", "vselect") and len(slot) >= 5:
                        new_slot[1] = slot[1] + offset
                        new_slot[2] = slot[2] + offset
                        new_slot[3] = slot[3] + offset
                        new_slot[4] = slot[4] + offset

                new_slots.append(tuple(new_slot))

            new_instr[engine] = new_slots

        return new_instr

    def validate_code(self, emitted: EmittedCode) -> EmittedCode:
        """Validate emitted code using constraint_validator."""
        try:
            from tools.constraint_validator.constraint_validator import validate_kernel
            result = validate_kernel(emitted.instruction_list)
            emitted.validation_passed = result.is_valid
            emitted.validation_errors = [i.message for i in result.issues if i.severity.value == "error"]
            if result.warning_count > 0:
                emitted.warnings.extend([i.message for i in result.issues if i.severity.value == "warning"])
        except ImportError:
            emitted.warnings.append("Could not import constraint_validator for validation")
        except Exception as e:
            emitted.warnings.append(f"Validation error: {str(e)}")
        return emitted


# ============== Output Formatting ==============

class PlainPrinter:
    """Plain text output without Rich."""

    def print_emitted(self, emitted: EmittedCode, show_code: bool = True):
        print("=" * 70)
        print("CODE EMITTER OUTPUT")
        print("=" * 70)
        print()

        status = "[OK]" if emitted.success else "[FAIL]"
        print(f"Status: {status}")
        print(f"Description: {emitted.description}")
        print(f"Cycles: {emitted.cycle_count}")
        print(f"Scratch Usage: {emitted.scratch_usage}")

        if emitted.validation_passed:
            print(f"Validation: PASSED")
        else:
            print(f"Validation: FAILED")
            for err in emitted.validation_errors[:5]:
                print(f"  - {err}")

        if emitted.warnings:
            print("\nWarnings:")
            for w in emitted.warnings[:5]:
                print(f"  - {w}")

        if show_code and emitted.python_code:
            print("\n" + "-" * 70)
            print("GENERATED CODE")
            print("-" * 70)
            print(emitted.python_code)

        if emitted.metadata:
            print("\n" + "-" * 70)
            print("METADATA")
            print("-" * 70)
            for k, v in emitted.metadata.items():
                print(f"  {k}: {v}")


class RichPrinter:
    """Rich colored output."""

    def __init__(self):
        self.console = Console()

    def print_emitted(self, emitted: EmittedCode, show_code: bool = True):
        self.console.print(Panel("CODE EMITTER OUTPUT", style="bold cyan", box=box.DOUBLE))

        # Status
        status_style = "green" if emitted.success else "red"
        self.console.print(f"[bold]Status:[/bold] [{status_style}]{'SUCCESS' if emitted.success else 'FAILED'}[/{status_style}]")
        self.console.print(f"[bold]Description:[/bold] {emitted.description}")

        # Stats table
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Cycles", f"{emitted.cycle_count:,}")
        table.add_row("Scratch Usage", f"{emitted.scratch_usage:,}")

        val_style = "green" if emitted.validation_passed else "red"
        table.add_row("Validation", f"[{val_style}]{'PASSED' if emitted.validation_passed else 'FAILED'}[/{val_style}]")

        self.console.print(table)

        if emitted.validation_errors:
            self.console.print("\n[bold red]Validation Errors:[/bold red]")
            for err in emitted.validation_errors[:5]:
                self.console.print(f"  [red]-[/red] {err}")

        if emitted.warnings:
            self.console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for w in emitted.warnings[:5]:
                self.console.print(f"  [yellow]-[/yellow] {w}")

        if show_code and emitted.python_code:
            self.console.print("\n[bold]Generated Code:[/bold]")
            syntax = Syntax(emitted.python_code, "python", theme="monokai", line_numbers=True)
            self.console.print(Panel(syntax, box=box.ROUNDED))

        if emitted.metadata:
            self.console.print("\n[bold dim]Metadata:[/bold dim]")
            for k, v in emitted.metadata.items():
                self.console.print(f"  [dim]{k}:[/dim] {v}")


def get_printer():
    """Get appropriate printer based on Rich availability."""
    if RICH_AVAILABLE:
        return RichPrinter()
    return PlainPrinter()


# ============== Demo and Main ==============

def run_demo():
    """Run demonstration of code emission capabilities."""
    printer = get_printer()

    print("=" * 70)
    print("CODE EMITTER DEMONSTRATION")
    print("=" * 70)
    print()

    emitter = CodeEmitter(base_scratch_addr=100)

    # Demo 1: Single vectorized hash
    print("\n1. SINGLE BATCH VECTORIZED HASH")
    print("-" * 40)

    const_vectors = [(200 + i*16, 208 + i*16) for i in range(6)]
    result = emitter.emit_vectorized_hash(
        val_base=100,
        tmp1_base=108,
        tmp2_base=116,
        const_vectors=const_vectors
    )
    printer.print_emitted(result, show_code=True)

    # Demo 2: Dual batch hash
    print("\n2. DUAL BATCH VECTORIZED HASH")
    print("-" * 40)

    result = emitter.emit_dual_hash(
        val_a=300, tmp1_a=308, tmp2_a=316,
        val_b=400, tmp1_b=408, tmp2_b=416,
        const_vectors=const_vectors
    )
    printer.print_emitted(result, show_code=False)

    # Demo 3: Loop unrolling
    print("\n3. LOOP UNROLLING")
    print("-" * 40)

    body_template = [
        {"alu": [("+", 10, 20, 30)]},
        {"valu": [("*", 40, 50, 60)]},
        {"store": [("store", 70, 80)]},
    ]
    result = emitter.emit_unrolled_loop(body_template, factor=4)
    printer.print_emitted(result, show_code=False)

    # Demo 4: Batch finish
    print("\n4. BATCH FINISH WITH BOUNDS CHECK")
    print("-" * 40)

    regs = {
        'v_val_a': 500, 'v_val_b': 508,
        'v_idx_a': 516, 'v_idx_b': 524,
        'v_htmp1_a': 532, 'v_htmp1_b': 540,
        'val_base_a': 550, 'val_base_b': 551,
        'idx_base_a': 552, 'idx_base_b': 553,
    }
    result = emitter.emit_batch_finish(
        regs=regs,
        v_one=600, v_two=608, v_zero=616, v_n_nodes=624,
        has_next=False
    )
    printer.print_emitted(result, show_code=True)


def emit_hash_code():
    """Emit complete hash function code."""
    emitter = CodeEmitter(base_scratch_addr=100)

    # Allocate constants first
    const_vectors = []
    base = 200
    for i in range(6):
        const_vectors.append((base + i*16, base + i*16 + 8))

    result = emitter.emit_dual_hash(
        val_a=100, tmp1_a=108, tmp2_a=116,
        val_b=124, tmp1_b=132, tmp2_b=140,
        const_vectors=const_vectors
    )

    printer = get_printer()
    printer.print_emitted(result, show_code=True)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Code Emitter for VLIW SIMD Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python code_emitter.py --demo              # Run demonstration
    python code_emitter.py --emit-hash         # Emit vectorized hash
    python code_emitter.py --json              # Output as JSON
    python code_emitter.py --validate          # Validate generated code
        """
    )
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--emit-hash", action="store_true", help="Emit vectorized hash code")
    parser.add_argument("--emit-dual-hash", action="store_true", help="Emit dual-batch hash code")
    parser.add_argument("--emit-triple-hash", action="store_true", help="Emit triple-batch hash code")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--validate", action="store_true", help="Validate generated code")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    emitter = CodeEmitter(base_scratch_addr=100)

    # Default: emit dual hash
    const_vectors = [(200 + i*16, 208 + i*16) for i in range(6)]

    if args.emit_hash:
        result = emitter.emit_vectorized_hash(
            val_base=100, tmp1_base=108, tmp2_base=116,
            const_vectors=const_vectors
        )
    elif args.emit_triple_hash:
        result = emitter.emit_triple_hash(
            val_a=100, tmp1_a=108, tmp2_a=116,
            val_b=124, tmp1_b=132, tmp2_b=140,
            val_c=148, tmp1_c=156, tmp2_c=164,
            const_vectors=const_vectors
        )
    else:  # Default or --emit-dual-hash
        result = emitter.emit_dual_hash(
            val_a=100, tmp1_a=108, tmp2_a=116,
            val_b=124, tmp1_b=132, tmp2_b=140,
            const_vectors=const_vectors
        )

    if args.validate:
        result = emitter.validate_code(result)

    if args.json:
        print(result.to_json())
    else:
        printer = PlainPrinter() if args.no_color else get_printer()
        printer.print_emitted(result, show_code=True)


if __name__ == "__main__":
    main()
