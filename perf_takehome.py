"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))
        return slots

    def build_vhash_dual(self, val_a, tmp1_a, tmp2_a, val_b, tmp1_b, tmp2_b, hash_const_vecs):
        """Process 2 batches' hash in parallel - uses 4 valu slots per cycle."""
        instrs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1_vec, const3_vec = hash_const_vecs[hi]
            instrs.append({"valu": [
                (op1, tmp1_a, val_a, const1_vec),
                (op3, tmp2_a, val_a, const3_vec),
                (op1, tmp1_b, val_b, const1_vec),
                (op3, tmp2_b, val_b, const3_vec),
            ]})
            instrs.append({"valu": [
                (op2, val_a, tmp1_a, tmp2_a),
                (op2, val_b, tmp1_b, tmp2_b),
            ]})
        return instrs

    def build_vhash_triple(self, val_a, tmp1_a, tmp2_a, val_b, tmp1_b, tmp2_b, val_c, tmp1_c, tmp2_c, hash_const_vecs):
        """Process 3 batches' hash in parallel - uses 6 valu slots per cycle."""
        instrs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1_vec, const3_vec = hash_const_vecs[hi]
            instrs.append({"valu": [
                (op1, tmp1_a, val_a, const1_vec), (op3, tmp2_a, val_a, const3_vec),
                (op1, tmp1_b, val_b, const1_vec), (op3, tmp2_b, val_b, const3_vec),
                (op1, tmp1_c, val_c, const1_vec), (op3, tmp2_c, val_c, const3_vec),
            ]})
            instrs.append({"valu": [
                (op2, val_a, tmp1_a, tmp2_a),
                (op2, val_b, tmp1_b, tmp2_b),
                (op2, val_c, tmp1_c, tmp2_c),
            ]})
        return instrs

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """Software-pipelined vectorized dual-batch implementation with finish-hash overlap.

        Key optimizations:
        - Overlap preparation of batch K+1 with hash computation of batch K
        - Overlap finish_late of batch K-1 with hash of batch K (using 3-way register rotation)
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Load header values
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.instrs.append({"flow": [("pause",)]})

        # Three complete register sets for 3-way rotation
        # This allows overlapping finish_late(K-1) during hash(K) while prepping K+1
        def alloc_batch_regs(prefix):
            return {
                'v_idx_a': self.alloc_scratch(f"{prefix}_v_idx_a", VLEN),
                'v_val_a': self.alloc_scratch(f"{prefix}_v_val_a", VLEN),
                'v_node_a': self.alloc_scratch(f"{prefix}_v_node_a", VLEN),
                'v_taddr_a': self.alloc_scratch(f"{prefix}_v_taddr_a", VLEN),
                'v_htmp1_a': self.alloc_scratch(f"{prefix}_v_htmp1_a", VLEN),
                'v_htmp2_a': self.alloc_scratch(f"{prefix}_v_htmp2_a", VLEN),
                'v_idx_b': self.alloc_scratch(f"{prefix}_v_idx_b", VLEN),
                'v_val_b': self.alloc_scratch(f"{prefix}_v_val_b", VLEN),
                'v_node_b': self.alloc_scratch(f"{prefix}_v_node_b", VLEN),
                'v_taddr_b': self.alloc_scratch(f"{prefix}_v_taddr_b", VLEN),
                'v_htmp1_b': self.alloc_scratch(f"{prefix}_v_htmp1_b", VLEN),
                'v_htmp2_b': self.alloc_scratch(f"{prefix}_v_htmp2_b", VLEN),
                'idx_base_a': self.alloc_scratch(f"{prefix}_idx_base_a"),
                'val_base_a': self.alloc_scratch(f"{prefix}_val_base_a"),
                'idx_base_b': self.alloc_scratch(f"{prefix}_idx_base_b"),
                'val_base_b': self.alloc_scratch(f"{prefix}_val_base_b"),
                'addr_a': [self.alloc_scratch(f"{prefix}_addr_a_{i}") for i in range(VLEN)],
                'addr_b': [self.alloc_scratch(f"{prefix}_addr_b_{i}") for i in range(VLEN)],
            }

        regs_0 = alloc_batch_regs("R0")  # Iterations 0, 3, 6, ...
        regs_1 = alloc_batch_regs("R1")  # Iterations 1, 4, 7, ...
        regs_2 = alloc_batch_regs("R2")  # Iterations 2, 5, 8, ...

        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)

        # Initialize vector constants
        self.instrs.append({"valu": [
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
        ]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
            ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]),
        ]})

        # Pre-compute hash constant vectors
        hash_const_vecs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.alloc_scratch(f"hc1_{hi}", VLEN)
            c3 = self.alloc_scratch(f"hc3_{hi}", VLEN)
            hash_const_vecs.append((c1, c3))
            self.instrs.append({"valu": [
                ("vbroadcast", c1, self.scratch_const(val1)),
                ("vbroadcast", c3, self.scratch_const(val3)),
            ]})

        n_vector_batches = batch_size // VLEN

        # Build list of all batch offsets
        batch_offsets = []
        for round_idx in range(rounds):
            for batch_idx in range(0, n_vector_batches, 2):
                batch_offsets.append((batch_idx * VLEN, (batch_idx + 1) * VLEN))

        total_iterations = len(batch_offsets)

        def emit_setup(r, off_a, off_b):
            """Emit full setup for a batch (no pipelining)."""
            const_a = self.scratch_const(off_a)
            const_b = self.scratch_const(off_b)
            # Address computation
            self.instrs.append({"alu": [
                ("+", r['idx_base_a'], self.scratch["inp_indices_p"], const_a),
                ("+", r['val_base_a'], self.scratch["inp_values_p"], const_a),
                ("+", r['idx_base_b'], self.scratch["inp_indices_p"], const_b),
                ("+", r['val_base_b'], self.scratch["inp_values_p"], const_b),
            ]})
            # Vector loads
            self.instrs.append({"load": [("vload", r['v_idx_a'], r['idx_base_a']), ("vload", r['v_val_a'], r['val_base_a'])]})
            self.instrs.append({"load": [("vload", r['v_idx_b'], r['idx_base_b']), ("vload", r['v_val_b'], r['val_base_b'])]})
            # Tree addresses
            self.instrs.append({"valu": [("+", r['v_taddr_a'], v_forest_p, r['v_idx_a']), ("+", r['v_taddr_b'], v_forest_p, r['v_idx_b'])]})
            # Extract addresses
            self.instrs.append({"alu": [("+", r['addr_a'][i], r['v_taddr_a'] + i, zero_const) for i in range(VLEN)]
                                + [("+", r['addr_b'][i], r['v_taddr_b'] + i, zero_const) for i in range(4)]})
            self.instrs.append({"alu": [("+", r['addr_b'][i], r['v_taddr_b'] + i, zero_const) for i in range(4, VLEN)]})
            # Scattered loads
            for i in range(0, VLEN, 2):
                self.instrs.append({"load": [("load", r['v_node_a'] + i, r['addr_a'][i]), ("load", r['v_node_a'] + i + 1, r['addr_a'][i + 1])]})
            for i in range(0, VLEN, 2):
                self.instrs.append({"load": [("load", r['v_node_b'] + i, r['addr_b'][i]), ("load", r['v_node_b'] + i + 1, r['addr_b'][i + 1])]})
            # XOR
            self.instrs.append({"valu": [("^", r['v_val_a'], r['v_val_a'], r['v_node_a']), ("^", r['v_val_b'], r['v_val_b'], r['v_node_b'])]})

        def emit_hash_only(r):
            """Emit hash without pipelining (for last iteration)."""
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                const1_vec, const3_vec = hash_const_vecs[hi]
                self.instrs.append({"valu": [
                    (op1, r['v_htmp1_a'], r['v_val_a'], const1_vec),
                    (op3, r['v_htmp2_a'], r['v_val_a'], const3_vec),
                    (op1, r['v_htmp1_b'], r['v_val_b'], const1_vec),
                    (op3, r['v_htmp2_b'], r['v_val_b'], const3_vec),
                ]})
                self.instrs.append({"valu": [
                    (op2, r['v_val_a'], r['v_htmp1_a'], r['v_htmp2_a']),
                    (op2, r['v_val_b'], r['v_htmp1_b'], r['v_htmp2_b']),
                ]})

        def emit_finish_early(r, r_nxt, has_next):
            """Emit finish cycles 1-2: index prep and XOR.

            These must happen before next hash starts.
            Cycle 1: VALU (4 slots) + LOAD (2 slots for remaining v_node_b[6:8])
            Cycle 2: VALU (2 slots for +) + VALU (2 slots for XOR)
            """
            # Cycle 1: index prep (VALU 4 slots) + remaining scattered loads
            instr1 = {"valu": [
                ("&", r['v_htmp1_a'], r['v_val_a'], v_one), ("&", r['v_htmp1_b'], r['v_val_b'], v_one),
                ("multiply_add", r['v_idx_a'], r['v_idx_a'], v_two, v_one),
                ("multiply_add", r['v_idx_b'], r['v_idx_b'], v_two, v_one),
            ]}
            if has_next:
                instr1["load"] = [("load", r_nxt['v_node_b'] + 6, r_nxt['v_taddr_b'] + 6), ("load", r_nxt['v_node_b'] + 7, r_nxt['v_taddr_b'] + 7)]
            self.instrs.append(instr1)

            # Cycle 2: + for index (VALU 2 slots) + XOR for next batch (VALU 2 slots)
            instr2 = {"valu": [("+", r['v_idx_a'], r['v_idx_a'], r['v_htmp1_a']), ("+", r['v_idx_b'], r['v_idx_b'], r['v_htmp1_b'])]}
            if has_next:
                instr2["valu"].extend([("^", r_nxt['v_val_a'], r_nxt['v_val_a'], r_nxt['v_node_a']), ("^", r_nxt['v_val_b'], r_nxt['v_val_b'], r_nxt['v_node_b'])])
            self.instrs.append(instr2)

        def emit_finish_late(r):
            """Emit finish cycles 3-5: bounds check and stores.

            These can be overlapped with next hash.
            """
            # Cycle 3: VALU (2 slots for <)
            self.instrs.append({"valu": [("<", r['v_htmp1_a'], r['v_idx_a'], v_n_nodes), ("<", r['v_htmp1_b'], r['v_idx_b'], v_n_nodes)]})

            # Cycle 4: VALU (2 slots for bounds) + STORE (2 slots for v_val)
            self.instrs.append({
                "valu": [
                    ("multiply_add", r['v_idx_a'], r['v_idx_a'], r['v_htmp1_a'], v_zero),
                    ("multiply_add", r['v_idx_b'], r['v_idx_b'], r['v_htmp1_b'], v_zero),
                ],
                "store": [("vstore", r['val_base_a'], r['v_val_a']), ("vstore", r['val_base_b'], r['v_val_b'])],
            })

            # Cycle 5: STORE (2 slots for v_idx)
            self.instrs.append({"store": [("vstore", r['idx_base_a'], r['v_idx_a']), ("vstore", r['idx_base_b'], r['v_idx_b'])]})

        def emit_finish_with_loads(r, r_nxt, has_next):
            """Emit full finish (cycles 1-5)."""
            emit_finish_early(r, r_nxt, has_next)
            emit_finish_late(r)

        def emit_hash_with_full_prep(r_cur, r_nxt, off_a, off_b):
            """Emit hash while preparing next batch.

            Skip extract step - use v_taddr directly for scattered loads.
            Pipeline:
            - Cycle 0: addr computation (ALU)
            - Cycles 2-3: vloads (LOAD)
            - Cycle 4: tree addr (VALU - 2 slots)
            - Cycles 5-11: scattered loads using v_taddr directly (14 loads)
            - Remaining 2 loads done in finish
            """
            const_a = self.scratch_const(off_a)
            const_b = self.scratch_const(off_b)

            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                const1_vec, const3_vec = hash_const_vecs[hi]
                cycle_num = hi * 2

                instr_a = {"valu": [
                    (op1, r_cur['v_htmp1_a'], r_cur['v_val_a'], const1_vec),
                    (op3, r_cur['v_htmp2_a'], r_cur['v_val_a'], const3_vec),
                    (op1, r_cur['v_htmp1_b'], r_cur['v_val_b'], const1_vec),
                    (op3, r_cur['v_htmp2_b'], r_cur['v_val_b'], const3_vec),
                ]}

                instr_b = {"valu": [
                    (op2, r_cur['v_val_a'], r_cur['v_htmp1_a'], r_cur['v_htmp2_a']),
                    (op2, r_cur['v_val_b'], r_cur['v_htmp1_b'], r_cur['v_htmp2_b']),
                ]}

                if cycle_num == 0:
                    # Cycle 0: addr computation (ALU)
                    instr_a["alu"] = [
                        ("+", r_nxt['idx_base_a'], self.scratch["inp_indices_p"], const_a),
                        ("+", r_nxt['val_base_a'], self.scratch["inp_values_p"], const_a),
                        ("+", r_nxt['idx_base_b'], self.scratch["inp_indices_p"], const_b),
                        ("+", r_nxt['val_base_b'], self.scratch["inp_values_p"], const_b),
                    ]
                elif cycle_num == 2:
                    # Cycles 2-3: vloads (LOAD)
                    instr_a["load"] = [("vload", r_nxt['v_idx_a'], r_nxt['idx_base_a']), ("vload", r_nxt['v_val_a'], r_nxt['val_base_a'])]
                    instr_b["load"] = [("vload", r_nxt['v_idx_b'], r_nxt['idx_base_b']), ("vload", r_nxt['v_val_b'], r_nxt['val_base_b'])]
                elif cycle_num == 4:
                    # Cycle 4: tree addr (VALU - 2 slots)
                    instr_a["valu"].extend([
                        ("+", r_nxt['v_taddr_a'], v_forest_p, r_nxt['v_idx_a']),
                        ("+", r_nxt['v_taddr_b'], v_forest_p, r_nxt['v_idx_b']),
                    ])
                    # Cycle 5: first scattered loads using v_taddr directly (skip extract)
                    instr_b["load"] = [("load", r_nxt['v_node_a'], r_nxt['v_taddr_a']), ("load", r_nxt['v_node_a'] + 1, r_nxt['v_taddr_a'] + 1)]
                elif cycle_num == 6:
                    # Cycles 6-7: v_node_a[2:6]
                    instr_a["load"] = [("load", r_nxt['v_node_a'] + 2, r_nxt['v_taddr_a'] + 2), ("load", r_nxt['v_node_a'] + 3, r_nxt['v_taddr_a'] + 3)]
                    instr_b["load"] = [("load", r_nxt['v_node_a'] + 4, r_nxt['v_taddr_a'] + 4), ("load", r_nxt['v_node_a'] + 5, r_nxt['v_taddr_a'] + 5)]
                elif cycle_num == 8:
                    # Cycles 8-9: v_node_a[6:8] + v_node_b[0:2]
                    instr_a["load"] = [("load", r_nxt['v_node_a'] + 6, r_nxt['v_taddr_a'] + 6), ("load", r_nxt['v_node_a'] + 7, r_nxt['v_taddr_a'] + 7)]
                    instr_b["load"] = [("load", r_nxt['v_node_b'], r_nxt['v_taddr_b']), ("load", r_nxt['v_node_b'] + 1, r_nxt['v_taddr_b'] + 1)]
                elif cycle_num == 10:
                    # Cycles 10-11: v_node_b[2:6] (remaining v_node_b[6:8] loaded in finish)
                    instr_a["load"] = [("load", r_nxt['v_node_b'] + 2, r_nxt['v_taddr_b'] + 2), ("load", r_nxt['v_node_b'] + 3, r_nxt['v_taddr_b'] + 3)]
                    instr_b["load"] = [("load", r_nxt['v_node_b'] + 4, r_nxt['v_taddr_b'] + 4), ("load", r_nxt['v_node_b'] + 5, r_nxt['v_taddr_b'] + 5)]

                self.instrs.append(instr_a)
                self.instrs.append(instr_b)

        def emit_hash_with_finish_overlap(r_cur, r_nxt, r_prev, off_a, off_b, do_prep, do_finish_late):
            """Hash current batch with overlapped prep and finish_late.

            Schedule:
            - Cycles 0-6: hash + prep (same as emit_hash_with_full_prep)
            - Cycle 7 (B): hash + scattered loads + finish_late < for prev
            - Cycle 9 (B): hash + scattered loads + finish_late multiply_add + vstore for prev
            - Cycle 11 (B): hash + scattered loads + finish_late vstore for prev
            """
            const_a = self.scratch_const(off_a) if do_prep else None
            const_b = self.scratch_const(off_b) if do_prep else None

            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                const1_vec, const3_vec = hash_const_vecs[hi]
                cycle_num = hi * 2

                instr_a = {"valu": [
                    (op1, r_cur['v_htmp1_a'], r_cur['v_val_a'], const1_vec),
                    (op3, r_cur['v_htmp2_a'], r_cur['v_val_a'], const3_vec),
                    (op1, r_cur['v_htmp1_b'], r_cur['v_val_b'], const1_vec),
                    (op3, r_cur['v_htmp2_b'], r_cur['v_val_b'], const3_vec),
                ]}

                instr_b = {"valu": [
                    (op2, r_cur['v_val_a'], r_cur['v_htmp1_a'], r_cur['v_htmp2_a']),
                    (op2, r_cur['v_val_b'], r_cur['v_htmp1_b'], r_cur['v_htmp2_b']),
                ]}

                # Prep operations (same schedule as emit_hash_with_full_prep)
                if do_prep:
                    if cycle_num == 0:
                        instr_a["alu"] = [
                            ("+", r_nxt['idx_base_a'], self.scratch["inp_indices_p"], const_a),
                            ("+", r_nxt['val_base_a'], self.scratch["inp_values_p"], const_a),
                            ("+", r_nxt['idx_base_b'], self.scratch["inp_indices_p"], const_b),
                            ("+", r_nxt['val_base_b'], self.scratch["inp_values_p"], const_b),
                        ]
                    elif cycle_num == 2:
                        instr_a["load"] = [("vload", r_nxt['v_idx_a'], r_nxt['idx_base_a']), ("vload", r_nxt['v_val_a'], r_nxt['val_base_a'])]
                        instr_b["load"] = [("vload", r_nxt['v_idx_b'], r_nxt['idx_base_b']), ("vload", r_nxt['v_val_b'], r_nxt['val_base_b'])]
                    elif cycle_num == 4:
                        instr_a["valu"].extend([
                            ("+", r_nxt['v_taddr_a'], v_forest_p, r_nxt['v_idx_a']),
                            ("+", r_nxt['v_taddr_b'], v_forest_p, r_nxt['v_idx_b']),
                        ])
                        instr_b["load"] = [("load", r_nxt['v_node_a'], r_nxt['v_taddr_a']), ("load", r_nxt['v_node_a'] + 1, r_nxt['v_taddr_a'] + 1)]
                    elif cycle_num == 6:
                        instr_a["load"] = [("load", r_nxt['v_node_a'] + 2, r_nxt['v_taddr_a'] + 2), ("load", r_nxt['v_node_a'] + 3, r_nxt['v_taddr_a'] + 3)]
                        instr_b["load"] = [("load", r_nxt['v_node_a'] + 4, r_nxt['v_taddr_a'] + 4), ("load", r_nxt['v_node_a'] + 5, r_nxt['v_taddr_a'] + 5)]
                    elif cycle_num == 8:
                        instr_a["load"] = [("load", r_nxt['v_node_a'] + 6, r_nxt['v_taddr_a'] + 6), ("load", r_nxt['v_node_a'] + 7, r_nxt['v_taddr_a'] + 7)]
                        instr_b["load"] = [("load", r_nxt['v_node_b'], r_nxt['v_taddr_b']), ("load", r_nxt['v_node_b'] + 1, r_nxt['v_taddr_b'] + 1)]
                    elif cycle_num == 10:
                        instr_a["load"] = [("load", r_nxt['v_node_b'] + 2, r_nxt['v_taddr_b'] + 2), ("load", r_nxt['v_node_b'] + 3, r_nxt['v_taddr_b'] + 3)]
                        instr_b["load"] = [("load", r_nxt['v_node_b'] + 4, r_nxt['v_taddr_b'] + 4), ("load", r_nxt['v_node_b'] + 5, r_nxt['v_taddr_b'] + 5)]

                # finish_late operations overlapped in hash B cycles
                if do_finish_late:
                    if cycle_num == 6:
                        # finish_late cycle 3: < comparison (VALU 2 slots, using B cycle)
                        instr_b["valu"].extend([
                            ("<", r_prev['v_htmp1_a'], r_prev['v_idx_a'], v_n_nodes),
                            ("<", r_prev['v_htmp1_b'], r_prev['v_idx_b'], v_n_nodes),
                        ])
                    elif cycle_num == 8:
                        # finish_late cycle 4: multiply_add + vstore (VALU 2 slots + STORE 2 slots)
                        instr_b["valu"].extend([
                            ("multiply_add", r_prev['v_idx_a'], r_prev['v_idx_a'], r_prev['v_htmp1_a'], v_zero),
                            ("multiply_add", r_prev['v_idx_b'], r_prev['v_idx_b'], r_prev['v_htmp1_b'], v_zero),
                        ])
                        instr_b["store"] = [
                            ("vstore", r_prev['val_base_a'], r_prev['v_val_a']),
                            ("vstore", r_prev['val_base_b'], r_prev['v_val_b']),
                        ]
                    elif cycle_num == 10:
                        # finish_late cycle 5: vstore (STORE 2 slots)
                        instr_b["store"] = [
                            ("vstore", r_prev['idx_base_a'], r_prev['v_idx_a']),
                            ("vstore", r_prev['idx_base_b'], r_prev['v_idx_b']),
                        ]

                self.instrs.append(instr_a)
                self.instrs.append(instr_b)

        def get_regs_3way(iter_idx):
            """Get (r_cur, r_nxt, r_prev) for 3-way rotation."""
            mod = iter_idx % 3
            if mod == 0:
                return regs_0, regs_1, regs_2
            elif mod == 1:
                return regs_1, regs_2, regs_0
            else:
                return regs_2, regs_0, regs_1

        # Main loop with 3-way rotation for finish-hash overlap
        # Timeline:
        # - iter 0: full setup + hash + full finish (no overlap)
        # - iter 1: hash + finish_early only (iter 0 did its own finish_late, iter 1's will overlap with iter 2)
        # - iter 2 to N-2: hash (finish_late for iter-1) + finish_early
        # - iter N-1: hash (finish_late for iter-2) + full finish
        for iter_idx in range(total_iterations):
            off_a, off_b = batch_offsets[iter_idx]
            r_cur, r_nxt, r_prev = get_regs_3way(iter_idx)
            has_next = iter_idx < total_iterations - 1

            if iter_idx == 0:
                # First iteration: full setup + hash + full finish
                emit_setup(r_cur, off_a, off_b)
                if has_next:
                    next_off_a, next_off_b = batch_offsets[iter_idx + 1]
                    emit_hash_with_full_prep(r_cur, r_nxt, next_off_a, next_off_b)
                else:
                    emit_hash_only(r_cur)
                emit_finish_with_loads(r_cur, r_nxt, has_next)
            elif iter_idx == 1:
                # Second iteration: hash + finish_early only (no finish_late to overlap, iter 0 did its own)
                if has_next:
                    next_off_a, next_off_b = batch_offsets[iter_idx + 1]
                    emit_hash_with_finish_overlap(r_cur, r_nxt, r_prev, next_off_a, next_off_b, do_prep=True, do_finish_late=False)
                else:
                    emit_hash_only(r_cur)
                emit_finish_early(r_cur, r_nxt, has_next)
            elif iter_idx == total_iterations - 1:
                # Last iteration: hash (with finish_late for iter-1) + full finish
                emit_hash_with_finish_overlap(r_cur, r_nxt, r_prev, 0, 0, do_prep=False, do_finish_late=True)
                emit_finish_with_loads(r_cur, r_nxt, has_next=False)
            else:
                # Middle iterations: hash (with finish_late for iter-1) + finish_early
                next_off_a, next_off_b = batch_offsets[iter_idx + 1]
                emit_hash_with_finish_overlap(r_cur, r_nxt, r_prev, next_off_a, next_off_b, do_prep=True, do_finish_late=True)
                emit_finish_early(r_cur, r_nxt, has_next)

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
