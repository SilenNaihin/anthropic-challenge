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
        """Software-pipelined vectorized dual-batch implementation.

        Key optimization: Overlap preparation of batch K+1 with hash computation of batch K.
        During 16 hash cycles, we use free ALU/LOAD engines for next batch prep.
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

        # Two complete register sets: even (E) and odd (O) iterations
        # Each set has everything needed for a dual-batch iteration
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

        regs_E = alloc_batch_regs("E")  # Even iterations
        regs_O = alloc_batch_regs("O")  # Odd iterations

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

        def emit_hash_with_prep(r_cur, r_nxt, off_a, off_b):
            """Emit hash for current batch while preparing next batch.

            Pipeline schedule during 16 hash cycles:
            - Cycle 0: addr comp for next (ALU)
            - Cycles 2-3: vloads for next (LOAD)
            - Cycle 4: tree addr for next (VALU - uses 2 free slots)
            - Cycles 5-6: extract for next (ALU)
            - Cycles 7-14: scattered loads for next (LOAD)
            """
            const_a = self.scratch_const(off_a)
            const_b = self.scratch_const(off_b)

            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                const1_vec, const3_vec = hash_const_vecs[hi]
                cycle_num = hi * 2  # 0, 2, 4, 6, 8, 10, 12, 14

                # Cycle A: op1 + op3 for hash (4 VALU slots)
                instr_a = {"valu": [
                    (op1, r_cur['v_htmp1_a'], r_cur['v_val_a'], const1_vec),
                    (op3, r_cur['v_htmp2_a'], r_cur['v_val_a'], const3_vec),
                    (op1, r_cur['v_htmp1_b'], r_cur['v_val_b'], const1_vec),
                    (op3, r_cur['v_htmp2_b'], r_cur['v_val_b'], const3_vec),
                ]}

                # Cycle B: op2 for hash (2 VALU slots)
                instr_b = {"valu": [
                    (op2, r_cur['v_val_a'], r_cur['v_htmp1_a'], r_cur['v_htmp2_a']),
                    (op2, r_cur['v_val_b'], r_cur['v_htmp1_b'], r_cur['v_htmp2_b']),
                ]}

                # Add pipelined prep operations
                if cycle_num == 0:
                    # Address computation for next batch
                    instr_a["alu"] = [
                        ("+", r_nxt['idx_base_a'], self.scratch["inp_indices_p"], const_a),
                        ("+", r_nxt['val_base_a'], self.scratch["inp_values_p"], const_a),
                        ("+", r_nxt['idx_base_b'], self.scratch["inp_indices_p"], const_b),
                        ("+", r_nxt['val_base_b'], self.scratch["inp_values_p"], const_b),
                    ]
                elif cycle_num == 2:
                    # vloads for next batch
                    instr_a["load"] = [("vload", r_nxt['v_idx_a'], r_nxt['idx_base_a']), ("vload", r_nxt['v_val_a'], r_nxt['val_base_a'])]
                    instr_b["load"] = [("vload", r_nxt['v_idx_b'], r_nxt['idx_base_b']), ("vload", r_nxt['v_val_b'], r_nxt['val_base_b'])]
                elif cycle_num == 4:
                    # Tree addresses for next batch (2 free VALU slots)
                    instr_a["valu"].extend([
                        ("+", r_nxt['v_taddr_a'], v_forest_p, r_nxt['v_idx_a']),
                        ("+", r_nxt['v_taddr_b'], v_forest_p, r_nxt['v_idx_b']),
                    ])
                    # Start extract in cycle B
                    instr_b["alu"] = [("+", r_nxt['addr_a'][i], r_nxt['v_taddr_a'] + i, zero_const) for i in range(VLEN)] + \
                                     [("+", r_nxt['addr_b'][i], r_nxt['v_taddr_b'] + i, zero_const) for i in range(4)]
                elif cycle_num == 6:
                    # Finish extract
                    instr_a["alu"] = [("+", r_nxt['addr_b'][i], r_nxt['v_taddr_b'] + i, zero_const) for i in range(4, VLEN)]
                    # Start scattered loads
                    instr_b["load"] = [("load", r_nxt['v_node_a'], r_nxt['addr_a'][0]), ("load", r_nxt['v_node_a'] + 1, r_nxt['addr_a'][1])]
                elif cycle_num == 8:
                    instr_a["load"] = [("load", r_nxt['v_node_a'] + 2, r_nxt['addr_a'][2]), ("load", r_nxt['v_node_a'] + 3, r_nxt['addr_a'][3])]
                    instr_b["load"] = [("load", r_nxt['v_node_a'] + 4, r_nxt['addr_a'][4]), ("load", r_nxt['v_node_a'] + 5, r_nxt['addr_a'][5])]
                elif cycle_num == 10:
                    instr_a["load"] = [("load", r_nxt['v_node_a'] + 6, r_nxt['addr_a'][6]), ("load", r_nxt['v_node_a'] + 7, r_nxt['addr_a'][7])]
                    instr_b["load"] = [("load", r_nxt['v_node_b'], r_nxt['addr_b'][0]), ("load", r_nxt['v_node_b'] + 1, r_nxt['addr_b'][1])]
                elif cycle_num == 12:
                    instr_a["load"] = [("load", r_nxt['v_node_b'] + 2, r_nxt['addr_b'][2]), ("load", r_nxt['v_node_b'] + 3, r_nxt['addr_b'][3])]
                    instr_b["load"] = [("load", r_nxt['v_node_b'] + 4, r_nxt['addr_b'][4]), ("load", r_nxt['v_node_b'] + 5, r_nxt['addr_b'][5])]
                elif cycle_num == 14:
                    instr_a["load"] = [("load", r_nxt['v_node_b'] + 6, r_nxt['addr_b'][6]), ("load", r_nxt['v_node_b'] + 7, r_nxt['addr_b'][7])]

                self.instrs.append(instr_a)
                self.instrs.append(instr_b)

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

        def emit_finish_with_loads(r, r_nxt, has_next):
            """Emit index computation, bounds check, and stores, overlapping scattered loads for next."""
            # Cycle 1: VALU (4 slots) + LOAD (2 slots)
            instr1 = {"valu": [
                ("&", r['v_htmp1_a'], r['v_val_a'], v_one), ("&", r['v_htmp1_b'], r['v_val_b'], v_one),
                ("multiply_add", r['v_idx_a'], r['v_idx_a'], v_two, v_one),
                ("multiply_add", r['v_idx_b'], r['v_idx_b'], v_two, v_one),
            ]}
            if has_next:
                instr1["load"] = [("load", r_nxt['v_node_a'], r_nxt['addr_a'][0]), ("load", r_nxt['v_node_a'] + 1, r_nxt['addr_a'][1])]
            self.instrs.append(instr1)

            # Cycle 2: VALU (2 slots) + LOAD (2 slots)
            instr2 = {"valu": [("+", r['v_idx_a'], r['v_idx_a'], r['v_htmp1_a']), ("+", r['v_idx_b'], r['v_idx_b'], r['v_htmp1_b'])]}
            if has_next:
                instr2["load"] = [("load", r_nxt['v_node_a'] + 2, r_nxt['addr_a'][2]), ("load", r_nxt['v_node_a'] + 3, r_nxt['addr_a'][3])]
            self.instrs.append(instr2)

            # Cycle 3: VALU (2 slots) + LOAD (2 slots)
            instr3 = {"valu": [("<", r['v_htmp1_a'], r['v_idx_a'], v_n_nodes), ("<", r['v_htmp1_b'], r['v_idx_b'], v_n_nodes)]}
            if has_next:
                instr3["load"] = [("load", r_nxt['v_node_a'] + 4, r_nxt['addr_a'][4]), ("load", r_nxt['v_node_a'] + 5, r_nxt['addr_a'][5])]
            self.instrs.append(instr3)

            # Cycle 4: FLOW (1 slot) + STORE (2 slots) + LOAD (2 slots)
            instr4 = {
                "flow": [("vselect", r['v_idx_a'], r['v_htmp1_a'], r['v_idx_a'], v_zero)],
                "store": [("vstore", r['val_base_a'], r['v_val_a']), ("vstore", r['val_base_b'], r['v_val_b'])],
            }
            if has_next:
                instr4["load"] = [("load", r_nxt['v_node_a'] + 6, r_nxt['addr_a'][6]), ("load", r_nxt['v_node_a'] + 7, r_nxt['addr_a'][7])]
            self.instrs.append(instr4)

            # Cycle 5: FLOW (1 slot) + STORE (1 slot) + LOAD (2 slots)
            instr5 = {
                "flow": [("vselect", r['v_idx_b'], r['v_htmp1_b'], r['v_idx_b'], v_zero)],
                "store": [("vstore", r['idx_base_a'], r['v_idx_a'])],
            }
            if has_next:
                instr5["load"] = [("load", r_nxt['v_node_b'], r_nxt['addr_b'][0]), ("load", r_nxt['v_node_b'] + 1, r_nxt['addr_b'][1])]
            self.instrs.append(instr5)

            # Cycle 6: STORE (1 slot) + LOAD (2 slots)
            instr6 = {"store": [("vstore", r['idx_base_b'], r['v_idx_b'])]}
            if has_next:
                instr6["load"] = [("load", r_nxt['v_node_b'] + 2, r_nxt['addr_b'][2]), ("load", r_nxt['v_node_b'] + 3, r_nxt['addr_b'][3])]
            self.instrs.append(instr6)

        def emit_xor(r):
            """Emit XOR for a batch after scattered loads complete."""
            self.instrs.append({"valu": [("^", r['v_val_a'], r['v_val_a'], r['v_node_a']), ("^", r['v_val_b'], r['v_val_b'], r['v_node_b'])]})

        def emit_hash_with_full_prep(r_cur, r_nxt, off_a, off_b):
            """Emit hash while preparing next batch (addr, vloads, tree addr, extract)."""
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

                # Pack operations on specific cycles
                if cycle_num == 0:
                    # Cycle 0: address computation (ALU)
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
                    # Cycle 4: tree addresses (VALU - 2 free slots)
                    instr_a["valu"].extend([
                        ("+", r_nxt['v_taddr_a'], v_forest_p, r_nxt['v_idx_a']),
                        ("+", r_nxt['v_taddr_b'], v_forest_p, r_nxt['v_idx_b']),
                    ])
                    # Cycle 5: extract part 1 (ALU - 12 ops)
                    instr_b["alu"] = [("+", r_nxt['addr_a'][i], r_nxt['v_taddr_a'] + i, zero_const) for i in range(VLEN)] + \
                                     [("+", r_nxt['addr_b'][i], r_nxt['v_taddr_b'] + i, zero_const) for i in range(4)]
                elif cycle_num == 6:
                    # Cycle 6: extract part 2 (ALU - 4 ops)
                    instr_a["alu"] = [("+", r_nxt['addr_b'][i], r_nxt['v_taddr_b'] + i, zero_const) for i in range(4, VLEN)]

                self.instrs.append(instr_a)
                self.instrs.append(instr_b)

        # Main loop with full pipelining
        for iter_idx in range(total_iterations):
            off_a, off_b = batch_offsets[iter_idx]
            is_even = (iter_idx % 2 == 0)
            r_cur = regs_E if is_even else regs_O
            r_nxt = regs_O if is_even else regs_E
            has_next = iter_idx < total_iterations - 1

            if iter_idx == 0:
                # Prologue: full setup for first iteration
                emit_setup(r_cur, off_a, off_b)

            if has_next:
                next_off_a, next_off_b = batch_offsets[iter_idx + 1]
                # Hash current batch while fully preparing next batch
                emit_hash_with_full_prep(r_cur, r_nxt, next_off_a, next_off_b)
            else:
                # Last iteration: just hash
                emit_hash_only(r_cur)

            # Finish current batch (overlaps 12 scattered loads for next)
            emit_finish_with_loads(r_cur, r_nxt, has_next)

            if has_next:
                # Remaining scattered loads: only addr_b[4:8] (4 loads = 2 cycles)
                self.instrs.append({"load": [("load", r_nxt['v_node_b'] + 4, r_nxt['addr_b'][4]), ("load", r_nxt['v_node_b'] + 5, r_nxt['addr_b'][5])]})
                self.instrs.append({"load": [("load", r_nxt['v_node_b'] + 6, r_nxt['addr_b'][6]), ("load", r_nxt['v_node_b'] + 7, r_nxt['addr_b'][7])]})
                emit_xor(r_nxt)

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
