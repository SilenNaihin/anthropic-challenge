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
        """Vectorized dual-batch implementation."""
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

        # Dual batch vector registers
        v_idx_a = self.alloc_scratch("v_idx_a", VLEN)
        v_val_a = self.alloc_scratch("v_val_a", VLEN)
        v_node_a = self.alloc_scratch("v_node_a", VLEN)
        v_tmp1_a = self.alloc_scratch("v_tmp1_a", VLEN)
        v_tmp2_a = self.alloc_scratch("v_tmp2_a", VLEN)
        v_tmp3_a = self.alloc_scratch("v_tmp3_a", VLEN)

        v_idx_b = self.alloc_scratch("v_idx_b", VLEN)
        v_val_b = self.alloc_scratch("v_val_b", VLEN)
        v_node_b = self.alloc_scratch("v_node_b", VLEN)
        v_tmp1_b = self.alloc_scratch("v_tmp1_b", VLEN)
        v_tmp2_b = self.alloc_scratch("v_tmp2_b", VLEN)
        v_tmp3_b = self.alloc_scratch("v_tmp3_b", VLEN)

        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)

        addr_a = [self.alloc_scratch(f"addr_a_{i}") for i in range(VLEN)]
        addr_b = [self.alloc_scratch(f"addr_b_{i}") for i in range(VLEN)]

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

        idx_base_a = self.alloc_scratch("idx_base_a")
        val_base_a = self.alloc_scratch("val_base_a")
        idx_base_b = self.alloc_scratch("idx_base_b")
        val_base_b = self.alloc_scratch("val_base_b")

        n_vector_batches = batch_size // VLEN

        for round_idx in range(rounds):
            for batch_idx in range(0, n_vector_batches, 2):
                const_a = self.scratch_const(batch_idx * VLEN)
                const_b = self.scratch_const((batch_idx + 1) * VLEN)

                # Compute addresses
                self.instrs.append({"alu": [
                    ("+", idx_base_a, self.scratch["inp_indices_p"], const_a),
                    ("+", val_base_a, self.scratch["inp_values_p"], const_a),
                    ("+", idx_base_b, self.scratch["inp_indices_p"], const_b),
                    ("+", val_base_b, self.scratch["inp_values_p"], const_b),
                ]})

                # Vector loads
                self.instrs.append({"load": [("vload", v_idx_a, idx_base_a), ("vload", v_val_a, val_base_a)]})
                self.instrs.append({"load": [("vload", v_idx_b, idx_base_b), ("vload", v_val_b, val_base_b)]})

                # Tree addresses
                self.instrs.append({"valu": [("+", v_tmp1_a, v_forest_p, v_idx_a), ("+", v_tmp1_b, v_forest_p, v_idx_b)]})

                # Extract (12 alu slots, 16 ops = 2 cycles)
                self.instrs.append({"alu": [("+", addr_a[i], v_tmp1_a + i, zero_const) for i in range(VLEN)]
                                    + [("+", addr_b[i], v_tmp1_b + i, zero_const) for i in range(4)]})
                self.instrs.append({"alu": [("+", addr_b[i], v_tmp1_b + i, zero_const) for i in range(4, VLEN)]})

                # Scattered loads (8 cycles for 16 loads)
                for i in range(0, VLEN, 2):
                    self.instrs.append({"load": [("load", v_node_a + i, addr_a[i]), ("load", v_node_a + i + 1, addr_a[i + 1])]})
                for i in range(0, VLEN, 2):
                    self.instrs.append({"load": [("load", v_node_b + i, addr_b[i]), ("load", v_node_b + i + 1, addr_b[i + 1])]})

                # XOR
                self.instrs.append({"valu": [("^", v_val_a, v_val_a, v_node_a), ("^", v_val_b, v_val_b, v_node_b)]})

                # Dual hash
                self.instrs.extend(self.build_vhash_dual(v_val_a, v_tmp1_a, v_tmp2_a, v_val_b, v_tmp1_b, v_tmp2_b, hash_const_vecs))

                # Index computation: idx = 2*idx + 1 + (val & 1)
                # & and multiply_add are independent - combine them (4 valu slots)
                self.instrs.append({"valu": [
                    ("&", v_tmp1_a, v_val_a, v_one), ("&", v_tmp1_b, v_val_b, v_one),
                    ("multiply_add", v_idx_a, v_idx_a, v_two, v_one),
                    ("multiply_add", v_idx_b, v_idx_b, v_two, v_one),
                ]})
                self.instrs.append({"valu": [("+", v_idx_a, v_idx_a, v_tmp1_a), ("+", v_idx_b, v_idx_b, v_tmp1_b)]})
                # Bounds check and store interleaved
                self.instrs.append({"valu": [("<", v_tmp1_a, v_idx_a, v_n_nodes), ("<", v_tmp1_b, v_idx_b, v_n_nodes)]})
                # vselect A + store val (val doesn't need the bounded idx)
                self.instrs.append({
                    "flow": [("vselect", v_idx_a, v_tmp1_a, v_idx_a, v_zero)],
                    "store": [("vstore", val_base_a, v_val_a), ("vstore", val_base_b, v_val_b)],
                })
                # vselect B + store idx A (idx_a is now bounded)
                self.instrs.append({
                    "flow": [("vselect", v_idx_b, v_tmp1_b, v_idx_b, v_zero)],
                    "store": [("vstore", idx_base_a, v_idx_a)],
                })
                # Final store idx B
                self.instrs.append({"store": [("vstore", idx_base_b, v_idx_b)]})

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
