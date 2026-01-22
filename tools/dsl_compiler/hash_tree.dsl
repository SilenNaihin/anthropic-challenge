# Hash+Tree Algorithm DSL Implementation
# ========================================
#
# This DSL file expresses the core algorithm from perf_takehome.py
# in a high-level, human-readable form.
#
# Algorithm:
# 1. For each round:
#    2. For each batch element:
#       3. idx = indices[i], val = values[i]
#       4. node_val = tree[idx]
#       5. mixed = val ^ node_val
#       6. hashed = hash(mixed)  # 6-stage hash function
#       7. new_idx = idx * 2 + (1 if hashed % 2 == 0 else 2)
#       8. new_idx = 0 if new_idx >= n_nodes else new_idx
#       9. Store back hashed value and new index

# ============== Constants ==============

# Architecture constants
var VLEN = 8
var n_nodes
var forest_p
var indices_p
var values_p
var rounds

# ============== Scalar Version (for reference) ==============

def scalar_iteration(i):
    # Load element data
    idx = load(indices_p + i)
    val = load(values_p + i)

    # Load tree node value
    node_addr = forest_p + idx
    node_val = load(node_addr)

    # XOR mix
    mixed = val ^ node_val

    # Hash (expands to 6 stages, 18 ALU ops)
    hashed = hash(mixed)

    # Compute new index
    # new_idx = 2*idx + (1 if hashed % 2 == 0 else 2)
    parity = hashed & 1      # 0 if even, 1 if odd
    offset = parity + 1      # 1 if even, 2 if odd
    new_idx = idx * 2 + offset

    # Bounds check
    in_bounds = new_idx < n_nodes
    if in_bounds:
        final_idx = new_idx
    else:
        final_idx = 0

    # Store results
    store(values_p + i, hashed)
    store(indices_p + i, final_idx)

# ============== Vectorized Version ==============

@vectorize(8)
def vector_batch(batch_offset):
    # Declare vectors
    vec v_idx
    vec v_val
    vec v_node_val
    vec v_mixed
    vec v_hashed
    vec v_addr
    vec v_new_idx
    vec v_parity
    vec v_offset
    vec v_in_bounds
    vec v_final_idx

    # Vector constants (broadcast)
    vec v_zero = broadcast(0)
    vec v_one = broadcast(1)
    vec v_two = broadcast(2)
    vec v_n_nodes = broadcast(n_nodes)
    vec v_forest_p = broadcast(forest_p)

    # Load batch of indices and values
    idx_base = indices_p + batch_offset
    val_base = values_p + batch_offset

    v_idx = vload(idx_base)
    v_val = vload(val_base)

    # Compute tree addresses (scattered - each element has different idx)
    v_addr = v_forest_p + v_idx

    # NOTE: Cannot use vload here because addresses are not contiguous!
    # Each element needs: tree[idx_0], tree[idx_1], ... tree[idx_7]
    # These are scattered addresses, must use scalar loads
    #
    # This is the fundamental bottleneck of tree traversal:
    # hash results determine next index, which are unpredictable

    # For now, express as 8 scalar loads (the DSL compiler should handle this)
    for lane in range(8):
        v_node_val[lane] = load(v_addr[lane])

    # XOR mix (fully vectorizable)
    v_mixed = v_val ^ v_node_val

    # Hash computation (fully vectorizable)
    # hash() expands to 6 stages:
    #   stage i: tmp1 = val op1 const1
    #            tmp2 = val op3 const3
    #            val = tmp1 op2 tmp2
    v_hashed = hash(v_mixed)

    # Index computation (fully vectorizable)
    v_parity = v_hashed & v_one
    v_offset = v_parity + v_one
    v_new_idx = v_idx * v_two + v_offset

    # Bounds check (fully vectorizable)
    v_in_bounds = v_new_idx < v_n_nodes

    # Select based on bounds (vectorizable using multiply_add trick)
    # final = in_bounds * new_idx + (1 - in_bounds) * 0
    #       = in_bounds * new_idx
    v_final_idx = v_new_idx * v_in_bounds

    # Store results (contiguous - vectorizable)
    vstore(val_base, v_hashed)
    vstore(idx_base, v_final_idx)

# ============== Main Loop ==============

def main_loop():
    # Process all rounds
    for round in range(rounds):
        # Process all batches in this round
        for batch in range(0, batch_size, VLEN):
            vector_batch(batch)

# ============== Notes for Optimization ==============
#
# Key insights for VLIW packing:
#
# 1. HASH COMPUTATION (hot path - 6 stages x 16 rounds x 16 batches = 1536 invocations)
#    - Each stage has 3 ops: tmp1 = val op1 const1
#                           tmp2 = val op3 const3
#                           val = tmp1 op2 tmp2
#    - tmp1 and tmp2 are INDEPENDENT - can run in parallel
#    - With VLEN=8, need 2 valu slots per stage (tmp1 || tmp2), then 1 for combine
#    - 6 stages * 2 cycles = 12 cycles per hash minimum
#    - But can pipeline: while computing hash for batch K, prepare batch K+1
#
# 2. MEMORY ACCESS (scattered reads are the bottleneck)
#    - vload/vstore: Great for indices/values arrays (contiguous)
#    - Tree lookups: MUST be scalar (indices are data-dependent, non-contiguous)
#    - 8 scalar loads per batch, only 2 load slots per cycle = 4 cycles minimum
#    - Can overlap with hash computation (different engines)
#
# 3. SOFTWARE PIPELINING
#    - While hash runs on VALU (12+ cycles), use ALU/LOAD for next batch prep
#    - Pipeline: setup(K+1) || hash(K) || finish(K-1)
#    - Double-buffering: Need two sets of registers (even/odd iterations)
#
# 4. SLOT UTILIZATION TARGET
#    - 12 ALU slots: Index computation, address math
#    - 6 VALU slots: Hash computation (uses 4-6 per cycle)
#    - 2 LOAD slots: Memory access (always saturated during scatter loads)
#    - 2 STORE slots: Result writeback
#    - 1 FLOW slot: Bounds check (can often be avoided with multiply_add trick)
