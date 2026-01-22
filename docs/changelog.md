# Changelog

Track every code change with the hypothesis behind it and the measured result.

## Format

```
## [Date] - Brief description
**Hypothesis**: What we expected to happen and why
**Change**: What we actually changed
**Result**: Cycle count, speedup, any issues
**Verdict**: Keep / Revert / Iterate
```

---

## [2025-01-22] - Replace vselect with multiply_add for bounds check

**Hypothesis**: The bounds check `idx < n_nodes ? idx : 0` uses vselect (FLOW engine, 1 slot). We can replace it with `idx * (idx < n_nodes)` using multiply_add (VALU engine, 6 slots). This eliminates the 2 FLOW operations that forced sequential execution.

**Change**: Replaced the two vselect operations with multiply_add:
- Old: `vselect(idx, mask, idx, zero)` using FLOW engine (1 slot)
- New: `multiply_add(idx, idx, mask, zero)` using VALU engine
- Reduced finish phase from 6 cycles to 5 cycles

**Result**: 4,692 → 4,436 cycles (33.30x speedup)

**Analysis**:
- Saved 256 cycles (1 cycle per iteration × 256 iterations)
- Both vselects now execute in parallel in cycle 4 instead of cycles 4 and 5
- Stores for v_idx_a and v_idx_b moved to cycle 5 together
- XOR for next batch remains in cycle 5

**Verdict**: Keep - eliminates flow engine bottleneck.

---

## [2025-01-22] - Combine XOR with last finish cycle

**Hypothesis**: XOR uses 2 VALU ops. Last finish cycle (cycle 6) only has 1 STORE op and VALU is free. Can combine them.

**Change**: Added XOR for next batch (r_nxt) during finish cycle 6, eliminating the separate emit_xor() call.

**Result**: 4,947 → 4,692 cycles (31.49x speedup)

**Analysis**:
- Saved 255 cycles (1 cycle per iteration × 255 iterations with next batch)
- XOR reads v_node_b which was loaded in cycles 1-4, so it's ready by cycle 6
- The last iteration has no XOR needed (no next batch)

**Verdict**: Keep - clean optimization with no downsides.

---

## [2025-01-22] - Pipeline v_node_a loads during hash cycles 8-11

**Hypothesis**: LOAD engine is free during hash cycles 8-15. Can move v_node_a scattered loads into cycles 8-11 (4 cycles × 2 loads = 8 loads), saving 4 cycles from finish phase.

**Change**: Added scattered loads for v_node_a[0:8] during hash cycles 8-11. Updated emit_finish_with_loads to only load v_node_b.

**Result**: 5,457 → 4,947 cycles (29.86x speedup)

**Analysis**:
- Saved 510 cycles (~2 cycles per iteration × 256 iterations)
- v_node_a loaded during hash, v_node_b loaded during finish
- Failed attempt: Adding v_node_b loads to hash cycles 12-15 caused correctness failure
  - addr_b[4:8] computed in cycle 6, should be ready by cycle 12
  - Root cause still unclear - possibly a timing issue with the store phase

**Verdict**: Keep - incremental improvement.

---

## [2025-01-22] - Overlap scattered loads with finish operations

**Hypothesis**: During emit_finish (6 cycles), LOAD engine is completely free. Can do 12 of 16 scattered loads in parallel, leaving only 4 for afterward.

**Change**: Created `emit_finish_with_loads` that overlaps scattered loads during finish:
- Cycles 1-6: finish operations (VALU, FLOW, STORE) + 2 loads each
- After finish: only 4 loads remain (addr_b[4:8])

**Result**: 6,987 → 5,457 cycles (27.07x speedup)

**Analysis**:
- Saved 1,530 cycles (6 cycles per iteration × 128 iterations - some overhead)
- Per iteration now: 16 hash + 6 finish (with 12 loads) + 2 remaining loads + 1 XOR = 25 cycles
- Failed attempt: Moving ALL scattered loads into hash cycles 7-15 caused correctness failures
  - Theory was sound (addr computed by cycle 6, loads in cycles 7-14)
  - Root cause unclear - possibly timing/dependency issue at round boundaries

**Verdict**: Keep - significant improvement with working correctness.

---

## [2025-01-22] - Partial software pipelining: addr, vloads, tree addr, extract

**Hypothesis**: Even if full pipelining with scattered loads failed, we can still pipeline the simpler parts:
- Address computation (ALU) during hash cycle 0
- vloads (LOAD) during hash cycles 2-3
- Tree addresses (VALU - 2 free slots) during hash cycle 4
- Extract (ALU) during hash cycles 5-6

This should save ~7 cycles per iteration.

**Change**: Implemented partial software pipelining using double-buffered register sets (regs_E, regs_O). The `emit_hash_with_full_prep` function overlaps addr/vloads/tree_addr/extract computation for batch K+1 during hash of batch K.

**Result**: 8,517 → 6,987 cycles (21.14x speedup)

**Analysis**:
- Saved ~1,530 cycles total (~12 cycles per iteration)
- Key insight: Tree address uses 2 VALU ops which fit in the 2 free slots during hash (hash uses 4 of 6)
- Scattered loads still happen sequentially after finish, but prep work is now overlapped
- Double-buffering avoids the buggy pointer-swapping from previous attempt

**Verdict**: Keep - significant improvement with clean implementation.

---

## [2025-01-22] - Failed: Full software pipelining attempt

**Hypothesis**: During 16 hash cycles (VALU), ALU and LOAD engines are free. Could overlap preparation of next batch:
- Cycle 0: addr comp for next (ALU)
- Cycles 1-2: vloads for next (LOAD)
- Cycle 3: tree addr for next (VALU - 2 free slots!)
- Cycles 4-5: extract for next (ALU)
- Cycles 6-13: scattered loads for next (LOAD)
Expected to save ~14 cycles per iteration = ~1800 cycles total.

**Change**: Implemented full software pipelining with register double-buffering (cur/nxt register sets).

**Result**: Correctness failure - "Incorrect result on round 1". The register swapping logic was buggy.

**Analysis**:
- The idea is sound (14 cycles of prep fit within 16 hash cycles)
- Tree addr can use 2 free VALU slots during hash (hash uses 4 of 6)
- Implementation complexity is high due to register handoff between iterations
- Likely bug in swap logic where v_tmp1_cur was reused for both tree addresses and intermediate values

**Verdict**: Reverted. Need cleaner implementation approach.

---

## [2025-01-22] - Combined vselect with stores and packed index ops

**Hypothesis**: vselect uses flow engine, vstore uses store engine - can run in same cycle. Also & and multiply_add are independent and can be packed.

**Changes**:
1. Combined & with multiply_add in same cycle (4 valu slots)
2. Interleaved vselect with vstore operations (flow + store engines)

**Result**: 9,029 → 8,773 → 8,517 cycles (17.35x speedup)

**Verdict**: Keep - good progress from multi-engine packing.

---

## [2025-01-22] - Eliminated vselect for index computation

**Hypothesis**: The flow engine only has 1 slot. Each vselect takes a full cycle. Replacing 2 vselects with arithmetic should save 2 cycles per batch pair.

**Change**: Rewrote `2*idx + (1 if val%2==0 else 2)` as `2*idx + 1 + (val & 1)` using multiply_add + add.

**Result**: 9,541 → 9,029 cycles (16.36x speedup)

**Verdict**: Keep - small but correct improvement.

---

## [2025-01-22] - Dual-batch parallel hash processing

**Hypothesis**: Processing 2 batches in parallel uses 4 of 6 valu slots, should nearly halve hash time.

**Changes**:
1. Added build_vhash_dual to process 2 batches' hash simultaneously
2. Doubled vector registers (v_idx_a/b, v_val_a/b, etc.)
3. Interleaved loads for both batches

**Result**: 14,403 → 9,541 cycles (15.49x speedup)

**Verdict**: Keep - significant improvement from better slot utilization.

---

## [2025-01-22] - Vectorization + hash constant pre-computation

**Hypothesis**: Vectorizing with VLEN=8 should give ~8x speedup. Pre-computing hash constants saves broadcast cycles.

**Changes**:
1. Vectorized main loop using vload/vstore for contiguous arrays
2. Used VALU operations for hash, XOR, index computation
3. Scattered tree loads use scalar loads (2 per cycle)
4. Packed 8 address extractions into 1 cycle (12 ALU slots available)
5. Pre-computed hash constant vectors outside loop (saves 6 broadcasts per iteration)

**Result**: 147,734 → 22,076 → 14,403 cycles (10.26x speedup)

**Verdict**: Keep - passes 3 tests. Need 7x more improvement for next threshold (2,164).

---

## [2025-01-22] - Initial setup

**Hypothesis**: N/A - baseline measurement

**Change**: No code changes, established baseline

**Result**: 147,734 cycles (baseline)

**Verdict**: Starting point established

---

<!-- Add new entries above this line -->
