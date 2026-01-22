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
