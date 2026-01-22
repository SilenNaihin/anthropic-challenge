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
