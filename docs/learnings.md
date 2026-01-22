# Learnings

Document insights, gotchas, and what worked vs didn't.

---

## Architecture Insights

### VLIW Execution Model
- All engine slots execute in parallel within a cycle
- Effects (writes) happen at END of cycle, after all reads
- This means you CAN read and write the same address in one cycle (read sees old value)

### Slot Limits Per Cycle
- ALU: 12 scalar ops
- VALU: 6 vector ops (each processes 8 elements)
- Load: 2 ops
- Store: 2 ops
- Flow: 1 op

### Vector Operations
- VLEN = 8 (vector length)
- `vload`/`vstore` require contiguous memory addresses
- Scattered access (like tree lookup by index) can't directly vectorize

---

## Gotchas

### Test Modification Warning
- LLMs frequently "cheat" by modifying tests to get better scores
- ALWAYS verify: `git diff origin/main tests/`
- The submission tests use a FROZEN copy of the simulator

### Multicore is Disabled
- `N_CORES = 1` intentionally
- Don't waste time trying to enable multicore

### Debug Instructions
- `debug` engine instructions are ignored by submission simulator
- Use them freely for development without affecting cycle count

---

## What Works

- **Vectorization**: VLEN=8 gives ~8x theoretical speedup
- **Dual-batch processing**: Processing 2 batches in parallel uses more valu slots
- **Multi-engine packing**: Combine flow+store, valu+alu in same cycle
- **Arithmetic instead of select**: `2*idx + (1 if even else 2)` → `2*idx + 1 + (val&1)`
- **Pre-computing constants**: Hash constant vectors outside loop saves broadcasts
- **Bounds check via multiply**: `vselect(idx, mask, idx, zero)` → `multiply_add(idx, idx, mask, zero)` (VALU instead of FLOW)
- **3-way register rotation**: Allows overlapping finish_late(K-1) with hash(K) while prepping K+1
- **Split finish into early/late**: finish_early (index prep + XOR) must happen before next hash, finish_late (bounds + stores) can overlap

## What Doesn't Work

- **Packing ops that share write/read in same cycle**: Reads see old values, writes happen at end
- **Triple batch hash**: 32 batches / 3 doesn't divide evenly, complex remainder handling
- **Overlapping scattered loads with valu**: Everything depends on loaded data

---

## Software Pipelining Strategy

### Theoretical Opportunity
During 16 hash cycles (uses 4 of 6 VALU slots), we have free capacity:
- ALU: 12 slots completely free
- LOAD: 2 slots completely free
- VALU: 2 slots free (hash uses 4)

For NEXT batch, we need:
- addr comp: 1 cycle ALU
- vloads: 2 cycles LOAD
- tree addr: 1 cycle VALU (only 2 ops - fits in free slots!)
- extract: 2 cycles ALU
- scattered loads: 8 cycles LOAD
Total: 14 cycles fits in 16 hash cycles

### Key Insight
Tree address computation (2 VALU ops) can run DURING hash cycles using the 2 free VALU slots. This is the critical enabler for full software pipelining.

### Implementation Complexity
- Requires double-buffering registers (cur/nxt sets)
- Register swap logic at end of each iteration is error-prone
- Careful tracking of which register holds what state
- Easy to accidentally share registers between unrelated values

### Current Bottleneck Analysis (3,674 cycles)
Per iteration at 3,674 cycles / 256 iterations = 14.3 cycles:
- Hash (with overlapped prep + finish_late): 12 cycles
  - Overlaps: addr comp, vloads, tree addr for next batch
  - Cycles 5-11: scattered loads (14 loads)
  - B cycles 7, 9, 11: finish_late operations for prev batch
- Finish_early (can't overlap): 2 cycles
  - Cycle 1: remaining 2 loads + index prep
  - Cycle 2: + for index + XOR for next
Total: 14 cycles per iteration (close to measured 14.3)

### Previous Bottleneck Analysis (4,436 cycles)
Per iteration at 4,436 cycles / 256 iterations = 17.3 cycles:
- Hash (with overlapped prep + v_node_a loads): 12 cycles
  - Overlaps: addr comp, vloads, tree addr, extract for next batch
  - Cycles 8-11: v_node_a scattered loads (8 loads)
- Finish (with v_node_b loads + XOR): 5 cycles
  - Cycles 1-4: v_node_b scattered loads (8 loads)
  - Cycle 5: Stores + XOR for next batch
Total: 17 cycles per iteration (close to measured 17.3)

### What Didn't Work: Scattered Loads During Hash
Attempted to move all 16 scattered loads into hash cycles 7-15 (9 cycles available). Theory was sound:
- addr_a computed at cycle 5, addr_b[0:4] at cycle 5, addr_b[4:8] at cycle 6
- Loads starting at cycle 7 should have all addresses available
- Result: Correctness failure ("Incorrect output values")
- Hypothesis: Timing issue at round boundaries or register dependency bug

### Remaining Optimization Opportunities
1. ~~**Finish-hash overlap**: Move XOR to end of hash (cycle 11), then overlap finish VALU ops with next hash B cycles. Theory: 12 cycles/iteration instead of 17 = ~3,100 cycles.~~ **DONE** - achieved 3,674 cycles (14 cycles/iter)
2. **Triple-batch processing**: Use all 6 VALU slots during hash instead of 4. Would need different pipeline structure.
3. **Overlap finish_early**: The remaining 2 finish_early cycles might be overlapable with hash A cycles (which have 2 free VALU slots).
4. **Reduce scattered loads**: If some tree lookups could be done with vector gather, might save cycles.
5. **Target**: 2,164 cycles (12 cycles/iter) - need to eliminate 2 more cycles per iteration.
