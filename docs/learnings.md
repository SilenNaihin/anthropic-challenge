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

### Current Bottleneck Analysis (6,987 cycles)
Per iteration at 6,987 cycles / 128 iterations = 54.6 cycles:
- Hash (with overlapped prep): 16 cycles
  - Overlaps: addr comp, vloads, tree addr, extract for next batch
- Finish (index comp, bounds, stores): 6 cycles
- Scattered loads: 8 cycles (NOT overlapped yet)
- XOR: 1 cycle
Total counted: ~31 cycles per iteration

### Remaining Optimization Opportunities
1. **Overlap scattered loads with finish**: LOAD engine is free during emit_finish() (6 cycles). Can do 12 of 16 scattered loads there, leaving only 4 for after.
2. **Triple-batch processing**: Use all 6 VALU slots during hash instead of 4.
3. **Loop overhead reduction**: Current unrolling may have inefficiencies at round boundaries.
