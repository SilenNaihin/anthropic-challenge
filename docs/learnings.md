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
