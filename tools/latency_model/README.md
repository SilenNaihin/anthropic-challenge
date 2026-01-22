# Instruction Latency Model Analyzer

Analyzes and documents the latency characteristics of the simulated VLIW SIMD architecture. This tool answers the fundamental question: **Do all operations complete in 1 cycle, or are there multi-cycle operations that require special scheduling?**

## Key Finding

**ALL operations complete in exactly 1 cycle.** The simulator uses a write-staging model where:

1. All reads happen at the START of a cycle (using current scratch values)
2. All writes are staged to `scratch_write` and `mem_write` dictionaries
3. All writes apply at the END of the cycle

This means:
- **Same-cycle write-read**: Reader gets the OLD value
- **Next-cycle read**: Reader gets the NEW value
- **No stalls or pipeline bubbles**: Value is available immediately the next cycle

## Usage

```bash
# Full analysis with empirical tests (default)
python tools/latency_model/latency_model.py

# Run empirical tests explicitly
python tools/latency_model/latency_model.py --empirical

# JSON output for scripting
python tools/latency_model/latency_model.py --json

# Plain text (no Rich formatting)
python tools/latency_model/latency_model.py --no-color

# Show all operation details
python tools/latency_model/latency_model.py --all-ops
```

## Architecture Summary

| Engine | Slots/Cycle | Latency | Effective Throughput |
|--------|-------------|---------|---------------------|
| alu    | 12          | 1 cycle | 12 scalar ops/cycle |
| valu   | 6           | 1 cycle | 48 elements/cycle (6 x VLEN=8) |
| load   | 2           | 1 cycle | 2 loads/cycle (16 with vload) |
| store  | 2           | 1 cycle | 2 stores/cycle (16 with vstore) |
| flow   | 1           | 1 cycle | 1 flow op/cycle |

## Empirical Tests

The analyzer runs 10 empirical tests to verify the latency model:

| Test | Purpose |
|------|---------|
| `single_alu` | Verify basic ALU operation takes 1 cycle |
| `alu_chain` | Verify RAW dependency works correctly |
| `load_use` | Verify loaded value is available next cycle |
| `single_valu` | Verify VALU vector ops take 1 cycle |
| `same_cycle_raw` | Verify same-cycle reads get old values |
| `parallel_alu` | Verify all 12 ALU slots can be used |
| `mixed_engines` | Verify different engines work in parallel |
| `store_load` | Verify store-then-load works correctly |
| `multiply_add` | Verify fused multiply-add takes 1 cycle |
| `integer_divide` | Verify integer division takes 1 cycle |

## Optimization Implications

### What This Means for Scheduling

1. **No Latency Hiding Needed**
   - All operations complete in 1 cycle
   - No need for out-of-order execution simulation
   - No need for software speculation

2. **Dependencies Are the Real Problem**
   - RAW (Read-After-Write) hazards force sequential execution
   - Break chains via: unrolling, pipelining, vectorization
   - Use dependency graph tools to identify critical paths

3. **Throughput Limits Matter**
   - Load/Store: Only 2 slots each - can bottleneck memory-heavy code
   - Flow: Only 1 slot - conditional operations serialize
   - ALU: 12 slots - rarely a problem
   - VALU: 6 slots x 8 elements = massive throughput

4. **VLIW Packing Is Key**
   - Maximum: 12 ALU + 6 VALU + 2 load + 2 store + 1 flow per cycle
   - Most kernels use <10% of available slots
   - Focus on packing independent operations together

## Integration with Other Tools

This latency model confirms that other tools' assumptions are correct:

- **dependency_graph**: Assumes 1-cycle latency for critical path analysis - CORRECT
- **vliw_packer**: Assumes operations can be packed freely - CORRECT
- **slot_analyzer**: Assumes slot limits are the only constraint - CORRECT
- **hash_pipeline**: Assumes 1-cycle per operation - CORRECT

## Technical Details

### Write-Staging Model (from problem.py)

```python
def step(self, instr: Instruction, core):
    # Stage writes - don't apply yet
    self.scratch_write = {}
    self.mem_write = {}

    # Execute all slots (reads use current values)
    for name, slots in instr.items():
        for slot in slots:
            ENGINE_FNS[name](core, *slot)  # Stages writes

    # Apply all writes at end of cycle
    for addr, val in self.scratch_write.items():
        core.scratch[addr] = val
    for addr, val in self.mem_write.items():
        self.mem[addr] = val
```

### Same-Cycle Semantics Example

```
Cycle 1:
  ALU: r1 = r0 + r0    (writes to scratch_write[1])
  ALU: r2 = r1 + r0    (reads r1, gets OLD value before write)

  After cycle: scratch_write applied
    r1 = new value
    r2 = (old_r1) + r0  <-- NOT (new_r1) + r0
```

## JSON Schema

```json
{
  "summary": {
    "all_single_cycle": true,
    "has_multi_cycle_ops": false
  },
  "operations": [
    {
      "engine": "alu",
      "operation": "+",
      "latency_cycles": 1,
      "latency_type": "single_cycle",
      "throughput_per_cycle": 12,
      "notes": "..."
    }
  ],
  "empirical_tests": [
    {
      "name": "single_alu",
      "description": "...",
      "expected_cycles": 3,
      "actual_cycles": 3,
      "passed": true,
      "conclusion": "..."
    }
  ],
  "throughput_bottlenecks": [...],
  "scheduling_notes": [...]
}
```

## Related Tools

- `tools/dependency_graph/` - Analyzes RAW/WAW/WAR hazards
- `tools/slot_analyzer.py` - Analyzes slot utilization
- `tools/vliw_packer/` - Packs instructions respecting dependencies
- `tools/hash_pipeline/` - Analyzes hash function ILP
