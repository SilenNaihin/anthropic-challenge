# CLAUDE.md

## Project Overview

Anthropic's performance engineering take-home challenge. Optimize a kernel for a simulated VLIW SIMD architecture to minimize clock cycles.

**Goal**: Beat Claude Opus 4.5's best of 1,487 cycles (baseline: 147,734 cycles)

## Key Files

- `perf_takehome.py` - Main file with `KernelBuilder.build_kernel()` to optimize
- `problem.py` - Simulator, architecture definition, reference kernel
- `mandate.md` - Challenge summary and benchmarks
- `tools/suggestions.md` - Detailed tool/optimization suggestions

## Commands

```bash
# Run submission tests (primary validation)
python tests/submission_tests.py

# Run with tracing for Perfetto visualization
python perf_takehome.py Tests.test_kernel_trace

# View trace (opens browser)
python watch_trace.py

# Verify tests unchanged (REQUIRED before submission)
git diff origin/main tests/
```

## Architecture Quick Reference

| Engine | Slots/cycle | Notes |
|--------|-------------|-------|
| alu    | 12          | Scalar ops: +, -, *, //, %, ^, &, \|, <<, >>, <, == |
| valu   | 6           | Vector ops (VLEN=8): vbroadcast, multiply_add, same as alu |
| load   | 2           | load, vload, const, load_offset |
| store  | 2           | store, vstore |
| flow   | 1           | select, vselect, cond_jump, jump, halt, pause |

**Key**: Effects apply at END of cycle (parallel reads before writes)

## Documentation Requirements

**ALWAYS update these files during work:**

- `docs/changelog.md` - Every code change with hypothesis and result
- `docs/learnings.md` - Insights, gotchas, what worked/didn't
- `docs/research.md` - Web findings, papers, external resources

## Critical Rules

1. **NEVER modify `tests/` folder** - LLMs have cheated this way
2. **Validate with `git diff origin/main tests/`** before any submission claim
3. **Use cycle count from `tests/submission_tests.py`** not other tests

## Optimization Strategy

Current baseline uses:
- 1 op per instruction (no VLIW packing)
- All scalar ops (no vectorization)
- Sequential dependencies (no pipelining)

Theoretical speedup available: ~100x via vectorization (8x), VLIW packing (5-10x), pipelining (2-3x)
