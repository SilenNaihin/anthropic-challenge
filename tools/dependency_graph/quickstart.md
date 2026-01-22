# Dependency Graph - Quick Reference

```bash
# Basic analysis
python tools/dependency_graph/dependency_graph.py

# JSON output
python tools/dependency_graph/dependency_graph.py --json

# Export DOT visualization
python tools/dependency_graph/dependency_graph.py --dot graph.dot

# Show more hot registers
python tools/dependency_graph/dependency_graph.py --top 20
```

**Key Metrics:**
- `Critical Path Length` = minimum possible cycles (true dependencies)
- `Theoretical Speedup` = total_cycles / critical_path
- `Hot Registers` = addresses causing most blocking (optimize these!)
- `Parallelism Potential` = average instructions that could run in parallel
