# VLIW Packer Quickstart

## TL;DR

```bash
# Pack current kernel
python tools/vliw_packer/vliw_packer.py

# Save packed result
python tools/vliw_packer/vliw_packer.py --output packed.json
```

## What It Does

Takes unpacked instructions (1 op per cycle) and combines independent ops into VLIW bundles while respecting:
- Slot limits (12 alu, 6 valu, 2 load, 2 store, 1 flow)
- Data dependencies (RAW, WAW, WAR hazards)

## Typical Results

- 1.4-2x speedup from automatic packing
- 5-10% utilization improvement
