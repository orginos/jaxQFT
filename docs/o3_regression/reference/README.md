# O(3) Regression Reference Data

This directory vendors a small set of `torchQFT` O(3) production summaries for
regression checks of the JAX O(3) implementation.

Scope:
- only compact `summary.json` files are copied here
- large torch checkpoints and history tensors are intentionally not copied

Current reference cases:
- `8x8 @ beta=1.05`
- `12x12 @ beta=1.187`
- `24x24 @ beta=1.375`

The canonical regression entry point is:

```bash
python scripts/on/run_o3_regression.py
```
