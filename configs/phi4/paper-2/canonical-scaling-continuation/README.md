# Paper 2 Canonical Scaling Continuations

These cards extend the fixed-architecture baseline runs to larger effective
batch sizes without changing the model.

Scientific purpose:
- test the hypothesis that the fixed-model baseline is limited by training
  quality rather than model capacity
- exploit the volume-versus-batch tradeoff observed in the NERSC runs
- compare volumes at more nearly matched effective gradient statistics

Interpretation:
- `L=128` at batch `64` is already the largest affordable batch in this family
- the smaller volumes are continued to larger batches so that the effective
  statistics per optimizer step are closer across volumes

Recommended workflow:
1. finish the baseline fixed-model runs
2. resume from the baseline checkpoint with the matching continuation card
3. keep the same architecture and validation settings
4. compare the improved `std(Delta S) / L` values against the baseline

## Continuation targets

- `L16_uniform_continue.toml`
  - continue from batch `64` up to `2048`
- `L32_uniform_continue.toml`
  - continue from batch `64` up to `1024`
- `L64_uniform_continue.toml`
  - continue from batch `64` up to `256`
- `L128_uniform_continue.toml`
  - keep batch `64` and extend the final low-learning-rate anneal

## Example

```bash
scripts/phi4/run_rg_coarse_eta_gaussian.sh \
  --config configs/phi4/paper-2/canonical-scaling-continuation/L32_uniform_continue.toml \
  --workdir runs/phi4/paper-2/canonical-scaling/L32/s0 \
  --gpu all \
  -- \
  --resume checkpoint.pkl
```

The trainer preserves the checkpoint architecture; these cards only change the
continued schedule.
