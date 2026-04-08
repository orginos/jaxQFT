# phi4 Reweighting Analysis Guidance

This note records how to interpret reweighting-based diagnostics for the
`phi^4` flow models and how to structure future analysis runs.

## Core Distinction

There are two different sources of variation that should not be mixed:

1. Variation across independent validation batches for one fixed trained model.
2. Variation across different seeds, where training may converge to different
   local minima in parameter space.

These answer different questions.

## 1. Within-Checkpoint Variation

For one fixed checkpoint `theta`, draw many independent validation batches and
evaluate on each batch:

- `std(Delta S)`
- `std(Delta S) / L`
- `ESS`
- tail-sensitive diagnostics such as:
  - max weight
  - max `Delta S`
  - selected quantiles of `Delta S` or `log w`

This measures the statistical uncertainty of the metric for that *particular*
model.

Recommended practice:

- save the per-batch metrics, not just one aggregate value
- report:
  - mean or median over batches
  - spread across batches
  - outlier frequency

## 2. Between-Seed Variation

Different random seeds can converge to different local minima even when the
architecture and schedule are fixed.

To study that effect:

- first average or robustly summarize the batch-level metrics *within each seed*
- then compare the seed-level summaries across seeds

This quantifies model-to-model variation due to optimization rather than
validation noise.

## ESS Interpretation

ESS is intentionally tail-sensitive:

```math
\mathrm{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
```

A single large reweighting outlier can collapse the ESS of an otherwise decent
checkpoint. Therefore:

- occasional isolated ESS collapses do not automatically imply a bad model
- frequent ESS collapses indicate a real heavy-tail mismatch problem

ESS should therefore be interpreted together with:

- `std(Delta S)` or `std(Delta S)/L` for bulk quality
- outlier/tail diagnostics for rare-event risk

## Suggested Reporting

For each checkpoint:

- evaluate many independent validation batches
- keep the full batchwise metric table
- summarize ESS with robust statistics:
  - median
  - interquartile range
  - min / max
- summarize tail behavior explicitly:
  - number of batches with catastrophic ESS
  - number of large-`Delta S` outliers

For a seed ensemble:

- report seed-to-seed distributions of the batch-aggregated metrics
- separate:
  - within-seed batch variation
  - between-seed training variation

## Practical Consequence

When a single seed has poor ESS but similar `std(Delta S)` to the rest, inspect
the tail behavior before concluding that the checkpoint is globally bad. A
single bad batch can dominate ESS, whereas frequent tail events signal a real
problem for reweighting.
