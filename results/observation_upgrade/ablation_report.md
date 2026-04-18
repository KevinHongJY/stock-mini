# AAPL Observation Upgrade

## Setup

- Data range: 2010-01-01 to latest available row
- Trading period per episode: 252 steps
- Training episodes per run: 20
- Evaluation episodes per run: 20
- Seeds: 7, 21, 42
- Variants: Dueling DDQN (raw), Dueling DDQN (returns + account), D3QN + PER + 3-step (raw), D3QN + PER + 3-step (returns + account), Noisy Full (raw), Noisy Full (returns + account)

## Best Variants

- Highest mean return rate: **Noisy Full (returns + account)** (15.21% +/- 10.83%)
- Highest annualized Sharpe ratio: **D3QN + PER + 3-step (raw)** (11.979 +/- 1.176)

## Files

- `ablation_runs.csv`: one row per variant-seed run
- `ablation_summary.csv`: aggregated mean and standard deviation by variant
- `training_curve_summary.csv`: mean episode reward curves across seeds
- `figures/`: SVG charts ready to use in the report
