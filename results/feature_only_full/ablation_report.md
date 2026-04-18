# AAPL Feature-Only Comparison

## Setup

- Data range: 2010-01-01 to latest available row
- Trading period per episode: 252 steps
- Training episodes per run: 20
- Evaluation episodes per run: 20
- Seeds: 7, 21, 42
- Variants: DQN, DDQN, Duel DDQN, D3QN (+PER+3-step), D3QN+Noisy

## Best Variants

- Highest mean return rate: **D3QN+Noisy** (16.20% +/- 11.45%)
- Highest mean Sharpe ratio: **D3QN (+PER+3-step)** (0.612 +/- 0.433)

## Files

- `ablation_runs.csv`: one row per variant-seed run
- `ablation_summary.csv`: aggregated mean and standard deviation by variant
- `training_curve_summary.csv`: mean episode reward curves across seeds
- `eval_curve_summary.csv`: mean evaluation cumulative return curves across seeds and windows
- `figures/`: SVG charts ready to use in the report
