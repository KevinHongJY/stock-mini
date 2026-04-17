# AAPL Noisy Full Follow-Up

## Setup

- Data range: 2010-01-01 to latest available row
- Trading period per episode: 252 steps
- Training episodes per run: 12
- Evaluation episodes per run: 12
- Seeds: 7, 21, 42
- Variants: Dueling DDQN, Noisy Full (sigma=0.5, lr=0.0005), Noisy Full (sigma=0.5, lr=0.00025), Noisy Full (sigma=0.3, lr=0.00025)

## Best Variants

- Highest mean return rate: **Noisy Full (sigma=0.5, lr=0.0005)** (13.69% +/- 10.09%)
- Highest annualized Sharpe ratio: **Dueling DDQN** (10.613 +/- 6.028)

## Files

- `ablation_runs.csv`: one row per variant-seed run
- `ablation_summary.csv`: aggregated mean and standard deviation by variant
- `training_curve_summary.csv`: mean episode reward curves across seeds
- `figures/`: SVG charts ready to use in the report
