# AAPL Final Confirmation

## Setup

- Data range: 2010-01-01 to latest available row
- Trading period per episode: 252 steps
- Training episodes per run: 30
- Evaluation episodes per run: 20
- Seeds: 7, 21, 42
- Variants: Dueling DDQN, D3QN + PER + 3-step, Noisy Full (sigma=0.5, lr=0.0005), Noisy Full (sigma=0.3, lr=0.00025)

## Best Variants

- Highest mean return rate: **Noisy Full (sigma=0.5, lr=0.0005)** (14.18% +/- 8.61%)
- Highest annualized Sharpe ratio: **Dueling DDQN** (13.316 +/- 0.970)

## Files

- `ablation_runs.csv`: one row per variant-seed run
- `ablation_summary.csv`: aggregated mean and standard deviation by variant
- `training_curve_summary.csv`: mean episode reward curves across seeds
- `figures/`: SVG charts ready to use in the report
