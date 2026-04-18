# Rainbow DQN Stock Mini Project

This repository is a compact course-ready stock trading project built around a Rainbow-style DQN pipeline.
It studies how core Rainbow improvements transfer to a single-stock trading setting with a discrete capital-aware environment.

The implementation includes these Rainbow-related components:

- Double DQN
- Dueling network architecture
- Prioritized experience replay (PER)
- Multi-step returns
- Noisy Networks

It does not implement the full original Rainbow stack because distributional RL (C51) is intentionally left out to keep the project lightweight and easy to explain in a course setting.

## Project Layout

- `src/d3qn_stock/`: package source code
- `configs/default.yaml`: default training configuration
- `configs/aapl_baseline.yaml`: AAPL baseline configuration
- `configs/aapl_returns_account.yaml`: AAPL feature-enhanced configuration
- `data/sample_stock.csv`: de-identified sample stock data
- `data/aapl_daily.csv`: AAPL daily OHLCV data used for the mini-project experiments
- `scripts/train.py`: training entrypoint
- `scripts/eval.py`: evaluation entrypoint
- `scripts/run_ablation.py`: ablation and figure generation script
- `tests/`: unit and smoke-oriented tests
- `results/`: generated experiment summaries and figures

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## CSV Format

The default sample file uses:

```text
Date,Open,High,Low,Close,Volume
```

The loader also supports custom column names through the YAML config:

- `data.date_column`
- `data.close_column`
- optional `data.open_column`
- optional `data.high_column`
- optional `data.low_column`
- optional `data.volume_column`

## Main Configurations

- `configs/default.yaml`: generic default config
- `configs/aapl_baseline.yaml`: baseline AAPL experiment
- `configs/aapl_noisy.yaml`: NoisyNet AAPL experiment
- `configs/aapl_noisy_tuned.yaml`: tuned NoisyNet AAPL experiment
- `configs/aapl_returns_account.yaml`: feature-enhanced observation experiment with returns and account-state features

## Train

```bash
./.venv/bin/python scripts/train.py --config configs/default.yaml
```

Expected outputs under `runs/<run_name>/`:

- `metrics.csv`
- `config_resolved.yaml`
- `checkpoints/checkpoint_latest.pt`
- `run.log`

## Evaluate

```bash
./.venv/bin/python scripts/eval.py \
  --config configs/default.yaml \
  --checkpoint runs/<run_name>/checkpoints/checkpoint_latest.pt \
  --output-dir runs/<run_name>/eval
```

Expected evaluation outputs:

- `eval_summary.json`
- `eval_metrics.csv`
- `cumulative_returns.json`
- `eval.log`

## Ablation Study

To regenerate the AAPL feature-based comparison figures:

```bash
./.venv/bin/python scripts/run_ablation.py \
  --config configs/aapl_baseline.yaml \
  --variant-set feature_only \
  --output-dir results/feature_only_full \
  --num-episodes 20 \
  --eval-episodes 20 \
  --seeds 7 21 42
```

The ablation outputs include:

- mean return bar chart
- mean Sharpe bar chart
- training reward curve
- mean cumulative return curve

## Notes

- The repository is intentionally limited to one stock CSV per run.
- The default action template uses five actions: hold, buy 50%, buy 100%, sell 50%, sell 100%.
- Reward options are `profit`, `sr`, and `sr_enhanced`. The default config uses `profit`.
- The feature-enhanced observation setting uses log returns together with account-state features such as cash ratio, position ratio, and equity return.

## Reference

Hessel, M., Modayil, J., van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M., and Silver, D. (2018). *Rainbow: Combining Improvements in Deep Reinforcement Learning*. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1).

- AAAI page: [https://ojs.aaai.org/index.php/AAAI/article/view/11796](https://ojs.aaai.org/index.php/AAAI/article/view/11796)
- arXiv: [https://arxiv.org/abs/1710.02298](https://arxiv.org/abs/1710.02298)
