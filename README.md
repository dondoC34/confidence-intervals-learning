# Confidence Intervals Learning

A small experiment exploring whether a neural network can directly learn **prediction / confidence intervals** alongside a pointwise regression estimate, adapting to **heteroscedastic** (input-dependent) noise without assuming a fixed parametric distribution.

## High-Level Idea

Instead of predicting a single value $\hat{y}$, the network (`PiNeuralNetwork` in PyTorch) outputs **three values** for each input $x$:
- **Upper bound** ($u(x)$)
- **Lower bound** ($l(x)$)
- **Regression estimate** ($\hat{y}(x)$)

Training optimizes a **Quality-Driven (QD)** style objective that balances four goals at once:
1. **Target coverage:** the fraction of samples falling inside $[l(x), u(x)]$ should reach the chosen confidence level (e.g. `0.90` or `0.95`), using a smooth sigmoid approximation so gradients can flow through coverage.
2. **Minimum width:** among covered samples, intervals are pushed to be as tight as possible.
3. **Bound ordering:** a smooth penalty prevents the lower bound from crossing above the upper bound ($l(x) \le u(x)$).
4. **Pointwise regression:** after an optional warm-up phase on the bounds, the regression head fits the conditional mean.

## Repository Overview

- `src/data/` — Synthetic 1D datasets with input-dependent noise (sinusoidal, cubic, linear, and inverse-quadratic *blob*).
- `src/model/` — PyTorch `PiNeuralNetwork` and learning-rate schedules.
- `src/plot/` — Utilities to visualize training snapshots (5x10 grids) and export animated GIFs.
- `training_shapshots_sin.ipynb` / `training_shapshots_blob.ipynb` — Notebooks showing how bounds and regression evolve over epochs.
- `main.py` — End-to-end training script on a cubic dataset that exports `cubic_training.gif`.

## Quick Start

```bash
uv sync
uv run python main.py
```
