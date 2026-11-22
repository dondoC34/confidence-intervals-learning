"""Train PiNeuralNetwork on a cubic heteroscedastic dataset and export a training GIF."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from src.data.data_generators import CubicYGenerator, UniformXGenerator
from src.data.dataset_generators import DatasetGenerator
from src.data.noise_generators import PowerNoiseGenerator
from src.model.lr_schedule import TimeDecreasingLr
from src.model.pi_neural_network import PiNeuralNetwork
from src.plot.training_plots import create_training_gif


def main() -> None:
    """Generate cubic data, train the model, and save ``cubic_training.gif`` in the repo root."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    x_min, x_max = -5.0, 5.0
    dataset_gen = DatasetGenerator(
        x_min=x_min,
        x_max=x_max,
        sample_size=500,
        validation_split=0.2,
        seed=seed,
    )
    x_train, y_train, x_val, y_val = dataset_gen.generate_custom_dataset(
        gen_y=CubicYGenerator(UniformXGenerator(sample_size=500)),
        noise_gen=PowerNoiseGenerator(exponent=3, rng=dataset_gen.rng),
    )

    model = PiNeuralNetwork(
        input_dim=1,
        hidden_layers=[(32, "relu"), (16, "relu")],
        confidence=0.95,
        softening_factor=130.0,
        lr=1e-3,
        weight_decay=1e-2,
        alpha=5.0,
        lambda_=0.02,
        mse_inactivity_epochs=30,
        lr_schedule=TimeDecreasingLr(decay=5e-4),
        verbose=False,
    )

    repo_root = Path(__file__).resolve().parent
    gif_path = repo_root / "cubic_training.gif"
    y_margin = 0.15 * float(y_train.max() - y_train.min())
    y_lim = (float(y_train.min() - y_margin), float(y_train.max() + y_margin))

    with tempfile.TemporaryDirectory() as tmp_dir:
        create_training_gif(
            gif_name=gif_path,
            epoch_interval=10,
            iterations=100,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_lim=(x_min, x_max),
            saving_folder=tmp_dir,
            batch_size=32,
            y_lim=y_lim,
        )

    val_ubs, val_lbs, val_regr = model.predict(x_val)
    covered = (y_val >= val_lbs) & (y_val <= val_ubs)
    print(f"Saved GIF: {gif_path} ({gif_path.stat().st_size / 1024:.1f} KB)")
    print(f"Device: {model.device} | Total epochs: {model.epoch}")
    print(f"Validation PICP (target 0.95): {np.mean(covered):.3f}")
    print(f"Validation MPIW (mean width):  {np.mean(val_ubs - val_lbs):.4f}")
    print(f"Validation Regression MSE:     {np.mean((val_regr - y_val) ** 2):.4f}")
    print(f"Inverted bounds (val):         {int(np.sum(val_lbs > val_ubs))}")


if __name__ == "__main__":
    main()
