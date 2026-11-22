"""Plotting utilities for visualizing Prediction Interval training progression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from src.model.pi_neural_network import PiNeuralNetwork

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class TrainingSnapshot:
    """Model predictions on a fixed evaluation grid at a specific training epoch."""

    epoch: int
    ubs: FloatArray
    lbs: FloatArray
    regr: FloatArray


def plot_bounds_on_ax(
    ax: Axes,
    x: FloatArray,
    y: FloatArray,
    x_grid: FloatArray,
    ubs: FloatArray,
    lbs: FloatArray,
    regr: FloatArray,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    title: str | None = None,
    show_legend: bool = True,
    scatter_size: float = 25.0,
    linewidth: float = 2.0,
) -> None:
    """Render data points, prediction interval bounds, and regression line on ``ax``.

    Args:
        ax: Target Matplotlib ``Axes`` instance.
        x: 1D array of input samples.
        y: 1D array of target values.
        x_grid: 1D sorted grid over which predictions were evaluated.
        ubs: Predicted upper bounds evaluated on ``x_grid``.
        lbs: Predicted lower bounds evaluated on ``x_grid``.
        regr: Predicted pointwise regression values evaluated on ``x_grid``.
        x_lim: Optional ``(x_min, x_max)`` limits for the x-axis.
        y_lim: Optional ``(y_min, y_max)`` limits for the y-axis.
        title: Optional subplot title.
        show_legend: Whether to attach a legend to ``ax``.
        scatter_size: Marker size for the data scatter plot.
        linewidth: Line width for bounds and regression curves.
    """
    sns.scatterplot(x=x, y=y, ax=ax, s=scatter_size, alpha=0.45, edgecolor="black")
    ax.fill_between(x_grid, lbs, ubs, color="steelblue", alpha=0.12)
    ax.plot(x_grid, ubs, color="red", linestyle="--", linewidth=linewidth, label="Upper Bound")
    ax.plot(x_grid, lbs, color="green", linestyle="--", linewidth=linewidth, label="Lower Bound")
    ax.plot(x_grid, regr, color="black", linewidth=linewidth, label="Regression")

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if title is not None:
        ax.set_title(title, fontsize=10)
    if show_legend:
        ax.legend(loc="upper left", fontsize=10)


def plot_training_snapshots_grid(
    snapshots: list[TrainingSnapshot],
    x: FloatArray,
    y: FloatArray,
    x_grid: FloatArray,
    ncols: int = 5,
    nrows: int = 10,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    suptitle: str = "Prediction Interval Training Progression",
) -> Figure:
    """Plot a grid of training snapshots (default: 5 columns x 10 rows).

    Args:
        snapshots: Ordered list of ``TrainingSnapshot`` objects.
        x: 1D array of dataset input values.
        y: 1D array of dataset target values.
        x_grid: 1D evaluation grid matching the snapshot predictions.
        ncols: Number of columns in the subplot grid.
        nrows: Number of rows in the subplot grid.
        x_lim: Optional ``(x_min, x_max)`` axis limits.
        y_lim: Optional ``(y_min, y_max)`` axis limits.
        figsize: Optional figure size; defaults to ``(ncols * 3.6, nrows * 2.6)``.
        suptitle: Overall figure title.

    Returns:
        The generated Matplotlib ``Figure``.
    """
    if figsize is None:
        figsize = (ncols * 3.6, nrows * 2.6)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = np.asarray(axes).ravel()

    for idx, ax in enumerate(axes_flat):
        if idx >= len(snapshots):
            ax.set_visible(False)
            continue

        snap = snapshots[idx]
        plot_bounds_on_ax(
            ax=ax,
            x=x,
            y=y,
            x_grid=x_grid,
            ubs=snap.ubs,
            lbs=snap.lbs,
            regr=snap.regr,
            x_lim=x_lim,
            y_lim=y_lim,
            title=f"Epoch {snap.epoch}",
            show_legend=False,
            scatter_size=14.0,
            linewidth=1.5,
        )
        ax.tick_params(labelsize=8)

    # Add a single shared legend at the top of the figure.
    if len(snapshots) > 0:
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            fontsize=11,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.suptitle(suptitle, fontsize=14, y=1.035)
    return fig


def plot_bounds_and_save(
    epoch: int,
    filenames: list[str],
    x: FloatArray,
    y: FloatArray,
    model: PiNeuralNetwork,
    x_lim: tuple[float, float],
    figsize: tuple[float, float] = (10.0, 7.0),
    saving_folder: str | Path | None = None,
    show: bool = False,
    y_lim: tuple[float, float] | None = None,
) -> None:
    """Evaluate ``model`` on a grid, plot bounds + regression, and optionally save to disk."""
    x_min, x_max = x_lim
    x_grid = np.linspace(x_min, x_max, 100)
    ubs, lbs, regr = model.predict(x_grid)

    fig, ax = plt.subplots(figsize=figsize)
    plot_bounds_on_ax(
        ax=ax,
        x=x,
        y=y,
        x_grid=x_grid,
        ubs=ubs,
        lbs=lbs,
        regr=regr,
        x_lim=x_lim,
        y_lim=y_lim,
        title=f"Epoch {epoch}",
        show_legend=True,
        scatter_size=90.0,
        linewidth=2.5,
    )

    if saving_folder is not None:
        out_dir = Path(saving_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Avoid ':' in filenames for Windows compatibility.
        filepath = out_dir / f"training_epoch_{epoch:04d}.png"
        fig.savefig(filepath, bbox_inches="tight")
        filenames.append(str(filepath))

    if show:
        plt.show()

    plt.close(fig)


def snapshot_training(
    epoch_interval: int,
    iterations: int,
    model: PiNeuralNetwork,
    x_train: FloatArray,
    y_train: FloatArray,
    x_lim: tuple[float, float],
    saving_folder: str | Path,
    batch_size: int = 32,
    figsize: tuple[float, float] = (10.0, 7.0),
    show: bool = False,
    y_lim: tuple[float, float] | None = None,
) -> list[str]:
    """Train ``model`` in chunks of ``epoch_interval`` epochs and save snapshot PNGs."""
    filenames: list[str] = []

    plot_bounds_and_save(
        epoch=model.epoch,
        filenames=filenames,
        x=x_train,
        y=y_train,
        model=model,
        x_lim=x_lim,
        figsize=figsize,
        saving_folder=saving_folder,
        show=show,
        y_lim=y_lim,
    )

    for _ in range(1, iterations):
        model.fit(x_train, y_train, epochs=epoch_interval, batch_size=batch_size)
        plot_bounds_and_save(
            epoch=model.epoch,
            filenames=filenames,
            x=x_train,
            y=y_train,
            model=model,
            x_lim=x_lim,
            figsize=figsize,
            saving_folder=saving_folder,
            show=show,
            y_lim=y_lim,
        )

    return filenames


def create_gif(image_paths: list[str], gif_name: str | Path, delete_images: bool = True) -> None:
    """Assemble a list of PNG file paths into an animated GIF."""
    frames: list[Any] = [imageio.imread(filepath) for filepath in image_paths]
    imageio.mimsave(str(gif_name), frames)

    if delete_images:
        for filepath in image_paths:
            Path(filepath).unlink(missing_ok=True)


def create_training_gif(
    gif_name: str | Path,
    epoch_interval: int,
    iterations: int,
    model: PiNeuralNetwork,
    x_train: FloatArray,
    y_train: FloatArray,
    x_lim: tuple[float, float],
    saving_folder: str | Path,
    batch_size: int = 32,
    figsize: tuple[float, float] = (10.0, 7.0),
    y_lim: tuple[float, float] | None = None,
) -> None:
    """Train ``model`` and produce an animated GIF of prediction-interval progression."""
    paths = snapshot_training(
        epoch_interval=epoch_interval,
        iterations=iterations,
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_lim=x_lim,
        saving_folder=saving_folder,
        batch_size=batch_size,
        figsize=figsize,
        show=False,
        y_lim=y_lim,
    )
    create_gif(paths, gif_name)
