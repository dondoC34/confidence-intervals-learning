"""Heteroscedastic noise generators for synthetic prediction-interval datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class NoiseGenerator(ABC):
    """Abstract base class for input-dependent noise injectors."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    @abstractmethod
    def __call__(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Inject noise into ``y`` conditioned on ``x`` and return ``(x, y_noisy)``."""


class LinearNoiseGenerator(NoiseGenerator):
    """Add zero-mean Gaussian noise with ``std = |x * intensification_factor|``."""

    def __init__(
        self,
        intensification_factor: float = 0.15,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)
        self.intensification = intensification_factor

    def __call__(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        scale = np.abs(x * self.intensification)
        y_noisy = self.rng.normal(loc=y, scale=scale, size=y.shape)
        return x, y_noisy


class PowerNoiseGenerator(NoiseGenerator):
    """Add zero-mean Gaussian noise with ``std = |x|^exponent``."""

    def __init__(self, exponent: int = 2, rng: np.random.Generator | None = None) -> None:
        super().__init__(rng=rng)
        self.exponent = exponent

    def __call__(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        scale = np.abs(x) ** self.exponent
        y_noisy = self.rng.normal(loc=y, scale=scale, size=y.shape)
        return x, y_noisy


class ExponentialNoiseGenerator(NoiseGenerator):
    """Draw non-negative exponential noise with ``scale = x^2``."""

    def __call__(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        scale = np.power(x, 2)
        y_noisy = self.rng.exponential(scale=scale, size=y.shape)
        return x, y_noisy


class BlobNoiseGenerator(NoiseGenerator):
    """Add inverse-quadratic Gaussian noise with bounded outlier resampling.

    Standard deviation grows as ``1 / max(x^2, eps)`` near zero; samples
    exceeding ``[min(y) - outlier_margin, max(y) + outlier_margin]`` are
    resampled from ``N(0, outlier_std^2)``.
    """

    def __init__(
        self,
        outlier_margin: float = 20.0,
        outlier_std: float = 5.0,
        eps: float = 1e-6,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)
        self.outlier_margin = outlier_margin
        self.outlier_std = outlier_std
        self.eps = eps

    def __call__(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        # Guard against division by zero when x == 0.
        scale = 1.0 / np.maximum(np.power(x, 2), self.eps)
        y_noisy = self.rng.normal(loc=y, scale=scale, size=y.shape)

        # Vectorized outlier resampling (avoids O(N^2) Python loops).
        upper_limit = float(np.max(y)) + self.outlier_margin
        lower_limit = float(np.min(y)) - self.outlier_margin
        outliers = (y_noisy >= upper_limit) | (y_noisy <= lower_limit)

        if np.any(outliers):
            resampled = self.rng.normal(loc=0.0, scale=self.outlier_std, size=y.shape)
            y_noisy = np.where(outliers, resampled, y_noisy)

        return x, y_noisy


# Backward-compatible alias.
PowNoiseGenerator = PowerNoiseGenerator
