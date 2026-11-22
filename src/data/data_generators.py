"""Synthetic 1D feature (X) and target (Y) data generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class DataGenerator(ABC):
    """Abstract base class for cached 1D NumPy data generators.

    Subclasses implement ``_generate()``; calling an instance caches the result
    so repeated calls return the same array unless ``use_cache=False`` is passed
    or ``reset_cache()`` is invoked.
    """

    def __init__(
        self,
        sample_size: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}.")
        self.sample_size = sample_size
        self.rng = rng if rng is not None else np.random.default_rng()
        self._data: FloatArray | None = None

    def reset_cache(self) -> None:
        """Clear any cached generated array."""
        self._data = None

    def __call__(self, use_cache: bool = True) -> FloatArray:
        """Return generated data, using the cached array when available."""
        if use_cache and self._data is not None:
            return self._data
        self._data = self._generate()
        return self._data

    @abstractmethod
    def _generate(self) -> FloatArray:
        """Generate a fresh 1D NumPy array of length ``self.sample_size``."""


class UniformXGenerator(DataGenerator):
    """Generate uniformly distributed 1D inputs in ``[lower_bound, upper_bound)``."""

    def __init__(
        self,
        sample_size: int,
        lower_bound: float = -3.0,
        upper_bound: float = 3.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if lower_bound >= upper_bound:
            raise ValueError(f"lower_bound ({lower_bound}) must be < upper_bound ({upper_bound}).")
        super().__init__(sample_size=sample_size, rng=rng)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def _generate(self) -> FloatArray:
        return self.rng.uniform(
            low=self.lower_bound, high=self.upper_bound, size=self.sample_size
        ).astype(np.float64)


class GaussianXGenerator(DataGenerator):
    """Generate normally distributed 1D inputs ``N(mean, standard_deviation^2)``."""

    def __init__(
        self,
        sample_size: int,
        mean: float = 0.0,
        standard_deviation: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if standard_deviation <= 0:
            raise ValueError(f"standard_deviation must be positive, got {standard_deviation}.")
        super().__init__(sample_size=sample_size, rng=rng)
        self.mean = mean
        self.std = standard_deviation

    def _generate(self) -> FloatArray:
        return self.rng.normal(loc=self.mean, scale=self.std, size=self.sample_size).astype(
            np.float64
        )


class BaseYGenerator(DataGenerator, ABC):
    """Base class for target generators conditioned on an ``x_generator``."""

    def __init__(self, x_generator: DataGenerator) -> None:
        super().__init__(sample_size=x_generator.sample_size, rng=x_generator.rng)
        self.x_gen = x_generator


class LinearYGenerator(BaseYGenerator):
    """Generate linear targets ``y = slope * x + intercept``."""

    def __init__(self, x_generator: DataGenerator, slope: float, intercept: float) -> None:
        super().__init__(x_generator)
        self.slope = slope
        self.intercept = intercept

    def _generate(self) -> FloatArray:
        return self.x_gen() * self.slope + self.intercept


class CubicYGenerator(BaseYGenerator):
    """Generate cubic targets ``y = multiplier * x^3``."""

    def __init__(self, x_generator: DataGenerator, multiplier: float = 1.0) -> None:
        super().__init__(x_generator)
        self.multiplier = multiplier

    def _generate(self) -> FloatArray:
        return self.multiplier * np.power(self.x_gen(), 3)


class SinYGenerator(BaseYGenerator):
    """Generate sinusoidal targets ``y = amplitude * sin(pi * x)``."""

    def __init__(self, x_generator: DataGenerator, amplitude: float = 1.0) -> None:
        super().__init__(x_generator)
        self.amplitude = amplitude

    def _generate(self) -> FloatArray:
        return self.amplitude * np.sin(np.pi * self.x_gen())


class FlatYGenerator(BaseYGenerator):
    """Generate constant targets ``y = value``."""

    def __init__(self, x_generator: DataGenerator, y: float = 1.0) -> None:
        super().__init__(x_generator)
        self.value = y

    def _generate(self) -> FloatArray:
        return np.full_like(self.x_gen(), fill_value=self.value, dtype=np.float64)


# Backward-compatible aliases for existing notebooks/scripts.
XGenerator = UniformXGenerator
XGeneratorGaussian = GaussianXGenerator
YLinearGenerator = LinearYGenerator
YCubicGenerator = CubicYGenerator
YSinGenerator = SinYGenerator
YFlatGenerator = FlatYGenerator
