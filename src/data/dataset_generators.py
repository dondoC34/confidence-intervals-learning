"""High-level dataset factory producing train/validation splits with heteroscedastic noise."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from src.data.data_generators import (
    CubicYGenerator,
    DataGenerator,
    FlatYGenerator,
    LinearYGenerator,
    SinYGenerator,
    UniformXGenerator,
)
from src.data.noise_generators import (
    BlobNoiseGenerator,
    ExponentialNoiseGenerator,
    NoiseGenerator,
    PowerNoiseGenerator,
)

FloatArray = NDArray[np.float64]
DatasetSplit = tuple[FloatArray, FloatArray, FloatArray, FloatArray]


class DatasetGenerator:
    """Factory for synthetic ``(x_train, y_train, x_val, y_val)`` regression splits."""

    def __init__(
        self,
        x_min: float,
        x_max: float,
        sample_size: int,
        validation_split: float = 0.2,
        seed: int | None = None,
    ) -> None:
        if not (0.0 < validation_split < 1.0):
            raise ValueError(f"validation_split must be in (0, 1), got {validation_split}.")
        if sample_size <= 1:
            raise ValueError(f"sample_size must be > 1, got {sample_size}.")

        self.x_min = x_min
        self.x_max = x_max
        self.val_split = validation_split
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _normalize_targets(y_train: FloatArray, y_val: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Scale train and validation targets by the training max absolute value."""
        scale = float(np.max(np.abs(y_train)))
        if scale == 0.0:
            return y_train, y_val
        return y_train / scale, y_val / scale

    def _make_x_generators(self) -> tuple[UniformXGenerator, UniformXGenerator]:
        """Create separate uniform X generators for train and validation splits."""
        val_size = max(1, int(self.val_split * self.sample_size))
        train_size = max(1, self.sample_size - val_size)

        x_gen_train = UniformXGenerator(
            sample_size=train_size,
            lower_bound=self.x_min,
            upper_bound=self.x_max,
            rng=self.rng,
        )
        x_gen_val = UniformXGenerator(
            sample_size=val_size,
            lower_bound=self.x_min,
            upper_bound=self.x_max,
            rng=self.rng,
        )
        return x_gen_train, x_gen_val

    def _make_dataset(
        self,
        x_gen: DataGenerator,
        x_gen_val: DataGenerator,
        y_gen: DataGenerator,
        y_gen_val: DataGenerator,
        noise_generator: NoiseGenerator,
    ) -> DatasetSplit:
        """Sample clean splits, inject noise, and normalize targets."""
        x_train, y_train = x_gen(use_cache=False), y_gen(use_cache=False)
        x_val, y_val = x_gen_val(use_cache=False), y_gen_val(use_cache=False)

        x_train, y_train_noisy = noise_generator(x_train, y_train)
        x_val, y_val_noisy = noise_generator(x_val, y_val)

        y_train_norm, y_val_norm = self._normalize_targets(y_train_noisy, y_val_noisy)
        return x_train, y_train_norm, x_val, y_val_norm

    def generate_custom_dataset(
        self,
        gen_y: DataGenerator,
        noise_gen: NoiseGenerator | None = None,
    ) -> DatasetSplit:
        """Generate a dataset from a user-supplied Y generator and optional noise."""
        if not hasattr(gen_y, "x_gen"):
            raise AttributeError("gen_y must expose an 'x_gen' attribute.")

        val_size = max(1, int(self.val_split * self.sample_size))
        train_size = max(1, self.sample_size - val_size)

        # Configure train generator and reset any stale cached arrays.
        gen_y.x_gen.lower_bound = self.x_min
        gen_y.x_gen.upper_bound = self.x_max
        gen_y.x_gen.sample_size = train_size
        gen_y.sample_size = train_size
        gen_y.x_gen.reset_cache()
        gen_y.reset_cache()

        gen_y_val = deepcopy(gen_y)
        gen_y_val.x_gen.sample_size = val_size
        gen_y_val.sample_size = val_size
        gen_y_val.x_gen.reset_cache()
        gen_y_val.reset_cache()

        x_train, y_train = gen_y.x_gen(), gen_y()
        x_val, y_val = gen_y_val.x_gen(), gen_y_val()

        if noise_gen is not None:
            x_train, y_train = noise_gen(x_train, y_train)
            x_val, y_val = noise_gen(x_val, y_val)

        y_train_norm, y_val_norm = self._normalize_targets(y_train, y_val)
        return x_train, y_train_norm, x_val, y_val_norm

    def generate_noisy_linear(self, slope: float, intercept: float) -> DatasetSplit:
        """Generate a linear dataset with quadratic Gaussian noise."""
        x_gen, x_gen_val = self._make_x_generators()
        y_gen = LinearYGenerator(x_gen, slope=slope, intercept=intercept)
        y_gen_val = LinearYGenerator(x_gen_val, slope=slope, intercept=intercept)
        noise_gen = PowerNoiseGenerator(rng=self.rng)
        return self._make_dataset(x_gen, x_gen_val, y_gen, y_gen_val, noise_gen)

    def generate_blob(self) -> DatasetSplit:
        """Generate a flat-mean dataset with inverse-quadratic blob noise."""
        x_gen, x_gen_val = self._make_x_generators()
        y_gen = FlatYGenerator(x_gen, y=0.0)
        y_gen_val = FlatYGenerator(x_gen_val, y=0.0)
        noise_gen = BlobNoiseGenerator(rng=self.rng)
        return self._make_dataset(x_gen, x_gen_val, y_gen, y_gen_val, noise_gen)

    def _generate_noisy_sin(
        self, amplitude: float, noise_generator: NoiseGenerator
    ) -> DatasetSplit:
        x_gen, x_gen_val = self._make_x_generators()
        y_gen = SinYGenerator(x_gen, amplitude=amplitude)
        y_gen_val = SinYGenerator(x_gen_val, amplitude=amplitude)
        return self._make_dataset(x_gen, x_gen_val, y_gen, y_gen_val, noise_generator)

    def generate_noisy_sin_gaussian(self, amplitude: float) -> DatasetSplit:
        """Generate a sinusoidal dataset with quadratic Gaussian noise."""
        return self._generate_noisy_sin(
            amplitude=amplitude,
            noise_generator=PowerNoiseGenerator(rng=self.rng),
        )

    def generate_noisy_sin_exponential(self, amplitude: float) -> DatasetSplit:
        """Generate a sinusoidal dataset with quadratic exponential noise."""
        return self._generate_noisy_sin(
            amplitude=amplitude,
            noise_generator=ExponentialNoiseGenerator(rng=self.rng),
        )

    def generate_noisy_cube(self, multiplier: float = 3.0) -> DatasetSplit:
        """Generate a cubic dataset with cubic-scale Gaussian noise."""
        x_gen, x_gen_val = self._make_x_generators()
        y_gen = CubicYGenerator(x_gen, multiplier=multiplier)
        y_gen_val = CubicYGenerator(x_gen_val, multiplier=multiplier)
        noise_gen = PowerNoiseGenerator(exponent=5, rng=self.rng)
        return self._make_dataset(x_gen, x_gen_val, y_gen, y_gen_val, noise_gen)
