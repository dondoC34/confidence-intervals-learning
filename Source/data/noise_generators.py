import numpy as np


class NoiseGenerator(object):
    def __call__(self, *args, **kwargs):
        pass


class LinearNoiseGenerator(NoiseGenerator):
    def __init__(self, intensification_factor: float = 0.15):
        super().__init__()
        self.intensification = intensification_factor

    def __call__(self, x: np.array, y: np.array):
        standard_deviations = np.abs(x * self.intensification)
        y_noisy = np.random.normal(size=y.shape, loc=y, scale=standard_deviations)

        return x, y_noisy


class PowNoiseGenerator(NoiseGenerator):
    def __init__(self, exponent: int = 2):
        super().__init__()
        self.exponent = exponent

    def __call__(self, x: np.array, y: np.array):
        standard_deviations = np.abs(x) ** self.exponent
        y_noisy = np.random.normal(size=y.shape, loc=y, scale=standard_deviations)

        return x, y_noisy


class ExponentialNoiseGenerator(NoiseGenerator):
    def __call__(self, x: np.array, y: np.array):
        standard_deviations = np.power(x, 2)
        y_noisy = np.random.exponential(size=y.shape, scale=standard_deviations)

        return x, y_noisy


class BlobNoiseGenerator(NoiseGenerator):
    def __call__(self, x: np.array, y: np.array):
        standard_deviations = 1.0 / np.power(x, 2)
        y_noisy = np.random.normal(size=y.shape, loc=y, scale=standard_deviations)
        y_noisy = np.array(
            [elem if elem < max(y) + 20 else np.random.normal(0, 5) for elem in y_noisy]
        )
        y_noisy = np.array(
            [elem if elem > min(y) - 20 else np.random.normal(0, 5) for elem in y_noisy]
        )

        return x, y_noisy
