import numpy as np


class DataGenerator(object):
    def __init__(self, sample_size: int):
        self.sample_size = sample_size
        self.x_gen = None

    def __call__(self, *args, **kwargs):
        pass


class XGenerator(DataGenerator):
    def __init__(
        self, sample_size: int, lower_bound: float = -3.0, upper_bound: float = 3.0
    ):
        super(XGenerator, self).__init__(sample_size)
        self.lb = lower_bound
        self.ub = upper_bound
        self.data = None

    def __call__(self, use_cache: bool = True):
        if use_cache and (self.data is not None):
            return self.data

        self.data = np.random.uniform(low=self.lb, high=self.ub, size=self.sample_size)
        return self.data


class XGeneratorGaussian(DataGenerator):
    def __init__(
        self, sample_size: int, mean: float = 0.0, standard_deviation: float = 1.0
    ):
        assert standard_deviation > 0
        super(XGeneratorGaussian, self).__init__(sample_size)
        self.data = None
        self.mean = mean
        self.std = standard_deviation

    def __call__(self, use_cache: bool = True):
        if use_cache and (self.data is not None):
            return self.data

        self.data = np.random.normal(
            loc=self.mean, scale=self.std, size=self.sample_size
        )
        return self.data


class YLinearGenerator(DataGenerator):
    def __init__(self, x_generator: XGenerator, slope: float, intercept: float):
        super(YLinearGenerator, self).__init__(x_generator.sample_size)
        self.x_gen = x_generator
        self.slope = slope
        self.intercept = intercept
        self.data = None

    def __call__(self, use_cache: bool = True):
        if use_cache and (self.data is not None):
            return self.data

        self.data = self.x_gen() * self.slope + self.intercept  # Linear function
        return self.data


class YCubicGenerator(DataGenerator):
    def __init__(self, x_generator: XGenerator, multiplier: float = 1.0):
        super(YCubicGenerator, self).__init__(x_generator.sample_size)
        self.mult = multiplier
        self.x_gen = x_generator
        self.data = None

    def __call__(self, use_cache: bool = True):
        if use_cache and (self.data is not None):
            return self.data

        self.data = self.mult * (self.x_gen() ** 3)
        return self.data


class YSinGenerator(DataGenerator):
    def __init__(self, x_generator: XGenerator, amplitude: float = 1.0):
        super(YSinGenerator, self).__init__(x_generator.sample_size)
        self.amplitude = amplitude
        self.x_gen = x_generator
        self.data = None

    def __call__(self, use_cache: bool = True):
        if use_cache and (self.data is not None):
            return self.data

        self.data = np.sin(np.pi * self.x_gen())
        return self.data


class YFlatGenerator(DataGenerator):
    def __init__(self, x_generator: XGenerator, y: float = 1.0):
        super(YFlatGenerator, self).__init__(x_generator.sample_size)
        self.Y = y
        self.x_gen = x_generator
        self.data = None

    def __call__(self, use_cache: bool = True):
        if use_cache and (self.data is not None):
            return self.data

        self.data = self.Y * np.ones_like(self.x_gen())
        return self.data
