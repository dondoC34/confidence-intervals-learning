from copy import deepcopy


from Source.data.data_generators import (DataGenerator, XGenerator,
                                         YCubicGenerator, YFlatGenerator,
                                         YLinearGenerator, YSinGenerator)
from Source.data.noise_generators import (BlobNoiseGenerator,
                                          ExponentialNoiseGenerator,
                                          NoiseGenerator, PowNoiseGenerator)


class DatasetGenerator:
    def __init__(
        self,
        x_min: float,
        x_max: float,
        sample_size: int,
        validation_split: float = 0.2,
    ):
        assert 0 < validation_split < 1
        self.x_min = x_min
        self.x_max = x_max
        self.val_split = validation_split
        self.sample_size = sample_size

    @staticmethod
    def __make_dataset(
        x_gen: DataGenerator,
        x_gen_val: DataGenerator,
        y_gen: DataGenerator,
        y_gen_val: DataGenerator,
        noise_generator: NoiseGenerator,
    ):
        x, y = x_gen(), y_gen()
        x_val, y_val = x_gen_val(), y_gen_val()

        x, y_noisy = noise_generator(x, y)
        x_val, y_val_noisy = noise_generator(x_val, y_val)

        y_noisy = y_noisy / max(y_noisy)
        y_val_noisy = y_val_noisy / max(y_val_noisy)

        return x, y_noisy, x_val, y_val_noisy

    def __make_x_generators(self):
        x_gen = XGenerator(
            sample_size=int((1 - self.val_split) * self.sample_size),
            lower_bound=self.x_min,
            upper_bound=self.x_max,
        )
        x_gen_val = XGenerator(
            sample_size=int(self.val_split * self.sample_size),
            lower_bound=self.x_min,
            upper_bound=self.x_max,
        )
        return x_gen, x_gen_val

    def generate_custom_dataset(
        self, gen_y: DataGenerator, noise_gen: NoiseGenerator = None
    ):
        gen_y.x_gen.lb = self.x_min
        gen_y.x_gen.ub = self.x_max
        gen_y.x_gen.sample_size = int((1 - self.val_split) * self.sample_size)

        gen_y_val = deepcopy(gen_y)
        gen_y_val.x_gen.sample_size = int(self.val_split * self.sample_size)

        x, y = gen_y.x_gen(), gen_y()
        x_val, y_val = gen_y_val.x_gen(), gen_y_val()

        if noise_gen:
            x, y = noise_gen(x, y)
            x_val, y_val = noise_gen(x_val, y_val)

        y = y / max(y)
        y_val = y_val / max(y)

        return x, y, x_val, y_val

    def generate_noisy_linear(self, slope: float, intercept: float):
        x_gen, x_gen_val = self.__make_x_generators()

        y_gen = YLinearGenerator(x_gen, slope=slope, intercept=intercept)
        y_gen_val = YLinearGenerator(x_gen_val, slope=slope, intercept=intercept)

        noise_generator = PowNoiseGenerator()

        return self.__make_dataset(
            x_gen=x_gen,
            x_gen_val=x_gen_val,
            y_gen=y_gen,
            y_gen_val=y_gen_val,
            noise_generator=noise_generator,
        )

    def generate_blob(self):
        x_gen, x_gen_val = self.__make_x_generators()

        y_gen = YFlatGenerator(x_gen, y=0.0)
        y_gen_val = YFlatGenerator(x_gen_val, y=0.0)

        noise_gen_blob = BlobNoiseGenerator()

        return self.__make_dataset(
            x_gen=x_gen,
            x_gen_val=x_gen_val,
            y_gen=y_gen,
            y_gen_val=y_gen_val,
            noise_generator=noise_gen_blob,
        )

    def __generate_noisy_sin(self, amplitude: float, noise_generator: NoiseGenerator):
        x_gen, x_gen_val = self.__make_x_generators()

        y_gen = YSinGenerator(x_gen, amplitude=amplitude)
        y_gen_val = YSinGenerator(x_gen_val, amplitude=amplitude)

        return self.__make_dataset(
            x_gen=x_gen,
            x_gen_val=x_gen_val,
            y_gen=y_gen,
            y_gen_val=y_gen_val,
            noise_generator=noise_generator,
        )

    def generate_noisy_sin_gaussian(self, amplitude: float):
        noise_gen = PowNoiseGenerator()
        return self.__generate_noisy_sin(amplitude=amplitude, noise_generator=noise_gen)

    def generate_noisy_sin_exponential(self, amplitude: float):
        noise_gen = ExponentialNoiseGenerator()
        return self.__generate_noisy_sin(amplitude=amplitude, noise_generator=noise_gen)

    def generate_noisy_cube(self):
        x_gen, x_gen_val = self.__make_x_generators()

        y_gen = YCubicGenerator(x_gen, multiplier=3.0)
        y_gen_val = YCubicGenerator(x_gen_val, multiplier=5.0)

        noise_generator = PowNoiseGenerator(exponent=3)

        return self.__make_dataset(
            x_gen=x_gen,
            x_gen_val=x_gen_val,
            y_gen=y_gen,
            y_gen_val=y_gen_val,
            noise_generator=noise_generator,
        )
