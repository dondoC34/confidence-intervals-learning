import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from Source.data.dataset_generators import (generate_random_cubic_data,
                                            generate_random_quadratic_data)
from Source.data.gen_dataset import DataGenerator
from Source.model.PiNeuralNet import PiNeuralNetwork

n_samples = 100
X = np.random.uniform(low=-2.0, high=2.0, size=(n_samples, 1))
y = 1.5 * np.sin(np.pi * X[:, 0]) + np.random.normal(
    loc=0.0, scale=1.0 * np.power(X[:, 0], 2)
)
y = y.reshape([-1, 1]) / 5.0
x_grid = np.linspace(-2, 2, 100)

nn = PiNeuralNetwork(
    input_shape=(1,),
    nn_hidden_layers=[(100, "relu")],
    confidence=0.95,
    softening_factor=10.0,
    lambda_=0.01,
    beta_=1.0,
    alpha_=0.0,
    verbose=True,
    batch_size=32,
)


tf_data = tf.data.Dataset.from_tensor_slices((X, y)).map(
    lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32))
)

nn.train(data=tf_data, epochs=600)

y_pred = nn(tf.expand_dims(x_grid, axis=1))
y_u_pred = y_pred[:, 0]
y_l_pred = y_pred[:, 1]

plt.scatter(X, y)
plt.plot(x_grid, y_u_pred, color="r")
plt.plot(x_grid, y_l_pred, color="g")
plt.title("BEFORE training")
plt.xlim(-2, 2)
plt.show()
