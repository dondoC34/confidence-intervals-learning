from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model


class PiNeuralNetwork(Model):
    def get_config(self):
        pass

    def __init__(
        self,
        input_shape: tuple,
        nn_hidden_layers: List[tuple],
        confidence: tf.constant,
        softening_factor: tf.constant,
        lr: float,
        alpha_: float,
        beta_: float,
        lambda_: float,
        batch_size: int,
        verbose: bool = False,
    ):
        super(PiNeuralNetwork, self).__init__()

        self.base_model = tf.keras.Sequential(name="base_model")
        self.base_model.add(tf.keras.layers.Input(shape=input_shape))

        for units, activation_function in nn_hidden_layers:
            self.base_model.add(
                tf.keras.layers.Dense(
                    units=units,
                    activation=activation_function,
                    kernel_initializer=tf.keras.initializers.RandomNormal(
                        mean=0.0, stddev=0.2
                    ),
                )
            )

        self.base_model.add(
            tf.keras.layers.Dense(
                units=3,
                activation="linear",
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.01
                ),
                bias_initializer=tf.keras.initializers.Constant(value=[1, -1, 0.0]),
            )
        )
        self.verbose = verbose
        self.softening_factor = softening_factor
        self.confidence = confidence
        self.lambda_ = tf.constant(lambda_)
        self.beta_ = tf.constant(beta_)
        self.alpha_ = {"warm-up": tf.constant(0.0), "training": tf.constant(alpha_)}
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.batch_size = batch_size
        self.epoch = 0
        self.epsilon = tf.constant(0.001)

    def call(self, inputs, training=None, mask=None):
        outputs = self.base_model(inputs, training=training)
        ubs, lbs, regr = tf.unstack(outputs, axis=1)
        return ubs, lbs, regr

    def loss_function(self, targets, ubs, lbs, regr):
        K_SU = tf.sigmoid(self.softening_factor * (ubs - targets))
        K_SL = tf.sigmoid(self.softening_factor * (targets - lbs))
        K_S = tf.multiply(K_SU, K_SL)

        MPIW_c = tf.reduce_sum(
            tf.multiply(tf.abs(ubs - lbs), K_S + self.epsilon / self.batch_size)
        ) / (tf.reduce_sum(K_S) + self.epsilon)

        MPIW_inv = tf.reduce_sum(
            tf.multiply(
                tf.maximum(lbs - ubs, -50.0), 1 - K_S + self.epsilon / self.batch_size
            )
        ) / (tf.reduce_sum(1 - K_S) + self.epsilon)

        PICP_S = tf.reduce_mean(K_S)
        w_MSE = tf.reduce_mean((regr - targets) ** 2 / tf.stop_gradient((ubs - lbs)))
        w_MSE = self.alpha_["warm-up" if self.epoch < 500 else "training"] * w_MSE

        Loss_S = (
            w_MSE
            + MPIW_c
            + self.beta_ * MPIW_inv
            + self.lambda_
            * self.batch_size
            / ((1 - self.confidence) * self.confidence)
            * tf.maximum(0.0, self.confidence - PICP_S)
        )

        # + tf.reduce_sum(tf.keras.activations.relu(lbs - ubs))

        return Loss_S

    def train_step(self, data: tf.data.Dataset):
        # processed = 0
        # ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        for batch_x, batch_y in data:
            with tf.GradientTape() as t:
                # Compute the output
                ubs, lbs, regr = self.call(batch_x, training=True)
                loss = self.loss_function(targets=batch_y, ubs=ubs, lbs=lbs, regr=regr)

            # compute & apply gradients
            gradient = t.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))
            # ta = ta.write(processed, loss)
            # processed += 1

        # tf.print(tf.reduce_mean(ta.stack())) if self.verbose else None

    def train(self, data: tf.data.Dataset, epochs: int = 10):
        data_batched = data.shuffle(1000).batch(self.batch_size, drop_remainder=True)

        for epoch in range(1, epochs + 1):
            # tf.print("EPOCH: " + str(epoch)) if self.verbose else None
            self.train_step(data=data_batched)
            self.epoch += 1
