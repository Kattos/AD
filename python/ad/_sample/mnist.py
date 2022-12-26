import os
import subprocess

from tensorflow import Module, TensorSpec, GradientTape, function, uint8, saved_model
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

__all__ = ["MnistSample"]

NUM_CLASSES = 10
NUM_ROWS, NUM_COLS = 28, 28
BATCH_SIZE = 1


class MnistSample(Module):
    def __init__(self):
        super().__init__()

        self.model = Sequential(
            [
                Flatten(input_shape=(NUM_ROWS, NUM_COLS)),
                Dense(16, activation="relu"),
                Dense(16, activation="relu"),
                Dense(10, activation="softmax"),
            ]
        )

        self.loss = SparseCategoricalCrossentropy()
        self.optimizer = SGD()

    @function(input_signature=[TensorSpec([BATCH_SIZE, NUM_ROWS, NUM_COLS, 1])])
    def forward(self, inputs):
        return self.model(inputs, training=False)

    @function(
        input_signature=[
            TensorSpec([BATCH_SIZE, NUM_ROWS, NUM_COLS, 1]),
            TensorSpec([BATCH_SIZE], uint8),
        ]
    )
    def learn(self, inputs, labels):
        with GradientTape() as tape:
            probs = self.model(inputs, training=True)
            loss = self.loss(labels, probs)

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def save(self, path):
        saved_model.save(self, path, signatures={"forward": self.forward})
