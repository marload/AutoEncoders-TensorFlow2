import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class VanillaVAE(Model):
    def __init__(self, input_shape=(32, 32, 3)):
        super(VanillaVAE, self).__init__()
        H, W, C = input_shape

        self.encoder = Sequential([
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(64),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        self.decoder = Sequential([
            layers.Dense(64),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(H * W * C, activation='sigmoid'),
        ])

    def generate(self, z):
        return self.decoder(z, training=False)

    def call(self, x, training=False):
        encoded = self.encoder(x, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded
