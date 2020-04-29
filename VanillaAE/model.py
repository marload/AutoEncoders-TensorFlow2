import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class VanillaAE(Model):
    def __init__(self, input_shape=(32, 32, 3)):
        super(VanillaAE, self).__init__()
        H, W, C = input_shape

        self.encoder = Sequential([
            layers.Input(input_shape),
            layers.Flatten(),
            layers.Dense(256, activation='relu')
        ])
        self.decoder = Sequential([
            layers.Dense(H * W * C, activation='sigmoid'),
            layers.Reshape(input_shape)
        ])

    def generate(self, z):
        return self.decoder(z, training=False)

    def call(self, x, training=False):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
