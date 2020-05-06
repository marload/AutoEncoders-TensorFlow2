import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class UpconvAE(Model):
    def __init__(self, input_shape=(32, 32, 3)):
        super(UpconvAE, self).__init__()
        H, W, C = input_shape

        self.encoder = Sequential([
            layers.Input(input_shape),
            layers.ZeroPadding2D(1),
            layers.Conv2D(12, 4, 2, activation='relu'),
            layers.ZeroPadding2D(1),
            layers.Conv2D(24, 4, 2, activation='relu'),
            layers.ZeroPadding2D(1),
            layers.Conv2D(48, 4, 2, activation='relu'),
        ])
        self.decoder = Sequential([
            layers.Conv2DTranspose(
                24, 4, 2, padding='same', activation='relu'),
            layers.Conv2DTranspose(
                12, 4, 2, padding='same', activation='relu'),
            layers.Conv2DTranspose(
                3, 4, 2, padding='same', activation='sigmoid'),
            layers.Reshape(input_shape)
        ])

    def generate(self, z):
        return self.decoder(z, training=False)

    def call(self, x, training=False):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
