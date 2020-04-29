import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class UpconvAE(Model):
    def __init__(self, input_shape=(32, 32, 3)):
        super(UpconvAE, self).__init__()
        H, W, C = input_shape

        self.noise = layers.GaussianNoise(0.2)
        self.encoder = Sequential([
            layers.Input(input_shape),
            layers.Conv2D(32, 3, padding='same', activaiton='relu'),
            layers.MaxPool2D(2, 2),
            layers.Conv2D(16, 3, padding='same', activaiton='relu'),
            layers.MaxPool2D(2, 2),
            layers.Conv2D(8, 3, padding='same', activaiton='relu'),
            layers.MaxPool2D(2, 2),
            layers.Conv2D(8, 3, padding='same', activaiton='relu'),
            layers.MaxPool2D(2, 2),
        ])
        self.decoder = Sequential([
            layers.Conv2DTranspose(8, 3, activation='relu'),
            layers.Conv2DTranspose(16, 3, activation='relu'),
            layers.Conv2DTranspose(32, 3, activation='relu'),
            layers.Conv2DTranspose(3, 3, activation='sigmoid'),
            layers.Reshape(input_shape)
        ])

    def call(self, x, training=False):
        noised = self.noise(x)
        encoded = self.encoder(noised)
        decoded = self.decoder(encoded)
        return decoded
