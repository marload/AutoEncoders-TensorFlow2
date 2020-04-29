import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class SparseAE(Model):
    def __init__(self, input_shape=(32, 32, 3)):
        super(SparseAE, self).__init__()
        H, W, C = input_shape

        self.encoder = Sequential([
            layers.Input(input_shape),
            layers.Flatten(),
            layers.Dense(256, activation='relu')
        ])
        self.l1 = layers.ActivityRegularization(l1=1e-5)
        self.decoder = Sequential([
            layers.Dense(H * W * C, activation='sigmoid'),
            layers.Reshape(input_shape)
        ])

    def generate(self, z):
        return self.decoder(z, training=False)

    def call(self, x, training=False):
        encoded = self.encoder(x)
        encoded = self.l1(encoded, training=training)
        decoded = self.decoder(encoded)
        return decoded
