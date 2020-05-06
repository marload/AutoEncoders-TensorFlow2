import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class ConvVAE(Model):
    def __init__(self, latent_dim, input_shape=(32, 32, 3)):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = Sequential([
            layers.Input(input_shape),
            layers.Conv2D(32, 3, 2, activation='relu'),
            layers.Conv2D(64, 3, 2, activation='relu'),
            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim)
        ])
        self.decoder = Sequential([
            layers.Dense(7*7*32, activation='relu'),
            layers.Reshape((7, 7, 32)),
            layers.Conv2DTranspose(64, 3, 2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, 3, 2, padding='same', activation='relu'),
            layers.Conv2DTranspose(3, 3, 1, padding='same', activation='sigmoid')
        ])

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, x):
        return self.decoder(x)
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps + tf.exp(logvar * 0.5) + mean

    def generate(self, z):
        return self.decoder(z, training=False)

    def call(self, x, training=False):
        encoded = self.encoder(x, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded
