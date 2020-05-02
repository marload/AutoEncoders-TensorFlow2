import tensorflow as tf
from model import VanillaAE

import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--iterations', type=int, default=5000)
args = parser.parse_args()

BATCH_SIZE = args.batch_size
LR = args.lr
ITERATIONS = args.iterations

cifar10 = tf.keras.datasets.cifar10
(x_train, _), (_, _) = cifar10.load_data()
x_train = x_train / 255.0

train_ds = tf.data.Dataset.from_tensor_slices(
    x_train).shuffle(10000).batch(BATCH_SIZE).repeat()
train_ds = iter(train_ds)

ae = VanillaAE()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(LR)

train_loss = tf.keras.metrics.Mean(name='train_loss')


@tf.function
def train_step(images):
    with tf.GradientTape() as tape:
        out = ae(images, training=True)
        loss = loss_object(out, images)
    grad = tape.gradient(loss, ae.trainable_variables)
    optimizer.apply_gradients(zip(grad, ae.trainable_variables))
    train_loss(loss)


for steps in range(ITERATIONS):
    images = next(train_ds)
    train_step(images)
    print('EPOCH[{}/{}] Loss={}'.format(steps,
                                        ITERATIONS, train_loss.result()))
    train_loss.reset_states()
