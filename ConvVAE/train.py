import tensorflow as tf
from model import ConvVAE

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--iterations', type=int, default=3000)
args = parser.parse_args()

BATCH_SIZE = args.batch_size
LR = args.lr
ITERATIONS = args.iterations
NUM_TEST = 8

cifar10 = tf.keras.datasets.cifar10
(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_test = x_test[np.array([51, 621, 66, 4211, 1156, 9512, 3011, 7877]), :]

train_ds = tf.data.Dataset.from_tensor_slices(
    x_train).shuffle(10000).batch(BATCH_SIZE).repeat()
train_ds = iter(train_ds)

ae = ConvVAE(50)

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

def test_result(steps):
    out = ae(x_test)

    figsize = (NUM_TEST, 2)
    fig = plt.figure(figsize=figsize)
    for idx in range(figsize[0]):
        plt.subplot(figsize[1], figsize[0], idx+1)
        plt.imshow(x_test[idx])
        plt.axis('off')
    for idx in range(figsize[0]):
        plt.subplot(figsize[1], figsize[0], idx+figsize[0]+1)
        plt.imshow(out[idx])
        plt.axis('off')
    plt.savefig(os.path.join('./result', str(
        steps).zfill(len(str(ITERATIONS))) + '.png'), bbox_inches='tight')

for steps in range(ITERATIONS):
    images = next(train_ds)
    train_step(images)
    test_result(steps)
    print('EPOCH[{}/{}] Loss={}'.format(steps,
                                        ITERATIONS, train_loss.result()))
    train_loss.reset_states()


