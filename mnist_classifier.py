# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic MNIST example using JAX with the mini-libraries stax and optimizers.

The mini-library jax.experimental.stax is for neural network building, and
the mini-library jax.experimental.optimizers is for first-order stochastic
optimization.
"""

import itertools
import time

import jax.numpy as jnp
import numpy.random as npr
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax

import config
import datasets


def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))


def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)


init_random_params, predict = stax.serial(
    Dense(config.num_hidden), Relu,
    Dense(config.num_hidden), Relu,
    Dense(10), LogSoftmax)

if __name__ == "__main__":

    writer = config.writer

    rng = random.PRNGKey(0)


    train_images, train_labels, test_images, test_labels = datasets.mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, config.batch_size)
    num_batches = num_complete_batches + bool(leftover)


    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * config.batch_size:(i + 1) * config.batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]


    batches = data_stream()

    opt_init, opt_update, get_params = optimizers.momentum(config.learning_rate, mass=config.momentum_mass)


    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)


    _, init_params = init_random_params(rng, (-1, 28 * 28))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    print("\nStarting training...")
    for epoch in range(config.num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))
        epoch_time = time.time() - start_time

        params = get_params(opt_state)
        train_acc = accuracy(params, (train_images, train_labels))
        test_acc = accuracy(params, (test_images, test_labels))
        writer.log({"train_acc": float(train_acc), "test_acc": float(test_acc)}, commit=True)
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))
