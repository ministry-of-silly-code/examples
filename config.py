import math
import os

import experiment_buddy
import jax.example_libraries.optimizers

initial_lr = .0001

decay_steps = 500000
num_hidden = 1024
decay_factor = .5

batch_size = 128
momentum_mass = 0.99
weight_norm = 0.00

num_epochs = 1000

# experiment_buddy.register(locals())

################################################################
# Derivative parameters
################################################################
learning_rate = jax.example_libraries.optimizers.inverse_time_decay(initial_lr, decay_steps, decay_factor,
                                                                    staircase=True)
eval_every = math.ceil(num_epochs / 1000)

writer = experiment_buddy.deploy(
    url="frosty",
    disabled=False,
    conda_env="py39",
    extra_modules=[],
    sweep_definition="sweep.yaml",
    wandb_run_name="example",
    wandb_kwargs={'entity': "ionelia", 'project': "homecredit"}
)
