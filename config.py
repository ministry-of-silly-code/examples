import math
import os

import jax.example_libraries.optimizers

import experiment_buddy

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
learning_rate = jax.example_libraries.optimizers.inverse_time_decay(initial_lr, decay_steps, decay_factor, staircase=True)
eval_every = math.ceil(num_epochs / 1000)

# tensorboard = experiment_buddy.deploy(host=os.environ.get('BUDDY_HOST', ""), sweep_yaml=os.environ.get('SWEEP', ""))
# tensorboard = experiment_buddy.deploy(host="milafrosty", sweep_yaml="sweep.yaml")
# tensorboard = experiment_buddy.deploy(url="milafrosty", sweep_yaml="")

experiment_buddy.deploy(url="milafrosty", disabled=False, conda_env="", extra_modules=[],
                                 # sweep_definition="sweep.yaml",
                                 wandb_run_name="homecredit",
                                 wandb_kwargs={'entity': "ionelia", 'project': "homecredit"})
