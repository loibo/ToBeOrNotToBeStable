# Import libraries
import numpy as np

import tensorflow as tf
from tensorflow import keras as ks

import IPPy.nn.models as NN_models
from IPPy.metrics import *
from IPPy.utils import *
from IPPy.operators import *
from IPPy.nn.datasets import *
from IPPy import stabilizers

import os


## ----------------------------------------------------------------------------------------------
## ---------- Initialization --------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------

# Load data
DATA_PATH = './data/'
TRAIN_PATH = os.path.join(DATA_PATH, 'GOPRO_train_small.npy')
TEST_PATH = os.path.join(DATA_PATH, 'GOPRO_test_small.npy')

train_data = np.load(TRAIN_PATH)
test_data = np.load(TEST_PATH)
N_train, m, n = train_data.shape
N_test, _, _ = test_data.shape
print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Define the setup for the forward problem
k_size = 11
sigma = 1.3
kernel = get_gaussian_kernel(k_size, sigma)

noise_level = 0.025

if noise_level == 0:
    suffix = '0'
elif noise_level == 0.025:
    suffix = '025'

print(f"Suffix: {suffix}")

# Number of epochs
n_epochs = 50

## ----------------------------------------------------------------------------------------------
## ---------- NN --------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
# Define dataloader
trainloader = Data2D(TRAIN_PATH, kernel, noise_level=noise_level, batch_size=8)

# Build model and compile it
model = NN_models.get_UNet(input_shape = (256, 256, 1), n_scales = 4, conv_per_scale = 2, final_relu=True, skip_connection=False)

# Define the Optimizer
initial_learning_rate = 5e-4
lr_schedule = ks.optimizers.schedules.PolynomialDecay(
    initial_learning_rate,
    decay_steps=1e4,
    end_learning_rate=1e-5)

model.compile(optimizer=ks.optimizers.Adam(learning_rate=initial_learning_rate),
              loss='mse',
              metrics=[SSIM, 'mse'])

# Train
model.fit(trainloader, epochs=n_epochs)
model.save(f"./model_weights/nn_unet_{suffix}.h5")
print(f"Training of NN model -> Finished.")

## ----------------------------------------------------------------------------------------------
## ---------- ReNN ------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
# Define dataloader
trainloader = Data2D(TRAIN_PATH, kernel, noise_level=noise_level, batch_size=8, convergence_path=os.path.join(DATA_PATH, f"GOPRO_convergence_small_0.npy"))

# Build model and compile it
model = NN_models.get_UNet(input_shape = (256, 256, 1), n_scales = 4, conv_per_scale = 2, final_relu=True, skip_connection=False)

# Define the Optimizer
initial_learning_rate = 1e-3
lr_schedule = ks.optimizers.schedules.PolynomialDecay(
    initial_learning_rate,
    decay_steps=1e4,
    end_learning_rate=1e-5)

model.compile(optimizer=ks.optimizers.Adam(learning_rate=lr_schedule),
              loss='mse',
              metrics=[SSIM, 'mse'])

# Train
model.fit(trainloader, epochs=n_epochs)
model.save(f"./model_weights/renn_unet_{suffix}.h5")
print(f"Training of ReNN model -> Finished.")

## ----------------------------------------------------------------------------------------------
## ---------- StNN ------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
# Define dataloader
reg_param = 1e-2
phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=3)
trainloader = Data2D(TRAIN_PATH, kernel, noise_level=noise_level, batch_size=8, phi=phi)

# Build model and compile it
model = NN_models.get_UNet(input_shape = (256, 256, 1), n_scales = 4, conv_per_scale = 2, final_relu=True, skip_connection=False)

# Define the Optimizer
initial_learning_rate = 1e-3
lr_schedule = ks.optimizers.schedules.PolynomialDecay(
    initial_learning_rate,
    decay_steps=1e4,
    end_learning_rate=1e-5)

model.compile(optimizer=ks.optimizers.Adam(learning_rate=lr_schedule),
              loss='mse',
              metrics=[SSIM, 'mse'])

# Train
model.fit(trainloader, epochs=n_epochs)
model.save(f"./model_weights/stnn_unet_{suffix}.h5")
print(f"Training of StNN model -> Finished.")

## ----------------------------------------------------------------------------------------------
## ---------- StReNN ----------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
# Define dataloader
reg_param = 1e-2
phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=3)
trainloader = Data2D(TRAIN_PATH, kernel, noise_level=noise_level, batch_size=8, phi=phi, convergence_path=os.path.join(DATA_PATH, f"GOPRO_convergence_small_0.npy"))

# Build model and compile it
model = NN_models.get_UNet(input_shape = (256, 256, 1), n_scales = 4, conv_per_scale = 2, final_relu=True, skip_connection=False)

# Define the Optimizer
initial_learning_rate = 1e-3
lr_schedule = ks.optimizers.schedules.PolynomialDecay(
    initial_learning_rate,
    decay_steps=1e4,
    end_learning_rate=1e-5)

model.compile(optimizer=ks.optimizers.Adam(learning_rate=lr_schedule),
              loss='mse',
              metrics=[SSIM, 'mse'])

# Train
model.fit(trainloader, epochs=n_epochs)
model.save(f"./model_weights/strenn_unet_{suffix}.h5")
print(f"Training of StReNN model -> Finished.")