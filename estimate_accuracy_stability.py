# Import libraries
import argparse
import os
import time

import numpy as np
import yaml
from tensorflow import keras as ks

from IPPy import operators, reconstructors, stabilizers
from IPPy.metrics import *
from IPPy.utils import *
from miscellaneous import utilities

parser = utilities.default_parsing()
args, setup = utilities.parse_arguments(parser)

## ----------------------------------------------------------------------------------------------
## ---------- Initialization --------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
utilities.initialization(seed=42)

# Load data
DATA_PATH = "./data/"
TEST_PATH = os.path.join(DATA_PATH, "GOPRO_test_small.npy")

test_data = np.load(TEST_PATH)
N_test, m, n = test_data.shape
print(f"Test data shape: {test_data.shape}")

# Define the setup for the forward problem
k_size = setup["k"]
sigma = setup["sigma"]
kernel = get_gaussian_kernel(k_size, sigma)

# Noise
noise_level = args.noise_inj if args.noise_inj is not None else args.noise_level
suffix = str(noise_level).split(".")[-1]

epsilon = args.epsilon

## ----------------------------------------------------------------------------------------------
## ---------- Accuracy --------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
# Create corrupted dataset
K = operators.ConvolutionOperator(kernel, (m, n))
corr_data = np.zeros_like(test_data)
epsilon_corr_data = np.zeros_like(test_data)

print("Generating Corrupted Data...")
start_time = time.time()
for i in range(len(test_data)):
    y_delta = K @ test_data[i]
    y_delta = np.reshape(y_delta, (m, n))

    corr_data[i] = y_delta

    y_delta = K @ test_data[i] + (noise_level + epsilon) * np.random.normal(0, 1, m * n)
    y_delta = np.reshape(y_delta, (m, n))

    epsilon_corr_data[i] = y_delta

print(f"...Done! (in {time.time() - start_time:0.4f}s)")

# Load model.
# Model name -> Choose in {nn, stnn, renn, strenn}
model_name_list = args.model

accuracies = []
stability_constants = []
for model_name in model_name_list:
    # Setting up the model given the name
    Psi = utilities.get_reconstructor(model_name, kernel, args, setup)

    # Reconstruct
    x_rec = Psi(corr_data)
    x_rec_epsilon = Psi(epsilon_corr_data)

    if x_rec.shape[-1] == 1:
        x_rec = x_rec[:, :, :, 0]

    if x_rec_epsilon.shape[-1] == 1:
        x_rec_epsilon = x_rec_epsilon[:, :, :, 0]

    ## We estimate the error eta by computing the maximum of || Psi(Ax_gt) - x_gt ||_2 and then the accuracy as eta^{-1}.
    err_vec = np.linalg.norm((x_rec - test_data).reshape(x_rec.shape[0], -1), axis=-1)
    idx_acc = np.argmax(err_vec)
    acc = 1 / err_vec[idx_acc]

    ## Given epsilon, we estimate C^epsilon_Psi by
    ##
    ## C^epsilon = (max|| Psi(Ax_gt) - x_gt ||_2 - eta) / epsilon
    stab_vec = np.linalg.norm(
        (x_rec_epsilon - test_data).reshape(x_rec_epsilon.shape[0], -1), axis=-1
    )
    idx_stab = np.argmax(stab_vec)
    stab = stab_vec[idx_stab]
    C_epsilon = (stab - (1 / acc)) / (
        np.linalg.norm((epsilon_corr_data[idx_stab] - corr_data[idx_stab]).flatten())
    )

    print("")
    print(f"Error of {model_name}: {1 / acc}")
    print(f"Accuracy of {model_name}: {acc}")
    print(f"{epsilon}-stability of {model_name}: {C_epsilon}")
    print(f"|| Psi(Ax + e) - x || = {stab}")
    print(f"Idx -> Acc: {idx_acc}, Stab: {idx_stab}")
    print("")

    accuracies.append(acc)
    stability_constants.append(C_epsilon)

# Visualize the results
import tabulate

accuracies.insert(0, "Accuracy")
stability_constants.insert(0, "Stability")
print(
    tabulate.tabulate(
        [accuracies, stability_constants],
        headers=[
            f"\u03B5 = {epsilon}",
        ]
        + args.model,
    )
)
