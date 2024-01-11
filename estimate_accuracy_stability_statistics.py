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

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    help="Name of the model to process. Can be used for multiple models to compare them.",
    required=True,
    action="append",
    choices=["nn", "renn", "stnn", "strenn", "is"],
)
stabilization = parser.add_mutually_exclusive_group(required=True)
stabilization.add_argument(
    "-ni",
    "--noise_inj",
    help="The amount of noise injection. Given as the variance of the Gaussian.",
    type=float,
    required=False,
)
stabilization.add_argument(
    "-nl",
    "--noise_level",
    help="The amount of noise level added to the input datum. Given as the variance of the Gaussian.",
    type=float,
    required=False,
)
parser.add_argument(
    "-e",
    "--epsilon",
    help="Noise level of additional corruption. Given as gaussian variance. Default: 0.",
    type=float,
    required=False,
    default=0,
)
parser.add_argument(
    "--n_tests",
    help="Number of times the computation will be performed. Default: 20.",
    type=int,
    required=False,
    default=20,
)
parser.add_argument(
    "--sample_per_test",
    help="Number of samples per test. Default: 50.",
    type=int,
    required=False,
    default=50,
)
parser.add_argument(
    "--config",
    help="The path for the .yml containing the configuration for the model.",
    type=str,
    required=False,
    default=None,
)
args = parser.parse_args()

if args.config is None:
    noise_level = args.noise_inj if args.noise_inj is not None else args.noise_level
    suffix = str(noise_level).split(".")[-1]
    args.config = f"./config/GoPro_{suffix}.yml"

with open(args.config, "r") as file:
    setup = yaml.safe_load(file)

## ----------------------------------------------------------------------------------------------
## ---------- Initialization --------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
utilities.initialization()

# Load data
DATA_PATH = "./data/"
TEST_PATH = os.path.join(DATA_PATH, "GOPRO_test_small.npy")

# Read Test Set
test_data = np.load(TEST_PATH)
N_test, m, n = test_data.shape

# Define the setup for the forward problem
k_size = setup["k"]
sigma = setup["sigma"]
kernel = get_gaussian_kernel(k_size, sigma)

# Noise
noise_level = args.noise_inj if args.noise_inj is not None else args.noise_level
suffix = str(noise_level).split(".")[-1]

epsilon = args.epsilon

# Set a seed
np.random.seed(seed=42)

# Statistics
N_population = args.n_tests
N_per_test = args.sample_per_test  # Number of test sample per sample test

pop_accuracies = []
pop_stability_constants = []

# Repeat N_population times
for n_test in range(N_population):
    print("")
    print(f"Sample test: {n_test+1}")

    # Sample from Test Set
    idx_sample = np.random.choice(np.arange(N_test), size=(N_per_test,))
    test_data_sample = test_data[idx_sample]
    N_test, m, n = test_data_sample.shape

    ## ----------------------------------------------------------------------------------------------
    ## ---------- Accuracy --------------------------------------------------------------------------
    ## ----------------------------------------------------------------------------------------------
    # Create corrupted dataset
    K = operators.ConvolutionOperator(kernel, (m, n))
    corr_data = np.zeros_like(test_data_sample)
    epsilon_corr_data = np.zeros_like(test_data_sample)

    print("Generating Corrupted Data...")
    start_time = time.time()
    for i in range(len(test_data_sample)):
        y_delta = K @ test_data_sample[i] + noise_level * np.random.normal(0, 1, m * n)
        y_delta = np.reshape(y_delta, (m, n))

        corr_data[i] = y_delta

        y_delta = K @ test_data_sample[i] + (noise_level + epsilon) * np.random.normal(
            0, 1, m * n
        )
        y_delta = np.reshape(y_delta, (m, n))

        epsilon_corr_data[i] = y_delta

    print(f"...Done! (in {time.time() - start_time}s)")

    # Load model.
    # Model name -> Choose in {nn, stnn, renn, strenn, is}
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
        err_vec = np.linalg.norm(
            (x_rec - test_data_sample).reshape(x_rec.shape[0], -1), axis=-1
        )
        idx_acc = np.argmax(err_vec)
        acc = 1 / err_vec[idx_acc]

        ## Given epsilon, we estimate C^epsilon_Psi by
        ##
        ## C^epsilon = (max|| Psi(Ax_gt) - x_gt ||_2 - eta) / epsilon
        stab_vec = np.linalg.norm(
            (x_rec_epsilon - test_data_sample).reshape(x_rec_epsilon.shape[0], -1),
            axis=-1,
        )
        idx_stab = np.argmax(stab_vec)
        stab = stab_vec[idx_stab]
        C_epsilon = (stab - (1 / acc)) / (
            np.linalg.norm(
                (epsilon_corr_data[idx_stab] - corr_data[idx_stab]).flatten()
            )
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

    # Save accuracies and stability_constants
    pop_accuracies.append(np.array(accuracies))
    pop_stability_constants.append(np.array(stability_constants))

# Save on a file
pop_accuracies = np.array(pop_accuracies)
pop_stability_constants = np.array(pop_stability_constants)

np.save(
    f"./plots/accuracies_noise_{suffix}_epsilon_{epsilon}.npy",
    pop_accuracies,
)
np.save(
    f"./plots/stabilities_noise_{suffix}_epsilon_{epsilon}.npy",
    pop_stability_constants,
)
