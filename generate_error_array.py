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
parser.add_argument(
    "-em",
    "--epsilon_min",
    help="Minimum noise level of additional corruption. Given as gaussian variance. Default: 0.01.",
    type=float,
    required=False,
    default=0.01,
)
parser.add_argument(
    "-eM",
    "--epsilon_max",
    help="Maximum noise level of additional corruption. Given as gaussian variance. Default: 0.1.",
    type=float,
    required=False,
    default=0.1,
)
parser.add_argument(
    "-en",
    "--epsilon_n",
    help="Number of noise level of additional corruption. Default: 10.",
    type=int,
    required=False,
    default=10,
)
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

# Utils
use_convergence = False
epsilon_min = args.epsilon_min
epsilon_max = args.epsilon_max
epsilon_n = args.epsilon_n

epsilon_vec = np.linspace(epsilon_min, epsilon_max, epsilon_n)

## ----------------------------------------------------------------------------------------------
## ---------- Accuracy --------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
# Create corrupted dataset
K = operators.ConvolutionOperator(kernel, (m, n))
corr_data = np.zeros_like(test_data)

print("Generating Corrupted Data...")
start_time = time.time()
for i in range(len(test_data)):
    y_delta = K @ test_data[i]
    y_delta = np.reshape(y_delta, (m, n))

    corr_data[i] = y_delta

print(f"...Done! (in {time.time() - start_time}s)")

# Load model.
# Model name -> Choose in {nn, stnn, renn, strenn, is}
model_name_list = args.model

for model_name in model_name_list:
    # Setting up the model given the name

    match model_name:
        case "nn":
            weights_name = "nn_unet"
            phi = stabilizers.PhiIdentity()
        case "stnn":
            weights_name = "stnn_unet"
            reg_param = setup[model_name]["reg_param"]
            phi = stabilizers.Tik_CGLS_stabilizer(
                kernel, reg_param, k=setup[model_name]["n_iter"]
            )
        case "renn":
            weights_name = "renn_unet"
            phi = stabilizers.PhiIdentity()
        case "strenn":
            weights_name = "strenn_unet"
            reg_param = setup[model_name]["reg_param"]
            phi = stabilizers.Tik_CGLS_stabilizer(
                kernel, reg_param, k=setup[model_name]["n_iter"]
            )
        case "is":
            use_convergence = True
            param_reg = setup[model_name]["reg_param"]
            algorithm = stabilizers.Tik_CGLS_stabilizer(
                kernel, param_reg, k=setup[model_name]["n_iter"]
            )

    if use_convergence:
        Psi = reconstructors.VariationalReconstructor(algorithm)
    else:
        if args.noise_level is not None:
            model = ks.models.load_model(
                f"./model_weights/{weights_name}_{suffix}.h5",
                custom_objects={"SSIM": SSIM},
            )
        elif args.noise_inj is not None:
            model = ks.models.load_model(
                f"./model_weights/{weights_name}_{suffix}_NI.h5",
                custom_objects={"SSIM": SSIM},
            )

        # Define reconstructor
        Psi = reconstructors.StabilizedReconstructor(model, phi)

    print("")
    error_vec = np.zeros((test_data.shape[0], epsilon_n, 2))
    for i, epsilon in enumerate(epsilon_vec):
        # Verbose
        print(
            f"Computing error vector for Psi = {model_name}, epsilon = {epsilon:0.3f}."
        )

        # Create vector for data
        epsilon_corr_data = corr_data + epsilon * np.random.normal(
            0, 1, corr_data.shape
        )

        # Reconstruct
        x_rec_epsilon = Psi(epsilon_corr_data)

        if x_rec_epsilon.shape[-1] == 1:
            x_rec_epsilon = x_rec_epsilon[:, :, :, 0]

        ## Given epsilon, we estimate C^epsilon_Psi by
        ##
        ## C^epsilon = (max|| Psi(Ax_gt + e) - x_gt ||_2 - eta) / epsilon

        stab_vec = np.linalg.norm(
            (x_rec_epsilon - test_data).reshape(x_rec_epsilon.shape[0], -1), axis=-1
        )
        noise_vec = np.linalg.norm(
            (epsilon_corr_data - corr_data).reshape(epsilon_corr_data.shape[0], -1),
            axis=-1,
        )

        error_vec[:, i, 0] = stab_vec
        error_vec[:, i, 1] = noise_vec

    np.save(
        f"./plots/{model_name}_unet_{suffix}_error_{epsilon_min}_to_{epsilon_max}.npy",
        error_vec,
    )
    print(error_vec.shape)
