# Import libraries
import argparse
import os

import numpy as np
import tensorflow as tf
import yaml
from tensorflow import keras as ks

import IPPy.nn.models as NN_models
from IPPy import stabilizers
from IPPy.metrics import *
from IPPy.nn.datasets import *
from IPPy.operators import *
from IPPy.utils import *
from miscellaneous import utilities

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    help="Name of the model to process. Can be used for multiple models.",
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
    "-nt",
    "--noise_type",
    help="Type of noise added to the input at training phace. Default: gaussian.",
    type=str,
    default="gaussian",
    required=False,
)
parser.add_argument(
    "--config",
    help="The path for the .yml containing the configuration for the model.",
    type=str,
    required=False,
    default=None,
)
parser.add_argument(
    "--verbose",
    help="Unable/Disable verbose for the code.",
    type=str,
    required=False,
    default="1",
    choices=["0", "1"],
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
TRAIN_PATH = os.path.join(DATA_PATH, "GOPRO_train_small.npy")
TEST_PATH = os.path.join(DATA_PATH, "GOPRO_test_small.npy")

train_data = np.load(TRAIN_PATH)
test_data = np.load(TEST_PATH)
N_train, m, n = train_data.shape
N_test, _, _ = test_data.shape
print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Define the setup for the forward problem
k_size = setup["k"]
sigma = setup["sigma"]
kernel = get_gaussian_kernel(k_size, sigma)

noise_level = args.noise_inj if args.noise_inj is not None else args.noise_level
suffix = str(noise_level).split(".")[-1]

if args.verbose == "1":
    print(f"Suffix: {suffix}")

# Number of epochs
n_epochs = setup["n_epochs"]
batch_size = setup["batch_size"]

# Model name -> Choose in {nn, stnn, renn, strenn}
model_name_list = args.model

for model_name in model_name_list:
    # Setting up the model given the name

    ## ----------------------------------------------------------------------------------------------
    ## ---------- NN --------------------------------------------------------------------------------
    ## ----------------------------------------------------------------------------------------------
    if model_name == "nn":
        # Define dataloader
        trainloader = Data2D(
            TRAIN_PATH,
            kernel,
            noise_level=noise_level,
            batch_size=batch_size,
            noise_type=args.noise_type,
        )

    ## ----------------------------------------------------------------------------------------------
    ## ---------- ReNN ------------------------------------------------------------------------------
    ## ----------------------------------------------------------------------------------------------
    if model_name == "renn":
        # Define dataloader
        trainloader = Data2D(
            TRAIN_PATH,
            kernel,
            noise_level=noise_level,
            batch_size=batch_size,
            convergence_path=os.path.join(
                DATA_PATH, f"GOPRO_convergence_small_{suffix}.npy"
            ),
            noise_type=args.noise_type,
        )

    ## ----------------------------------------------------------------------------------------------
    ## ---------- StNN ------------------------------------------------------------------------------
    ## ----------------------------------------------------------------------------------------------
    if model_name == "stnn":
        # Define dataloader
        reg_param = setup[model_name]["reg_param"]
        phi = stabilizers.Tik_CGLS_stabilizer(
            kernel, reg_param, k=setup[model_name]["n_iter"]
        )
        trainloader = Data2D(
            TRAIN_PATH,
            kernel,
            noise_level=noise_level,
            batch_size=batch_size,
            phi=phi,
            noise_type=args.noise_type,
        )

    ## ----------------------------------------------------------------------------------------------
    ## ---------- StReNN ----------------------------------------------------------------------------
    ## ----------------------------------------------------------------------------------------------
    if model_name == "strenn":
        # Define dataloader
        reg_param = setup[model_name]["reg_param"]
        phi = stabilizers.Tik_CGLS_stabilizer(
            kernel, reg_param, k=setup[model_name]["n_iter"]
        )
        trainloader = Data2D(
            TRAIN_PATH,
            kernel,
            noise_level=noise_level,
            batch_size=batch_size,
            phi=phi,
            convergence_path=os.path.join(
                DATA_PATH, f"GOPRO_convergence_small_{suffix}.npy"
            ),
            noise_type=args.noise_type,
        )

    # TRAIN
    # Build model and compile it
    model = NN_models.get_UNet(
        input_shape=(256, 256, 1),
        n_scales=4,
        conv_per_scale=2,
        final_relu=True,
        skip_connection=False,
    )

    # Define the Optimizer
    model.compile(
        optimizer=ks.optimizers.Adam(learning_rate=setup["learning_rate"]),
        loss="mse",
        metrics=[SSIM, "mse"],
    )

    # Train
    model.fit(trainloader, epochs=setup["n_epochs"])
    model.save(f"./model_weights/{model_name}_unet_{suffix}.h5")
