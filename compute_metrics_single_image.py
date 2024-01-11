# Import libraries
import argparse
import os

import numpy as np
import tensorflow as tf
import yaml
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tensorflow import keras as ks

from IPPy import operators, reconstructors, stabilizers
from IPPy.metrics import *
from IPPy.nn import models as NN_models
from IPPy.utils import *
from miscellaneous import utilities

#### TODO:
# - Gestire parsing
# - Semplificare il match model
# - Introdurre file .py con gli esperimenti A e B.

## ----------------------------------------------------------------------------------------------
## ---------- Initialization --------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
utilities.initialization()
args, setup = utilities.parse_arguments()

# Load data
DATA_PATH = "./data/"
TEST_PATH = os.path.join(DATA_PATH, "GOPRO_train_small.npy")

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

# Set a seed
np.random.seed(seed=42)

## ----------------------------------------------------------------------------------------------
## ---------- Model evaluation ------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
# Load the test image
if args.path.isdigit():
    idx = int(args.path)
    x_gt = test_data[idx]
else:
    # Load the given image
    x_gt = Image.open(args.path)[:, :, 0]
    x_gt = np.array(x_gt.resize((256, 256)))

# Corrupt it
K = operators.ConvolutionOperator(kernel, (m, n))
y_delta = K @ x_gt + (noise_level + epsilon) * np.random.normal(0, 1, m * n)
y_delta = np.reshape(y_delta, (m, n))

# Visualize
save_output = True
if save_output:
    plt.imsave("./images/corr_image.png", y_delta, cmap="gray")
    plt.imsave("./images/gt_image.png", x_gt, cmap="gray")

# Load model.
# Model name -> Choose in {nn, stnn, renn, strenn}
model_name_list = args.model

RE_list = []
PSNR_list = []
SSIM_list = []


for model_name in model_name_list:
    # Setting up the model given the name
    Psi = utilities.get_reconstructor(model_name, kernel, args, setup)

    # Reconstruct
    x_rec = Psi(y_delta)

    # Metrics
    RE_list.append(rel_err(x_gt, x_rec))
    SSIM_list.append(ssim(x_gt, x_rec, data_range=1))
    PSNR_list.append(PSNR(x_gt, x_rec))

    # Save reconstruction
    if save_output:
        plt.imsave(
            f"./images/recon_{model_name}_{suffix}_eps_{str(epsilon)}.png",
            x_rec,
            cmap="gray",
        )

# Print out the metrics
import tabulate

errors = [
    [
        "Corrupted",
        rel_err(x_gt, y_delta),
        PSNR(x_gt, y_delta),
        ssim(x_gt, y_delta, data_range=1),
    ]
]
for i in range(len(model_name_list)):
    errors.append(
        [model_name_list[i].capitalize(), RE_list[i], PSNR_list[i], SSIM_list[i]]
    )

print(
    tabulate.tabulate(
        errors, headers=[f"\u03B5 = {epsilon}", "Rel. Err.", "PSNR", "SSIM"]
    )
)
