# Import libraries
import numpy as np
import os
from tensorflow import keras as ks

import time

from IPPy.metrics import *
from IPPy.utils import *
from IPPy import operators
from IPPy import stabilizers
from IPPy import reconstructors

import argparse
import yaml
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model",
                           help="Name of the model to process. Can be used for multiple models to compare them.",
                           required=True,
                           action='append',
                           choices=["nn", "renn", "stnn", "strenn", "is"]
                           )
stabilization = parser.add_mutually_exclusive_group(required=True)
stabilization.add_argument('-ni', '--noise_inj',
                    help="The amount of noise injection. Given as the variance of the Gaussian.",
                    type=float,
                    required=False)
stabilization.add_argument('-nl', '--noise_level',
                    help="The amount of noise level added to the input datum. Given as the variance of the Gaussian.",
                    type=float,
                    required=False)
parser.add_argument("-e", "--epsilon",
                           help="Noise level of additional corruption. Given as gaussian variance. Default: 0.",
                           type=float,
                           required=False,
                           default=0
                           )
parser.add_argument('--config',
                    help="The path for the .yml containing the configuration for the model.",
                    type=str,
                    required=False,
                    default=None)
args = parser.parse_args()

if args.config is None:
    noise_level = args.noise_inj if args.noise_inj is not None else args.noise_level
    suffix = str(noise_level).split('.')[-1]
    args.config = f"./config/GoPro_{suffix}.yml"

with open(args.config, 'r') as file:
    setup = yaml.safe_load(file)

## ----------------------------------------------------------------------------------------------
## ---------- Initialization --------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
# Load data
DATA_PATH = './data/'
TEST_PATH = os.path.join(DATA_PATH, 'GOPRO_test_small.npy')

test_data = np.load(TEST_PATH)
N_test, m, n = test_data.shape
print(f"Test data shape: {test_data.shape}")

# Define the setup for the forward problem
k_size = setup['k']
sigma = setup['sigma']
kernel = get_gaussian_kernel(k_size, sigma)

# Noise
noise_level = args.noise_inj if args.noise_inj is not None else args.noise_level
suffix = str(noise_level).split('.')[-1]

epsilon = args.epsilon

# Utils
use_convergence = False

# Set a seed
np.random.seed(seed=42)

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

    y_delta = K @ test_data[i] + (noise_level + epsilon) * np.random.normal(0, 1, m*n)
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
    match model_name:
        case 'nn': 
            weights_name = 'nn_unet'
            phi = stabilizers.PhiIdentity()
        case 'stnn':
            weights_name = 'stnn_unet'
            reg_param = setup[model_name]['reg_param']
            phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=setup[model_name]['n_iter'])
        case 'renn':
            weights_name = 'renn_unet'
            phi = stabilizers.PhiIdentity()
        case 'strenn':
            weights_name = 'strenn_unet'
            reg_param = setup[model_name]['reg_param']
            phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=setup[model_name]['n_iter'])
        case 'is':
            use_convergence = True
            param_reg = setup[model_name]['reg_param']
            algorithm = stabilizers.Tik_CGLS_stabilizer(kernel, param_reg, k=setup[model_name]['n_iter'])

    if use_convergence:
        Psi = reconstructors.VariationalReconstructor(algorithm)
    else:
        if args.noise_level is not None:
            model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}.h5", custom_objects={'SSIM': SSIM})
        elif args.noise_inj is not None:
            model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}_NI.h5", custom_objects={'SSIM': SSIM})

        # Define reconstructor
        Psi = reconstructors.StabilizedReconstructor(model, phi)

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
    stab_vec = np.linalg.norm((x_rec_epsilon - test_data).reshape(x_rec.shape[0], -1), axis=-1)
    idx_stab = np.argmax(stab_vec)
    stab = stab_vec[idx_stab]
    C_epsilon = (stab - (1 / acc)) / (np.linalg.norm((epsilon_corr_data[idx_stab] - corr_data[idx_stab]).flatten()))
    
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
accuracies.insert(0, 'Accuracy')
stability_constants.insert(0, 'Stability')
print(tabulate.tabulate([accuracies, stability_constants], headers=[f"\u03B5 = {epsilon}",] + args.model))