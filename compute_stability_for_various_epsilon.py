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

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import yaml
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model",
                    help="Name of the model to process.",
                    required=True,
                    choices=["nn", "renn", "stnn", "strenn", "is"]
                    )
parser.add_argument('-ni', '--noise_inj',
                    help="The amount of noise injection. Given as the variance of the Gaussian. Default: 0.",
                    type=float,
                    default=0,
                    required=False)
parser.add_argument("-em", "--epsilon_min",
                    help="Minimum noise level of additional corruption. Given as gaussian variance. Default: 0.01.",
                    type=float,
                    required=False,
                    default=0.01
                    )
parser.add_argument("-eM", "--epsilon_max",
                    help="Maximum noise level of additional corruption. Given as gaussian variance. Default: 0.1.",
                    type=float,
                    required=False,
                    default=0.1
                    )
parser.add_argument("-en", "--epsilon_n",
                    help="Number of noise level of additional corruption. Default: 10.",
                    type=int,
                    required=False,
                    default=10
                    )
parser.add_argument('--config',
                    help="The path for the .yml containing the configuration for the model.",
                    type=str,
                    required=False,
                    default=None)
args = parser.parse_args()

if args.config is None:
    suffix = str(args.noise_inj).split('.')[-1]
    args.config = f"./config/GoPro_{suffix}.yml"

with open(args.config, 'r') as file:
    setup = yaml.safe_load(file)
    
## ----------------------------------------------------------------------------------------------
## ---------- Initialization --------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
# Load data
DATA_PATH = './data/'
TEST_PATH = os.path.join(DATA_PATH, 'GOPRO_test_small.npy')

test_data = np.load(TEST_PATH)[10:51]
N_test, m, n = test_data.shape
print(f"Test data shape: {test_data.shape}")

# Define the setup for the forward problem
k_size = setup['k']
sigma = setup['sigma']
kernel = get_gaussian_kernel(k_size, sigma)

# Noise
noise_level = args.noise_inj
suffix = str(noise_level).split('.')[-1]

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
# np.random.seed(42)
for i in range(len(test_data)):
    y_delta = K @ test_data[i]
    y_delta = np.reshape(y_delta, (m, n))
    
    corr_data[i] = y_delta

print(f"...Done! (in {time.time() - start_time}s)")

# Load model.
# Model name -> Choose in {nn, stnn, renn, strenn, is}
model_name = args.model

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

model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}.h5", custom_objects={'SSIM': SSIM})

if use_convergence:
    Psi = reconstructors.VariationalReconstructor(algorithm)
else:
    model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}.h5", custom_objects={'SSIM': SSIM})

    # Define reconstructor
    Psi = reconstructors.StabilizedReconstructor(model, phi)

# Reconstruct
x_rec = Psi(corr_data)

if x_rec.shape[-1] == 1:
    x_rec = x_rec[:, :, :, 0]

## We estimate the error eta by computing the maximum of || Psi(Ax_gt) - x_gt ||_2 and then the accuracy as eta^{-1}.
err_vec = []
for i in range(len(test_data)):
    err = x_rec[i] - test_data[i]
    err_vec.append(np.linalg.norm(err.reshape((m, n)).flatten()))

idx_acc = np.argmax(np.array(err_vec))
acc = 1 / err_vec[idx_acc]

C_epsilon_vec = []
error_norm_vec = []
for epsilon in epsilon_vec:
    # Create vector for data
    epsilon_corr_data = np.zeros_like(test_data)

    print(f"Generating Corrupted Data for epsilon = {epsilon}...")
    start_time = time.time()
    # np.random.seed(42)
    for i in range(len(test_data)):
        y_delta = K @ test_data[i] + (noise_level + epsilon) * np.random.normal(0, 1, m*n)
        y_delta = np.reshape(y_delta, (m, n))
        
        epsilon_corr_data[i] = y_delta

    print(f"...Done! (in {time.time() - start_time}s)")

    # Reconstruct
    x_rec_epsilon = Psi(epsilon_corr_data)

    if x_rec_epsilon.shape[-1] == 1:
        x_rec_epsilon = x_rec_epsilon[:, :, :, 0]

    ## Given epsilon, we estimate C^epsilon_Psi by 
    ##
    ## C^epsilon = (max|| Psi(Ax_gt + e) - x_gt ||_2 - eta) / epsilon

    stab_vec = []
    for i in range(len(test_data)):
        stab = x_rec_epsilon[i] - test_data[i]
        stab_vec.append(np.linalg.norm(stab.reshape((m, n)).flatten()))

    idx_stab = np.argmax(np.array(stab_vec))
    stab = stab_vec[idx_stab]
    C_epsilon = (stab - (1 / acc)) / (np.linalg.norm((epsilon_corr_data[idx_stab] - corr_data[idx_stab]).flatten()))
    C_epsilon_vec.append(C_epsilon)
    error_norm_vec.append(stab)

    print("")
    print(f"Error of {model_name}: {1 / acc}")
    print(f"Accuracy of {model_name}: {acc}")
    print(f"{epsilon}-stability of {model_name}: {C_epsilon}")
    print(f"|| Psi(Ax + e) - x || = {stab}")
    print(f"Idx -> Acc: {idx_acc}, Stab: {idx_stab}")
    print("")

# Save the results
C_epsilon_vec = np.array(C_epsilon_vec)
error_norm_vec = np.array(error_norm_vec)

print(C_epsilon_vec)
print(error_norm_vec)

np.save('./plots/' + model_name + '_' + suffix + '_stability_different_epsilon.npy', C_epsilon_vec)
np.save('./plots/' + model_name + '_' + suffix + '_perturbed_error_different_epsilon.npy', error_norm_vec)