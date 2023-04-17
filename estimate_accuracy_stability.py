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
k_size = 11
sigma = 1.3
kernel = get_gaussian_kernel(k_size, sigma)

# Noise
epsilon = 0.05
noise_level = 0.025

if noise_level == 0:
    suffix = '0'
elif noise_level == 0.025:
    suffix = '025'

# Utils
use_convergence = False

## ----------------------------------------------------------------------------------------------
## ---------- Accuracy --------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
# Create corrupted dataset
K = operators.ConvolutionOperator(kernel, (m, n))
corr_data = np.zeros_like(test_data)
epsilon_corr_data = np.zeros_like(test_data)

print("Generating Corrupted Data...")
start_time = time.time()
# np.random.seed(42)
for i in range(len(test_data)):
    y_delta = K @ test_data[i] # + noise_level * np.random.normal(0, 1, m*n)
    y_delta = np.reshape(y_delta, (m, n))
    
    corr_data[i] = y_delta

    y_delta = K @ test_data[i] + (noise_level + epsilon) * np.random.normal(0, 1, m*n)
    y_delta = np.reshape(y_delta, (m, n))
    
    epsilon_corr_data[i] = y_delta

print(f"...Done! (in {time.time() - start_time}s)")

# Load model.
# Model name -> Choose in {nn, stnn, renn, strenn}
model_name_list = ('renn', 'strenn', 'is') # ('nn', 'stnn', 'renn', 'strenn') # , 'is')

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
            reg_param = 1e-2
            phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=3)
        case 'renn':
            weights_name = 'renn_unet'
            phi = stabilizers.PhiIdentity()
        case 'strenn':
            weights_name = 'strenn_unet'
            reg_param = 1e-2
            phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=3)
        case 'is':
            use_convergence = True
            param_reg = 0.31 # 8e-2
            algorithm = stabilizers.Tik_CGLS_stabilizer(kernel, param_reg, k=100)

    if use_convergence:
        Psi = reconstructors.VariationalReconstructor(algorithm)
    else:
        model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}_noise_data.h5", custom_objects={'SSIM': SSIM})

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
    err_vec = []
    for i in range(len(test_data)):
        err = x_rec[i] - test_data[i]
        err_vec.append(np.linalg.norm(err.reshape((m, n)).flatten()))

    idx_acc = np.argmax(np.array(err_vec))
    acc = 1 / err_vec[idx_acc]

    ## Given epsilon, we estimate C^epsilon_Psi by 
    ##
    ## C^epsilon = (max|| Psi(Ax_gt) - x_gt ||_2 - eta) / epsilon

    stab_vec = []
    for i in range(len(test_data)):
        stab = x_rec_epsilon[i] - test_data[i]
        stab_vec.append(np.linalg.norm(stab.reshape((m, n)).flatten()))

    idx_stab = np.argmax(np.array(stab_vec))
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
print(tabulate.tabulate([accuracies, stability_constants], headers=[f"\u03B5 = {epsilon}", 'NN', 'StNN', 'ReNN', 'StReNN', 'Conv.']))