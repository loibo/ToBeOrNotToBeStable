# Import libraries
import numpy as np
import os
from tensorflow import keras as ks

import time

##### TO DO: add ... to the import.
from IPPy.metrics import *
from IPPy.utils import *
from IPPy import operators
from IPPy import stabilizers
from IPPy import reconstructors

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
noise_level = 0.025

if noise_level == 0:
    suffix = '0'
elif noise_level == 0.025:
    suffix = '025'

# Utils
use_convergence = False
epsilon_min = 0.01
epsilon_max = 0.1
epsilon_n = 10

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
    y_delta = K @ test_data[i] # + noise_level * np.random.normal(0, 1, m*n)
    y_delta = np.reshape(y_delta, (m, n))
    
    corr_data[i] = y_delta

print(f"...Done! (in {time.time() - start_time}s)")

# Load model.
# Model name -> Choose in {nn, stnn, renn, strenn, is}
model_name_list = ('renn', 'strenn', 'is') #('nn', 'stnn', 'renn', 'strenn') # , 'is')

models_error = []
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
            param_reg = 0.27 # 1e-1 # 0.31 # 8e-2
            algorithm = stabilizers.Tik_CGLS_stabilizer(kernel, param_reg, k=100)

    if use_convergence:
        Psi = reconstructors.VariationalReconstructor(algorithm)
    else:
        model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}_noise_data.h5", custom_objects={'SSIM': SSIM})

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

    error_norm_vec = []
    for epsilon in epsilon_vec:
        # Create vector for data
        epsilon_corr_data = np.zeros_like(test_data)

        print(f"Generating Corrupted Data for epsilon = {epsilon}...")
        start_time = time.time()
        # np.random.seed(42)
        for i in range(len(test_data)):
            y_delta = K @ test_data[i] + epsilon * np.random.normal(0, 1, m*n)
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
        error_norm_vec.append(stab)

        print("")
        print(f"Accuracy of {model_name}: {acc}")
        print(f"For epsilon={epsilon}, || Psi(Ax + e) - x || = {stab}")
        print(f"Idx -> Acc: {idx_acc}, Stab: {idx_stab}")
        print("")

    # Save the results
    error_norm_vec = [1/acc] + error_norm_vec
    error_norm_vec = np.array(error_norm_vec)
    print(error_norm_vec)

    models_error.append(error_norm_vec)

models_error = np.array(models_error)
np.save('./plots/perturbed_error_different_epsilon_' + suffix + '_noise_data.npy', models_error)