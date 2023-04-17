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

test_data = np.load(TEST_PATH)[10:11]
N_test, m, n = test_data.shape
print(f"Test data shape: {test_data.shape}")

# Define the setup for the forward problem
k_size = 11
sigma = 1.3
kernel = get_gaussian_kernel(k_size, sigma)

# Noise
epsilon = 0.01
noise_level = 0

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
model_name_list = ('nn', 'stnn', 'renn', 'strenn', 'is')

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
            param_reg = 1e-1 # 8e-2
            algorithm = stabilizers.Tik_CGLS_stabilizer(kernel, param_reg, k=100)

    if use_convergence:
        Psi = reconstructors.VariationalReconstructor(algorithm)
    else:
        model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}.h5", custom_objects={'SSIM': SSIM})

        # Define reconstructor
        Psi = reconstructors.StabilizedReconstructor(model, phi)

    # Reconstruct
    x_rec = Psi(corr_data)
    x_rec_epsilon = Psi(epsilon_corr_data)

    if x_rec.shape[-1] == 1:
        x_rec = x_rec[:, :, :, 0]

    if x_rec_epsilon.shape[-1] == 1:
        x_rec_epsilon = x_rec_epsilon[:, :, :, 0]

    # Save the results
    plt.imsave(f"./images/x_true.png", test_data[0, :, :], cmap='gray', dpi=400)
    plt.imsave(f"./images/y_true.png", corr_data[0, :, :], cmap='gray', dpi=400)
    plt.imsave(f"./images/y_corr.png", epsilon_corr_data[0, :, :], cmap='gray', dpi=400)
    plt.imsave(f"./images/rec_{model_name}_true.png", x_rec[0, :, :], cmap='gray', dpi=400)
    plt.imsave(f"./images/rec_{model_name}_corr.png", x_rec_epsilon[0, :, :], cmap='gray', dpi=400)