# Change current directory to the correct directory
import os
print(f"Current directory: {os.getcwd()}")

# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks

from skimage.metrics import structural_similarity as ssim

from IPPy.metrics import *
from IPPy.utils import *
from IPPy import NN_models
from IPPy import stabilizers
from IPPy import operators
from IPPy import reconstructors


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
k_size = 11
sigma = 1.3
kernel = get_gaussian_kernel(k_size, sigma)

# Noise
epsilon = 0.05
noise_level = 0.025 + epsilon


## ----------------------------------------------------------------------------------------------
## ---------- Model evaluation ------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
# Load a test image
idx = 30
x_gt = test_data[idx]

# Corrupt it
K = operators.ConvolutionOperator(kernel, (m, n))
y_delta = K @ x_gt + noise_level * np.random.normal(0, 1, m*n)
y_delta = np.reshape(y_delta, (m, n))

# Visualize
save_output = True
if save_output:
    plt.imsave('./results/corr_image.png', y_delta, cmap='gray')
    plt.imsave('./results/gt_image.png', x_gt, cmap='gray')

# Load model.
# Model name -> Choose in {nn, stnn, renn, strenn}
model_name_list = ('nn', 'stnn', 'renn', 'strenn')

RE_list = []
PSNR_list = []
SSIM_list = []
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

    model = ks.models.load_model(f"./model_weights/{weights_name}.h5", custom_objects={'SSIM': SSIM})

    # Define reconstructor
    Psi = reconstructors.StabilizedReconstructor(model, phi)

    # Reconstruct
    x_rec = Psi(y_delta)

    # Metrics
    RE_list.append(rel_err(x_gt, x_rec))
    SSIM_list.append(ssim(x_gt, x_rec))
    PSNR_list.append(PSNR(x_gt, x_rec))

    # Save reconstruction
    if save_output:
        plt.imsave(f"./results/recon_{model_name}.png", x_rec, cmap='gray')

# Print out the metricss
import tabulate
errors = [["Corrupted", rel_err(x_gt, y_delta), PSNR(x_gt, y_delta), ssim(x_gt, y_delta)]]
for i in range(len(model_name_list)):
    errors.append([model_name_list[i].capitalize(), RE_list[i], PSNR_list[i], SSIM_list[i]])

print(tabulate.tabulate(errors, headers=[f"\u03B5 = {epsilon}", 'Rel. Err.', 'PSNR', 'SSIM']))