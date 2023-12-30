# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks

from skimage.metrics import structural_similarity as ssim

from PIL import Image
import os

from IPPy.metrics import *
from IPPy.utils import *
from IPPy.nn import models as NN_models
from IPPy import stabilizers
from IPPy import operators
from IPPy import reconstructors

import argparse
import yaml
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path",
                    help="Path to the image you want to process. If an int is given, then the corresponding test image will be processed.",
                    required=True)
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
TEST_PATH = os.path.join(DATA_PATH, 'GOPRO_train_small.npy')

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
y_delta = K @ x_gt + (noise_level + epsilon) * np.random.normal(0, 1, m*n)
y_delta = np.reshape(y_delta, (m, n))

# Visualize
save_output = True
if save_output:
    plt.imsave('./images/corr_image.png', y_delta, cmap='gray')
    plt.imsave('./images/gt_image.png', x_gt, cmap='gray')

# Utils
use_convergence = False

# Load model.
# Model name -> Choose in {nn, stnn, renn, strenn}
model_name_list = args.model

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
    
    print(use_convergence)
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
    x_rec = Psi(y_delta)

    # Metrics
    RE_list.append(rel_err(x_gt, x_rec))
    SSIM_list.append(ssim(x_gt, x_rec,data_range=1))
    PSNR_list.append(PSNR(x_gt, x_rec))

    # Save reconstruction
    if save_output:
        plt.imsave(f"./images/recon_{model_name}_{suffix}_eps_{str(epsilon)}.png", x_rec, cmap='gray')

# Print out the metricss
import tabulate
errors = [["Corrupted", rel_err(x_gt, y_delta), PSNR(x_gt, y_delta), ssim(x_gt, y_delta,data_range=1)]]
for i in range(len(model_name_list)):
    errors.append([model_name_list[i].capitalize(), RE_list[i], PSNR_list[i], SSIM_list[i]])

print(tabulate.tabulate(errors, headers=[f"\u03B5 = {epsilon}", 'Rel. Err.', 'PSNR', 'SSIM']))