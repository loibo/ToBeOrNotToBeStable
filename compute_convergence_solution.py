import numpy as np
import os

from IPPy import operators
from IPPy import solvers
from IPPy.utils import *
from IPPy import stabilizers
from IPPy.metrics import *

from skimage.metrics import structural_similarity as ssim

# Parse input
import argparse
import yaml
parser = argparse.ArgumentParser()
parser.add_argument('--test', 
                    action='store_true',
                    help="If used, process the test set. The  training set is processed otherwise.")
parser.add_argument('-nl', '--noise_level',
                    help="The noise level injected into the y datum, given as gaussian variance. Default: 0.",
                    type=float,
                    default=0,
                    required=False)
parser.add_argument('--config',
                    help="The path for the .yml containing the configuration for the model.",
                    type=str,
                    required=False,
                    default=None)
args = parser.parse_args()

if args.config is None:
    suffix = str(args.noise_level).split('.')[-1]
    args.config = f"./config/GoPro_{suffix}.yml"

with open(args.config, 'r') as file:
    setup = yaml.safe_load(file)

## ----------------------------------------------------------------------------------------------
## ---------- Initialization --------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------

# Load data
DATA_PATH = './data/'
TRAIN_PATH = os.path.join(DATA_PATH, 'GOPRO_train_small.npy')
TEST_PATH = os.path.join(DATA_PATH, 'GOPRO_test_small.npy')

train_data = np.load(TRAIN_PATH)
test_data = np.load(TEST_PATH)
N_train, m, n = train_data.shape
N_test, _, _ = test_data.shape
print(f"Training data shape: {train_data.shape}")

# Define the setup for the forward problem
k_size = setup['k']
sigma = setup['sigma']
kernel = get_gaussian_kernel(k_size, sigma)

noise_level = args.noise_level
suffix = str(noise_level).split('.')[-1]

## ----------------------------------------------------------------------------------------------
## ---------- Setup Problem  --------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
K = operators.ConvolutionOperator(kernel, (m, n))
Psi_k = stabilizers.Tik_CGLS_stabilizer(kernel, setup['is']['reg_param'], k=setup['is']['n_iter'])


## ----------------------------------------------------------------------------------------------
## ---------- Compute solutions -----------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
if not args.test:
    # Train data
    import time
    start_time = time.time()
    tot_ssim = 0
    convergence_train_data = np.zeros_like(train_data)
    for i in range(len(train_data)):
        if (i+1) % 100 == 0:
            print(f"Done {i+1} image(s). Time passed: {time.time() - start_time:0.4f}s. Avg. SSIM: {tot_ssim/(i+1)}.")

        # Load ground truth
        x_gt = train_data[i]

        # Compute corrupted data
        y_delta = K @ x_gt + noise_level * np.random.normal(0, 1, m*n)

        # Compute convergence solution
        x_rec = Psi_k(y_delta.reshape((m, n)))

        # Append the result
        convergence_train_data[i] = x_rec

        # Compute the SSIM
        tot_ssim = tot_ssim + ssim(x_gt, x_rec, data_range=1)

    # Saving
    np.save(f'./data/GOPRO_convergence_small_{suffix}.npy', convergence_train_data)

else:
    # Test data
    import time
    start_time = time.time()
    convergence_test_data = np.zeros_like(test_data)
    for i in range(len(test_data)):
        if i % 100 == 0:
            print(f"Done {i+1} image(s). Time passed: {time.time() - start_time}s")

        # Load ground truth
        x_gt = test_data[i]

        # Compute corrupted data
        y_delta = K @ x_gt + noise_level * np.random.normal(0, 1, m*n)

        # Compute convergence solution
        x_rec = Psi_k(y_delta.reshape((m, n)))

        # Append the result
        convergence_test_data[i] = x_rec

    # Saving
    np.save(f'./data/GOPRO_convergence_test_small_{suffix}.npy', convergence_test_data)