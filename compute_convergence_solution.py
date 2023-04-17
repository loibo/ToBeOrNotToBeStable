import numpy as np
import os

from IPPy import operators
from IPPy import solvers
from IPPy.utils import *
from IPPy import stabilizers
from IPPy.metrics import *

# Parse input
import argparse
import yaml
parser = argparse.ArgumentParser()
parser.add_argument('--test', 
                    action='store_true',
                    help="If used, process the test set. The  training set is processed otherwise.")
parser.add_argument('--noise_level',
                    help="The noise level injected into the y datum, given as gaussian variance. Default: 0.",
                    type=float,
                    default=0,
                    required=False)
parser.add_argument('--param_reg',
                    help="The regularization parameter for the inverse problem. Default: 1e-2.",
                    type=float,
                    default=1e-2,
                    required=False)
parser.add_argument('--n_iter',
                    help="The maximum number of iterations for the reconstruction algorithm. Default: 100",
                    default=100,
                    type=int,
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
param_reg = args.param_reg
Psi_k = stabilizers.Tik_CGLS_stabilizer(kernel, param_reg, k=args.n_iter)


## ----------------------------------------------------------------------------------------------
## ---------- Compute solutions -----------------------------------------------------------------
## ----------------------------------------------------------------------------------------------

if not args.test:
    # Train data
    import time
    start_time = time.time()
    convergence_train_data = np.zeros_like(train_data)
    for i in range(len(train_data)):
        if i % 100 == 0:
            print(f"Done {i+1} image(s). Time passed: {time.time() - start_time:0.4f}s")

        # Load ground truth
        x_gt = train_data[i]

        # Compute corrupted data
        y_delta = K @ x_gt + noise_level * np.random.normal(0, 1, m*n)

        # Compute convergence solution
        x_rec = Psi_k(y_delta.reshape((m, n)))

        # Append the result
        convergence_train_data[i] = x_rec

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