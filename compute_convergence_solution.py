import numpy as np
import os

from IPPy import operators
from IPPy import solvers
from IPPy.utils import *
from IPPy import stabilizers
from IPPy.metrics import *

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
k_size = 11
sigma = 1.3
kernel = get_gaussian_kernel(k_size, sigma)

noise_level = 0.025
suffix = str(noise_level).split('.')[-1]

## ----------------------------------------------------------------------------------------------
## ---------- Setup Problem  --------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
K = operators.ConvolutionOperator(kernel, (m, n))
param_reg = 1e-2
Psi_k = stabilizers.Tik_CGLS_stabilizer(kernel, param_reg, k=100)

convergence_train_data = np.zeros_like(train_data)

## ----------------------------------------------------------------------------------------------
## ---------- Compute solutions -----------------------------------------------------------------
## ----------------------------------------------------------------------------------------------

# Train data
import time
start_time = time.time()
for i in range(len(train_data)):
    if i % 100 == 0:
        print(f"Done {i+1} image(s). Time passed: {time.time() - start_time}s")

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