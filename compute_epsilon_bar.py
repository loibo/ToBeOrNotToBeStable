# Import libraries
import numpy as np
import os
from tensorflow import keras as ks

import time

from IPPy.utils import *
from IPPy import operators
from IPPy.metrics import *
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
noise_level = 0

if noise_level == 0:
    suffix = '0'
elif noise_level == 0.025:
    suffix = '025'

## ----------------------------------------------------------------------------------------------
## ---------- Utility  --------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
def compute_stability(Psi, epsilon):
    # Create corrupted dataset
    K = operators.ConvolutionOperator(kernel, (m, n))
    corr_data = np.zeros_like(test_data)
    epsilon_corr_data = np.zeros_like(test_data)

    print("Generating Corrupted Data...")
    start_time = time.time()
    for i in range(len(test_data)):
        y_delta = K @ test_data[i] + noise_level * np.random.normal(0, 1, m*n)
        y_delta = np.reshape(y_delta, (m, n))
        
        corr_data[i] = y_delta

        y_delta = K @ test_data[i] + (noise_level + epsilon) * np.random.normal(0, 1, m*n)
        y_delta = np.reshape(y_delta, (m, n))
        
        epsilon_corr_data[i] = y_delta

    print(f"...Done! (in {time.time() - start_time}s)")

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

    return C_epsilon

def bisection(Psi, a, b, tol, kmax):
    def f(epsilon):
        return compute_stability(Psi, epsilon) - 1
    fa = f(a)
    fb = f(b)

    print(f"f(a) = {fa}, f(b) = {fb}")
    for i in range(kmax):
        c = (a+b)/2
        fc = f(c)

        print(f"f(c) = {fc}, c = {c}")

        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        if np.abs(fa) < tol:
            return a
        elif np.abs(fb) < tol:
            return b

## ----------------------------------------------------------------------------------------------
## ---------- Compute it  -----------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
model = ks.models.load_model(f"./model_weights/nn_unet_0.h5", custom_objects={'SSIM': SSIM})
phi = stabilizers.PhiIdentity()

Psi = reconstructors.StabilizedReconstructor(model, phi)

#epsilon_bar = bisection(Psi, 1e-5, 1e-7, 1e-3, 50)
# print(f"Epsilon_bar = {epsilon_bar}")
epsilon = 1e-4
print(compute_stability(Psi, epsilon))
