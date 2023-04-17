import numpy as np
import matplotlib.pyplot as plt
from skimage import data

from tensorflow import keras as ks

import IPPy.NN_models as NN_models
from IPPy import reconstructors
from IPPy.metrics import *
from IPPy.utils import *
from IPPy.operators import *
from IPPy.NN_utils import *
from IPPy import stabilizers


## ----------------------------------------------------------------------------------------------
## ---------- Initialization --------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------

# Load data
DATA_PATH = './data/'
TRAIN_PATH = os.path.join(DATA_PATH, 'GOPRO_train_small.npy')

train_data = np.load(TRAIN_PATH)
N_train, m, n = train_data.shape
print(f"Training data shape: {train_data.shape}")

# Define the setup for the forward problem
k_size = 11
sigma = 1.3

kernel_type = 'gaussian'
if kernel_type == 'gaussian':
    kernel = get_gaussian_kernel(k_size, sigma)
elif kernel_type == 'motion':
    kernel = get_motion_blur_kernel(k_size) 

noise_level = 0

if noise_level == 0:
    suffix = '0'
elif noise_level == 0.025:
    suffix = '025'
elif noise_level == 0.02:
    suffix = '02'

print(f"Suffix: {suffix}")

# Corrupt
x_true = train_data[700]
K = ConvolutionOperator(kernel, (m, n))
y = K @ x_true
y_delta = y.reshape((m, n)) + noise_level * np.random.normal(0, 1, (m, n))

# Save the results
plt.imsave(f'./results2/corr_{str(noise_level)[2:]}.png', y_delta.reshape((m, n)), cmap='gray', dpi=400)


# Reconstruct with algorithm
weights_name = 'is'
reg_param = 1e-2
phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=200)

x_is = phi(y_delta)

# Save the results
plt.imsave(f'./results2/{weights_name}_{suffix}_{kernel_type}_recon.png', x_is.reshape((m, n)), cmap='gray', dpi=400)

# Reconstruct with NN
weights_name = 'nn_unet'
phi = stabilizers.PhiIdentity()

model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}.h5", custom_objects={'SSIM': SSIM})
Psi = reconstructors.StabilizedReconstructor(model, phi)

x_nn = Psi(y_delta)

# Save the results
plt.imsave(f'./results2/{weights_name}_{suffix}_{kernel_type}_recon.png', x_nn, cmap='gray', dpi=400)

# Reconstruct with StNN
weights_name = 'stnn_unet'
param_reg = 1e-2
phi = stabilizers.Tik_CGLS_stabilizer(kernel, reg_param, k=3)

# Save the results
plt.imsave(f'./results2/{weights_name}_{suffix}_{kernel_type}_preprocess.png', phi(y_delta), cmap='gray', dpi=400)

model = ks.models.load_model(f"./model_weights/{weights_name}_{suffix}.h5", custom_objects={'SSIM': SSIM})
Psi = reconstructors.StabilizedReconstructor(model, phi)

x_stnn = Psi(y_delta)

# Save the results
plt.imsave(f'./results2/{weights_name}_{suffix}_{kernel_type}_recon.png', x_stnn, cmap='gray', dpi=400)

# Quanitative results
print(f"SSIM (start, is, nn, stnn): {SSIM(x_true, y_delta), SSIM(x_true, x_is), SSIM(x_true, x_nn), SSIM(x_true, x_stnn)}")
print(f"PSNR (start, is, nn, stnn): {PSNR(x_true, y_delta), PSNR(x_true, x_is), PSNR(x_true, x_nn), PSNR(x_true, x_stnn)}")

