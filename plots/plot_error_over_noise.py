import numpy as np
import matplotlib.pyplot as plt

# Read data
error_data = np.load('./plots/perturbed_error_different_epsilon_025_noise_data.npy')
epsilon_vec = np.linspace(0.01, 0.1, 10)
epsilon_vec = np.insert(epsilon_vec, 0, 0)

# Plot specifications
colors = ['g', 'k', 'purple'] #['b', 'r', 'g', 'k', 'purple']

for i in range(error_data.shape[0]):
    plt.semilogy(epsilon_vec, error_data[i, :], c=colors[i])
plt.legend(['ReNN', 'StReNN', 'IS']) # (['NN', 'StNN', 'ReNN', 'StReNN', 'IS'])
plt.grid()
plt.ylim([13, 35])
plt.xlabel(r'$\delta$')
plt.ylabel(r'$||\Psi(Ax^{gt} + e) - x^{gt}||$')
plt.savefig('./plots/error_over_delta_025_noise_data.png', bbox_inches="tight", dpi=400)