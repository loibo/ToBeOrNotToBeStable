import numpy as np
import matplotlib.pyplot as plt

# Read data
error_data = np.load('./perturbed_error_different_epsilon_0.npy')
#error_data = np.load('./plots/perturbed_error_different_epsilon_025_noise_data.npy')
epsilon_vec = np.linspace(0.01, 0.1, 10)
epsilon_vec = np.insert(epsilon_vec, 0, 0)

# Plot specifications
colors = ['b', 'r', 'g', 'k', 'purple']
#colors = ['g', 'k', 'purple'] 

for i in range(error_data.shape[0]):
    #plt.semilogy(epsilon_vec, error_data[i, :], c=colors[i])
    plt.plot(epsilon_vec, error_data[i, :], c=colors[i])
# plt.legend(['ReNN', 'StReNN', 'IS'])
plt.legend(['NN', 'StNN', 'ReNN', 'StReNN', 'IS'])
plt.grid()
plt.ylim([8, 18])
plt.xlabel(r'$\delta$')
plt.ylabel(r'$||\Psi(Ax^{gt} + e) - x^{gt}||$')
plt.savefig('./plots/error_over_delta_025_noise_data.png', bbox_inches="tight", dpi=400)