import numpy as np
import matplotlib.pyplot as plt

def change_median(bplot, colors):
    # Change medians color
    for patch, color in zip(bplot['medians'], colors):
        patch.set_color(color)

# Read the data
BASE_PATH = './' #V'./statistics'

noise_level = 0
epsilon = 0.01

suffix = str(noise_level).split(".")[-1]
epsilon_suffix = str(epsilon).split(".")[-1]

acc = np.load(f"{BASE_PATH}/accuracies_noise_{suffix}_epsilon_{epsilon_suffix}.npy")
stab = np.load(f"{BASE_PATH}/stabilities_noise_{suffix}_epsilon_{epsilon_suffix}.npy")

colors = ['b', 'r', 'g', 'k', 'purple']

# Set the limit of the axis
xmin, xmax = 0.074, 0.078
ymin, ymax = 0, 1

# Visualize the shape
print(f"Shape of Acc: {acc.shape}")
print(f"Shape of Stab: {stab.shape}")

# Boxplot Acc
plt.figure(figsize=[16,4])
bplot_acc = plt.boxplot(acc[1:, :], vert=False)
plt.grid()
plt.xlabel(r'$\hat{\eta}^{-1}$')
plt.yticks([1, 2, 3, 4, 5], ['NN', 'StNN', 'ReNN', 'StReNN', 'IS'])
# plt.set_xtickslabels(["NN", "StNN", "ReNN", "StReNN"])
# plt.title(f"Accuracy in delta_hat = {suffix}, delta = {epsilon_suffix}")
change_median(bplot_acc, colors)
plt.savefig(f"{BASE_PATH}/box_acc_noise_{suffix}_epsilon_{epsilon_suffix}.png", dpi=600)

# Boxplot Stab
plt.figure(figsize=[16,4])
bplot_stab = plt.boxplot(stab[1:, :], vert=False)
# plt.xscale('log')
plt.grid()
plt.xlabel(r'$\hat{C}^\epsilon_\Psi$')
plt.yticks([1, 2, 3, 4, 5], ['NN', 'StNN', 'ReNN', 'StReNN', 'IS'])
# plt.xticks([-1, 0, 1], [r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
# plt.set_xtickslabels(["NN", "StNN", "ReNN", "StReNN"])
plt.title(f"Stability in delta_hat = {suffix}, delta = {epsilon_suffix}")
change_median(bplot_stab, colors)
plt.savefig(f"{BASE_PATH}/box_stab_noise_{suffix}_epsilon_{epsilon_suffix}.png", dpi=600)

"""
# Points Acc
plt.figure()
plt.scatter(np.max(acc[:], axis=0), np.max(stab[1:], axis=0), c=colors, marker='o')
plt.plot(np.linspace(xmin, xmax, 20), np.ones((20, )), 'b--')
plt.xlabel(r'$\hat{\eta}^{-1}$')
plt.ylabel(r'$\hat{C}_\Psi^\epsilon$')
plt.title(r'$\delta = 0.01$')
plt.grid()
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.savefig(f"{BASE_PATH}/parallel_noise_{suffix}_epsilon_{epsilon_suffix}.png")
"""
# Tables
print(f"Accuracy: {np.max(acc[:], axis=0)}")
print(f"Stability: {np.max(stab[1:], axis=0)}")

