import numpy as np
import matplotlib.pyplot as plt

labels = ['NN', 'StNN', 'ReNN', 'StReNN'] #, 'IS']
accuracies = [0.0764895, 0.0710975, 0.0761493, 0.0705434]
stabilities = [0.963241, 0.360895, 0.951613, 0.384699]
stabilities05 = [0.771942, 0.240862, 0.767473, 0.249572]
colors = ['b', 'r', 'g', 'k'] #, 'purple']

# Set the limit of the axis
xmin, xmax = 0.05, 0.08 # 0.07, 0.14
ymin, ymax = 0, 1# -1, 39

# Draw points
plt.scatter(accuracies, stabilities, c=colors, marker='o')
plt.scatter(accuracies, stabilities05, c=colors, marker='s')
# plt.scatter(accuracies[4:], stabilities[4:], c=colors[4:], marker='s')

# Draw horizontal line at 1
plt.hlines(1, xmin, xmax, linestyles='dashed')

# Other specifications
plt.xlabel(r'$\hat{\eta}^{-1}$')
plt.ylabel(r'$\hat{C}_\Psi^\epsilon$')
plt.title(r'$\delta = 0.075 \to \delta = 0.105$')
plt.grid()
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])

# Save the figure
plt.savefig("./plots/parallelogram_delta_05_08_noise_025.png", dpi=800)