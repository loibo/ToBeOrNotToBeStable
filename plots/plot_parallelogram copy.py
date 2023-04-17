import numpy as np
import matplotlib.pyplot as plt

labels = ['NN', 'StNN', 'ReNN', 'StReNN', 'IS'] * 2
accuracies = [0.0735836, 0.0740386, 0.0626336, 0.062956, 0.0563983] * 2
stabilities = [0.705377, 0.548397, 0.137442, 0.105945, 1.24885, 0.934706, 0.804639, 0.240679, 0.177288, 1.51544]
colors = ['b', 'r', 'g', 'k', 'purple'] * 2

# Set the limit of the axis
xmin, xmax = 0.05, 0.08 # 0.09, 0.14
ymin, ymax = 0, 1.8 # -1, 39

# Draw points
plt.scatter(accuracies[:5], stabilities[:5], c=colors[:5], marker='o')
plt.scatter(accuracies[5:], stabilities[5:], c=colors[5:], marker='s')

# Draw horizontal line at 1
plt.hlines(1, xmin, xmax, linestyles='dashed')

# Annotate points
for i in range(len(accuracies)):
    if i != 3 and i != 8:
        plt.annotate(labels[i], (accuracies[i], stabilities[i]), xytext=(accuracies[i]+0.0005, stabilities[i]+0.005), color=colors[i])
    else:
        plt.annotate(labels[i], (accuracies[i], stabilities[i]), xytext=(accuracies[i]+0.0005, stabilities[i]-0.013), color=colors[i])

# Other specifications
plt.xlabel(r'$\eta$')
plt.ylabel(r'$C_\Psi^\epsilon$')
plt.title(r'$\epsilon = 0.05 \to 0.08$')
plt.grid()
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])

# Save the figuer
plt.savefig("./plots/parallelogram_epsilon_05_noise_025.png", dpi=500)