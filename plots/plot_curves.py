import matplotlib.pyplot as plt
import numpy as np

labels = ["NN", "StNN", "ReNN", "StReNN", "IS"]
model_names = ["nn", "stnn", "renn", "strenn", "is"]
colors = ["b", "r", "g", "k", "purple"]
accuracies = [1 / 0.1287, 1 / 0.1029, 1 / 0.1296, 1 / 0.0980, 1 / 0.0940]

# Choose plot type, in {stability, perturbed_error}
plot_type = "perturbed_error"

if plot_type == "perturbed_error":
    ylabel = r"$|| \Psi(Ax + e) - x ||$"
elif plot_type == "stability":
    ylabel = r"$C^\epsilon_\Psi$"

# Noise
noise_level = 0

# Create epsilon vector
epsilon_min = 1e-4
epsilon_max = 0.02
epsilon_n = 10

epsilon_vec = np.linspace(epsilon_min, epsilon_max, epsilon_n)
epsilon_vec = np.concatenate(
    [
        np.array(
            0,
        ),
        epsilon_vec,
    ]
)

if noise_level == 0:
    suffix = "0"
elif noise_level == 0.025:
    suffix = "025"

for i, model_name in enumerate(model_names):
    stab_const = np.load(
        "./plots/"
        + model_name
        + "_"
        + suffix
        + "_"
        + plot_type
        + "_different_epsilon.npy"
    )
    plt.semilogy(epsilon_vec, stab_const, c=colors[i], linewidth=1)
plt.legend(labels)
plt.xlabel(r"$\delta$")
plt.ylabel(ylabel)
plt.grid()

# Save the figure
plt.savefig("./plots/" + plot_type + "_over_epsilon_noise_" + suffix + ".png", dpi=500)
