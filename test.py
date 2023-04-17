# Import libraries
import numpy as np
import os

# Noise
noise_level = 0.025 # in {0, 0.025}
epsilon = 0.05 # if noise_level = 0 -> epsilon = 0.01
               # if noise_level = 0.025 -> epsilon = 0.05 or 0.08
               
epsilon_suffix = str(epsilon).split(".")[-1]
if noise_level == 0:
    suffix = '0'
elif noise_level == 0.025:
    suffix = '025'

pop_accuracies = np.load(f"./statistics/accuracies_noise_{suffix}_epsilon_{epsilon_suffix}.npy")
pop_stability_constants = np.load(f"./statistics/stabilities_noise_{suffix}_epsilon_{epsilon_suffix}.npy")

pop_accuracies_old = np.load(f"./statistics/accuracies_noise_{suffix}_epsilon_{epsilon_suffix}_old.npy")
pop_stability_constants_old = np.load(f"./statistics/stabilities_noise_{suffix}_epsilon_{epsilon_suffix}_old.npy")

pop_accuracies_new = np.zeros_like(pop_accuracies_old)
pop_stability_constants_new = np.zeros_like(pop_stability_constants_old)

pop_accuracies_new[:, :-1] = pop_accuracies
pop_accuracies_new[:, -1] = pop_accuracies_old[:, -1]

pop_stability_constants_new[:, :-1] = pop_stability_constants
pop_stability_constants_new[:, -1] = pop_stability_constants_old[:, -1]

np.save(f"./statistics/accuracies_noise_{suffix}_epsilon_{epsilon_suffix}.npy", pop_accuracies_new)
np.save(f"./statistics/stabilities_noise_{suffix}_epsilon_{epsilon_suffix}.npy", pop_stability_constants_new)






