# Import libraries
import matplotlib.pyplot as plt
import numpy as np


## ----------------------------------------------------------------------------------------------
## ----------   Functions    --------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
def plot_error_curves(noise_level, em, eM):
    suffix = str(noise_level).split(".")[-1]
    basename = f"unet_{suffix}_error_{em}_to_{eM}.npy"
    model_name_list = ["NN", "ReNN", "StNN", "StReNN", "IS"]
    color_list = ["b", "g", "r", "orange", "purple"]

    plt.figure()
    for i, model_name in enumerate(model_name_list):
        error_array = np.load(f"./plots/{model_name.lower()}_{basename}")[:, :, 0].mean(
            axis=0
        )
        train_noise_idx = np.where(np.linspace(em, eM, len(error_array)) == noise_level)

        # Add error to the plot
        plt.plot(
            np.linspace(em, eM, len(error_array)),
            error_array,
            ".-",
            c=color_list[i],
            markersize=8,
            label=model_name
        )
        plt.plot(noise_level, error_array[train_noise_idx], "d", c=color_list[i], label=f"_{model_name}")
    plt.xlabel(r"$\delta$", fontsize=15)
    plt.ylabel(r"$\| \| \Psi(Ax + e) - x \| \|$", fontsize=12)
    plt.grid()
    plt.legend(fontsize=12)
    plt.show()

def boxplot_accuracy_stability(noise_level, epsilon):
    suffix = str(noise_level).split(".")[-1]
    model_name_list = ["NN", "ReNN", "StNN", "StReNN", "IS"]
    color_list = ["b", "g", "r", "orange", "purple"]

    # Load accuracies vector
    accuracies_fname = f"./plots/accuracies_noise_{suffix}_epsilon_{epsilon}.npy"
    acc_vec = np.load(accuracies_fname)
    
    # Load stabilities vector
    stabilities_fname = f"./plots/stabilities_noise_{suffix}_epsilon_{epsilon}.npy"
    stab_vec = np.load(stabilities_fname)

    bplot_acc = plt.boxplot(acc_vec, vert=False)
    plt.grid()
    plt.xlabel(r'$\hat{\eta}^{-1}$', fontsize=12)
    plt.yticks(np.arange(1, len(model_name_list)+1), model_name_list, fontsize=12)
    # TODO: change_median(bplot_acc, colors)
    plt.show()

## ----------------------------------------------------------------------------------------------
## ----------   Experiment A    -----------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
# plot_error_curves(0, 0.0, 0.03)
boxplot_accuracy_stability(0, 0.01)


## ----------------------------------------------------------------------------------------------
## ----------   Experiment V    -----------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
# plot_error_curves(0.025, 0.0, 0.1)