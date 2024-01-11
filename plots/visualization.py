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

        # Add error to the plot
        plt.plot(
            np.linspace(em, eM, len(error_array)),
            error_array,
            ".-",
            c=color_list[i],
            markersize=8,
        )
        plt.plot(noise_level, error_array[5], "d", c=color_list[i])
    plt.xlabel(r"$\delta$", fontsize=15)
    plt.ylabel(r"$\| \Psi(Ax + e) - x \|$", fontsize=12)
    plt.grid()
    plt.legend(model_name_list, fontsize=12)
    plt.show()


plot_error_curves(0.025, 0.0, 0.1)
