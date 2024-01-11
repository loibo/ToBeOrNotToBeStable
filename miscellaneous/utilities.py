import argparse
import os

import numpy as np
import tensorflow as tf
import yaml
from tensorflow import keras as ks

from IPPy import operators, reconstructors, stabilizers
from IPPy.metrics import *


def initialization():
    # Disable TensorFlow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Name of the model to process. Can be used for multiple models to compare them.",
        required=True,
        action="append",
        choices=["nn", "renn", "stnn", "strenn", "is"],
    )
    stabilization = parser.add_mutually_exclusive_group(required=True)
    stabilization.add_argument(
        "-ni",
        "--noise_inj",
        help="The amount of noise injection. Given as the variance of the Gaussian.",
        type=float,
        required=False,
    )
    stabilization.add_argument(
        "-nl",
        "--noise_level",
        help="The amount of noise level added to the input datum. Given as the variance of the Gaussian.",
        type=float,
        required=False,
    )
    parser.add_argument(
        "-nt",
        "--noise_type",
        help="Type of noise added to the input at training phace. Default: gaussian.",
        type=str,
        default="gaussian",
        required=False,
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        help="Noise level of additional corruption. Given as gaussian variance. Default: 0.",
        type=float,
        required=False,
        default=0,
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Path to the image you want to process. If an int is given, then the corresponding test image will be processed.",
        required=True,
    )
    parser.add_argument(
        "--config",
        help="The path for the .yml containing the configuration for the model.",
        type=str,
        required=False,
        default=None,
    )
    args = parser.parse_args()

    if args.config is None:
        noise_level = args.noise_inj if args.noise_inj is not None else args.noise_level
        suffix = str(noise_level).split(".")[-1]
        args.config = f"./config/GoPro_{suffix}.yml"

    with open(args.config, "r") as file:
        setup = yaml.safe_load(file)

    return args, setup


def get_reconstructor(model_name, kernel, args, setup):
    # Utils
    use_convergence = False

    match model_name:
        case "nn":
            weights_name = "nn_unet"
            phi = stabilizers.PhiIdentity()
        case "stnn":
            weights_name = "stnn_unet"
            reg_param = setup[model_name]["reg_param"]
            phi = stabilizers.Tik_CGLS_stabilizer(
                kernel, reg_param, k=setup[model_name]["n_iter"]
            )
        case "renn":
            weights_name = "renn_unet"
            phi = stabilizers.PhiIdentity()
        case "strenn":
            weights_name = "strenn_unet"
            reg_param = setup[model_name]["reg_param"]
            phi = stabilizers.Tik_CGLS_stabilizer(
                kernel, reg_param, k=setup[model_name]["n_iter"]
            )
        case "is":
            use_convergence = True
            param_reg = setup[model_name]["reg_param"]
            algorithm = stabilizers.Tik_CGLS_stabilizer(
                kernel, param_reg, k=setup[model_name]["n_iter"]
            )

    if use_convergence:
        Psi = reconstructors.VariationalReconstructor(algorithm)
    else:
        noise_level = args.noise_inj if args.noise_inj is not None else args.noise_level
        suffix = str(noise_level).split(".")[-1]

        if args.noise_level is not None:
            model = ks.models.load_model(
                f"./model_weights/{weights_name}_{suffix}.h5",
                custom_objects={"SSIM": SSIM},
            )
        elif args.noise_inj is not None:
            model = ks.models.load_model(
                f"./model_weights/{weights_name}_{suffix}_NI.h5",
                custom_objects={"SSIM": SSIM},
            )

        # Define reconstructor
        Psi = reconstructors.StabilizedReconstructor(model, phi)
        return Psi
