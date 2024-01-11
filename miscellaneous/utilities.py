import os

import numpy as np
import tensorflow as tf


def initialization():
    # Disable TensorFlow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
