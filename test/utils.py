import numpy as np
import tensorflow as tf


def equal_tensors(a, b):
    """
    Given tensor a and tensor b, return true if all values match element-wise
    """
    return np.all(tf.math.equal(a, b).numpy().flatten())
