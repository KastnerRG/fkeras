"""
Export modules
"""

from .hessian import *


assert tf.executing_eagerly(), "FKeras requires TF with eager execution mode on"
