from qkeras import QDense
from keras import backend
import fkeras as fk
import numpy as np
from fkeras.utils import gen_lbi_region_at_layer_level, quantize_and_bitflip
import tensorflow.compat.v2 as tf

assert tf.executing_eagerly(), "QKeras requires TF with eager execution mode on"


class FQDense(QDense):
    """
    Implements a faulty QDense layer

    Parameters:
    * ber (float): Bit Error Rate, or how often you want a fault to occur
    * bit_loc (list of tuples): Target ranges for the bit errors, e.g., (0, 3) targets bits at index 0 through 3, where 0 is the LSB.

    Please refer to the documentation of QDense in QKeras for the other
    parameters.
    """

    def __init__(self, units, ber=0.0, bit_loc=0, accum_faults=False, **kwargs):
        self.ber = ber
        self.units = units
        self.bit_loc = bit_loc
        self.flbrs = list()
        self.accum_faults = accum_faults

        super(FQDense, self).__init__(units=units, **kwargs)

        # self.og_kernel = None

    def set_ber(self, ber):
        self.ber = ber

    def get_ber(self):
        return self.ber

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "units": self.units,
                "ber": self.ber,
                "bit_loc": self.bit_loc,
            }
        )
        return config

    def call(self, inputs):
        return super().call(inputs)