from qkeras import QDense


class FQDense(QDense):
    """
    Implements a faulty QDense layer

    Parameters:
    * ber (float): Bit Error Rate, or how often you want a fault to occur
    * bit_loc (list of tuples): Target ranges for the bit errors, e.g., (0, 3) targets bits at index 0 through 3, where 0 is the LSB. 

    Please refer to the documentation of QDense in QKeras for the other
    parameters.
    """

    def __init__(self, ber, bit_loc, **kwargs):
        super().__init__(**kwargs)
        self.ber = ber
        self.bit_loc = bit_loc


    def call(self, inputs):
        # TODO: Induce bitflips at ber at the indicated bit_loc
        pass
