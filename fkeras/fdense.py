from qkeras import QDense
from keras import backend
import tensorflow.compat.v2 as tf


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


    # def call(self, inputs):
        # original_kernel = self.kernel

        # self.kernel = faulty_kernel

        # super_outputs = super().call(inputs)

        # self.kernel = original_kernel


        # super_outputs = super().call(inputs)
        # # TODO: Induce bitflips at ber at the indicated bit_loc
        # pass
        # #backend.learning_phase() (0 is Test | 1 is Train)
        # if backend.learning_phase() == 0:
            
        # return super_outputs

    def call(self, inputs):
        # if self.kernel_quantizer:
        #     quantized_kernel = self.kernel_quantizer_internal(self.kernel)
        # else:
        #     quantized_kernel = self.kernel
        
        faulty_qkernel = ?
        output = tf.keras.backend.dot(inputs, faulty_qkernel)
        if self.use_bias:
            if self.bias_quantizer:
                quantized_bias = self.bias_quantizer_internal(self.bias)
            else:
                quantized_bias = self.bias
            output = tf.keras.backend.bias_add(output, quantized_bias,
                                            data_format="channels_last")
        if self.activation is not None:
            output = self.activation(output)
        return output



