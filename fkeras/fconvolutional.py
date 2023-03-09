from qkeras import QConv2D
from keras import backend
import fkeras as fk
from fkeras.utils import gen_lbi_region_at_layer_level, quantize_and_bitflip
import tensorflow.compat.v2 as tf

assert tf.executing_eagerly(), "QKeras requires TF with eager execution mode on"


class FQConv2D(QConv2D):
    """
    Implements a faulty QConv2D layer

    Parameters:
    * ber (float): Bit Error Rate, or how often you want a fault to occur
    * bit_loc (list of tuples): Target ranges for the bit errors, e.g., (0, 3) targets bits at index 0 through 3, where 0 is the LSB.

    Please refer to the documentation of QDense in QKeras for the other
    parameters.
    """

    def __init__(self, filters, kernel_size, ber=0.0, bit_loc=0, **kwargs):
        self.ber = ber
        self.bit_loc = bit_loc
        self.filters = filters
        self.kernel_size = kernel_size
        self.flbrs = list()

        super(FQConv2D, self).__init__(
            filters=filters, kernel_size=kernel_size, **kwargs
        )

    def set_ber(self, ber):
        self.ber = ber

    def get_ber(self):
        return self.ber

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "ber": self.ber,
                "bit_loc": self.bit_loc,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
            }
        )
        return config

    def call(self, inputs):
        if self.ber == 0:  # For speed
            return super().call(inputs)

        quant_config = self.kernel_quantizer_internal.get_config()
        faulty_layer_bit_region = gen_lbi_region_at_layer_level(
            self.kernel,
            quant_config["bits"],
            self.ber,
        )[0]

        faulty_qkernel = fk.utils.quantize_and_bitflip_deterministic_v3(
            self.kernel,
            self.kernel_quantizer_internal,
            self.flbrs, #[(faulty_layer_bit_region.start_lbi, faulty_layer_bit_region.end_lbi)],
            [faulty_layer_bit_region.ber],
        )

        # Useful for debugging purposes
        # tf.print("\n\n###########")
        # tf.print("self.kernel")
        # tf.print(self.kernel)
        # tf.print("Qkernel")
        # tf.print(self.kernel_quantizer_internal(self.kernel))
        # tf.print("Faulty Qkernal")
        # tf.print(faulty_qkernel)
        # tf.print("###########\n\n")
        # qkernel = self.kernel_quantizer_internal(self.kernel)
        # equality_tensor = tf.math.equal(qkernel, faulty_qkernel)
        # tf.print("Equality tensor:")
        # tf.print(equality_tensor)
        # tf.print("Reduced CONV equality tensor:")
        # tf.print(tf.math.reduce_all(equality_tensor)) # Logical and across all elements of tensor

        outputs = tf.keras.backend.conv2d(
            inputs,
            faulty_qkernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        if self.use_bias:
            if self.bias_quantizer:
                quantized_bias = self.bias_quantizer_internal(self.bias)
            else:
                quantized_bias = self.bias

        outputs = tf.keras.backend.bias_add(
            outputs, quantized_bias, data_format=self.data_format
        )

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
