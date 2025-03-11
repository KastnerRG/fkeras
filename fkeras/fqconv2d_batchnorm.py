from qkeras.qconv2d_batchnorm import QConv2DBatchnorm
import fkeras as fk
from fkeras.utils import gen_lbi_region_at_layer_level
import tensorflow.compat.v2 as tf
from .quantizers import *
from tensorflow.python.framework import smart_cond as tf_utils
from tensorflow.python.ops import math_ops

assert tf.executing_eagerly(), "QKeras requires TF with eager execution mode on"


class FQConv2DBatchnorm(QConv2DBatchnorm):
    """
    NOTE: Not fully implemented yet. Note TODO in call() function.

    Implements a fault QConv2DBatchnorm layer

    Parameters:
    * ber (float): Bit Error Rate, or how often you want a fault to occur
    * bit_loc (list of tuples): Target ranges for the bit errors, e.g., (0, 3)
      targets bits at index 0 through 3, where 0 is the LSB.

    Please refer to the documentation of QConv2DBatchnorm in QKeras for the
    other parameters.
    """

    def __init__(
        self, filters, kernel_size, ber=0.0, bit_loc=0, accum_faults=False, **kwargs
    ):
        self.ber = ber
        self.bit_loc = bit_loc
        self.filters = filters
        self.kernel_size = kernel_size
        self.accum_faults = accum_faults

        self.flbrs = list()

        super(FQConv2DBatchnorm, self).__init__(
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

    def call(self, inputs, training=None):

        if self.ber == 0:  # For speed
            return super().call(inputs, training=training)

        # numpy value, mark the layer is in training
        training = self.batchnorm._get_training_value(
            training
        )  # pylint: disable=protected-access

        # checking if to update batchnorm params
        if self.ema_freeze_delay is None or self.ema_freeze_delay < 0:
            # if ema_freeze_delay is None or a negative value, do not freeze bn stats
            bn_training = tf.cast(training, dtype=bool)
        else:
            bn_training = tf.math.logical_and(
                training, tf.math.less_equal(self._iteration, self.ema_freeze_delay)
            )

        kernel = self.kernel
        # NOTE: These conv2d + batchnorm calls are for training
        # (and computing the gradient for backpropagation).
        # The actual inference is done with the folded weights
        # and biases at the bottom of this function.
        # TODO: Need to figure out how to do this in FKeras
        # because there are effectively two different kernels:
        # one for training (trainable) and one for inference (not trainable).

        # run conv to produce output for the following batchnorm
        conv_outputs = tf.keras.backend.conv2d(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        if self.use_bias:
            bias = self.bias
            conv_outputs = tf.keras.backend.bias_add(
                conv_outputs, bias, data_format=self.data_format
            )
        else:
            bias = 0

        _ = self.batchnorm(conv_outputs, training=bn_training)
        if training is True:
            # The following operation is only performed during training

            self._iteration.assign_add(
                tf_utils.smart_cond(
                    training,
                    lambda: tf.constant(1, tf.int64),
                    lambda: tf.constant(0, tf.int64),
                )
            )

            # calcuate mean and variance from current batch
            bn_shape = conv_outputs.shape
            ndims = len(bn_shape)
            reduction_axes = [i for i in range(ndims) if i not in self.batchnorm.axis]
            keep_dims = len(self.batchnorm.axis) > 1
            (
                mean,
                variance,
            ) = self.batchnorm._moments(  # pylint: disable=protected-access
                math_ops.cast(
                    conv_outputs, self.batchnorm._param_dtype
                ),  # pylint: disable=protected-access
                reduction_axes,
                keep_dims=keep_dims,
            )
            # get batchnorm weights
            gamma = self.batchnorm.gamma
            beta = self.batchnorm.beta
            moving_mean = self.batchnorm.moving_mean
            moving_variance = self.batchnorm.moving_variance

            if self.folding_mode == "batch_stats_folding":
                # using batch mean and variance in the initial training stage
                # after sufficient training, switch to moving mean and variance
                new_mean = tf_utils.smart_cond(
                    bn_training, lambda: mean, lambda: moving_mean
                )
                new_variance = tf_utils.smart_cond(
                    bn_training, lambda: variance, lambda: moving_variance
                )

                # get the inversion factor so that we replace division by multiplication
                inv = math_ops.rsqrt(new_variance + self.batchnorm.epsilon)
                if gamma is not None:
                    inv *= gamma
                # fold bias with bn stats
                folded_bias = inv * (bias - new_mean) + beta

            elif self.folding_mode == "ema_stats_folding":
                # We always scale the weights with a correction factor to the long term
                # statistics prior to quantization. This ensures that there is no jitter
                # in the quantized weights due to batch to batch variation. During the
                # initial phase of training, we undo the scaling of the weights so that
                # outputs are identical to regular batch normalization. We also modify
                # the bias terms correspondingly. After sufficient training, switch from
                # using batch statistics to long term moving averages for batch
                # normalization.

                # use batch stats for calcuating bias before bn freeze, and use moving
                # stats after bn freeze
                mv_inv = math_ops.rsqrt(moving_variance + self.batchnorm.epsilon)
                batch_inv = math_ops.rsqrt(variance + self.batchnorm.epsilon)

                if gamma is not None:
                    mv_inv *= gamma
                    batch_inv *= gamma
                folded_bias = tf_utils.smart_cond(
                    bn_training,
                    lambda: batch_inv * (bias - mean) + beta,
                    lambda: mv_inv * (bias - moving_mean) + beta,
                )
                # moving stats is always used to fold kernel in tflite; before bn freeze
                # an additional correction factor will be applied to the conv2d output
                inv = mv_inv
            else:
                assert ValueError

            # wrap conv kernel with bn parameters
            folded_kernel = inv * kernel
            # quantize the folded kernel
            if self.kernel_quantizer is not None:
                q_folded_kernel = self.kernel_quantizer_internal(folded_kernel)
            else:
                q_folded_kernel = folded_kernel

            # If loaded from a ckpt, bias_quantizer is the ckpt value
            # Else if bias_quantizer not specified, bias
            #   quantizer is None and we need to calculate bias quantizer
            #   type according to accumulator type. User can call
            #   bn_folding_utils.populate_bias_quantizer_from_accumulator(
            #      model, input_quantizer_list]) to populate such bias quantizer.
            if self.bias_quantizer_internal is not None:
                q_folded_bias = self.bias_quantizer_internal(folded_bias)
            else:
                q_folded_bias = folded_bias

            # set value for the folded weights
            self.folded_kernel_quantized.assign(q_folded_kernel, read_value=False)
            self.folded_bias_quantized.assign(q_folded_bias, read_value=False)

            applied_kernel = q_folded_kernel
            applied_bias = q_folded_bias
        else:
            applied_kernel = self.folded_kernel_quantized
            applied_bias = self.folded_bias_quantized

        # quantize and bit flip quantized folded kernel
        quant_config = self.kernel_quantizer_internal.get_config()
        faulty_layer_bit_region = gen_lbi_region_at_layer_level(
            self.folded_kernel_quantized,
            quant_config["bits"],
            self.ber,
        )[0]

        faulty_folded_kernel_quantized = fk.utils.quantize_and_bitflip_deterministic_v3(
            self.folded_kernel_quantized,
            self.kernel_quantizer_internal,
            self.flbrs,
            [faulty_layer_bit_region],
        )

        self.folded_kernel_quantized.assign(
            faulty_folded_kernel_quantized, read_value=False
        )

        # calculate conv2d output using the quantized folded kernel
        folded_outputs = tf.keras.backend.conv2d(
            inputs,
            self.folded_kernel_quantized,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if training is True and self.folding_mode == "ema_stats_folding":
            batch_inv = math_ops.rsqrt(variance + self.batchnorm.epsilon)
            y_corr = tf_utils.smart_cond(
                bn_training,
                lambda: (
                    math_ops.sqrt(moving_variance + self.batchnorm.epsilon)
                    * math_ops.rsqrt(variance + self.batchnorm.epsilon)
                ),
                lambda: tf.constant(1.0, shape=moving_variance.shape),
            )
            folded_outputs = math_ops.mul(folded_outputs, y_corr)

        folded_outputs = tf.keras.backend.bias_add(
            folded_outputs, applied_bias, data_format=self.data_format
        )
        if self.activation is not None:
            return self.activation(folded_outputs)

        return folded_outputs
