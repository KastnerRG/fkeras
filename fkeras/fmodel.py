import random
import numpy as np
import tensorflow.compat.v2 as tf

# TODO: Create Keras model wrapper to set up model level faults??

SUPPORTED_LAYERS = ["FQDense", "FQConv2D"]  # TODO: Get list from fkeras itself?
# Though not all layers have BERs...


class FModel:
    def __init__(self, model, model_param_ber=0):
        self.model = model
        self.model_param_ber = model_param_ber
        self._set_model_param_ber() 

    def set_model_param_ber(self, ber):
        self.model_param_ber = ber
        self._set_model_param_ber() 

    def _set_model_param_ber(self):
        # Set layer BERs based on the model BER
        self.random_select_model_param_bitflip()

    def random_select_model_param_bitflip(self):
        """
        Randomly select which bit(s) to flip in the model's parameters at the
        model's BER. This function will choose the appropriate layer BERs needed
        to carry out the specified model BER randomly.

        If you want to run multiple trials of the same model BER, call this
        function for each trial. TODO: In future, use LBI regions to specify
        start and end LBI indices so that this function can be called
        'random_select_param_bitflip()' or something. 

        The number of faults to be injected into the model's parameters = ber *
        num_model_param_bits.
        """
        # for layer in model.layers:
        #     TODO: Count number of model parameter bits
        #     Number bit flips = model_ber * num_model_param_bits
        #     Randomly choice/select from total model parameter bits
        #     Set layer BERs accordingly
        num_model_param_bits = 0
        num_bits_per_layer = {}
        # TODO: Randomly select bit index from total model parameter bits
        # E.g., set layer1's BER accordingly if bit_idx in layer1's bit range
        """
        |   Layer  |   Layer2  |    etc.    |
        | (0 - 20) | (21 - 50) | (51 - ...) |
        """
        for layer in self.model.layers:
            # Check if layer is (supported) FQKeras layer
            if (
                hasattr(layer, "quantizers")
                and layer.__class__.__name__ in SUPPORTED_LAYERS
            ):
                quant_config = layer.kernel_quantizer_internal.get_config()
                num_param_bits = quant_config["bits"] * np.array(layer.kernel).size
                num_model_param_bits += num_param_bits
                num_bits_per_layer[layer.name] = num_param_bits
            # TODO: Need separate branch for when SUPPORTED_LAYERS includes
            # regular float Keras layers, i.e., shouldn't check if layer has any
            # quantizers.
            else:
                # Float Keras or unsupported QKeras layer
                raise NotImplementedError(
                    f"Injecting faults in {layer.__class__.__name__} layer not yet supported."
                )
