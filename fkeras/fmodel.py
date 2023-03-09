import random
import numpy as np
import tensorflow.compat.v2 as tf
from collections import OrderedDict, defaultdict
import fkeras as fk


SUPPORTED_LAYERS = ["FQDense", "FQConv2D"]  # TODO: Get list from fkeras itself?
# Though not all layers have BERs...

# Layers that have no parameters to bit flip
NON_PARAM_LAYERS = ["InputLayer", "QActivation", "Flatten"]


class FModel:
    def __init__(self, model, model_param_ber=0):
        self.model = model
        self.model_param_ber = model_param_ber
        self._set_layer_bit_ranges()
        self._set_model_param_ber()
        # self.layer_bit_ranges = {}
        # self.num_model_param_bits = 0
        # TODO: Set layer_bit_ranges and num_model_param_bits

    def set_model_param_ber(self, ber):
        self.model_param_ber = ber
        self._set_model_param_ber()

    def _set_model_param_ber(self):
        # Set layer BERs based on the model BER
        self.random_select_model_param_bitflip()

    def _set_layer_bit_ranges(self):
        self.num_model_param_bits = 0
        self.layer_bit_ranges = OrderedDict()
        self.num_bits_per_layer = defaultdict(int)
        # self.bits_to_flip_per_layer = defaultdict(int)
        # TODO: Randomly select bit index from total model parameter bits
        # E.g., set layer1's BER accordingly if bit_idx in layer1's bit range
        """
        |   Layer  |   Layer2  |    etc.    |
        |  [0, 20) |  [20, 50) | [50 - ...) |
        """
        for layer in self.model.layers:
            # Check if layer is (supported) FQKeras layer
            if (
                hasattr(layer, "quantizers")
                and layer.__class__.__name__ in SUPPORTED_LAYERS
            ):
                quant_config = layer.kernel_quantizer_internal.get_config()
                num_param_bits = quant_config["bits"] * np.array(layer.kernel).size
                self.num_bits_per_layer[layer.name] = num_param_bits
                start_idx = self.num_model_param_bits
                end_idx = self.num_model_param_bits + num_param_bits
                self.layer_bit_ranges[(start_idx, end_idx)] = layer
                self.num_model_param_bits += num_param_bits
            # TODO: Need separate branch for when SUPPORTED_LAYERS includes
            # regular float Keras layers, i.e., shouldn't check if layer has any
            # quantizers.
            elif layer.__class__.__name__ in NON_PARAM_LAYERS:
                # Skip layers that have no parameters
                continue
            else:
                # Float Keras or unsupported QKeras layer
                raise NotImplementedError(
                    f"Injecting faults in {layer.__class__.__name__} layer not yet supported."
                )
        print(f"[fkeras.fmodel._set_layer_bit_ranges] {self.num_model_param_bits}")

    def explicit_select_model_param_bitflip(self, bits_to_flip):
        """
        Explicitly select which bits to flip in the model's parameters at the
        model's BER. This function will choose the appropriate layer BERs needed
        to carry out the specified bit flips deterministically.

        If you want to run multiple trials of the same faulty model
        (defined by the flipped bits), call this function for each trial.

        The number of faults to be injected into the model's parameters = len(bits_to_flip).
        """
        bits_to_flip_per_layer = defaultdict(int)
        # num_faults = int(self.num_model_param_bits * self.model_param_ber)
        print(
            f"[fkeras.fmodel.explicit_select_model_param_bitflip] num_faults = {len(bits_to_flip)}"
        )
        # bits_to_flip = random.sample(list(range(self.num_model_param_bits)), num_faults)
        bits_to_flip.sort()
        # print(f"[fkeras.fmodel.explicit_select_model_param_bitflip] {bits_to_flip}")
        for bit in bits_to_flip:
            # print(f"[fkeras.fmodel.explicit_select_model_param_bitflip] {bit}")
            # print(f"[fkeras.fmodel.explicit_select_model_param_bitflip] self.layer_bit_ranges = {self.layer_bit_ranges}")
            for r in self.layer_bit_ranges.keys():
                # print(f"[fkeras.fmodel.explicit_select_model_param_bitflip] {bit} {r}")
                if bit >= r[0] and bit < r[1]:  # If bit within range
                    layer = self.layer_bit_ranges[r]
                    bits_to_flip_per_layer[layer.name] += 1
                    # print(f"[fkeras.fmodel.explicit_select_model_param_bitflip] bits_to_flip_per_layer = {bits_to_flip_per_layer[layer.name]}")
                    layer.flbrs = [ fk.utils.FaultyLayerBitRegion(bit-r[0], bit-r[0], 1.0) ]

        for layer in self.model.layers:
            if layer.__class__.__name__ in SUPPORTED_LAYERS:
                layer.set_ber(
                    bits_to_flip_per_layer[layer.name]
                    / self.num_bits_per_layer[layer.name]
                )

    def random_select_model_param_bitflip(self):
        """
        Randomly select which bit to flip in the model's parameters at the
        model's BER. This function will choose the appropriate layer BERs needed
        to carry out the specified model BER randomly.

        If you want to run multiple trials of the same model BER, call this
        function for each trial. TODO: In future, use LBI regions to specify
        start and end LBI indices so that this function can be called
        'random_select_param_bitflip()' or something.

        The number of faults to be injected into the model's parameters = ber *
        num_model_param_bits.
        """
        bits_to_flip_per_layer = defaultdict(int)
        num_faults = int(self.num_model_param_bits * self.model_param_ber)
        print(f"num_faults = {num_faults}")
        bits_to_flip = random.sample(list(range(self.num_model_param_bits)), num_faults)
        bits_to_flip.sort()
        for bit in bits_to_flip:
            for r in self.layer_bit_ranges.keys():
                if bit >= r[0] and bit < r[1]:  # If bit within range
                    bits_to_flip_per_layer[layer.name] += 1

        for layer in self.model.layers:
            if layer.__class__.__name__ in SUPPORTED_LAYERS:
                layer.set_ber(
                    bits_to_flip_per_layer[layer.name]
                    / self.num_bits_per_layer[layer.name]
                )
