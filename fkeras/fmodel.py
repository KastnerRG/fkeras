import random
import numpy as np
import tensorflow.compat.v2 as tf
from collections import OrderedDict, defaultdict
import fkeras as fk


SUPPORTED_LAYERS = ["FQDense", "FQConv2D"]  # TODO: Get list from fkeras itself?
# Though not all layers have BERs...

# Layers that have no parameters to bit flip
NON_PARAM_LAYERS = [
    "InputLayer",
    "QActivation",
    "Flatten",
    "BatchNormalization",
    "Activation",
    "Add",
]

class FModelAlt:
    def __init__(self, model, verbose=0, incl_biases=False):
        self.model = model

        # TODO: If support is added for only flipping bits in bias,
        # add option to not include weights
        self.incl_weights = True
        self.incl_biases = incl_biases
        self.flipped_bi_to_layer_and_flbri = {}
        self.quant_config_to_bin_float_dicts = dict()

        self.num_model_param_bits = 0
        self.layer_bit_ranges = OrderedDict()
        self.layer_name_to_num_total_bits_wb = defaultdict(list)
        self.layer_name_to_num_flipped_bits_wb = defaultdict(list)
        self.weight_bis = set()
        self.bias_bis = set()

        self.layer_name_to_lis = dict()

        self.__build_bin_to_float__()
        self.__set_layer_bit_ranges__()

    def __get_bin_float_dicts__(self,curr_quant):
        curr_quant_config = curr_quant.get_config()
        binstr_to_float = dict()
        float_to_binstr = dict()
        for qvi, qval in enumerate(curr_quant.range()):
            qval_binstr = bin(qvi)[2:].zfill(curr_quant_config["bits"])
            binstr_to_float[qval_binstr] = qval
            float_to_binstr[qval] = qval_binstr

        return binstr_to_float, float_to_binstr

    def __build_bin_to_float__(self):
        model_weights = self.model.get_weights()
        mwi = 0
        for li, layer in enumerate(self.model.layers):
            self.layer_name_to_lis[layer.name] = list()
            for w_or_b_arr_i in range(len(layer.get_weights())):
                self.layer_name_to_lis[layer.name] += [mwi]
                mwi += 1

            # Check if layer is (supported) QKeras layer
            if (
                hasattr(layer, "quantizers")
                and layer.__class__.__name__ in SUPPORTED_LAYERS
            ):
                mwi_k = self.layer_name_to_lis[layer.name][0]
                assert np.all(model_weights[mwi_k] == layer.kernel.numpy())
                
                if layer.kernel_quantizer_internal:
                    curr_quant = layer.kernel_quantizer_internal
                    curr_quant_config = curr_quant.get_config()
                    cqc_hashable = tuple([(k,v) for k,v in curr_quant_config.items()])
                    
                    if cqc_hashable not in self.quant_config_to_bin_float_dicts:
                        self.quant_config_to_bin_float_dicts[cqc_hashable] = self.__get_bin_float_dicts__(curr_quant)

                if layer.use_bias:
                    mwi_b = self.layer_name_to_lis[layer.name][1]
                    assert np.all(model_weights[mwi_b] == layer.bias.numpy())

                    if layer.bias_quantizer_internal:
                        curr_quant = layer.bias_quantizer_internal
                        curr_quant_config = curr_quant.get_config()
                        cqc_hashable = tuple([(k,v) for k,v in curr_quant_config.items()])

                        if cqc_hashable not in self.quant_config_to_bin_float_dicts:
                            self.quant_config_to_bin_float_dicts[cqc_hashable] = self.__get_bin_float_dicts__(curr_quant)


    def __set_layer_bit_ranges__(self):
        """
        |   Layer1  |   Layer2  |    etc.    |
        |  [0, 20)  |  [20, 50) | [50 - ...) |
        """
        for li, layer in enumerate(self.model.layers):
            # Check if layer is (supported) FQKeras layer
            if (
                hasattr(layer, "quantizers")
                and layer.__class__.__name__ in SUPPORTED_LAYERS
            ):
                num_param_bits = 0
                num_weight_bits = 0
                num_bias_bits = 0

                if self.incl_weights:
                    assert layer.quantizers[0] == layer.kernel_quantizer
                    num_weights = layer.kernel.numpy().size
                    num_weight_bits = layer.kernel_quantizer.get_config()["bits"] * num_weights
                    num_param_bits += num_weight_bits
                    for bi in range(self.num_model_param_bits,
                                    self.num_model_param_bits+num_weight_bits
                    ):
                        self.weight_bis.add(bi) 

                if layer.use_bias and self.incl_biases:
                    assert layer.quantizers[1] == layer.bias_quantizer
                    num_biases  = layer.bias.numpy().size
                    num_bias_bits = layer.bias_quantizer.get_config()["bits"] * num_biases
                    num_param_bits += num_bias_bits
                    for bi in range(self.num_model_param_bits+num_weight_bits,
                                    self.num_model_param_bits+num_weight_bits+num_bias_bits
                    ):
                        self.bias_bis.add(bi)
                
                self.layer_name_to_num_total_bits_wb[layer.name] = (num_weight_bits, num_bias_bits)
                self.layer_name_to_num_flipped_bits_wb[layer.name] = [0, 0]
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
    
    def explicitly_flip_bits(self, bits_to_flip):
        internal_bits_to_flip = sorted(bits_to_flip)

        actual_bits_to_flip = list()

        for bit in internal_bits_to_flip:

            if bit not in self.flipped_bi_to_layer_and_flbri:

                for r in self.layer_bit_ranges.keys():
                    if bit >= r[0] and bit < r[1]:  # If bit within range
                        layer = self.layer_bit_ranges[r]

                        assoc_lis = self.layer_name_to_lis[layer.name]
                        num_w_bits, num_b_bits = self.layer_name_to_num_total_bits_wb[layer.name]
                        num_ws = layer.kernel.numpy().size
                        num_bs = layer.bias.numpy().size
                        w_bitwidth = num_w_bits/num_ws
                        b_bitwidth = num_b_bits/num_bs

                        layer_bi = bit-r[0]


                        if bit in self.weight_bis:
                            layer_wi = int(layer_bi // w_bitwidth)
                            layer_wbi = int(layer_bi % w_bitwidth)
                            curr_quant = layer.kernel_quantizer_internal
                            curr_quant_config = curr_quant.get_config()
                            cqc_hashable = tuple([(k,v) for k,v in curr_quant_config.items()])

                            self.layer_name_to_num_flipped_bits_wb[layer.name][0] += 1
                            self.flipped_bi_to_layer_and_flbri[bit] = (
                                layer,
                                cqc_hashable,
                                assoc_lis[0],
                                layer_bi,
                                layer_wi,
                                layer_wbi,
                            )
                            actual_bits_to_flip.append(
                                self.flipped_bi_to_layer_and_flbri[bit]
                            )

                        elif bit in self.bias_bis:
                            layer_bi = bit-r[0]-num_w_bits
                            layer_wi = int(layer_bi // b_bitwidth)
                            layer_wbi = int(layer_bi % b_bitwidth)

                            curr_quant = layer.bias_quantizer_internal
                            curr_quant_config = curr_quant.get_config()
                            cqc_hashable = tuple([(k,v) for k,v in curr_quant_config.items()])

                            self.layer_name_to_num_flipped_bits_wb[layer.name][1] += 1
                            self.flipped_bi_to_layer_and_flbri[bit] = (
                                layer, 
                                cqc_hashable,
                                assoc_lis[1],
                                layer_bi,
                                layer_wi,
                                layer_wbi,
                            )
                            actual_bits_to_flip.append(
                                self.flipped_bi_to_layer_and_flbri[bit]
                            )
        
        if len(actual_bits_to_flip) > 0:
            new_params = [np.copy(arr) for arr in self.model.get_weights()]
            
            for l, cqc_hashable, li_adj, _, wi, wbi in actual_bits_to_flip:
                assoc_quant = None
                if li_adj == self.layer_name_to_lis[l.name][0]:
                    assoc_quant = l.kernel_quantizer_internal
                else:
                    assoc_quant = l.bias_quantizer_internal

                orig_shape = new_params[li_adj].shape
                faulty_w_or_b_arr = new_params[li_adj].flatten()
                qval = faulty_w_or_b_arr[wi]
                qval = assoc_quant(qval).numpy()
                qbin = self.quant_config_to_bin_float_dicts[cqc_hashable][1][qval]
                flipped_bit = "1" if qbin[wbi] == "0" else "0"
                qbin_f = qbin[:wbi] + flipped_bit + qbin[wbi+1:]
                qval_f = self.quant_config_to_bin_float_dicts[cqc_hashable][0][qbin_f]
                faulty_w_or_b_arr[wi] = qval_f
                new_params[li_adj] = faulty_w_or_b_arr.reshape(orig_shape)
            
            self.model.set_weights(new_params)


    def explicitly_reset_bits(self, bits_to_reset):
        internal_bits_to_reset = sorted(bits_to_reset)

        actual_bits_to_reset = list()

        for bit in internal_bits_to_reset:

            if bit in self.flipped_bi_to_layer_and_flbri:
                actual_bits_to_reset.append(
                    self.flipped_bi_to_layer_and_flbri[bit]
                )

                self.flipped_bi_to_layer_and_flbri.pop(bit)
        
        if len(actual_bits_to_reset) > 0:
            new_params = [np.copy(arr) for arr in self.model.get_weights()]
            
            for l, cqc_hashable, li_adj, _, wi, wbi in actual_bits_to_reset:
                assoc_quant = None
                if li_adj == self.layer_name_to_lis[l.name][0]:
                    assoc_quant = l.kernel_quantizer_internal
                else:
                    assoc_quant = l.bias_quantizer_internal
                orig_shape = new_params[li_adj].shape
                faulty_w_or_b_arr = new_params[li_adj].flatten()
                qval = faulty_w_or_b_arr[wi]
                qval = assoc_quant(qval).numpy()
                qbin = self.quant_config_to_bin_float_dicts[cqc_hashable][1][qval]
                flipped_bit = "1" if qbin[wbi] == "0" else "0"
                qbin_f = qbin[:wbi] + flipped_bit + qbin[wbi+1:]
                qval_f = self.quant_config_to_bin_float_dicts[cqc_hashable][0][qbin_f]
                faulty_w_or_b_arr[wi] = qval_f
                new_params[li_adj] = faulty_w_or_b_arr.reshape(orig_shape)

                if li_adj == self.layer_name_to_lis[l.name][0]:
                    self.layer_name_to_num_flipped_bits_wb[l.name][0] -= 1
                else:
                    self.layer_name_to_num_flipped_bits_wb[l.name][1] -= 1
            
            self.model.set_weights(new_params)