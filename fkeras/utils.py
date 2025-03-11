import tensorflow.compat.v2 as tf
import numpy as np
import time

assert tf.executing_eagerly(), "QKeras requires TF with eager execution mode on"
import random


def float_to_fp(x, scaling_exponent):
    """x is a float"""
    return int(x * (2**scaling_exponent))


def fp_to_float(x, scaling_exponent):
    """x is an integer"""
    return x * (2**-scaling_exponent)


def int_to_binstr(x, bit_width):
    """x is an python int object"""
    return format(x, "b").rjust(bit_width, "0")


def binstr_to_int(x):
    """x is a little-endian binary string"""
    return int(x, 2)


def get_tensor_size(i_tensor):
    num_of_elems = 1
    for x in list(i_tensor.shape):
        num_of_elems = num_of_elems * x
    return num_of_elems


def get_fault_indices(i_tensor, num_faults):
    # S: Create a set to store the fault indices
    indices = set()

    # S: Get bounds for indices
    bounds = i_tensor.shape

    # S: Generate num_faults-many indices
    while len(indices) < num_faults:
        # S: Define the next fault index
        curr_fault = list()

        for j, b in enumerate(bounds):
            curr_fault.append(int(np.random.uniform(0, b)))

        indices.add(tuple(curr_fault))

    return list(indices)


class LayerBitRegion:
    def __init__(self, i_start_lbi, end_lbi):
        self.start_lbi = i_start_lbi
        self.end_lbi = end_lbi


class FaultyLayerBitRegion(LayerBitRegion):
    def __init__(self, i_start_lbi, end_lbi, ber):
        super().__init__(i_start_lbi, end_lbi)
        self.ber = ber
    
    def __repr__(self):
        repr_str = "fkeras.utils.FaultyLayerBitRegion"
        repr_str += f"({self.start_lbi}, {self.end_lbi}, {self.ber})"

        return repr_str


def gen_lbi_region_from_weight_level(
    i_tensor, i_bit_width, i_bit_pos, little_endian=True, ber=None
):
    # TODO: Fix this. This is just copied code from other places.
    # gen_lbi_region_from_weight_level(test_tensor, 8, 0)
    lbi_regions = list()
    region_bers = list()
    for i in range(0, get_tensor_size(i_tensor)):
        if little_endian:
            inclusive_reg_start = wb_index_to_lb_index((i, i_bit_pos), i_bit_width)
            inclusive_reg_end = wb_index_to_lb_index((i, i_bit_pos), i_bit_width)
        else:
            inclusive_reg_start = wb_index_to_lb_index((i, i_bit_pos), i_bit_width)
            inclusive_reg_end = wb_index_to_lb_index((i, i_bit_pos), i_bit_width)
        inclusive_reg_start = wb_index_to_lb_index((i, i_bit_pos), i_bit_width)
        inclusive_reg_end = wb_index_to_lb_index((i, i_bit_pos), i_bit_width)
        lbi_regions.append((inclusive_reg_start, inclusive_reg_end))
        region_bers.append(1.0)
    return lbi_regions, region_bers


def gen_lbi_region_at_layer_level(i_tensor, i_bit_width, ber):
    """
    Given a BER for a given layer, generate a single LayerBitRegion
    """
    return [FaultyLayerBitRegion(0, get_tensor_size(i_tensor) * i_bit_width, ber)]


def lb_index_to_wb_index(i_lbi, i_bit_width, little_endian=True):
    """
    Translates a layer-bit index (LBI) to a weight-bit index (WBI)

    Example:
    If you have a layer with two 3-bit weights

    LBI =    0     1     2     3     4     5
          |  0  |  0  |  0  |  0  |  0  |  0  |
    WBI =  (0,0) (0,1) (0,2) (1,0) (1,1) (1,2) [   big-endian]
    WBI =  (0,2) (0,1) (0,0) (1,2) (1,1) (1,0) [little-endian]
    """
    if little_endian:
        return (int(i_lbi / i_bit_width), (i_bit_width - 1) - (i_lbi % i_bit_width))
    else:
        return (int(i_lbi / i_bit_width), i_lbi % i_bit_width)


def wb_index_to_lb_index(i_wbi, i_bit_width, little_endian=True):
    """
    Translates a weight-bit index (WBI) to a layer-bit index (LBI)

    Example:
    If you have a layer with two 3-bit weights

    LBI =    0     1     2     3     4     5
          |  0  |  0  |  0  |  0  |  0  |  0  |
    WBI =  (0,0) (0,1) (0,2) (1,0) (1,1) (1,2) [   big-endian]
    WBI =  (0,2) (0,1) (0,0) (1,2) (1,1) (1,0) [little-endian]
    """
    if little_endian:
        return (i_wbi[0] * i_bit_width) + (i_bit_width - i_wbi[1] - 1)
    else:
        return (i_wbi[0] * i_bit_width) + (i_wbi[1])


def gen_mask_tensor_deterministic(i_tensor, i_ber, i_qbits):
    # S: Generate the mask array (default value is 0)
    mask_array = np.full(i_tensor.shape, 0).flatten()

    # S: Determine the number of bits in region
    num_rbits = mask_array.size * i_qbits

    # S: Determine the number of faults to inject
    num_rfaults = int(num_rbits * i_ber)

    # S: Inject faults
    faults_injected = 0
    while faults_injected < num_rfaults:
        mask_array[faults_injected % mask_array.size] = mask_array[
            faults_injected % mask_array.size
        ] + 2 ** (faults_injected // mask_array.size)
        faults_injected = faults_injected + 1

    return tf.convert_to_tensor(np.reshape(mask_array, i_tensor.shape), dtype=tf.int64)


def gen_mask_tensor_deterministic_v2(i_tensor, i_ber, i_qbits, i_keep_negative=0):
    # S: Generate the mask array (default value is 0)
    mask_array = np.full(i_tensor.shape, 0).flatten()

    # S: Determine the number of bits in region
    num_rbits = mask_array.size * i_qbits

    # S: Determine the number of faults to inject
    num_rfaults = int(num_rbits * i_ber)

    # S: Inject faults
    faults_injected = 0
    while faults_injected < num_rfaults:
        mask_array[faults_injected % mask_array.size] = mask_array[
            faults_injected % mask_array.size
        ] + 2 ** (faults_injected // mask_array.size)
        faults_injected = faults_injected + 1

    sign_mask = 2 ** (i_qbits - i_keep_negative)
    rval_mask = sign_mask - 1
    for i in range(mask_array.shape[0]):
        mask_array[i] = (mask_array[i] & rval_mask) - (mask_array[i] & sign_mask)

    return tf.convert_to_tensor(np.reshape(mask_array, i_tensor.shape), dtype=tf.int64)


def gen_mask_tensor_deterministic_v3(i_tensor, i_flbirs, i_qbits, i_keep_negative=0):
    # S: Generate the mask array (default value is 0)
    mask_array = np.full(i_tensor.shape, 0).flatten()

    # S: Inject faults
    for curr_flbir in i_flbirs:
        # S: Get the start LBI and end LBI
        s_lbi, e_lbi, rber = (curr_flbir.start_lbi, curr_flbir.end_lbi, curr_flbir.ber)

        # S: Get the weight bit index representation of the LBI
        ### TODO: This code assumes a region of a single bit (s_lbi == e_lbi)
        ### Update this code to be more general.
        s_wbi = lb_index_to_wb_index(s_lbi, i_qbits)

        # S: Flip the bit of indicated weight
        mask_array[s_wbi[0]] = mask_array[s_wbi[0]] | (1 << s_wbi[1])

    sign_mask = 2 ** (i_qbits - i_keep_negative)
    rval_mask = sign_mask - 1
    for i in range(mask_array.shape[0]):
        mask_array[i] = (mask_array[i] & rval_mask) - (mask_array[i] & sign_mask)

    # tf.print(mask_array)

    return tf.convert_to_tensor(np.reshape(mask_array, i_tensor.shape), dtype=tf.int64)


def gen_mask_tensor_random(i_tensor, i_ber, i_qbits):
    # S: Generate the mask array (default value is 0)
    mask_array = np.full(i_tensor.shape, 0).flatten()

    # S: Determine the number of bits in region
    num_rbits = mask_array.size * i_qbits

    # S: Determine the number of faults to inject
    num_rfaults = int(num_rbits * i_ber)

    # S: Generate list of randomly sampled bit indices
    faults_to_inject = random.sample(range(num_rbits), num_rfaults)

    # S: Inject random faults
    for fault_loc in faults_to_inject:
        mask_array[fault_loc % mask_array.size] = mask_array[
            fault_loc % mask_array.size
        ] + 2 ** (fault_loc // mask_array.size)

    return tf.convert_to_tensor(np.reshape(mask_array, i_tensor.shape), dtype=tf.int64)


def full_tensor_quantize_and_bit_flip_deterministic(
    i_tensor, i_scaling_exp, i_ber, i_qbits
):
    og_dtype = i_tensor.dtype
    i_tensor = i_tensor * (2**i_scaling_exp)
    i_tensor = tf.cast(i_tensor, tf.int64)
    i_tensor = tf.bitwise.bitwise_xor(
        i_tensor, gen_mask_tensor_deterministic(i_tensor, i_ber, i_qbits)
    )
    i_tensor = tf.cast(i_tensor, og_dtype)
    i_tensor = i_tensor * (2**-i_scaling_exp)

    return i_tensor


def full_tensor_quantize_and_bit_flip_deterministic_v2(
    i_tensor, i_scaling_exp, i_ber, i_qbits, i_keep_negative=0
):
    og_dtype = i_tensor.dtype
    i_tensor = i_tensor * (2**i_scaling_exp)
    i_tensor = tf.cast(i_tensor, tf.int64)
    i_tensor = tf.bitwise.bitwise_xor(
        i_tensor,
        gen_mask_tensor_deterministic_v2(i_tensor, i_ber, i_qbits, i_keep_negative),
    )
    i_tensor = tf.cast(i_tensor, og_dtype)
    i_tensor = i_tensor * (2**-i_scaling_exp)

    return i_tensor


def full_tensor_quantize_and_bit_flip_deterministic_v3(
    i_tensor, i_scaling_exp, i_flbirs, i_qbits, i_keep_negative=0
):
    og_dtype = i_tensor.dtype
    i_tensor = i_tensor * (2**i_scaling_exp)
    i_tensor = tf.cast(i_tensor, tf.int64)
    i_tensor = tf.bitwise.bitwise_xor(
        i_tensor,
        gen_mask_tensor_deterministic_v3(i_tensor, i_flbirs, i_qbits, i_keep_negative),
    )
    i_tensor = tf.cast(i_tensor, og_dtype)
    i_tensor = i_tensor * (2**-i_scaling_exp)

    return i_tensor


def full_tensor_quantize_and_bit_flip(i_tensor, i_scaling_exp, i_ber, i_qbits):
    og_dtype = i_tensor.dtype
    i_tensor = i_tensor * (2**i_scaling_exp)
    i_tensor = tf.cast(i_tensor, tf.int64)
    i_tensor = tf.bitwise.bitwise_xor(
        i_tensor, gen_mask_tensor_random(i_tensor, i_ber, i_qbits)
    )
    i_tensor = tf.cast(i_tensor, og_dtype)
    i_tensor = i_tensor * (2**-i_scaling_exp)

    return i_tensor


def quantize_and_bitflip_deterministic(
    i_values, i_quantizer, regions, bers, i_keep_negative=0
):
    """
    i_values:    a float matrix of non-quantized model parameters
    i_quantizer: qkeras quantizer
    """

    # S: Get quantization configuration
    quant_config = i_quantizer.get_config()
    scaling_exponent = quant_config["bits"] - quant_config["integer"] - i_keep_negative

    # S: Get quantized values (represented as floats)
    result = i_quantizer(i_values)

    # result = result * (2**scaling_exponent)
    new_result = full_tensor_quantize_and_bit_flip_deterministic(
        result, scaling_exponent, bers[0], quant_config["bits"]
    )
    result = new_result
    # Convert back to float
    # result = result * (2**-scaling_exponent)

    return result


def quantize_and_bitflip_deterministic_v2(
    i_values, i_quantizer, regions, bers, i_keep_negative=0
):
    """
    i_values:    a float matrix of non-quantized model parameters
    i_quantizer: qkeras quantizer
    """

    # S: Get quantization configuration
    quant_config = i_quantizer.get_config()
    scaling_exponent = quant_config["bits"] - quant_config["integer"] - i_keep_negative

    # S: Get quantized values (represented as floats)
    result = i_quantizer(i_values)

    # result = result * (2**scaling_exponent)
    new_result = full_tensor_quantize_and_bit_flip_deterministic_v2(
        result, scaling_exponent, bers[0], quant_config["bits"], i_keep_negative
    )
    result = new_result
    # Convert back to float
    # result = result * (2**-scaling_exponent)

    return result


def quantize_and_bitflip_deterministic_v3(i_values, i_quantizer, regions, bers):
    """
    i_values:    a float matrix of non-quantized model parameters
    i_quantizer: qkeras quantizer
    """

    # S: Get quantization configuration
    quant_config = i_quantizer.get_config()
    scaling_exponent = (
        quant_config["bits"] - quant_config["integer"] - quant_config["keep_negative"]
    )

    # S: Get quantized values (represented as floats)
    result = i_quantizer(i_values)

    new_result = full_tensor_quantize_and_bit_flip_deterministic_v3(
        result,
        scaling_exponent,
        regions,
        quant_config["bits"],
        quant_config["keep_negative"],
    )
    result = new_result

    return result


def quantize_and_bitflip(i_values, i_quantizer, regions, bers):
    """
    i_values:    a float matrix of non-quantized model parameters
    i_quantizer: qkeras quantizer
    """

    # S: Get quantization configuration
    quant_config = i_quantizer.get_config()
    scaling_exponent = quant_config["bits"] - quant_config["integer"]

    # S: Get quantized values (represented as floats)
    result = i_quantizer(i_values)

    # result = result * (2**scaling_exponent)
    new_result = full_tensor_quantize_and_bit_flip(
        result, scaling_exponent, bers[0], quant_config["bits"]
    )
    result = new_result
    # Convert back to float
    # result = result * (2**-scaling_exponent)

    return result
