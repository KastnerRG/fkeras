import tensorflow.compat.v2 as tf
import numpy as np
import time
assert tf.executing_eagerly(), "QKeras requires TF with eager execution mode on"
import random
    
def float_to_fp(x, scaling_exponent):
    """x is a float"""
    return int(x * (2 ** scaling_exponent))

def fp_to_float(x, scaling_exponent):
    """x is an integer"""
    return x * (2 ** -scaling_exponent)

def int_to_binstr(x, bit_width): 
    """x is an python int object""" 
    return format(x, "b").rjust(bit_width, '0')

def binstr_to_int(x):
    """x is a little-endian binary string"""
    return int(x,2)

def get_tensor_size(i_tensor):
    num_of_elems = 1
    for x in list(i_tensor.shape):
        num_of_elems = num_of_elems*x
    return num_of_elems

def get_fault_indices(i_tensor, num_faults):
    #S: Create a set to store the fault indices
    indices = set()
    
    #S: Get bounds for indices
    bounds = i_tensor.shape
    
    #S: Generate num_faults-many indices
    while len(indices) < num_faults:
        
        #S: Define the next fault index
        curr_fault = list()
        
        for j, b  in enumerate(bounds):
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

def gen_lbi_region_from_weight_level(
    i_tensor, 
    i_bit_width, 
    i_bit_pos, 
    little_endian=True, 
    ber=None): 
    # TODO: Fix this. This is just copied code from other places.
    # gen_lbi_region_from_weight_level(test_tensor, 8, 0) 
    lbi_regions = list() 
    region_bers = list() 
    for i in range(0, get_tensor_size(i_tensor)): 
        if little_endian: 
            inclusive_reg_start = wb_index_to_lb_index((i,i_bit_pos), i_bit_width)
            inclusive_reg_end = wb_index_to_lb_index((i,i_bit_pos), i_bit_width) 
        else: 
            inclusive_reg_start = wb_index_to_lb_index((i,i_bit_pos), i_bit_width) 
            inclusive_reg_end = wb_index_to_lb_index((i,i_bit_pos), i_bit_width) 
        inclusive_reg_start = wb_index_to_lb_index((i, i_bit_pos), i_bit_width)
        inclusive_reg_end = wb_index_to_lb_index((i, i_bit_pos), i_bit_width)
        lbi_regions.append( (inclusive_reg_start, inclusive_reg_end) ) 
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
        return (int(i_lbi/i_bit_width), (i_bit_width-1) - (i_lbi%i_bit_width))
    else:
        return (int(i_lbi/i_bit_width), i_lbi%i_bit_width)
    
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
        return (i_wbi[0]*i_bit_width) + (i_bit_width - i_wbi[1]-1)
    else:
        return (i_wbi[0]*i_bit_width) + (i_wbi[1])

class FKERAS_quantize_and_bitflip(tf.Module):
    def __init__(self, i_quantizer, regions, bers):
    # def __init__(self, i_quantizer, regions, bers, Name=None):
        # super(FKERAS_quantize_and_bitflip, self).__init__(name=name)
        self.built = False
        self.i_quantizer = i_quantizer
        self.regions = regions
        self.bers = bers
        
    def _set_trainable_parameter(self):
        pass
    # Override not to expose the quantizer variables.
    @property
    def variables(self):
        return ()

    # Override not to expose the quantizer variables.
    @property
    def trainable_variables(self):
        return ()

    # Override not to expose the quantizer variables.
    @property
    def non_trainable_variables(self):
        return ()
    
    def __call__(self, x):
        i_values = x
        i_quantizer = self.i_quantizer
        regions = self.regions
        bers = self.bers
        print(f"[fkeras] {tf.executing_eagerly()}")
        if tf.executing_eagerly():
            print(f"[fkeras] {type(x.numpy())}")
        
        return i_quantizer(i_values)
    
    
def gen_mask_tensor_deterministic(i_tensor, i_ber, i_qbits):
    #S: Generate the mask array (default value is 0)
    mask_array = np.full(i_tensor.shape, 0).flatten()
    
    #S: Determine the number of bits in region
    num_rbits = mask_array.size * i_qbits
    
    #S: Determine the number of faults to inject
    num_rfaults = int(num_rbits * i_ber)
    
    #S: Inject faults
    faults_injected = 0
    while faults_injected < num_rfaults:
        mask_array[faults_injected%mask_array.size] = mask_array[faults_injected%mask_array.size]\
                                                    + 2**(faults_injected//mask_array.size)
        faults_injected = faults_injected + 1
        
        
    return tf.convert_to_tensor(np.reshape(mask_array, i_tensor.shape), dtype=tf.int64)

def gen_mask_tensor_random(i_tensor, i_ber, i_qbits):
    #S: Generate the mask array (default value is 0)
    mask_array = np.full(i_tensor.shape, 0).flatten()
    
    #S: Determine the number of bits in region
    num_rbits = mask_array.size * i_qbits
    
    #S: Determine the number of faults to inject
    num_rfaults = int(num_rbits * i_ber)
    
    #S: Generate list of randomly sampled bit indices
    faults_to_inject = random.sample(range(num_rbits), num_rfaults)
    
    #S: Inject random faults
    faults_injected = 0
    for fault_loc in faults_to_inject:
        mask_array[fault_loc%mask_array.size] = mask_array[fault_loc%mask_array.size]\
                                                    + 2**(fault_loc//mask_array.size)        
        
    return tf.convert_to_tensor(np.reshape(mask_array, i_tensor.shape), dtype=tf.int64)

def full_tensor_quantize_and_bit_flip(i_tensor, i_scaling_exp, i_ber, i_qbits):
    og_dtype = i_tensor.dtype
    i_tensor = i_tensor*(2**i_scaling_exp)
    i_tensor = tf.cast(i_tensor, tf.int64)
    # i_tensor = tf.bitwise.bitwise_xor(i_tensor, tf.convert_to_tensor(np.full(i_tensor.shape, 1<<63), dtype=tf.int64))
    i_tensor = tf.bitwise.bitwise_xor(i_tensor, gen_mask_tensor_random(i_tensor, i_ber, i_qbits))
    i_tensor = tf.cast(i_tensor, og_dtype)
    i_tensor = i_tensor*(2** -i_scaling_exp)
    
    return i_tensor

def quantize_and_bitflip(i_values, i_quantizer, regions, bers):
    """
    i_values:    a float matrix of non-quantized model parameters
    i_quantizer: qkeras quantizer
    """

    #S: Get quantization configuration
    quant_config = i_quantizer.get_config()
    scaling_exponent = quant_config["bits"] - quant_config["integer"]
    
    #S: Get quantized values (represented as floats)
    result = i_quantizer(i_values)
    
    og_dtype = result.dtype
    result = result*(2**scaling_exponent)
    # result = tf.stop_gradient(tf.cast(result, tf.int64))
    # new_result = tf.stop_gradient(tf.cast( 
    #                                   tf.bitwise.bitwise_xor( tf.cast(result, tf.int64), tf.convert_to_tensor(np.full(result.shape, 0), dtype=tf.int64)),
    #                                   og_dtype
    #                                  ) 
    #                          )
    new_result = tf.stop_gradient(full_tensor_quantize_and_bit_flip(result, scaling_exponent, bers[0], quant_config["bits"]))
    
    # tf.print("Before")
    # tf.print(result)
          
    result = new_result

    # tf.print("After")
    # tf.print(result)
    
    result = result*(2** -scaling_exponent)

    
    return result