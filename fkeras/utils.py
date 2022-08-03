import tensorflow.compat.v2 as tf
import numpy as np

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

def quantize_and_bitflip(i_values, i_quantizer, regions, bers):
    """
    i_values:    a float matrix of non-quantized model parameters
    i_quantizer: qkeras quantizer
    """
    # NOTE: i_value should be matrix of weights/a kernel. Therefore, we should
    # 1) quantize everything, 2) select the weights to be fault injected, 3)
    # inject faults, 4) write fault-injected weights back into quantized kernel
    
    #S: Get quantization configuration
    quant_config = i_quantizer.get_config()
    scaling_exponent = quant_config["bits"] - quant_config["integer"]
    
    # #S: Determine the number of faults to inject
    # #S: TODO: Update the way we determine the num_faults
    # # num_faults = int(get_tensor_size(i_values) * quant_config["bits"] * ber)
    # num_faults = int(get_tensor_size(i_values) * ber)
    # print(f"Num_faults: {num_faults}")
    
    #S: Get quantized values (represented as floats)
    result = i_quantizer(i_values)
    
    #S: Flatten i_values
    result = result.numpy().flatten()
    print(f"Flattened Result = {result}")
    

    #S:  Iterate through every region
    for r_i, region in enumerate(regions):
        
        #S: Determine the number of bits in region
        ### Assumptions: (1) region bounds are inclusive
        ### and (2) bounds are in layer-bit index form
        num_rbits = region[1] - region[0] + 1
        
        #S: Determine the number of faults in region
        num_rfaults = int(num_rbits * bers[r_i])
        
        for lbi in range(region[0], region[0]+num_rfaults):
            #S: Translate layer-bit index to weight-bit index
            w_i, b_pos = lb_index_to_wb_index(lbi, quant_config["bits"], True)
            
            #S: Get the weight containing the bit
            ### to be flipped/updated
            curr_val = result[w_i]

            #S: Turn float to fixed-point representation (an integer)
            curr_val = float_to_fp(curr_val, scaling_exponent)

            #S: Integer to binary string
            curr_val = int_to_binstr(curr_val, quant_config["bits"])

            #S: Update position to be little-endian
            curr_pos = -1 - b_pos 

            #S: Turn binary string to list of characters
            curr_val = list(curr_val)

            #S: Flip the indicated bit
            curr_val[curr_pos] = "0" if "1" == curr_val[curr_pos] else "1"

            #S: Turn list of characters to binary string
            curr_val = "".join(curr_val)

            #S: Turn binary string to integer
            curr_val = binstr_to_int(curr_val)

            #S: Turn to integer into float
            curr_val = fp_to_float(curr_val, scaling_exponent)

            #S: Update i_values/result
            result[w_i] = curr_val
        
        
    #S: Reshape i_values
    result = tf.reshape(result, i_values.shape)
    
    return result