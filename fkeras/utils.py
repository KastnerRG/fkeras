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


def quantize_and_bitflip(i_values, i_quantizer, pos=0, ber=0.0):
    """
    i_values:    a float matrix of non-quantized model parameters
    i_quantizer: qkeras quantizer
    """ 
    #S: Get quantization configuration
    quant_config = i_quantizer.get_config()
    scaling_exponent = quant_config["bits"] - quant_config["integer"]
    
    #S: Determine the number of faults to inject
    num_faults = int(get_tensor_size(i_values) * quant_config["bits"] * ber)
    # vvvvvvvvvv - Number of faults determined on weight level
    # num_faults = int(get_tensor_size(i_values) * ber) 
    
    #S: Get quantized values (represented as floats)
    result = i_quantizer(i_values)
    
    #S: Flatten i_values
    result = result.numpy().flatten()
    
    #TODO: Turn for-loop into while-loop so that
    #####  we iterate through bits instead of weights
    for i in range(num_faults):
        curr_val = result[i]
        # print(f"FL#1: curr_val = {curr_val} | {i}")
        
        # TODO: Make this stuff below its own function
        #S: Turn float to fixed-point representation (an integer)
        curr_val = float_to_fp(curr_val, scaling_exponent)

        #S: Integer to binary string
        curr_val = int_to_binstr(curr_val, quant_config["bits"])

        #S: Update position to be little-endian
        curr_pos = -1 - pos 

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
        result[i] = curr_val
        
        
    #S: Reshape i_values
    result = tf.reshape(result, i_values.shape)
    
    return result

# For later, do things at weight-level?
# If you want to specify bit position(s), then ber = 1 and you must provide a Weight error
# rate. 

# If you want to specify bit region + ber, then provide (list of ) bit regions
# and a (list of) bit error rates, where bit region at index i has bit error
# rate at index i. 