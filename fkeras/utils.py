def float_to_fp(x, scaling_exponent):
    """x is a float"""
    return int(x * (2 ** scaling_exponent))

def fp_to_float(x, scaling_exponent):
    """x is an integer"""
    return x * (2 ** -scaling_exponent)

def int_to_binstr(x):
    """x is an python int object"""
    return format(x, "b")#BitArray(hex=x).bin

def binstr_to_int(x):
    """x is a little-endian binary string"""
    return int(x,2)


def quantize_and_bitflip(i_value, i_quantizer, pos=0):
    """
    i_value:     float that is not quantized
    i_quantizer: qkeras quantizer
    """
    quant_config = i_quantizer.get_config()
    scaling_exponent = quant_config["bits"] - quant_config["integer"]
    
    #S: Get quantized value (a float)
    result = i_quantizer(i_value)
    
    #S: Turn float to fixed-point representation (an integer)
    result = float_to_fp(result, scaling_exponent)
    
    #S: Integer to binary string
    result = int_to_binstr(result)
    
    #S: Update position to be little-endian
    pos = -1 - pos 
    
    #S: Turn binary string to list of characters
    result = list(result)
    
    #S: Flip the indicated bit
    result[pos] = "0" if "1" == result[pos] else "1"
    
    #S: Turn list of characters to binary string
    result = "".join(result)
    
    #S: Turn binary string to integer
    result = binstr_to_int(result)
    
    #S: Turn to integer into float
    result = fp_to_float(result, scaling_exponent)
    
    return result