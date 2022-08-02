import pytest 
import fkeras as fk
# from fkeras.utils import quantize_and_bitflip
import tensorflow as tf

import qkeras 
from qkeras.quantizers import quantized_bits 
from utils import equal_tensors


def create_test_tensors():
    pass

def test_bitflip_zero_tensor():
    golden_tensor = tf.fill((2, 2, 4), 0.5)
    test_quantizer = quantized_bits(bits=8, integer=7, keep_negative=1, alpha=1) 
    test_tensor = tf.zeros((2,2,4)) 
    qb_tensor = fk.utils.quantize_and_bitflip(
        test_tensor, 
        test_quantizer, 
        pos=0, 
        ber=1
    ) 
    assert equal_tensors(qb_tensor, golden_tensor)

def test_bitflip_zero_tensor_pos_2():
    pass

def test_zero_ber():
    golden_tensor = tf.fill((2, 2, 4), 0.0)
    test_tensor = tf.zeros((2,2,4)) 
    test_quantizer = quantized_bits(bits=8, integer=7, keep_negative=1, alpha=1) 
    qb_tensor = fk.utils.quantize_and_bitflip(
        test_tensor, 
        test_quantizer, 
        pos=0, 
        ber=0
    )
    assert equal_tensors(qb_tensor, golden_tensor)

