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
    test_tensor = tf.zeros((2, 2, 4))

    # TODO: Update the following code block with function call that
    ###### returns the same lbi region
    my_bit_pos = 0
    my_test_regions = list()
    my_test_bers = list()
    for i in range(0, 16):
        inclusive_reg_start = fk.utils.wb_index_to_lb_index((i, my_bit_pos), 8)
        inclusive_reg_end = fk.utils.wb_index_to_lb_index((i, my_bit_pos), 8)
        my_test_regions.append((inclusive_reg_start, inclusive_reg_end))
        my_test_bers.append(1.0)

    qb_tensor = fk.utils.quantize_and_bitflip(
        test_tensor, test_quantizer, my_test_regions, my_test_bers
    )

    assert equal_tensors(qb_tensor, golden_tensor)


def test_bitflip_zero_tensor_pos_2():
    pass


def test_zero_ber():
    golden_tensor = tf.fill((2, 2, 4), 0.0)
    test_tensor = tf.zeros((2, 2, 4))
    test_quantizer = quantized_bits(bits=8, integer=7, keep_negative=1, alpha=1)

    # TODO: Update the following code block with function call that
    ###### returns the same lbi region
    my_bit_pos = 0
    my_test_regions = list()
    my_test_bers = list()
    for i in range(0, 16):
        inclusive_reg_start = fk.utils.wb_index_to_lb_index((i, my_bit_pos), 8)
        inclusive_reg_end = fk.utils.wb_index_to_lb_index((i, my_bit_pos), 8)
        my_test_regions.append((inclusive_reg_start, inclusive_reg_end))
        my_test_bers.append(0.0)

    qb_tensor = fk.utils.quantize_and_bitflip(
        test_tensor, test_quantizer, my_test_regions, my_test_bers
    )
    assert equal_tensors(qb_tensor, golden_tensor)
