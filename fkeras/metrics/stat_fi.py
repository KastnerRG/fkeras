import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.keras import backend
from tensorflow.python.ops import gradients

from ..fmodel import SUPPORTED_LAYERS

from itertools import chain


class StatFI:
    """
    Class for computing Hessian metrics:
        - The top 1 (k) eigenvalues of the Hessian
        - The top 1 (k) eigenvectors of the Hessian
        - The trace of the Hessian
    """

    def __init__(self, model):
        """
        model: Keras model
        loss_fn: loss function
        x: input data
        y: target data

        NOTE: For now, just operates on a single batch of data
        """
        self.model = model
        self.layer_indices = self.get_layers_with_trainable_params(model)
        np.random.seed(83158011)

    def get_layers_with_trainable_params(self, model):
        """
        Get the indices of the model layers that have trainable parameters
        """
        layer_indices = []
        for i, layer in enumerate(model.layers):
            if len(layer.trainable_variables) > 0:
                layer_indices.append(i)
        return layer_indices

    def get_supported_layer_indices(self):
        # Get indices of parameters in supported layers
        supported_indices = []
        running_idx = 0
        for i in self.layer_indices:
            if self.model.layers[i].__class__.__name__ in SUPPORTED_LAYERS:
                supported_indices.append(running_idx)
            running_idx += self.model.layers[i].trainable_variables.__len__()
        return supported_indices

    def get_params_and_quantizers(self):
        """
        Return tuple (list of parameters layer-wise, list of quantizers layer-wise),
        e.g., ([param1, param2, ...], [quantizer1, quantizer2, ...]) for layers
        1 and 2.
        """

        # Compute the gradients of the loss function with respect to the model parameters
        params = [
            v.numpy()
            for i in self.layer_indices
            if self.model.layers[i].__class__.__name__ in SUPPORTED_LAYERS
            for v in self.model.layers[i].trainable_variables
        ]

        quantizers = [
            self.model.layers[i].kernel_quantizer_internal
            for i in self.layer_indices
            if self.model.layers[i].__class__.__name__ in SUPPORTED_LAYERS
        ]

        return np.array(params, dtype="object"), np.array(quantizers, dtype="object")
