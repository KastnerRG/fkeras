import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.keras import backend
from tensorflow.python.ops import gradients

from ..fmodel import SUPPORTED_LAYERS

from itertools import chain


class HessianMetrics:
    """
    Class for computing Hessian metrics:
        - The top 1 (k) eigenvalues of the Hessian
        - The top 1 (k) eigenvectors of the Hessian
        - The trace of the Hessian
    """

    def __init__(self, model, loss_fn, x, y, batch_size=32, layer_precision_info=None):
        """
        model: Keras model
        loss_fn: loss function
        x: input data
        y: target data

        NOTE: For now, just operates on a single batch of data
        """
        self.model = model
        self.loss_fn = loss_fn
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.batched_x = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        self.batched_y = tf.data.Dataset.from_tensor_slices(y).batch(batch_size)
        self.layer_indices = self.get_layers_with_trainable_params(model)
        self.layer_precision_info = layer_precision_info
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

    def get_supported_layer_indices(self, include_bias=False):
        # Get indices of parameters in supported layers
        supported_indices = []
        running_idx = 0
        for i in self.layer_indices:
            print(f"layer: {self.model.layers[i].__class__.__name__}")
            if self.model.layers[i].__class__.__name__ in SUPPORTED_LAYERS:
                supported_indices.append(running_idx)
                if include_bias and self.model.layers[i].trainable_variables.__len__() > 1:
                    # Add bias if it exists
                    supported_indices.append(running_idx + 1)
            # print(f"len trainable variables: {self.model.layers[i].trainable_variables.__len__()}")
            print(f"trainable variables[0] shape: {self.model.layers[i].trainable_variables[0].shape}")
            print(f"trainable variables[1] shape: {self.model.layers[i].trainable_variables[1].shape}")
            running_idx += self.model.layers[i].trainable_variables.__len__()
        return supported_indices

    def hessian_vector_product(self, v):
        """
        Compute the Hessian vector product of Hv, where
        H is the Hessian of the loss function with respect to the model parameters
        v is a vector of the same size as the model parameters

        Based on: https://github.com/tensorflow/tensorflow/blob/47f0e99c1918f68daa84bd4cac1b6011b2942dac/tensorflow/python/eager/benchmarks/resnet50/hvp_test.py#L62
        """
        # Compute the gradients of the loss function with respect to the model parameters
        params = [
            v
            for i in self.layer_indices
            for v in self.model.layers[i].trainable_variables
        ]
        num_data = 0
        temp_hv = [tf.zeros_like(p) for p in params]
        for batch_x, batch_y in zip(self.batched_x, self.batched_y):
            with tf.GradientTape() as outer_tape:
                with tf.GradientTape() as inner_tape:
                    loss = self.loss_fn(batch_y, self.model(batch_x, training=True))
                # Compute the gradient inside the outer `out_tape` context manager
                # which means the gradient computation is differentiable as well.
                grads = inner_tape.gradient(loss, params)
            hv = outer_tape.gradient(grads, params, output_gradients=v)
            temp_hv = [
                THv1 + Hv1 * float(len(batch_x)) for THv1, Hv1 in zip(temp_hv, hv)
            ]
            num_data += len(batch_x)
        temp_hv = [THv1 / float(num_data) for THv1 in temp_hv]
        eigenvalue = tf.reduce_sum(
            [tf.reduce_sum(THv1 * v1) for THv1, v1 in zip(temp_hv, v)]
        )
        # Compute the Hessian vector product
        return temp_hv, eigenvalue

    def hessian_vector_product_hack(self, v, super_layer_idx=None):
        """
        Compute the Hessian vector product of Hv, where
        H is the Hessian of the loss function with respect to the model parameters
        v is a vector of the same size as the model parameters

        Based on: https://github.com/tensorflow/tensorflow/blob/47f0e99c1918f68daa84bd4cac1b6011b2942dac/tensorflow/python/eager/benchmarks/resnet50/hvp_test.py#L62
        """
        # Compute the gradients of the loss function with respect to the model parameters
        layer_indices = self.get_layers_with_trainable_params(
            self.model.layers[super_layer_idx]
        )
        params = [
            v
            for i in layer_indices
            for v in self.model.layers[super_layer_idx].layers[i].trainable_variables
        ]
        num_data = 0
        temp_hv = [tf.zeros_like(p) for p in params]
        for batch_x, batch_y in zip(self.batched_x, self.batched_y):
            with tf.GradientTape() as outer_tape:
                with tf.GradientTape() as inner_tape:
                    loss = self.loss_fn(self.model(batch_x, training=True), batch_y)
                # Compute the gradient inside the outer `out_tape` context manager
                # which means the gradient computation is differentiable as well.
                grads = inner_tape.gradient(loss, params)
            hv = outer_tape.gradient(grads, params, output_gradients=v)
            temp_hv = [
                THv1 + Hv1 * float(len(batch_x)) for THv1, Hv1 in zip(temp_hv, hv)
            ]
            num_data += len(batch_x)
        temp_hv = [THv1 / float(num_data) for THv1 in temp_hv]
        eigenvalue = tf.reduce_sum(
            [tf.reduce_sum(THv1 * v1) for THv1, v1 in zip(temp_hv, v)]
        )
        # Compute the Hessian vector product
        return temp_hv, eigenvalue

    def trace(self, max_iter=100, tolerance=1e-3):
        """
        Compute the trace of the Hessian using Hutchinson's method
        max_iter: maximimum number of iterations used to compute trace
        tolerance: tolerance for convergence
        """
        trace = 0.0
        trace_vhv = []
        layer_trace_vhv = []
        params = [
            v
            for i in self.layer_indices
            for v in self.model.layers[i].trainable_variables
        ]
        # for i in range(max_iter):
        while True:
            v = [np.random.uniform(size=p.shape) for p in params]
            # Generate Rademacher random variables
            for vi in v:
                vi[vi < 0.5] = -1
                vi[vi >= 0.5] = 1
            v = [tf.convert_to_tensor(vi, dtype=tf.dtypes.float32) for vi in v]
            # Compute the Hessian vector product
            hv, _ = self.hessian_vector_product(v)
            # Compute the trace
            curr_trace_vhv = [tf.reduce_sum(vi * hvi) for (vi, hvi) in zip(v, hv)]
            layer_trace_vhv.append(curr_trace_vhv)
            trace_vhv.append(tf.reduce_sum(curr_trace_vhv))
            if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tolerance:
                break
            else:
                trace = np.mean(trace_vhv)
        # TODO: Create a dictionary of layer names and their traces
        # trace_dict = {}
        # for i, l_i in enumerate(layer_indices):
        # TODO: Take dot product of weights and biases for a layer across all
        # iterations and take the mean
        #     layer_name = self.model.layers[sl_i].layers[l_i].name
        #     trace_dict[layer_name] = np.mean(layer_trace_vhv, axis=0)[i]
        return np.mean(trace_vhv)

    def trace_hack(self, max_iter=100, tolerance=1e-3):
        """
        Compute the trace of the Hessian using Hutchinson's method
        max_iter: maximimum number of iterations used to compute trace
        tolerance: tolerance for convergence
        """
        trace = 0.0
        trace_vhv = []
        layer_trace_vhv = []
        for sl_i in self.layer_indices:
            super_layer = self.model.layers[sl_i]
            layer_indices = self.get_layers_with_trainable_params(super_layer)
            # tf.print(f"\n\n#########HessianDebug{self.get_layers_with_trainable_params(super_layer)}#########\n\n")
            params = [
                v
                for i in layer_indices
                for v in self.model.layers[sl_i].layers[i].trainable_variables
            ]
            # for i in range(max_iter):
            while True:
                v = [np.random.uniform(size=p.shape) for p in params]
                # Generate Rademacher random variables
                for vi in v:
                    vi[vi < 0.5] = -1
                    vi[vi >= 0.5] = 1
                v = [tf.convert_to_tensor(vi, dtype=tf.dtypes.float32) for vi in v]
                # Compute the Hessian vector product
                hv, _ = self.hessian_vector_product_hack(v, super_layer_idx=sl_i)
                # Compute the trace
                curr_trace_vhv = [tf.reduce_sum(vi * hvi) for (vi, hvi) in zip(v, hv)]
                layer_trace_vhv.append(curr_trace_vhv)
                trace_vhv.append(tf.reduce_sum(curr_trace_vhv))
                if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tolerance:
                    break
                else:
                    trace = np.mean(trace_vhv)
            break  # Compute for encoder only
        # TODO: Create a dictionary of layer names and their traces
        # trace_dict = {}
        # for i, l_i in enumerate(layer_indices):
        # TODO: Take dot product of weights and biases for a layer across all
        # iterations and take the mean
        #     layer_name = self.model.layers[sl_i].layers[l_i].name
        #     trace_dict[layer_name] = np.mean(layer_trace_vhv, axis=0)[i]
        return np.mean(trace_vhv)

    def normalize_vector_list(self, v_list):
        """
        Normalize a list of vectors
        """
        s = tf.reduce_sum([tf.reduce_sum(v * v) for v in v_list])
        s = s**0.5
        return [v / (s + 1e-6) for v in v_list]

    def top_k_eigenvalues_hack(self, k=1, max_iter=100, tolerance=1e-3):
        """
        Compute the top k eigenvalues and eigenvectors of the Hessian using the
        power iteration method. The eigenvalues are sorted in descending order.
        k: number of eigenvalues to compute
        max_iter: maximum number of iterations used to compute eigenvalues
        tolerance: tolerance for convergence
        """
        eigenvalues = []
        eigenvectors = []
        for sl_i in self.layer_indices:
            layer_indices = self.get_layers_with_trainable_params(
                self.model.layers[sl_i]
            )
            params = [
                v
                for i in layer_indices
                for v in self.model.layers[sl_i].layers[i].trainable_variables
            ]
            for i in range(k):
                eigenvalue = None
                # Initialize the eigenvector
                v = [np.random.uniform(size=p.shape) for p in params]
                v = [tf.convert_to_tensor(vi, dtype=tf.dtypes.float32) for vi in v]
                # Normalize the eigenvector
                v = self.normalize_vector_list(v)
                for j in range(max_iter):
                    # Make v orthonormal to eigenvectors
                    for ei in eigenvectors:
                        v = [vi - tf.reduce_sum(vi * e) * e for (vi, e) in zip(v, ei)]
                    v = self.normalize_vector_list(v)
                    # Compute the Hessian vector product
                    hv, tmp_eigenvalue = self.hessian_vector_product_hack(
                        v, super_layer_idx=sl_i
                    )
                    # Normalize the eigenvector
                    v = self.normalize_vector_list(hv)
                    if eigenvalue == None:
                        eigenvalue = tmp_eigenvalue
                    else:
                        if (
                            abs(tmp_eigenvalue - eigenvalue) / (abs(eigenvalue) + 1e-6)
                            < tolerance
                        ):
                            break
                        else:
                            eigenvalue = tmp_eigenvalue
                eigenvalues.append(eigenvalue)
                eigenvectors.append(v)
            break  # Compute for encoder only
        return eigenvalues, eigenvectors

    def top_k_eigenvalues(self, k=1, max_iter=100, tolerance=1e-3, rank_BN=True):
        """
        Compute the top k eigenvalues and eigenvectors of the Hessian using the
        power iteration method. The eigenvalues are sorted in descending order.
        k: number of eigenvalues to compute
        max_iter: maximum number of iterations used to compute eigenvalues
        tolerance: tolerance for convergence
        """
        params = [
            v
            for i in self.layer_indices
            for v in self.model.layers[i].trainable_variables
        ]

        eigenvalues = []
        eigenvectors = []

        for _ in range(k):
            eigenvalue = None
            # Initialize the eigenvector
            v = [np.random.uniform(size=p.shape) for p in params]
            v = [tf.convert_to_tensor(vi, dtype=tf.dtypes.float32) for vi in v]
            # Normalize the eigenvector
            v = self.normalize_vector_list(v)
            for j in range(max_iter):
                # Make v orthonormal to eigenvectors
                for ei in eigenvectors:
                    v = [vi - tf.reduce_sum(vi * e) * e for (vi, e) in zip(v, ei)]
                v = self.normalize_vector_list(v)
                # Compute the Hessian vector product
                hv, tmp_eigenvalue = self.hessian_vector_product(v)
                # Normalize the eigenvector
                v = self.normalize_vector_list(hv)
                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if (
                        abs(tmp_eigenvalue - eigenvalue) / (abs(eigenvalue) + 1e-6)
                        < tolerance
                    ):
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(np.array(v, dtype=object))

        if not rank_BN:
            supported_indices = self.get_supported_layer_indices()
            print(f"supported_indices: {supported_indices}")

            sanitized_evs = []
            for i in range(k):
                curr_evs = []
                for j in range(len(eigenvectors[i])):
                    if j in supported_indices:
                        curr_evs.append(np.array(eigenvectors[i][j]))
                sanitized_evs.append(np.array(curr_evs, dtype=object))

            eigenvectors = sanitized_evs

        return eigenvalues, eigenvectors

    def do_sum_hessian_rank(self, params, eigenvectors, eigenvalues, k, iter_by=1):
        """
        Given flattened list of parameters, list of eigenvectors, and list of
        eigenvalues, compute the eigenvector/value scores.

        Combine the weight eigenvectors into a single vector for model-wide
        parameter sensitivity ranking using weighted sum strategy.
        iter_by = 2 if ignoring biases
        Return a list of eigenvector/eigenvalue
        scores, one score for each parameter.
        Current method: weighted sum of eigenvectors
        """
        combined_eigenvector_score = np.zeros(params.size)
        for i in range(k):
            combined_eigenvector = []
            for j in range(0, len(eigenvectors[i]), iter_by):
                # TODO: Exception is used due to the inconsistent
                # usage of numpy arrays and tensorflow tensors.
                # To be fixed in the future by only using np arrays.
                try:
                    ev = eigenvectors[i][j].numpy()  # Weight eigenvector
                except AttributeError:
                    ev = np.array(eigenvectors[i][j])
                combined_eigenvector.extend(ev.flatten())
            combined_eigenvector = np.array(combined_eigenvector)
            if eigenvalues:
                curr_eigenvalue = eigenvalues[i].numpy()
                combined_eigenvector = combined_eigenvector * curr_eigenvalue
            scalar_rank = np.dot(combined_eigenvector, params)
            combined_eigenvector_score += np.abs(scalar_rank * combined_eigenvector)
        return combined_eigenvector_score

    def do_max_hessian_rank(self, params, eigenvectors, eigenvalues, k):
        """
        Given flattened list of parameters, list of eigenvectors, and list of
        eigenvalues, compute the eigenvector/value scores, using the "max"
        strategy:
            max_i sum_j^k (eigenvector_j_i * param_i)
        Sort by max, max-1, max-2, etc.
        Return a list of eigenvector/eigenvalue scores, one score for each
        parameter.
        """
        combined_eigenvector_score = np.zeros(params.size)
        # Reshape eigenvectors into a matrix where row i is the flattened ith
        # weight eigenvector
        flat_eigenvector_matrix = np.zeros((k, params.size))
        for j in range(k):
            combined_eigenvector = []
            for l in range(0, len(eigenvectors[j]), 2):
                # Go every 2 to ignore biases
                ev = eigenvectors[j][l].numpy()  # Weight eigenvector
                combined_eigenvector.extend(ev.flatten())
            combined_eigenvector = np.array(combined_eigenvector)
            if eigenvalues:
                curr_eigenvalue = eigenvalues[j].numpy()
                combined_eigenvector = combined_eigenvector * curr_eigenvalue
            flat_eigenvector_matrix[j] = combined_eigenvector
        # Compute score for each parameter
        for i in range(combined_eigenvector_score.size):
            # print(f"Flat eigenvector matrix[{i}][{j}]: {flat_eigenvector_matrix[i][j]}")
            print(f"Params[{i}]: {params[i]}")
            combined_eigenvector_score[i] = np.sum(
                [np.abs(flat_eigenvector_matrix[j][i]) * params[i] for j in range(k)]
            )
        return combined_eigenvector_score

    def rank_bits(self, param_scores, num_bits):
        """
        Given a list of parameter scores, return a list of bit indices sorted by
        the highest scoring bits.
        Decompose the parameter scores bitwise into powers of 2 then multiply
        these components
        """
        # param _scores is len n and want an array of n * m (m = num bits)
        bitwise_scores = []

        bitwise_weights = np.array([2**x for x in range(num_bits - 1, -1, -1)])

        for param_score in param_scores:
            curr_scores = param_score * bitwise_weights
            bitwise_scores.extend(curr_scores)

        bitwise_scores = np.array(bitwise_scores)
        bitwise_rank = np.flip(np.argsort(bitwise_scores))
        bitwise_scores = bitwise_scores[bitwise_rank]

        return bitwise_rank, bitwise_scores

    def sort_bits_MSB_to_LSB(self, param_bit_order_ranking, num_bits):
        """
        Given a bit ranking in which the bits are ordered from MSB to LSB in
        parameter order, return a list of bit indices where all the MSBs are first,
        followed by all the MSBs - 1, etc.
        """

        bit_rank = []
        for i in range(num_bits):
            bit_group = param_bit_order_ranking[i::num_bits]
            bit_rank.extend(bit_group)

        return np.array(bit_rank)

    ###############################
    def convert_param_ranking_to_msb_bit_ranking_mixed_precision(self, param_ranking):
        layer_info_head = 0
        num_of_params_seen = 0
        last_bit_idx = -1
        param_to_bit_indices_dict = dict()
        # Build a dictionary such that (key, val) = param_index, ([bit_indices])
        for wi in range(len(param_ranking)):
            bit_indices_associated_with_param = []
            bit_width = self.layer_precision_info[layer_info_head][1]
            for wbi in range(bit_width):
                last_bit_idx += 1
                bit_indices_associated_with_param.append((last_bit_idx, wbi))

            param_to_bit_indices_dict[wi] = bit_indices_associated_with_param

            num_of_params_seen += 1
            if num_of_params_seen == self.layer_precision_info[layer_info_head][0]:
                layer_info_head += 1
                num_of_params_seen = 0

        # Add the bit indices into the appropriate list based on
        # the parameter ranking
        max_bit_width = max(self.layer_precision_info, key=lambda x: x[1])[1]
        sorted_msb_lsb_lists = [[] for _ in range(max_bit_width)]
        for param_idx in param_ranking:
            for bit_idx, wbi in param_to_bit_indices_dict[param_idx]:
                sorted_msb_lsb_lists[wbi].append(bit_idx)

        merged_msb_to_lsb_list = []
        for l in sorted_msb_lsb_lists:
            merged_msb_to_lsb_list.extend(l)

        return merged_msb_to_lsb_list

    def convert_param_ranking_to_msb_bit_ranking_single_precision(
        self, param_ranking, num_bits
    ):
        # Convert param ranking to bit ranking
        bit_level_rank = []

        for param in param_ranking:
            bit_idx = param * num_bits
            bit_level_rank.append(bit_idx)

            for j in range(1, num_bits):
                bit_level_rank.append(bit_idx + j)
        # Sort from MSB to LSB
        bit_level_rank = self.sort_bits_MSB_to_LSB(bit_level_rank, num_bits)
        return bit_level_rank

    def convert_param_ranking_to_msb_bit_ranking(self, param_ranking, num_bits):
        if self.layer_precision_info == None:
            return self.convert_param_ranking_to_msb_bit_ranking_single_precision(
                param_ranking, num_bits
            )
        else:
            return self.convert_param_ranking_to_msb_bit_ranking_mixed_precision(
                param_ranking
            )

    ###############################

    def hessian_ranking_hack(self, eigenvectors, eigenvalues=None, k=1, strategy="sum", iter_by=1):
        """
        Given list of eigenvectors and eigenvalues, compute the sensitivity.
        Use Hessian to rank parameters based on sensitivity to bit flips with
        respect to all parameters (not by layer like layer_hessian_ranking).
        """
        # Get all parameters of the model and flatten and concat into one list
        params = []
        for sl_i in self.layer_indices:
            super_layer = self.model.layers[sl_i]
            for l_i in self.get_layers_with_trainable_params(super_layer):
                params.append( 
                    self.model.layers[sl_i].layers[l_i].trainable_variables[0].numpy()
                )
                if self.model.layers[sl_i].layers[l_i].trainable_variables.__len__() > 1:
                    # Add bias if it exists
                    params.append(
                        self.model.layers[sl_i].layers[l_i].trainable_variables[1].numpy()
                    )
            break  # Compute for encoder only
        # Flatten and concatenate all eigenvectors into one list
        params = np.concatenate(params, axis=None)
        params = params.flatten()
        if strategy == "sum":
            eigenvector_rank = self.do_sum_hessian_rank(
                params, eigenvectors, eigenvalues, k, iter_by=iter_by
            )
        elif strategy == "max":
            eigenvector_rank = self.do_max_hessian_rank(
                params, eigenvectors, eigenvalues, k
            )
        param_ranking = np.flip(np.argsort(np.abs(eigenvector_rank)))
        param_scores = eigenvector_rank[param_ranking]
        return param_ranking, param_scores

    def hessian_ranking_general(
        self, eigenvectors, eigenvalues=None, k=1, strategy="sum", iter_by=1
    ):
        """
        Given list of eigenvectors and eigenvalues, compute the sensitivity.
        Use Hessian to rank parameters based on sensitivity to bit flips with
        respect to all parameters (not by layer like layer_hessian_ranking).
        """
        # Get all parameters of the model and flatten and concat into one list
        params = [
            v
            for i in self.layer_indices
            for v in self.model.layers[i].trainable_variables
        ]

        # Get indices of parameters in supported layers
        supported_indices = self.get_supported_layer_indices()

        # Sanitize params (i.e., remove any params from unsupported layers)
        sanitized_params = list()
        for i in range(len(params)):
            if i in supported_indices:
                sanitized_params.append(np.array(params[i]))
        params = sanitized_params

        # Flatten and concatenate all eigenvectors into one list
        params = np.concatenate(params, axis=None)
        params = params.flatten()
        # print(f"params shape: {params.shape}")
        if strategy == "sum":
            eigenvector_rank = self.do_sum_hessian_rank(
                params, eigenvectors, eigenvalues, k, iter_by=iter_by
            )
        elif strategy == "max":
            eigenvector_rank = self.do_max_hessian_rank(
                params, eigenvectors, eigenvalues, k
            )
        param_ranking = np.flip(np.argsort(np.abs(eigenvector_rank)))
        param_scores = eigenvector_rank[param_ranking]

        return param_ranking, param_scores

    def gradient_ranking_hack_OLD_CODE(self):
        """
        Rank parameters based on gradient magnitude per layer
        """
        grad_ranking = {}
        grad_dict = {}
        for sl_i in self.layer_indices:
            super_layer = self.model.layers[sl_i]
            for l_i in self.get_layers_with_trainable_params(super_layer):
                layer_name = self.model.layers[sl_i].layers[l_i].name
                print(f"Gradient ranking by sensitivity for layer {layer_name}")
                # Compute gradient ranking
                with tf.GradientTape() as inner_tape:
                    loss = self.loss_fn(self.model(self.x), self.y)
                    params = (
                        self.model.trainable_variables
                        if l_i is None
                        else self.model.layers[sl_i].layers[l_i].trainable_variables
                    )
                    grads = inner_tape.gradient(loss, params)
                grads = grads[0].numpy()
                grad_dict[layer_name] = grads
                print(f"grads shape: {grads.shape}")
                grads = grads.flatten()
                print(f"flat grads shape: {grads.shape}")
                param_ranking = np.flip(np.argsort(np.abs(grads)))
                param_rank_score = grads[param_ranking]
                print(f"grad parameter_ranking: {param_ranking[:10]}")
                grad_ranking[layer_name] = [
                    (param_ranking[i], param_rank_score[i])
                    for i in range(len(param_ranking))
                ]
            break  # Compute for encoder only
        return grad_ranking, grad_dict

    def gradient_ranking_hack(self, super_layer_idx=0):
        """
        Rank parameters based on gradient magnitude per layer
        """
        # TODO: Consider implementing gradient ranking in batches?
        # Get all parameters of the model and flatten and concat into one list
        params = []
        for sl_i in self.layer_indices:
            super_layer = self.model.layers[sl_i]
            for l_i in self.get_layers_with_trainable_params(super_layer):
                params.append(  # Weights only
                    self.model.layers[sl_i].layers[l_i].trainable_variables[0]
                )
            break  # Compute for encoder only

        # Calculate grads over all params
        with tf.GradientTape() as inner_tape:
            loss = self.loss_fn(self.model(self.x), self.y)
            grads = inner_tape.gradient(loss, params)
        grads = np.array(grads, dtype=object)

        # Flatten grads
        flattened_grads = list()
        for g in grads:
            # print(f"grad shape: {g.shape}")
            flattened_grads.extend(np.array(g).flatten())
        grads = np.array(flattened_grads)

        # Compute ranking and rank score
        # print(f"flat grads shape: {grads.shape}")
        param_ranking = np.flip(np.argsort(np.abs(grads)))
        param_rank_score = grads[param_ranking]
        # print(f"grad parameter_ranking: {param_ranking[:10]}")

        return param_ranking, param_rank_score

    def gradient_ranking(self):
        """
        Rank parameters based on gradient magnitude per layer
        """
        # TODO: Consider implementing gradient ranking in batches?
        # Get all parameters of the model and flatten and concat into one list
        params = [
            v
            for i in self.layer_indices
            for v in self.model.layers[i].trainable_variables
        ]

        # Calculate grads over all params
        with tf.GradientTape() as inner_tape:
            loss = self.loss_fn(self.y, self.model(self.x))
            grads = inner_tape.gradient(loss, params)
        grads = np.array(grads, dtype=object)

        # Get indices of parameters in supported layers
        supported_indices = self.get_supported_layer_indices()
        # Sanitize grads (i.e., remove any grads from unsupported layers)
        sanitized_grads = list()
        for i in range(len(grads)):
            if i in supported_indices:
                sanitized_grads.append(np.array(grads[i]))
        grads = np.array(sanitized_grads, dtype=object)

        # Flatten grads
        flattened_grads = list()
        for g in grads:
            # print(f"grad shape: {g.shape}")
            flattened_grads.extend(g.flatten())
        grads = np.array(flattened_grads)

        # Compute ranking and rank score
        # print(f"flat grads shape: {grads.shape}")
        param_ranking = np.flip(np.argsort(np.abs(grads)))
        param_rank_score = grads[param_ranking]
        # print(f"grad parameter_ranking: {param_ranking[:10]}")

        return param_ranking, param_rank_score
    
    def aspis_taylor_ranking_hack(self):
        """
        Rank parameters based on (gradient * magnitude)^2 
        Modified for encoder/decoder architecture

        Based on: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10771492
        """
       # Get all parameters of the model and flatten and concat into one list
        params = []
        for sl_i in self.layer_indices:
            super_layer = self.model.layers[sl_i]
            for l_i in self.get_layers_with_trainable_params(super_layer):
                params.append( 
                    self.model.layers[sl_i].layers[l_i].trainable_variables[0]
                )
                # if self.model.layers[sl_i].layers[l_i].trainable_variables.__len__() > 1:
                #     # Add bias if it exists
                #     params.append(
                #         self.model.layers[sl_i].layers[l_i].trainable_variables[1]
                #     )
            break  # Compute for encoder only

        # Calculate grads over all params
        with tf.GradientTape() as inner_tape:
            loss = self.loss_fn(self.model(self.x), self.y)
            grads = inner_tape.gradient(loss, params)
        grads = np.array(grads, dtype=object)

        # Flatten grads
        flattened_grads = list()
        flattened_params = list()
        for g, p in zip(grads, params):
            # print(f"grad shape: {g.shape}")
            flattened_grads.extend(np.array(g).flatten())
            flattened_params.extend(np.array(p).flatten())
        grads = np.array(flattened_grads)
        params = np.array(flattened_params)

        # Compute ranking and rank score
        score = (grads * params) ** 2

        # print(f"score shape: {grads.shape}")
        param_ranking = np.flip(np.argsort(np.abs(score)))
        param_rank_score = score[param_ranking]
        # print(f"grad parameter_ranking: {param_ranking[:10]}")

        return param_ranking, param_rank_score

    def aspis_taylor_ranking(self):
        """
        Rank parameters based on (gradient * magnitude)^2 

        Based on: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10771492
        """
        params = [
            v
            for i in self.layer_indices
            for v in self.model.layers[i].trainable_variables
        ]

        # Calculate grads over all params
        with tf.GradientTape() as inner_tape:
            loss = self.loss_fn(self.y, self.model(self.x))
            grads = inner_tape.gradient(loss, params)
        grads = np.array(grads, dtype=object)

        # Get indices of parameters in supported layers
        supported_indices = self.get_supported_layer_indices()
        # Sanitize grads (i.e., remove any grads from unsupported layers)
        sanitized_grads = list()
        # Collect santized parameters so that we can multiply them with the gradients
        sanitized_params = list()
        for i in range(len(grads)):
            if i in supported_indices:
                sanitized_grads.append(np.array(grads[i]))
                sanitized_params.append(np.array(params[i])) 
        grads = np.array(sanitized_grads, dtype=object)
        params = np.array(sanitized_params, dtype=object)


        # Flatten grads & params
        flattened_grads = list()
        flattened_params = list()
        for g, p in zip(grads, params):
            # print(f"grad shape: {g.shape}")
            flattened_grads.extend(g.flatten())
            flattened_params.extend(p.flatten())
        grads = np.array(flattened_grads)
        params = np.array(flattened_params)

        score = (grads * params) ** 2

        # Compute ranking and rank score
        # print(f"score shape: {grads.shape}")
        param_ranking = np.flip(np.argsort(np.abs(score)))
        param_rank_score = score[param_ranking]
        # print(f"grad parameter_ranking: {param_ranking[:10]}")

        return param_ranking, param_rank_score