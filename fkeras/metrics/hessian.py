import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.keras import backend
from tensorflow.python.ops import gradients


class HessianMetrics:
    """
    Class for computing Hessian metrics:
        - The top 1 (k) eigenvalues of the Hessian
        - The top 1 (k) eigenvectors of the Hessian
        - The trace of the Hessian
    """

    def __init__(self, model, loss_fn, x, y, batch_size=32):
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

    # def hessian_vector_product(self, v, layer_idx=None):
    #     """
    #     Compute the Hessian vector product of Hv, where
    #     H is the Hessian of the loss function with respect to the model parameters
    #     v is a vector of the same size as the model parameters

    #     Based on: https://github.com/tensorflow/tensorflow/blob/47f0e99c1918f68daa84bd4cac1b6011b2942dac/tensorflow/python/eager/benchmarks/resnet50/hvp_test.py#L62
    #     """
    #     # Compute the gradients of the loss function with respect to the model parameters

    #     with tf.GradientTape() as outer_tape:
    #         with tf.GradientTape() as inner_tape:
    #             loss = self.loss_fn(self.model(self.x), self.y)
    #         # params = self.model.trainable_variables if layer_idx is None else self.model.layers[layer_idx].trainable_variables
    #         # tf.print(f"\n\n##{self.model.trainable_variables}##\n\n")
    #         params = (
    #             self.model.trainable_variables
    #             if layer_idx is None
    #             else self.model.trainable_variables[layer_idx]
    #         )
    #         grads = inner_tape.gradient(loss, params)
    #     # Compute the Hessian vector product
    #     return outer_tape.gradient(grads, params, output_gradients=v)

    def layer_hessian_vector_product_hack(
        self, v, super_layer_idx=None, layer_idx=None
    ):
        """
        Compute the Hessian vector product of Hv for a given layer, where
        H is the Hessian of the loss function with respect to the model parameters
        v is a vector of the same size as the model parameters

        Based on: https://github.com/tensorflow/tensorflow/blob/47f0e99c1918f68daa84bd4cac1b6011b2942dac/tensorflow/python/eager/benchmarks/resnet50/hvp_test.py#L62
        """
        # Compute the gradients of the loss function with respect to the model parameters
        params = (
            self.model.trainable_variables
            if layer_idx is None
            else self.model.layers[super_layer_idx]
            .layers[layer_idx]
            .trainable_variables
        )
        num_data = 0
        temp_hv = [tf.zeros_like(p) for p in params]
        for batch_x, batch_y in zip(self.batched_x, self.batched_y):
            with tf.GradientTape() as outer_tape:
                with tf.GradientTape() as inner_tape:
                    loss = self.loss_fn(self.model(batch_x), batch_y)
                    grads = inner_tape.gradient(loss, params)
                hv = outer_tape.gradient(grads, params, output_gradients=v)
            num_data += len(batch_x)
            temp_hv = [
                THv1 + Hv1 * float(len(batch_x)) for THv1, Hv1 in zip(temp_hv, hv)
            ]
        temp_hv = [THv1 / float(num_data) for THv1 in temp_hv]
        eigenvalue = tf.reduce_sum(
            [tf.reduce_sum(THv1 * v1) for THv1, v1 in zip(temp_hv, v)]
        )
        # Compute the Hessian vector product
        return temp_hv, eigenvalue

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

    def layer_top_k_eigenvalues_hack(self, k=1, max_iter=100, tolerance=1e-3):
        """
        Compute the top k eigenvalues and eigenvectors of the Hessian using the
        power iteration method per layer. The eigenvalues are sorted in
        descending order.
        k: number of eigenvalues to compute
        max_iter: maximum number of iterations used to compute eigenvalues
        tolerance: tolerance for convergence
        """
        layer_eigenvalues = {}
        layer_eigenvectors = {}
        for sl_i in self.layer_indices:
            super_layer = self.model.layers[sl_i]
            # tf.print(f"\n\n#########HessianDebug{self.get_layers_with_trainable_params(super_layer)}#########\n\n")
            for l_i in self.get_layers_with_trainable_params(super_layer):
                layer_name = self.model.layers[sl_i].layers[l_i].name
                print(f"Computing top {k} eigenvalues for layer {layer_name}")
                eigenvalues = []
                eigenvectors = []
                for i in range(k):
                    eigenvalue = None
                    # Initialize the eigenvector
                    v = [
                        np.random.uniform(
                            size=self.model.layers[sl_i]
                            .layers[l_i]
                            .trainable_variables[i]
                            .shape
                        )
                        for i in range(
                            len(self.model.layers[sl_i].layers[l_i].trainable_variables)
                        )
                    ]
                    v = [tf.convert_to_tensor(vi, dtype=tf.dtypes.float32) for vi in v]
                    # Normalize the eigenvector
                    # v = [vi / tf.norm(vi) for vi in v]
                    v = self.normalize_vector_list(v)
                    for j in range(max_iter):
                        # Make v orthonormal to eigenvectors
                        for ei in eigenvectors:
                            v = [
                                vi - tf.reduce_sum(vi * e) * e for (vi, e) in zip(v, ei)
                            ]
                        # v = [vi / tf.norm(vi) for vi in v]
                        v = self.normalize_vector_list(v)
                        # Compute the Hessian vector product
                        hv, tmp_eigenvalue = self.layer_hessian_vector_product_hack(
                            v, super_layer_idx=sl_i, layer_idx=l_i
                        )
                        # Normalize the eigenvector
                        # v = [vi / tf.norm(vi) for vi in hv]
                        v = self.normalize_vector_list(hv)
                        if eigenvalue is None:
                            eigenvalue = tmp_eigenvalue
                        else:
                            if (
                                abs(tmp_eigenvalue - eigenvalue)
                                / (abs(eigenvalue) + 1e-6)
                                < tolerance
                            ):
                                break
                            else:
                                eigenvalue = tmp_eigenvalue
                    eigenvalues.append(eigenvalue)
                    eigenvectors.append(v)
                layer_eigenvalues[layer_name] = eigenvalues
                layer_eigenvectors[layer_name] = eigenvectors
            break  # Compute for encoder only
        return layer_eigenvalues, layer_eigenvectors

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

    def layer_hessian_ranking(self, layer_eigenvectors, layer_eigenvalues=None, k=1):
        """
        Use the top eigenvalues and eigenvectors of the hessian to rank the
        parameters based on sensitivity to bit flips. To do so, we:
            1. Retrieve the top eigenvalues and eigenvectors of the Hessian
            2. Compute sensitivity ranking by taking the inner product of the
               top k eigenvectors with the parameters and then we are left with
               k scalars which we can use to rank the parameters. In this sense,
               the parameters that get scaled the most by the k-th eigenvector
               are associated with rank k. Ex: paraemeters X, Y, and Z are
               scaled the most by the top 1 eigenvector, so those parameters
               are ranked the top 1 most sensitive to bit flips.
        NOTE: We can try different combinations of how to balance the
        "eigenvalue ranking" and the "importance-in-each-eigenvector" scores and
        give a design. Here we are restricted to just one dimension of the
        eigenvector, so there is some freedom to design it.
        """
        # Compute sensitivity ranking
        # TODO: Provide bit-level ranking based on qkeras layer quantizer info
        # and log2
        sensitivity_ranking = {}
        for sl_i in self.layer_indices:
            super_layer = self.model.layers[sl_i]
            for l_i in self.get_layers_with_trainable_params(super_layer):
                layer_name = self.model.layers[sl_i].layers[l_i].name
                print(f"Ranking by sensitivity for layer {layer_name}")
                # Compute ranking for kth eigenvector
                combined_eigenvector_score = np.zeros(
                    layer_eigenvectors[layer_name][0][0].numpy().size
                )
                for i in range(k):
                    # Get eigenvector for weights only (ignore biases)
                    curr_eigenvector = layer_eigenvectors[layer_name][i][0]
                    curr_eigenvector = curr_eigenvector.numpy()
                    curr_eigenvector = curr_eigenvector.flatten()
                    if layer_eigenvalues:
                        curr_eigenvalue = layer_eigenvalues[layer_name][i].numpy()
                        print(f"{i}th eigenvalue: {curr_eigenvalue}")
                        curr_eigenvector = curr_eigenvector * curr_eigenvalue
                    params = (
                        self.model.layers[sl_i]
                        .layers[l_i]
                        .trainable_variables[0]
                        .numpy()
                    )
                    params = params.flatten()
                    # Compute dot product of eigenvector with this layer's
                    # params to get overall ranking and then sort params based
                    # on eigenvector values to get ranking within the overall
                    # ranking
                    scalar_rank = np.dot(curr_eigenvector, params)
                    combined_eigenvector_score += np.abs(scalar_rank * curr_eigenvector)
                param_ranking = np.flip(np.argsort(np.abs(combined_eigenvector_score)))
                param_rank_score = combined_eigenvector_score[param_ranking]
                print(f"parameter_ranking: {param_ranking[:10]}")
                sensitivity_ranking[layer_name] = [
                    (param_ranking[i], param_rank_score[i])
                    for i in range(len(param_ranking))
                ]
            break  # Compute for encoder only
        return sensitivity_ranking

    def do_sum_hessian_rank(self, params, eigenvectors, eigenvalues, k):
        """
        Given flattened list of parameters, list of eigenvectors, and list of
        eigenvalues, compute the eigenvector/value scores.

        Combine the weight eigenvectors into a single vector for model-wide
        parameter sensitivity ranking using weighted sum strategy.
        Return a list of eigenvector/eigenvalue
        scores, one score for each parameter.
        Curren method: weighted sum of eigenvectors
        """
        combined_eigenvector_score = np.zeros(params.size)
        for i in range(k):
            combined_eigenvector = []
            for j in range(0, len(eigenvectors[i]), 2):
                # Go every 2 to ignore biases
                ev = eigenvectors[i][j].numpy()  # Weight eigenvector
                combined_eigenvector.extend(ev.flatten())
            combined_eigenvector = np.array(combined_eigenvector)
            if eigenvalues:
                curr_eigenvalue = eigenvalues[i].numpy()
                combined_eigenvector = combined_eigenvector * curr_eigenvalue
            # scalar_rank = np.dot(combined_eigenvector, params)
            scalar_rank = 1
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
        # I need the QKeras quantizer to get the proper fixed point bit
        # representation
        pass

    def hessian_ranking(self, eigenvectors, eigenvalues=None, k=1, strategy="sum"):
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
                params.append(  # Weights only
                    self.model.layers[sl_i].layers[l_i].trainable_variables[0].numpy()
                )
            break  # Compute for encoder only
        # Flatten and concatenate all eigenvectors into one list
        params = np.concatenate(params, axis=None)
        params = params.flatten()
        print(f"params shape: {params.shape}")
        if strategy == "sum":
            eigenvector_rank = self.do_sum_hessian_rank(
                params, eigenvectors, eigenvalues, k
            )
        elif strategy == "max":
            eigenvector_rank = self.do_max_hessian_rank(
                params, eigenvectors, eigenvalues, k
            )
        # conv_rank = eigenvector_rank[: 72]
        # dense_rank = eigenvector_rank[72:]
        # conv_ranking = np.flip(np.argsort(np.abs(conv_rank)))
        # dense_ranking = np.flip(np.argsort(np.abs(dense_rank)))
        # print(f"conv_ranking: {conv_ranking[:10]}")
        # print(f"dense_ranking: {dense_ranking[:10]}")
        param_ranking = np.flip(np.argsort(np.abs(eigenvector_rank)))
        param_scores = eigenvector_rank[param_ranking]
        print(f"parameter_ranking: {param_ranking[:15]}")
        return param_ranking, param_scores

    def layer_gradient_ranking(self):
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
