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

    def __init__(self, model, loss_fn, x, y):
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

    def hessian_vector_product(self, v, layer_idx=None):
        """
        Compute the Hessian vector product of Hv, where
        H is the Hessian of the loss function with respect to the model parameters
        v is a vector of the same size as the model parameters

        Based on: https://github.com/tensorflow/tensorflow/blob/47f0e99c1918f68daa84bd4cac1b6011b2942dac/tensorflow/python/eager/benchmarks/resnet50/hvp_test.py#L62
        """
        # Compute the gradients of the loss function with respect to the model parameters

        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                loss = self.loss_fn(self.model(self.x), self.y)
            # params = self.model.trainable_variables if layer_idx is None else self.model.layers[layer_idx].trainable_variables
            # tf.print(f"\n\n##{self.model.trainable_variables}##\n\n")
            params = (
                self.model.trainable_variables
                if layer_idx is None
                else self.model.trainable_variables[layer_idx]
            )
            grads = inner_tape.gradient(loss, params)
        # Compute the Hessian vector product
        return outer_tape.gradient(grads, params, output_gradients=v)

    def hessian_vector_product_hack_reusable_values(
        self, super_layer_idx=None, layer_idx=None
    ):
        """
        Compute the Hessian vector product of Hv, where
        H is the Hessian of the loss function with respect to the model parameters
        v is a vector of the same size as the model parameters

        Based on: https://github.com/tensorflow/tensorflow/blob/47f0e99c1918f68daa84bd4cac1b6011b2942dac/tensorflow/python/eager/benchmarks/resnet50/hvp_test.py#L62
        """
        # Compute the gradients of the loss function with respect to the model parameters

        with tf.GradientTape() as inner_tape:
            loss = self.loss_fn(self.model(self.x), self.y)

            params = (
                self.model.trainable_variables
                if layer_idx is None
                else self.model.layers[super_layer_idx]
                .layers[layer_idx]
                .trainable_variables
            )
            grads = inner_tape.gradient(loss, params)

        return (params, grads)

    def hessian_vector_product_hack(self, v, super_layer_idx=None, layer_idx=None):
        """
        Compute the Hessian vector product of Hv, where
        H is the Hessian of the loss function with respect to the model parameters
        v is a vector of the same size as the model parameters

        Based on: https://github.com/tensorflow/tensorflow/blob/47f0e99c1918f68daa84bd4cac1b6011b2942dac/tensorflow/python/eager/benchmarks/resnet50/hvp_test.py#L62
        """
        # Compute the gradients of the loss function with respect to the model parameters

        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                loss = self.loss_fn(self.model(self.x), self.y)
            # params = self.model.trainable_variables if layer_idx is None else self.model.layers[layer_idx].trainable_variables
            # tf.print(f"\n\n##{self.model.trainable_variables}##\n\n")
            params = (
                self.model.trainable_variables
                if layer_idx is None
                else self.model.layers[super_layer_idx]
                .layers[layer_idx]
                .trainable_variables
            )
            grads = inner_tape.gradient(loss, params)
        # Compute the Hessian vector product
        # tf.print(f"\n\n##{params}\n\n Grads:{grads}##\n\n")
        # tf.print(f"\n\n##{params.shape}\n{grads.shape}##\n\n")
        return outer_tape.gradient(grads, params, output_gradients=v)

    def trace(self, max_iter=100, tolerance=1e-3):
        """
        Compute the trace of the Hessian using Hutchinson's method
        max_iter: maximimum number of iterations used to compute trace
        tolerance: tolerance for convergence
        """
        layer_traces = {}
        for l_i in self.layer_indices:
            layer_name = self.model.layers[l_i].name
            # print(f"Computing trace for layer {layer_name}")
            trace_vhv = []
            trace = 0.0

            for i in range(max_iter):
                v = [
                    np.random.uniform(
                        size=self.model.layers[l_i].trainable_variables[i].shape
                    )
                    for i in range(len(self.model.layers[l_i].trainable_variables))
                ]
                # Generate Rademacher random variables
                for vi in v:
                    vi[vi < 0.5] = -1
                    vi[vi >= 0.5] = 1
                v = [tf.convert_to_tensor(vi, dtype=tf.dtypes.float32) for vi in v]
                # Compute the Hessian vector product
                hv = self.hessian_vector_product(v, layer_idx=l_i)
                # Compute the trace
                trace_vhv.append(
                    tf.reduce_sum([tf.reduce_sum(vi * hvi) for (vi, hvi) in zip(v, hv)])
                )
                if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tolerance:
                    layer_traces[layer_name] = np.mean(trace_vhv)
                else:
                    trace = np.mean(trace_vhv)
        return layer_traces

    def trace_hack(self, max_iter=100, tolerance=1e-3):
        """
        Compute the trace of the Hessian using Hutchinson's method
        max_iter: maximimum number of iterations used to compute trace
        tolerance: tolerance for convergence
        """
        layer_traces = {}
        for sl_i in self.layer_indices:
            super_layer = self.model.layers[sl_i]
            # tf.print(f"\n\n#########HessianDebug{self.get_layers_with_trainable_params(super_layer)}#########\n\n")

            for l_i in self.get_layers_with_trainable_params(super_layer):
                layer_name = self.model.layers[sl_i].layers[l_i].name
                # print(f"Computing trace for layer {layer_name}")
                trace_vhv = []
                trace = 0.0

                for i in range(max_iter):
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
                    # Generate Rademacher random variables
                    for vi in v:
                        vi[vi < 0.5] = -1
                        vi[vi >= 0.5] = 1
                    v = [tf.convert_to_tensor(vi, dtype=tf.dtypes.float32) for vi in v]
                    # Compute the Hessian vector product
                    hv = self.hessian_vector_product_hack(
                        v, super_layer_idx=sl_i, layer_idx=l_i
                    )
                    # Compute the trace
                    trace_vhv.append(
                        tf.reduce_sum(
                            [tf.reduce_sum(vi * hvi) for (vi, hvi) in zip(v, hv)]
                        )
                    )
                    if (
                        abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6)
                        < tolerance
                    ):
                        layer_traces[layer_name] = np.mean(trace_vhv)
                        break
                    else:
                        trace = np.mean(trace_vhv)
        return layer_traces

    def top_k_eigenvalues_hack(self, k=1, max_iter=100, tolerance=1e-3):
        """
        Compute the top k eigenvalues and eigenvectors of the Hessian using the
        power iteration method. The eigenvalues are sorted in descending order.
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
                    # Initialize the eigenvector
                    # v = [
                    #     np.random.uniform(
                    #         size=self.model.layers[l_i].trainable_variables[i].shape
                    #     )
                    #     for i in range(len(self.model.layers[l_i].trainable_variables))
                    # ]
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
                    v = [vi / tf.norm(vi) for vi in v]
                    for j in range(max_iter):
                        eigenvalue = None
                        # Make v orthonormal to eigenvectors
                        for ei in eigenvectors:
                            v = [
                                vi - tf.reduce_sum(vi * e) * e for (vi, e) in zip(v, ei)
                            ]
                        # Normalize the eigenvector
                        v = [vi / tf.norm(vi) for vi in v]
                        # Compute the Hessian vector product
                        hv = self.hessian_vector_product_hack(
                            v, super_layer_idx=sl_i, layer_idx=l_i
                        )
                        # Compute the eigenvalue
                        tmp_eigenvalue = tf.reduce_sum(
                            [tf.reduce_sum(vi * hvi) for (vi, hvi) in zip(v, hv)]
                        )
                        # Normalize the eigenvector
                        v = [hvi / tf.norm(hvi) for hvi in hv]
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

    def sensitivity_ranking(self, layer_eigenvectors, k=1):
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
        sensitivity_ranking = {}
        for sl_i in self.layer_indices:
            super_layer = self.model.layers[sl_i]
            for l_i in self.get_layers_with_trainable_params(super_layer):
                layer_name = self.model.layers[sl_i].layers[l_i].name
                print(f"Ranking by sensitivity for layer {layer_name}")
                # NOTE: Just using the top 1 eigenvector for now
                # Get eigenvector for weights only (ignore biases)
                curr_eigenvector = layer_eigenvectors[layer_name][0][0]
                # TODO: Compute dot product of eigenvector with this layer's params to get overall ranking and then sort params based on eigenvector values to get ranking within the overall ranking
                # print(f"parameters: {self.model.layers[sl_i].layers[l_i].trainable_variables[0]}")
                # Operate on numpy arrays now
                curr_eigenvector = curr_eigenvector.numpy()
                print(f"eigenvector shape: {curr_eigenvector.shape}")
                curr_eigenvector = curr_eigenvector.flatten()
                # curr_eigenvector = tf.reshape(curr_eigenvector, [-1]) # Flatten
                print(f"flat eigenvector shape: {curr_eigenvector.shape}")
                param_ranking = np.flip(np.argsort(np.abs(curr_eigenvector)))
                param_rank_score = curr_eigenvector[param_ranking]
                print(f"parameter_ranking: {param_ranking[:10]}")
                sensitivity_ranking[layer_name] = [
                    (param_ranking[i], param_rank_score[i])
                    for i in range(len(param_ranking))
                ]
                # ranking = tf.reduce_sum(curr_eigenvector * self.model.layers[sl_i].layers[l_i].trainable_variables, axis=1)
                # print(f"ranking shape: {ranking.shape}")
                # print(f"ranking: {ranking}")
                # Sort parameters based on eigenvector values
                # sorted_indices = tf.argsort(ranking, direction="DESCENDING")
                # print(f"sorted_indices shape: {sorted_indices.shape}")
                # print(f"sorted_indices: {sorted_indices}")
            break  # Compute for encoder only
        return sensitivity_ranking

    def gradient_ranking(self):
        """
        Rank parameters based on gradient magnitude
        """
        grad_ranking = {}
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
        return grad_ranking
