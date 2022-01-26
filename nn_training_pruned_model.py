import keras
import numpy as np
from numpy import ndarray

from nn_evaluate import huber_loss
from nn_build_prunable_model import CustomModel

def generate_pruning_mask (k_weights ,
                           k_sparsity ,
                           pruning_type="unstructured") :
    """
    Takes in matrix of kernel weights (for a dense layer) and returns the pruning mask for it
    :param k_weights: 2D matrix of the weights
    :param k_sparsity: percentage of weights to set to 0
    :param pruning_type: can be "unstructured" or "structured". In case of latter, entire channel/unit is pruned.
    :return: sparse matrix with same shape as the original kernel weight matrix
    """
    # Copy the kernel weights and get ranked indices of the abs
    if pruning_type == "unstructured" :
        kernel_weights_mask = np.ones (shape=k_weights.shape)
        ind = np.unravel_index (
            np.argsort (
                np.abs (k_weights) ,
                axis=None) ,
            k_weights.shape)

        # Number of indexes to set to 0
        cutoff = int (len (ind [0]) * k_sparsity)
        # The indexes in the 2D kernel weight matrix to set to 0
        sparse_cutoff_inds = (ind [0] [0 :cutoff] , ind [1] [0 :cutoff])
        kernel_weights_mask [sparse_cutoff_inds] = 0.
        return kernel_weights_mask

    else :

        # Copy the kernel weights and get ranked indeces of the
        # column-wise L2 Norms
        kernel_weights_mask: ndarray = np.ones (shape=k_weights.shape)
        ind = np.argsort (np.linalg.norm (k_weights , axis=0))

        # Number of indexes to set to 0
        cutoff = int (len (ind) * k_sparsity)
        # The indexes in the 2D kernel weight matrix to set to 0
        sparse_cutoff_inds = ind [0 :cutoff]
        kernel_weights_mask [: , sparse_cutoff_inds] = 0.

        return kernel_weights_mask


def generate_layer_masks(init_model,
                         k_sparsity=0.1,
                         pruning_type = "unstructured"):
    """
    Takes in a model made of dense layers and prunes the weights
    :param init_model: Keras model
    :param k_sparsity: target sparsity of the model
    :param pruning_type: can be "unstructured" or "structured". In case of latter, entire channel/unit is pruned.
    :return: List of mask for each of dense layers
    """
    # Getting a list of the names of each component (w + b) of each layer
    names = [weight.name for layer in init_model.layers for weight in layer.weights]
    # Getting the list of the weights for weight of each layer
    weights = init_model.get_weights ()

    # Initializing list that will contain the new masks
    newMaskList = []

    # Iterate over all dense layers but the final layer
    for i in range (0, len (weights) - 2, 1):
        if len (weights [i].shape) == 2:
            mask = generate_pruning_mask (weights [i],
                                          k_sparsity,
                                          pruning_type = pruning_type)
            newMaskList.append (mask)
    return newMaskList


def create_sparse_model(model,
                        input_dim,
                        output_dim,
                        k_sparsity=0.1,
                        bn_epsilon=1e-4,
                        bn_momentum=0.9,
                        l1_reg=0.0,
                        l2_reg=0.0,
                        kernel_initializer="glorot_uniform",
                        pruning_type="unstructured",
                        optimizer=None):
    """
    Builds a sparse version of the input model.
    :param model: Keras Model object for which pruning will be implemented
    :param input_dim: The total features that will be input to the model
    :param output_dim: The total number of output variables expected
    :param k_sparsity: Pruning Fraction for the model
    :param bn_epsilon:  Small float added to variance to avoid dividing by zero in batchnorm computation
    :param bn_momentum: Momentum for the batchnorm layer
    :param l1_reg: L1 regularization term
    :param l2_reg: L2 regularization term
    :param kernel_initializer: initialization scheme for the dense layers
    :param pruning_type: can be "unstructured" or "structured". In case of latter, entire channel/unit is pruned.
    :param optimizer: optimization algorithm for the training.
    :return: Compiled Keras Model Object
    """
    if k_sparsity >= 1.0 or k_sparsity <= 0.0:
        raise ValueError ("Sparsity can only be within the range of (0.0,1.0) endpoint values not included.")

    # generate the masks for pruning
    masksList = generate_layer_masks (init_model=model,
                                      k_sparsity=k_sparsity,
                                      pruning_type = pruning_type)
    sparse_model = CustomModel.build (input_dim=input_dim,
                                      output_dim=output_dim,
                                      masks=masksList,
                                      eps=bn_epsilon,
                                      momentum=bn_momentum,
                                      l1_reg=l1_reg,
                                      l2_reg=l2_reg,
                                      initializer=kernel_initializer)
    # Load the existing model's weights to the custom model's kernels
    # MaskedDense layer's set_weights() ONLY updates the kernel not the mask
    init_weights = []
    for layer in model.layers:
        init_weights.append (layer.get_weights ())
    # If input model is a custom pruned model, the top layer would be an InputLayer
    if isinstance (model.layers [0], keras.layers.InputLayer):
        for i in range (1, len (sparse_model.layers)):
            sparse_model.layers [i].set_weights (init_weights [i])
    else:
        for i in range (1, len (sparse_model.layers)):
            sparse_model.layers [i].set_weights (init_weights [i - 1])

            # compile the sparse model
    sparse_model.compile (optimizer=optimizer,
                          loss=huber_loss,
                          metrics=['acc', 'mse', 'mae'])
    sparse_model.summary ()

    return sparse_model