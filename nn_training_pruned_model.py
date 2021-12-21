import keras
import numpy as np
from nn_evaluate import huber_loss
from nn_training import train_model
from nn_build_pruning_model import CustomModel

def generate_pruning_mask(k_weights, k_sparsity):
    """
    Takes in matrix of kernel weights (for a dense
      layer) and returns the pruning mask for it
    Args:
      k_weights: 2D matrix of the weights
      k_sparsity: percentage of weights to set to 0
    Returns:
      kernel_weights_mask: sparse matrix with same shape as the original
        kernel weight matrix
    """
    # Copy the kernel weights and get ranked indices of the abs
    kernel_weights_mask = np.ones (shape=k_weights.shape)
    ind = np.unravel_index (
        np.argsort (
            np.abs (k_weights),
            axis=None),
        k_weights.shape)

    # Number of indexes to set to 0
    cutoff = int (len (ind [0]) * k_sparsity)
    # The indexes in the 2D kernel weight matrix to set to 0
    sparse_cutoff_inds = (ind [0] [0:cutoff], ind [1] [0:cutoff])
    kernel_weights_mask [sparse_cutoff_inds] = 0.
    return kernel_weights_mask


def generate_layer_masks(init_model, k_sparsity=0.5):
    """
    Takes in a model made of dense layers and prunes the weights
    Args:
      model: Keras model
      k_sparsity: target sparsity of the model
    Returns:
      List of mask for each of dense layers
    """
    # Getting a list of the names of each component (w + b) of each layer
    names = [weight.name for layer in init_model.layers for weight in layer.weights]
    # Getting the list of the weights for weight of each layer
    weights = init_model.get_weights ()

    # Initializing list that will contain the new sparse weights
    newMaskList = []

    # Iterate over all dense layers but the final layer
    for i in range (0, len (weights) - 2, 1):
        if len (weights [i].shape) == 2:
            mask = generate_pruning_mask (weights [i], k_sparsity)
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
                        optimizer=None):
    """
    Builds and Compiles a sparse version of the input model.
    Args:
        model: Keras Model object for which pruning will be implemented
        input_dim: The total features that will be input to the model
        output_dim: The total number of output variables expected
        k_sparsity: Pruning Fraction for the model
        bn_epsilon: Small float added to variance to avoid dividing by zero
        bn_momentum: Momentum for the moving average
        l1_reg: L1 regularization term
        l2_reg: L2 regularization term
        kernel_initializer: initialization scheme for the dense layers
        optimizer: optimization algorithm for the training.

    Returns: Compiled Keras Model Object

    """
    if k_sparsity >= 1.0 or k_sparsity <= 0.0:
        raise ValueError ("Sparsity can only be within the range of (0.0,1.0) endpoint values not included.")

    # generate the masks for pruning
    masksList = generate_layer_masks (init_model=model, k_sparsity=k_sparsity)
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


def train_sparse_model(sparse_model,
                       x,
                       y,
                       dxy,
                       retrain_epochs=10,
                       batch_size=1000,
                       callbacks=None,
                       verbose=False,
                       validation_split=0.1):
    """
    Calls the train_model for the sparse model input and returns the results of the training.
    Args:
        sparse_model: Pruned Keras Model class object
        x: Feature array for training
        y: Truth values for 1st regression variable, momentum
        dxy: Truth values for 2nd regression variable, displacement
        retrain_epochs: Total epochs for fine-tuning
        batch_size: Batch size for fine-tuning
        callbacks: Training callbacks list
        verbose: True or False, if True, the training log will be displaced.
        validation_split: The fraction of split for the training-validation data

    Returns: re-trained sparse_model, and associated training history

    """
    history = None
    sparse_model, history = train_model (sparse_model,
                                         x,
                                         np.column_stack((y, dxy)),
                                         save_model=False,
                                         epochs=retrain_epochs,
                                         batch_size=batch_size,
                                         callbacks=callbacks,
                                         validation_split=validation_split,
                                         verbose=verbose)
    return sparse_model, history
