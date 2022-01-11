from keras import backend as K
from keras.engine.base_layer import Layer
from keras.engine import base_layer_utils
from keras import initializers, regularizers, activations


class MaskedDense(Layer):
    """
    Custom Keras Layer implementing weights masking for the Dense Layer to prune the layer weights using the input mask.
    Inputs:
    units -> The number of output neurons in the layer used to initialize the kernel per dimension (inputs.shape[0],units)
    mask -> A binary numpy matrix where 0 indicates the indices where the weight are to be pruned. Must have the same shape
            as the kernel (inputs.shape[0],units) to perform element-wise multiplication with the kernel.
    """

    def __init__(self,
                 num_outputs,
                 mask,
                 activation=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):

        self.num_outputs = num_outputs
        self.layer_mask = K.constant(mask, dtype=K.floatx())
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activation = activations.get(activation)
        super(MaskedDense, self).__init__(**kwargs)

    def get_config(self):

        config = {
            'num_outputs': self.num_outputs,
            'mask': self.layer_mask.numpy(),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activation': activations.serialize(self.activation)
        }
        base_config = super(MaskedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):

        last_dim = int(input_shape[-1])
        self.kernel = self.add_weight("kernel",
                                      shape=(last_dim, self.num_outputs),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      dtype=K.floatx(),
                                      trainable=True)

        super(MaskedDense, self).build(input_shape)
        self.built = True

    def call(self,
             inputs,
             *args,
             **kwargs):

        if not self.activation:
            return K.dot(inputs, self.kernel * self.layer_mask)
        else:
            outputs = K.dot(inputs, self.kernel * self.layer_mask)
            return self.activation(outputs)

    def compute_output_shape(self, input_shape):
        return [int(input_shape[-1]), self.num_outputs]

    def set_weights(self, weights):

        params = [self.kernel]

        expected_num_weights = 0
        for param in params:
            if isinstance(param, base_layer_utils.TrackableWeightHandler):
                expected_num_weights += param.num_tensors
            else:
                expected_num_weights += 1

        assert expected_num_weights == len(weights)

        weight_value_tuples = []
        weight_index = 0
        for param in params:
            # multiplication of input weights to layer mask is necessary to ensure mask is applied to initial kernels
            weight_value_tuples.append((param, weights[weight_index] * self.layer_mask.numpy()))
            weight_index += 1

        K.batch_set_value(weight_value_tuples)

    def get_weights(self):
        # return a masked kernel to ensure the next one is set
        return K.batch_get_value([self.kernel * self.layer_mask])
