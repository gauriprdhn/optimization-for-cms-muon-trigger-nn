from keras.models import Model
from keras.layers import (Dense,
                          BatchNormalization,
                          Input,
                          Activation)
from keras import regularizers
from custom_dense_layer import MaskedDense

class CustomModel:
    """
    Keras Model
    """
    @staticmethod
    def build(input_dim,
              output_dim,
              masks,
              nodes1 = 20,
              nodes2 = 15,
              nodes3 = 10,
              layerActivation = "tanh",
              finalActivation = "linear",
              eps = 1e-4,
              momentum = 0.9,
              l1_reg = 0.0,
              l2_reg = 0.0,
              initializer = "glorot_uniform"):
        # set regularization per layer ONLY if l1 or l2 is not 0.0
        if l1_reg == 0 and l2_reg == 0:
            regularizer = None
        else:
            regularizer = regularizers.L1L2(l1=l1_reg, l2=l2_reg)
        # input pre-processing layers
        inputs = Input(shape = (input_dim,),name = "input_layer")
        x = BatchNormalization(epsilon = eps, momentum  = momentum, name="batchnorm-1")(inputs)
        # Hidden Masked Dense Unit 1
        x = MaskedDense(num_outputs = nodes1, mask = masks[0],kernel_regularizer=regularizer,
                        kernel_initializer = initializer, name = "masked_dense-1")(x)
        x = BatchNormalization(epsilon = eps, momentum  = momentum, name = "batchnorm-2")(x)
        x = Activation(activation = layerActivation, name = "layer_activation-1")(x)
        if nodes2:
            # Hidden Masked Dense Unit 2
            x = MaskedDense(num_outputs = nodes2, mask = masks[1],kernel_regularizer=regularizer,
                            kernel_initializer = initializer, name = "masked_dense-2")(x)
            x = BatchNormalization(epsilon = eps, momentum  = momentum, name = "batchnorm-3")(x)
            x = Activation(activation = layerActivation, name = "layer_activation-2")(x)
            if nodes3:
                # Hidden Masked Dense Unit 3
                x = MaskedDense(num_outputs = nodes3, mask = masks[2],kernel_regularizer=regularizer,
                                kernel_initializer = initializer, name = "masked_dense-3")(x)
                x = BatchNormalization(epsilon = eps, momentum  = momentum, name = "batchnorm-4")(x)
                x = Activation(activation = layerActivation, name = "layer_activation-3")(x)
        # Output layer
        outputs = Dense(units = output_dim, activation = finalActivation, kernel_initializer = initializer,
                        use_bias = True, name = "output_dense")(x)
        model = Model(inputs=inputs, outputs=[outputs])
        return model