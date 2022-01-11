from qkeras.qlayers import QDense,QActivation
from custom_Qdense_layer import CustomQDense
from keras import regularizers
from keras.layers import Input, BatchNormalization, Activation
from keras.models import Model

class CustomQModel :
    @staticmethod
    def build ( input_dim ,
                output_dim ,
                bits ,
                int_bits ,
                masks ,
                nodes1: int = 20 ,
                nodes2: int = 15 ,
                nodes3: int = 10 ,
                eps: float = 1e-4 ,
                momentum: float = 0.9 ,
                l1_reg: float = 0.0 ,
                l2_reg: float = 0.0 ,
                finalActivation="linear" ,
                initializer="he_uniform" ) :
        # set regularization per layer ONLY if l1 or l2 is not 0.0
        if l1_reg == 0 and l2_reg == 0 :
            regularizer = None
        else :
            regularizer = regularizers.L1L2 (l1 = l1_reg , l2 = l2_reg)

        # define quantizers
        kernelQuantizer = "quantized_bits(bits={},integer={},symmetric = True,use_stochastic_rounding=True, qnoise_factor = 1.0, alpha = 1.0)".format (
            bits , int_bits)
        activationQuantizer = "quantized_tanh(bits ={}, symmetric = True, use_stochastic_rounding=True, use_real_tanh =True)".format (
            bits)

        # input pre-processing layers
        inputs = Input (shape = (input_dim ,) , name = "input_layer")
        x = BatchNormalization (epsilon = eps , momentum = momentum , name = "bn-in") (inputs)

        x = CustomQDense (units = nodes1 ,
                          mask = masks [0] ,
                          kernel_regularizer = regularizer ,
                          kernel_initializer = initializer ,
                          kernel_quantizer = "quantized_bits(bits={},integer={},symmetric = False,use_stochastic_rounding=True, qnoise_factor = 1.0, alpha = 1.0)".format (
                              bits , int_bits) ,
                          name = "masked-qdense-1") (x)
        x = BatchNormalization (epsilon = eps , momentum = momentum , name = "bn-1") (x)
        x = QActivation (activation = activationQuantizer ,
                         name = "act_1") (x)

        x = CustomQDense (units = nodes2 ,
                          mask = masks [1] ,
                          kernel_regularizer = regularizer ,
                          kernel_initializer = initializer ,
                          kernel_quantizer = kernelQuantizer ,
                          name = "masked-qdense-2") (x)
        x = BatchNormalization (epsilon = eps , momentum = momentum , name = "bn-2") (x)
        x = QActivation (activation = activationQuantizer ,
                         name = "act_2") (x)

        x = CustomQDense (units = nodes3 ,
                          mask = masks [2] ,
                          kernel_regularizer = regularizer ,
                          kernel_initializer = initializer ,
                          kernel_quantizer = kernelQuantizer ,
                          name = "masked-qdense-3") (x)
        x = BatchNormalization (epsilon = eps , momentum = momentum , name = "bn-3") (x)
        x = QActivation (activation = activationQuantizer ,
                         name = "act_3") (x)

        x = QDense (output_dim ,
                    kernel_quantizer = kernelQuantizer ,
                    kernel_initializer = initializer ,
                    kernel_regularizer = regularizer ,
                    use_bias = True ,
                    name = "dense-output") (x)
        outputs = Activation (finalActivation , name = "final-activation") (x)

        model = Model (inputs = inputs ,
                       outputs = [outputs] ,
                       name = "quantized-model")
        return model