from keras.layers import Input, BatchNormalization, Activation
from qkeras.qlayers import QDense, QActivation
from keras.regularizers import L1L2
from nn_evaluate import huber_loss
from keras.optimizer_v2.adam import Adam
from keras.models import Model
from nn_quantization_module_support import set_weights_from_baseline

def create_model_quantized_randomized (
        bits ,
        int_bits ,
        nvariables ,
        lr = 0.001 ,
        clipnorm = 10. ,
        initializer = "he_uniform" ,
        nodes1 = 64 ,
        nodes2 = 32 ,
        nodes3 = 16 ,
        outnodes = 2 ,
        l1_reg = 0.0 ,
        l2_reg = 0.0 ,
        eps = 1e-4,
        momentum = 0.9,
        quantization_type = "S" ) :  # 'S' for symmetric, 'A' for asymmetric, 'AS' for Hybrid

    # intializing the quantizers and layer variables
    if quantization_type == "S" :
        kernelQuantizer = "quantized_bits(bits={},integer={},symmetric = True,use_stochastic_rounding=True, qnoise_factor = 1.0, alpha = 1.0)".format (
            bits , int_bits )
        activationQuantizer = "quantized_tanh(bits ={}, symmetric = True, use_stochastic_rounding=True, use_real_tanh =True)".format (
            bits )
    if quantization_type == "A" :
        kernelQuantizer = "quantized_bits(bits={},integer={},symmetric = False,use_stochastic_rounding=True, qnoise_factor = 1.0, alpha = 1.0)".format (
            bits , int_bits )
        activationQuantizer = "quantized_tanh(bits ={}, symmetric = False, use_stochastic_rounding=True, use_real_tanh =True)".format (
            bits )
    else :
        firstLayerQuantizer = "quantized_bits(bits={},integer={},symmetric = False,use_stochastic_rounding=True, qnoise_factor = 1.0, alpha = 1.0)".format (
            bits , int_bits )
        kernelQuantizer = "quantized_bits(bits={},integer={},symmetric = True,use_stochastic_rounding=True, qnoise_factor = 1.0, alpha = 1.0)".format (
            bits , int_bits )
        activationQuantizer = "quantized_tanh(bits ={}, symmetric = True, use_stochastic_rounding=True, use_real_tanh =True)".format (
            bits )

    regularizer = L1L2 ( l1 = l1_reg , l2 = l2_reg )
    x = x_in = Input ( (nvariables ,) )
    x = BatchNormalization ( epsilon = eps , momentum = momentum , name = "bn-input" ) ( x )
    if quantization_type == "AS" :
        x = QDense ( nodes1 ,
                     kernel_quantizer = firstLayerQuantizer ,
                     kernel_initializer = initializer ,
                     use_bias = False ,
                     kernel_regularizer = regularizer ,
                     name = "hidden-dense-1" ) ( x )
    else :
        x = QDense ( nodes1 ,
                     kernel_quantizer = kernelQuantizer ,
                     kernel_initializer = initializer ,
                     use_bias = False ,
                     kernel_regularizer = regularizer ,
                     name = "hidden-dense-1" ) ( x )
    x = BatchNormalization ( epsilon = eps , momentum = momentum , name = "bn-1" ) ( x )
    x = QActivation ( activation = activationQuantizer ,
                       name = "act_1" ) ( x )

    if nodes2 :

        x = QDense ( nodes2 ,
                     kernel_quantizer = kernelQuantizer ,
                     kernel_initializer = initializer ,
                     use_bias = False ,
                     kernel_regularizer = regularizer ,
                     name = "hidden-dense-2" ) ( x )
        x = BatchNormalization ( epsilon = eps , momentum = momentum , name = "bn-2" ) ( x )
        x = QActivation ( activation = activationQuantizer ,
                          name = "act_2" ) ( x )
        if nodes3 :
            x = QDense ( nodes3 ,
                         kernel_quantizer = kernelQuantizer ,
                         kernel_initializer = initializer ,
                         kernel_regularizer = regularizer ,
                         use_bias = False ,
                         name = "hidden-dense-3" ) ( x )
            x = BatchNormalization ( epsilon = eps , momentum = momentum , name = "bn-3" ) ( x )
            x = QActivation ( activation = activationQuantizer ,
                              name = "act_3" ) ( x )

    x = QDense ( outnodes ,
                 kernel_quantizer = kernelQuantizer ,
                 kernel_initializer = initializer ,
                 use_bias = True ,
                 name = "dense-output" ) ( x )
    x = Activation ( "linear" ) ( x )

    model = Model ( inputs = x_in , outputs = x , name = "quantized-model" )

    adam = Adam ( lr = lr ,
                  clipnorm = clipnorm )
    model.compile ( optimizer = adam ,
                    loss = huber_loss ,
                    metrics = [ 'acc' , 'mse' , 'mae' ] )

    model.summary ()
    return model


def create_model_quantized(baseline,
                           bits,
                           int_bits,
                           nvariables,
                           lr=0.001,
                           clipnorm=10.,
                           initializer = "he_uniform",
                           nodes1=64,
                           nodes2=32,
                           nodes3=16,
                           outnodes=2,
                           l1_reg = 0.0,
                           l2_reg = 0.0,
                           eps = 1e-4,
                           momentum = 0.9,
                           quantization_type = "S"):

    # intializing the quantizers and layer variables
    if quantization_type == "S" :
        kernelQuantizer = "quantized_bits(bits={},integer={},symmetric = True,use_stochastic_rounding=True, qnoise_factor = 1.0, alpha = 1.0)".format (
            bits , int_bits )
        activationQuantizer = "quantized_tanh(bits ={}, symmetric = False, use_stochastic_rounding=True, use_real_tanh =True)".format (
            bits )
    if quantization_type == "A" :
        kernelQuantizer = "quantized_bits(bits={},integer={},symmetric = False,use_stochastic_rounding=True, qnoise_factor = 1.0, alpha = 1.0)".format (
            bits , int_bits )
        activationQuantizer = "quantized_tanh(bits ={}, symmetric = False, use_stochastic_rounding=True, use_real_tanh =True)".format (
            bits )
    else :
        firstLayerQuantizer = "quantized_bits(bits={},integer={},symmetric = False,use_stochastic_rounding=True, qnoise_factor = 1.0, alpha = 1.0)".format (
            bits , int_bits )
        kernelQuantizer = "quantized_bits(bits={},integer={},symmetric = True,use_stochastic_rounding=True, qnoise_factor = 1.0, alpha = 1.0)".format (
            bits , int_bits )
        activationQuantizer = "quantized_tanh(bits ={}, symmetric = True, use_stochastic_rounding=True, use_real_tanh =True)".format (
            bits )

    regularizer = L1L2 ( l1 = l1_reg , l2 = l2_reg )

    x = x_in = Input ( (nvariables ,) )
    x = BatchNormalization ( epsilon = eps , momentum = momentum , name = "bn-input" ) ( x )
    if quantization_type == "AS" :
        x = QDense ( nodes1 ,
                     kernel_quantizer = firstLayerQuantizer ,
                     kernel_initializer = initializer ,
                     use_bias = False ,
                     kernel_regularizer = regularizer ,
                     name = "hidden-dense-1" ) ( x )
    else :
        x = QDense ( nodes1 ,
                     kernel_quantizer = kernelQuantizer ,
                     kernel_initializer = initializer ,
                     use_bias = False ,
                     kernel_regularizer = regularizer ,
                     name = "hidden-dense-1" ) ( x )
    x = BatchNormalization ( epsilon = eps , momentum = momentum , name = "bn-1" ) ( x )
    x = QActivation ( activation = activationQuantizer ,
                       name = "act_1" ) ( x )

    if nodes2:

        x = QDense(nodes2,
                   kernel_quantizer = kernelQuantizer,
                   kernel_initializer=initializer,
                   use_bias = False,
                   kernel_regularizer = regularizer,
                   name="hidden-dense-2")(x)
        x = BatchNormalization(epsilon = eps, momentum  = momentum, name = "bn-2")(x)
        x = QActivation(activation = activationQuantizer,
                    name="act_2")(x)
        if nodes3:

            x = QDense(nodes3,
                       kernel_quantizer = kernelQuantizer,
                       kernel_initializer=initializer,
                       kernel_regularizer = regularizer,
                       use_bias = False,
                       name="hidden-dense-3")(x)
            x = BatchNormalization(epsilon = eps, momentum  = momentum, name = "bn-3")(x)
            x = QActivation(activation = activationQuantizer,
                            name="act_3")(x)

    x = QDense(outnodes,
                kernel_quantizer = kernelQuantizer,
                kernel_initializer = initializer,
                use_bias = True,
                name="dense-output")(x)
    x = Activation("linear")(x)

    model = Model(inputs=x_in, outputs=x,name="quantized-model")

    adam = Adam(lr=lr,
                clipnorm=clipnorm)
    model.compile(optimizer=adam,
                  loss=huber_loss,
                  metrics=['acc','mse','mae'])
    # transfer the pre-trained model's final weights to the quantized model
    model = set_weights_from_baseline(baseline, model)
    model.summary()
    return model

