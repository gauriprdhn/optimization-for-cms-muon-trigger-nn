from nn_globals import *
from qkeras import utils
import pickle

def save_qmodel( model_for_export ,
                 bits: int = 8 ,
                 int_bits: int = 0 ,
                 sparsity: int = 0 ,
                 l1_reg_val: float = 0.0 ,
                 l2_reg_val: float = 0.0 ,
                 epochs: int = 100,
                 batch_size: int = 1,
                 quantization_type: str = "AS" ,
                 additional_prefix: str = None ) :
    if additional_prefix is not None:
        qkeras_file = MODELFILEPATH + "/{}_quantized_model_{}_bits_{}_int_{}_sparsity_{}_l1_reg_{}_l2_reg_{}_epochs_{}_batch_{}.h5".format (
            additional_prefix ,
            quantization_type ,
            bits ,
            int_bits ,
            sparsity ,
            l1_reg_val ,
            l2_reg_val,
            epochs,
            batch_size)
        weights_file = MODELFILEPATH + "/{}_quantized_model_{}_bits_{}_int_{}_sparsity_{}_l1_reg_{}_l2_reg_{}_epochs_{}_batch_{}_weights.h5".format (
            additional_prefix ,
            quantization_type ,
            bits ,
            int_bits ,
            sparsity ,
            l1_reg_val ,
            l2_reg_val,
            epochs,
            batch_size)
    else:
        qkeras_file = MODELFILEPATH + "/quantized_model_{}_bits_{}_int_{}_sparsity_{}_l1_reg_{}_l2_reg_{}_epochs_{}_batch_{}.h5".format (
            quantization_type ,
            bits ,
            int_bits ,
            sparsity ,
            l1_reg_val ,
            l2_reg_val,
            epochs,
            batch_size)
        weights_file = MODELFILEPATH + "/quantized_model_{}_bits_{}_int_{}_sparsity_{}_l1_reg_{}_l2_reg_{}_epochs_{}_batch_{}_weights.h5".format (
            quantization_type ,
            bits ,
            int_bits ,
            sparsity ,
            l1_reg_val ,
            l2_reg_val,
            epochs,
            batch_size)
    model_for_export.save ( filepath = qkeras_file , include_optimizer = False )
    utils.model_save_quantized_weights ( model_for_export , weights_file )

def save_training_history( training_history_dict ,
                           bits: int = 8 ,
                           int_bits: int = 0 ,
                           sparsity: int = 0,
                           additional_prefix: str = None) :
    if additional_prefix is not None:
        filename = TRAINING_LOGS + "/{}_quantizedTrainingHistoryDict_bits_{}_ints_{}_sparsity_{}".format (
                                                                                            additional_prefix,
                                                                                            bits ,
                                                                                            int_bits ,
                                                                                            sparsity )
    else:
        filename = TRAINING_LOGS + "/quantizedTrainingHistoryDict_bits_{}_ints_{}_sparsity_{}".format (
                                                                                            bits ,
                                                                                            int_bits ,
                                                                                            sparsity )
    with open ( filename , 'wb' ) as file_pi :
        pickle.dump ( training_history_dict.history , file_pi )


def set_weights_from_baseline(baseline, qmodel):
    """
    Helper function to clone weights from baseline to the quantized model.
    :param baseline: Keras Model for baseline
    :param qmodel: Keras Model for quantized version of the baseline
    :return: Quantized model with the weights for Dense and BN layers set same as in the baseline
    """
    layer_config = []
    for layer in baseline.layers:
        layer_config.append(layer.get_weights())
    if isinstance (baseline.layers [0], keras.layers.InputLayer):
        for i in range(1,len(qmodel.layers)):
            layer = qmodel.layers[i]
            if "act" in layer.name:
                i+=1
            else:
                layer.set_weights(layer_config[i])
    else:
        for i in range(1,len(qmodel.layers)-1):
            layer = qmodel.layers[i]
            if "act" in layer.name:
                i+=1
            else:
                layer.set_weights(layer_config[i-1])
    return qmodel
