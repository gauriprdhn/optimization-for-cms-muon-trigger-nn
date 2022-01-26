import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import  model_from_json
from tensorflow.keras.utils import custom_object_scope
from qkeras import utils

eps = 1e-7
my_cmap = plt.cm.viridis
my_cmap.set_under('w',1)
my_palette = ("#377eb8", "#e41a1c", "#984ea3", "#ff7f00", "#4daf4a")

def saving_model(model,
                 filepath:str ='',
                 model_filename:str ='model'):
    """
    Helper function to save the trained pruned models in the following format:
        Config -> JSON file format
        Weights -> H5PY file format
    :param model: Keras Model object
    :param filepath: path to store the file in the directory
    :param model_filename: name for the model
    :return: None
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open(filepath + "/" + model_filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filepath + "/" + model_filename + "_weights.h5")
    print("Saved model to disk")

# def loading_trained_model(filepath='',
#                          model_filename='model',
#                          custom_objects=None):
#     """
#     Helper function to load the trained models USING THE CUSTOM LAYER from the models directory.
#     :param filepath: Path to the folder/ directory where the model is stored
#     :param model_filename: Name of the file.
#     :param custom_objects: List of custom layer objects used in the model so that they can be loaded using the config.
#     :return: Keras Model loaded from the file.
#     """
#     try:
#         model_path = filepath + "/" + model_filename + ".json"
#         print(model_path)
#         model_weights_path = filepath + "/" + model_filename + "_weights.h5"
#         print(model_weights_path)
#         with open(model_path, 'r') as f:
#             loaded_model_json = f.read()
#         f.close()
#         if custom_objects:
#             with custom_object_scope(custom_objects):
#                 loaded_model = model_from_json(loaded_model_json)
#             # load weights into new model
#             loaded_model.load_weights(model_weights_path)
#         else:
#             loaded_model = model_from_json (loaded_model_json)
#             # load weights into new model
#             loaded_model.load_weights (model_weights_path)
#         print("Loaded model from disk")
#
#         return loaded_model
#
#     except:
#         print("ERROR: The model doesn't exist at the address provided.")

def loading_trained_model ( filepath = '' ,
                            model_filename = 'model' ,
                            custom_objects = None ,
                            is_quantized = False ) :
    """
    Helper function to load the trained models USING THE CUSTOM LAYER from the models directory.
    :param filepath: Path to the folder/ directory where the model is stored
    :param model_filename: Name of the file.
    :param custom_objects: List of custom layer objects used in the model so that they can be loaded using the config.
    :param is_quantized: Enables use of qkeras library to load the quantized model
    :return: Model object loaded from the file.
    """
    if is_quantized :
        qkeras_file = filepath + "/" + model_filename + ".h5"
        qkeras_weights_file = filepath + "/" + model_filename + "_weights.h5"
        if custom_objects :
            loaded_qmodel = utils.load_qmodel ( filepath = qkeras_file )
            loaded_qmodel.load_weights ( qkeras_weights_file )
        else :
            loaded_qmodel = utils.load_qmodel ( filepath = qkeras_file , custom_objects = custom_objects )
            loaded_qmodel.load_weights ( qkeras_weights_file )

        print ( "Loaded quantized model from disk" )
        return loaded_qmodel
    else :
        model_path = filepath + "/" + model_filename + ".json"
        print ( model_path )
        model_weights_path = filepath + "/" + model_filename + "_weights.h5"
        print ( model_weights_path )
        with open ( model_path , 'r' ) as f :
            loaded_model_json = f.read ( )
        f.close ( )
        if custom_objects :
            with custom_object_scope ( custom_objects ) :
                loaded_model = model_from_json ( loaded_model_json )
            # load weights into new model
            loaded_model.load_weights ( model_weights_path )
        else :
            loaded_model = model_from_json ( loaded_model_json )
            # load weights into new model
            loaded_model.load_weights ( model_weights_path )

        print ( "Loaded model from disk" )
        return loaded_model

def __get_weights__(model, layer_type="dense"):
    """
    Helper function to get all the weights associated with selected layers from the keras model
    Args:
        model: Keras Model
        layer_type: Nature of the layer such as "dense" or "batchnorm" for which weights are to be returned.

    Returns:
        weights: Dictionary mapping layer names to a list of weights.
    """
    weights = dict()
    if layer_type == "dense":
        for layer in model.layers:
            if layer_type in layer.name:
                weights[layer.name] = layer.get_weights()
    elif layer_type == "all":
        for layer in model.layers:
            weights[layer.name] = layer.get_weights()
    return weights

def __layer_statistics__(layer_weights):
    """
    Helper function to print statistical properties, such as mean, median, etc. for the layer weights.
    Args:
        layer_weights: Input numpy array of the weights for a layer.

    Returns: None

    """
    arr = np.ndarray.flatten(layer_weights)
    stats = {"min": min(arr),
             "max": max(arr),
             "mean": np.mean(arr),
             "median": np.median(arr),
             "range": abs(max(arr) - min(arr)),
             "zeros": (arr == 0.).sum(),
             "negatives": (arr < 0.).sum(),
             "positives": (arr > 0.).sum(),
             "var": np.std(arr) ** 2
             }
    print(pd.DataFrame.from_dict(stats, orient="index", columns=["measure"]))

