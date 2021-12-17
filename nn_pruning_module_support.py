import numpy as np
import pandas as pd
from nn_globals import *
import matplotlib.pyplot as plt
from nn_plotting import gaus, fit_gaus
from keras.models import  model_from_json
from tensorflow.keras.utils import custom_object_scope

# global variables
#plt.style.use('tdrstyle.mplstyle')
eps = 1e-7
my_cmap = plt.cm.viridis
my_cmap.set_under('w',1)
my_palette = ("#377eb8", "#e41a1c", "#984ea3", "#ff7f00", "#4daf4a")

def saving_pruned_model(model,
                        filepath='',
                        model_filename='model'):
    """

    Helper function to save the trained pruned models in the following format:
        Config -> JSON file format
        Weights -> H5PY file format
    Args:
        model: Keras Model
        filepath: Path on the system where the model is to be saved.
        model_filename: Name string for the model

    Returns:

    """
    # serialize model to JSON
    model_json = model.to_json()
    with open(filepath + "/" + model_filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filepath + "/" + model_filename + "_weights.h5")
    print("Saved model to disk")

def loading_trained_model(filepath='', model_filename='model'):
    """
    Helper function to load the trained models (NOT USING THE CUSTOM LAYER) from the models directory.
    Args:
        filepath: Path to the folder/ directory where the model is stored
        model_filename: Name of the file.

    Returns:
        loaded_model: Keras Model loaded from the file.
    """
    try:
        model_path = filepath + "/" + model_filename + ".json"
        model_weights_path = filepath + "/" + model_filename + "_weights.h5"
        with open(model_path, 'r') as f:
            json_model_file = f.read()
        f.close()
        loaded_model = model_from_json(json_model_file)
        # load weights into new model
        loaded_model.load_weights(model_weights_path)
        print("Loaded model from disk")

        return loaded_model

    except:
        print("ERROR: The model doesn't exist at the address {}".format(model_path))

def loading_pruned_model(filepath='',
                         model_filename='model',
                         custom_objects=None):
    """
    Helper function to load the trained models USING THE CUSTOM LAYER from the models directory.
    Args:
        filepath: Path to the folder/ directory where the model is stored
        model_filename: Name of the file.
        custom_objects: List of custom layer objects used in the model so that they can be loaded using the config.

    Returns:
        loaded_model: Keras Model loaded from the file.
    """
    try:
        model_path = filepath + "/" + model_filename + ".json"
        print(model_path)
        model_weights_path = filepath + "/" + model_filename + "_weights.h5"
        print(model_weights_path)
        with open(model_path, 'r') as f:
            loaded_model_json = f.read()
        f.close()
        with custom_object_scope(custom_objects):
            loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_weights_path)
        print("Loaded model from disk")

        return loaded_model

    except:
        print("ERROR: The model doesn't exist at the address provided.")

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

def __plot_dist__(weights, title="", color="purple", alpha=0.5):
    """
    Helper function to plot the distribution of layer weights.
    Args:
        weights: Input numpy array of layer weights
        title: Title for the plot
        color: Color for the plot
        alpha: Intensity of the color.

    Returns: None

    """
    w = np.ndarray.flatten(weights)
    plt.figure(figsize=(5, 5), dpi=75)
    plt.hist(w, color=color, alpha=alpha)
    plt.title("weights for layer " + title)
    plt.show()

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

def __generate_delta_plots__(model,
                             x,
                             y,
                             dxy,
                             bins_y: list = [-2.0,2.0],
                             bins_dxy: list = [-50.0,50.0],
                             color = "red",
                             batch_size: int = 4096,
                             min_y_val: float = 20.):
    """
    Helper function to generate diff plots for the resolution of the model on the test inputs.
    We check for the relative change in predictions for momentum and displacement when plotted against a
    gaussian distribution that mimics the ideal fit of values.
    Args:
        model: The trained keras model
        x: Test feature space
        y: Truth values for the 1st regression output, momentum
        dxy:Truth values for the 2nd regression output, displacement
        color: Color for the plots generated
        batch_size: Batch size for predictions
        min_y_val: Minimum value for y, beyond this predictions are considered pertaining to displaced muons

    Returns: None

    """
    # Predictions
    y_test_true = y.copy()
    y_test_true /= reg_pt_scale

    y_test_sel = (np.abs(1.0 / y) >= min_y_val / reg_pt_scale)

    y_test_meas_ = model.predict(x, batch_size = batch_size)
    y_test_meas = y_test_meas_[:, 0]
    y_test_meas /= reg_pt_scale
    y_test_meas = y_test_meas.reshape(-1)

    dxy_test_true = dxy.copy()
    dxy_test_true /= reg_dxy_scale

    # dxy_test_sel = (np.abs(dxy_test_true) >= 25)

    dxy_test_meas = y_test_meas_[:, 1]
    dxy_test_meas /= reg_dxy_scale
    dxy_test_meas = dxy_test_meas.reshape(-1)

    # Plot Delta(q/pT)/pT
    plt.figure(figsize=(5, 5), dpi=75)
    yy = ((np.abs(1.0 / y_test_meas) - np.abs(1.0 / y_test_true)) / np.abs(1.0 / y_test_true))
    hist, edges, _ = plt.hist(yy, bins=100, range=(bins_y[0], bins_y[1] - eps), histtype='stepfilled', facecolor=color, alpha=0.3)
    plt.xlabel(r'$\Delta(p_{T})_{\mathrm{meas-true}}/{(p_{T})}_{true}$ [1/GeV]')
    plt.ylabel(r'entries')
    logger.info('# of entries: {0}, mean: {1}, std: {2}'.format(len(yy), np.mean(yy), np.std(yy[np.abs(yy) < 0.4])))

    popt = fit_gaus(hist, edges, mu=np.mean(yy), sig=np.std(yy[np.abs(yy) < 2.0]))
    logger.info('gaus fit (a, mu, sig): {0}'.format(popt))
    xdata = (edges[1:] + edges[:-1]) / 2
    plt.plot(xdata, gaus(xdata, popt[0], popt[1], popt[2]), color=color)
    plt.show()

    # Plot Delta(dxy)
    plt.figure(figsize=(5, 5), dpi=75)
    yy = (dxy_test_meas - dxy_test_true)[y_test_sel]
    hist, edges, _ = plt.hist(yy, bins=100, range=(bins_dxy[0],bins_dxy[1]), histtype='stepfilled', facecolor=color, alpha=0.3)
    plt.xlabel(r'$\Delta(d_{0})_{\mathrm{meas-true}}$ [cm]')
    plt.ylabel(r'entries')
    logger.info('# of entries: {0}, mean: {1}, std: {2}'.format(len(yy), np.mean(yy), np.std(yy)))

    popt = fit_gaus(hist, edges, mu=np.mean(yy), sig=np.std(yy))
    logger.info('gaus fit (a, mu, sig): {0}'.format(popt))
    xdata = (edges[1:] + edges[:-1]) / 2
    plt.plot(xdata, gaus(xdata, popt[0], popt[1], popt[2]), color=color)
    plt.show()