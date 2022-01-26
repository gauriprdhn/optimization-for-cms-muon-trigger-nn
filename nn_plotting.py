import contextlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nn_globals import *
from scipy.optimize import curve_fit

#plt.style.use('tdrstyle.mplstyle')
eps = 1e-7
my_cmap = plt.cm.viridis
my_cmap.set_under('w',1)
my_palette = ("#377eb8", "#e41a1c", "#984ea3", "#ff7f00", "#4daf4a")

def gaus(x,a,mu,sig):
    """

    :param x: Input array for which a gaussian approximation is to be generated.
    :param a: Number of values in the input array, x.
    :param mu: mean of the distribution.
    :param sig: standard deviation for the distribution.
    :return: Array with gaussian representation = a*np.exp(-0.5*np.square((x-mu)/sig))
    """
    return a*np.exp(-0.5*np.square((x-mu)/sig))

def fit_gaus(hist, edges, mu=0., sig=1.):
    """
    Helper function to fit the gaussian approximation to a curve.
    :param hist: histogram data entries on y-axis
    :param edges: histogram data for the x-axis
    :param mu: mean of the distribution
    :param sig: standard deviation for the distribution
    :return: plot values for the guassian fitted data
    """

    hist = hist.astype ('float64')
    edges = edges.astype ('float64')
    xdata = (edges [1:] + edges [:-1]) / 2
    ydata = hist
    popt, pcov = curve_fit (gaus, xdata, ydata, p0=[np.max (hist), mu, sig])
    if not np.isfinite (pcov).all ():
        raise Exception ('Fit has failed to converge.')
    return popt


# Answer from https://stackoverflow.com/a/2891805
@contextlib.contextmanager
def np_printoptions(*args, **kwargs):

    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

def corr_plot(x,
              y,
              dxy,
              columns="all"):
    """
    Helper function to plot correlation between input features and the predicted outputs [using truth values]
    :param x: input feature space array
    :param y: 1st predicted variable truth values
    :param dxy:  2nd predicted variable truth values
    :param columns: list of columns to be considered for correlation plots
    :return: None
    """

    df = pd.DataFrame(x, columns=INPUTFEATURES)
    df["momentum"] = y
    df["displacement"] = dxy

    if isinstance(columns, list):
        df_cont = df[columns]
    else:
        df_cont = df

    C_mat = df_cont.corr()
    fig = plt.figure(figsize=(6, 6), dpi=100)
    sns.heatmap(C_mat, vmax=.75, square=True, cmap="YlGnBu")
    plt.show()

def __generate_delta_plots__(model,
                             x,
                             y,
                             dxy,
                             num_bins:int = 100,
                             bins_y: list = [-2.0,2.0],
                             bins_dxy: list = [-50.0,50.0],
                             color:str = "red",
                             alpha: float = 0.5,
                             batch_size: int = 4096,
                             min_y_val: float = 20.,
                             return_vals:bool = False):
    """
    Helper function to generate diff plots for the resolution of the model on the test inputs.
    We check for the relative change in predictions for momentum and displacement when plotted against a
    gaussian distribution that mimics the ideal fit of values.

    :param model: The trained keras model
    :param x: Test feature space
    :param y: Truth values for the 1st regression output, momentum
    :param dxy: Truth values for the 2nd regression output, displacement
    :param bins_y: bin range for y
    :param bins_dxy: bin range for dxy
    :param color:  Color for the plots generated
    :param batch_size: Batch size for predictions
    :param min_y_val: Threshold value for y. Predictions are considered pertaining to displaced muons if > min_y_val
    :return: Tuple of (mu_y, sig_y, mu_dxy, sig_dxy) if return_vals = True, else None
    """
    # Predictions
    y_test_true = y.copy()
    y_test_true /= REG_PT_SCALE

    y_test_sel = (np.abs(1.0 / y) >= min_y_val / REG_PT_SCALE)

    y_test_meas_ = model.predict(x, batch_size = batch_size)
    y_test_meas = y_test_meas_[:, 0]
    y_test_meas /= REG_PT_SCALE
    y_test_meas = y_test_meas.reshape(-1)

    dxy_test_true = dxy.copy()
    dxy_test_true /= REG_DXY_SCALE

    dxy_test_meas = y_test_meas_[:, 1]
    dxy_test_meas /= REG_DXY_SCALE
    dxy_test_meas = dxy_test_meas.reshape(-1)

    # Plot Delta(q/pT)/pT
    plt.figure(figsize=(5, 5), dpi=75)
    yy = ((np.abs(1.0 / y_test_meas) - np.abs(1.0 / y_test_true)) / np.abs(1.0 / y_test_true))
    hist, edges, _ = plt.hist(yy,
                              bins=num_bins,
                              range=(bins_y[0], bins_y[1] - eps),
                              histtype='stepfilled',
                              facecolor=color,
                              alpha=alpha)
    plt.xlabel(r'$\Delta(p_{T})_{\mathrm{meas-true}}/{(p_{T})}_{true}$ [1/GeV]')
    plt.ylabel(r'entries')
    logger.info('# of entries: {0}, mean: {1}, std: {2}'.format(len(yy), np.mean(yy), np.std(yy[np.abs(yy) < 0.4])))
    mu_yy = np.mean(yy)
    sig_yy = np.std(yy[np.abs(yy) < 2.0])
    popt = fit_gaus(hist, edges, mu=mu_yy, sig=sig_yy)
    logger.info('gaus fit (a, mu, sig): {0}'.format(popt))
    xdata = (edges[1:] + edges[:-1]) / 2
    plt.plot(xdata, gaus(xdata, popt[0], popt[1], popt[2]), color=color)
    plt.show()

    # Plot Delta(dxy)
    plt.figure(figsize=(5, 5), dpi=75)
    dx_yy = (dxy_test_meas - dxy_test_true)[y_test_sel]
    hist, edges, _ = plt.hist(dx_yy,
                              bins=num_bins,
                              range=(bins_dxy[0],bins_dxy[1]),
                              histtype='stepfilled',
                              facecolor=color,
                              alpha=alpha)
    plt.xlabel(r'$\Delta(d_{0})_{\mathrm{meas-true}}$ [cm]')
    plt.ylabel(r'entries')
    logger.info('# of entries: {0}, mean: {1}, std: {2}'.format(len(dx_yy), np.mean(dx_yy), np.std(dx_yy)))
    mu_dxyy = np.mean(dx_yy)
    sig_dxyy = np.std(dx_yy)
    popt = fit_gaus(hist, edges, mu=mu_dxyy, sig=sig_dxyy)
    logger.info('gaus fit (a, mu, sig): {0}'.format(popt))
    xdata = (edges[1:] + edges[:-1]) / 2
    plt.plot(xdata, gaus(xdata, popt[0], popt[1], popt[2]), color=color)
    plt.show()

    if return_vals:
        return (mu_yy, sig_yy, mu_dxyy, sig_dxyy)
    else:
        return None

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
    plt.hist(w, color=color, alpha=alpha,bins=100)
    plt.title("weights for layer " + title)
    plt.show()
