import contextlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nn_globals import features
from scipy.optimize import curve_fit

#plt.style.use('tdrstyle.mplstyle')
# eps = 1e-7
my_cmap = plt.cm.viridis
my_cmap.set_under('w',1)
my_palette = ("#377eb8", "#e41a1c", "#984ea3", "#ff7f00", "#4daf4a")

def gaus(x,a,mu,sig):
    """

    Args:
        x: Input array for which a gaussian approximation is to be generated.
        a: Number of values in the input array, x.
        mu: mean of the distribution.
        sig: standard deviation for the distribution.

    Returns:
        Array with gaussian representation = a*np.exp(-0.5*np.square((x-mu)/sig))
    """

    return a*np.exp(-0.5*np.square((x-mu)/sig))

def fit_gaus(hist, edges, mu=0., sig=1.):
    """
    Helper function to fit the gaussian approximation to a curve.
    Args:
        hist: histogram data entries on y-axis
        edges: histogram data for the x-axis
        mu: mean of the distribution
        sig: standard deviation for the distribution

    Returns:
        plot values for the guassian fitted data
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

    Args:
        x:
        y:
        dxy:
        columns:

    Returns:

    """
    df = pd.DataFrame(x, columns=features)
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
