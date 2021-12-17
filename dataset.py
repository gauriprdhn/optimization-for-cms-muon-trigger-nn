import numpy as np
from nn_globals import *
from sklearn.model_selection import train_test_split


def _handle_nan_in_x (x) :
    """
    :param x: Input array suspected to contain NaNs
    :return: array with NaNs substituted by 0.0
    """
    x [np.isnan (x)] = 0.0
    x [x == -999.0] = 0.0
    return x


# def _zero_out_x (x) :
#     x = 0.0
#     return x
#
#
# def _fixME1Ring (x) :
#     for i in range (len (x)) :
#         if (x [i , 0] != 0.0) : x [i , 18] = x [i , 18] + 1
#     return x

def muon_data(filename, reg_pt_scale=1.0,
              reg_dxy_scale=1.0,
              correct_for_eta=False):
    try:
        logger.info('Loading muon data from {0} ...'.format(filename))
        loaded = np.load(filename)
        the_variables = loaded['variables']
        the_parameters = loaded['parameters']
        # print(the_variables.shape)
        the_variables = the_variables[:nentries]
        the_parameters = the_parameters[:nentries]
        logger.info('Loaded the variables with shape {0}'.format(the_variables.shape))
        logger.info('Loaded the parameters with shape {0}'.format(the_parameters.shape))
    except:
        logger.error('Failed to load data from file: {0}'.format(filename))

    assert(the_variables.shape[0] == the_parameters.shape[0])
    _handle_nan_in_x(the_variables)
      #_fixME1Ring(the_variables)
    _handle_nan_in_x(the_parameters)
    mask = np.logical_or(np.logical_or( np.logical_or((the_variables[:,23] == 11), (the_variables[:,23] == 13)), (the_variables[:,23] == 14)),(the_variables[:,23] == 15))

    the_variables = the_variables[mask]
    the_parameters = the_parameters[mask]
    assert(the_variables.shape[0] == the_parameters.shape[0])

    x = the_variables[:,0:23]
    y = reg_pt_scale*the_parameters[:,0]
    phi = the_parameters[:,1]
    #eta = the_parameters[:,2]
    vx = the_parameters[:,3]
    vy = the_parameters[:,4]
    #vz = the_parameters[:,5]
    dxy = vy * np.cos(phi) - vx * np.sin(phi)
    logger.info('Loaded the encoded variables with shape {0}'.format(x.shape))
    logger.info('Loaded the encoded parameters with shape {0}'.format(y.shape))

    return x, y, dxy


def muon_data_split (filename , reg_pt_scale=1.0 , reg_dxy_scale=1.0 , test_size=0.5,
                     batch_size=128) :
    """
    Function to preprocess the raw data input from .npz file and divide it into train-test splits.
    :param filename: Absolute address of the .npz file
    :param reg_pt_scale: term for momentum scaling
    :param reg_dxy_scale: term for displacement scaling
    :param test_size: Fraction of the data samples that are to be considered for testing. If 0.0 the function returns the whole dataset divided into x ,y ,dxy
    :param batch_size: Number of sample for uniform batches.
    :return: Each of x,y, and dxy split into train-test arrays if test_size != 0.0 else x,y,dxy
    """
    x , y , dxy = muon_data (filename ,
                             reg_pt_scale=reg_pt_scale ,
                             reg_dxy_scale=reg_dxy_scale)

    if test_size :

        # Split dataset in training and testing
        x_train , x_test , y_train , y_test , dxy_train , dxy_test = train_test_split (x , y , dxy ,
                                                                                       test_size=test_size)
        logger.info ('Loaded # of training and testing events: {0}'.format ((x_train.shape [0] , x_test.shape [0])))

        # Check for cases where the number of events in the last batch could be too few
        validation_split = 0.1
        train_num_samples = int (x_train.shape [0] * (1.0 - validation_split))
        val_num_samples = x_train.shape [0] - train_num_samples
        if (train_num_samples % batch_size) < 100 :
            logger.warning (
                'The last batch for training could be too few! ({0}%{1})={2}. Please change test_size.'.format (
                    train_num_samples , batch_size , train_num_samples % batch_size))
            logger.warning ('Try this formula: int(int({0}*{1})*{2}) % 128'.format (x.shape [0] , 1.0 - test_size ,
                                                                                    1.0 - validation_split))
        train_num_samples = int (x_train.shape [0] * 2 * (1.0 - validation_split))
        val_num_samples = x_train.shape [0] - train_num_samples
        if (train_num_samples % batch_size) < 100 :
            logger.warning (
                'The last batch for training after mixing could be too few! ({0}%{1})={2}. Please change test_size.'.format (
                    train_num_samples , batch_size , train_num_samples % batch_size))
            logger.warning ('Try this formula: int(int({0}*{1})*2*{2}) % 128'.format (x.shape [0] , 1.0 - test_size ,
                                                                                      1.0 - validation_split))
        y_train = np.abs(y_train)
        y_test = np.abs(y_test)
        return x_train , x_test , y_train , y_test , dxy_train , dxy_test

    else :
        y = np.abs (y)
        return x , y , dxy