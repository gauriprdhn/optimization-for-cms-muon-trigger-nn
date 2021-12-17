import datetime
from nn_globals import logger

def lr_schedule(epoch, lr):
    """
    Reduces the learning rate by 0.9 if the epoch % 10 == 0 that is for every 10th epoch.
    Args:
        epoch: Training epoch number
        lr: learning rate for the optimizer.

    Returns: Updated learning rate

    """
    if (epoch % 10) == 0 and epoch != 0:
        lr *= 0.9
    return lr

def save_the_model(model, name='model'):
    """
    Save the trained model.
    Args:
        model: Keras Model class object
        name:  Name for the model file.

    Returns: None

    """
    model.save_weights(name + '_weights.h5')
    # Store model to json
    with open(name + '.json', 'w') as outfile:
        outfile.write(model.to_json())
    return None

def train_model(model,
                x,
                y,
                model_name='model',
                save_model=False,
                batch_size=1,
                epochs=100,
                verbose=False,
                callbacks=None,
                validation_split=0.1,
                shuffle=True):
    """
    Function to call .fit() function for any model (be it sparse or not)
    Args:
        model: Keras Model class object
        x: Array of features
        y: Array of truth value for model's output
        model_name: filename to save the model, only to be specified if save_model = True
        save_model: if True, save the model at the filename specified in model_name argument
        batch_size: Number of data entries to be processed as a single batch during each epoch
        epochs: Number of steps in the training cycle
        verbose: Print the training epoch's details
        callbacks: training callbacks
        validation_split: train-valiation data split
        shuffle: Shuffle data entries in x,y during training for each batch

    Returns: trained keras model and associated training history dict.

    """
    start_time = datetime.datetime.now()
    logger.info('Begin training ...')

    history = model.fit(x, y, batch_size=batch_size,
                        epochs=epochs, verbose=verbose, callbacks=callbacks,
                        validation_split=validation_split, shuffle=shuffle)

    logger.info('Done training. Time elapsed: {0} sec'.format(str(datetime.datetime.now() - start_time)))
    if save_model:
        save_the_model(model, name=model_name)

    return model, history