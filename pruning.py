from typing import Union, List

import numpy as np
from nn_globals import *
from dataset import muon_data_split
from nn_pruning_module_support import loading_trained_model
from nn_evaluate import  k_fold_validation
from nn_plotting import __generate_delta_plots__
from nn_training_pruned_model import create_sparse_model
from nn_training import train_model, lr_schedule
from tensorflow.keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, TerminateOnNaN, EarlyStopping

def run_iterative_pruning_v2 (X_train ,
                              y_train ,
                              dxy_train ,
                              X_test ,
                              y_test ,
                              dxy_test ,
                              baseline_model,
                              pruning_type:str = "unstructured",
                              init_sparsity: float = 0.1 ,
                              target_sparsity: float = 1.0 ,
                              pruning_fraction_step: float = 0.1 ,
                              training_params: List[dict] = None ,
                              cv_folds: int = 1 ,
                              plot_colors: Union[str,List[str], None] = "red") :
    if target_sparsity > 1.0 or target_sparsity <= 0 :
        print ("INVALID value entered for target sparsity, it can only be in the range [0,1]")
    if pruning_fraction_step > target_sparsity :
        print ("INVALID value entered for pruning fraction, it has to be <= target_sparsity")

    # list of models, new pruned models get appended to it while training
    pruned_models , training_historys = [] , []
    i = 0
    while (init_sparsity <= target_sparsity) :

        print ("-----------------------------------------------------------------------------------------------")
        print ("-----------------------------------------------------------------------------------------------")
        print ("Currently pruning the model upto {} % of the baseline".format (round (init_sparsity * 100)))
        print ("-----------------------------------------------------------------------------------------------")
        print ("-----------------------------------------------------------------------------------------------")

        # training loop begins
        lr = training_params [i] ['lr']
        clipnorm = training_params [i] ['clipnorm']
        eps = training_params [i] ['eps']
        momentum = training_params [i] ['momentum']
        retrain_epochs = training_params [i] ['epochs']
        retrain_batch_size = training_params [i] ['batch_size']
        l1_reg = training_params [i] ['l1_reg']
        l2_reg = training_params [i] ['l2_reg']
        sparsity = init_sparsity

        # define optimizer, callbacks here
        adam = Adam (lr=lr , clipnorm=clipnorm)
        lr_decay = LearningRateScheduler (lr_schedule , verbose=1)
        terminate_on_nan = TerminateOnNaN ()
        early_stopping = EarlyStopping (min_delta=1e-5 ,
                                        monitor='val_loss' ,
                                        patience=25 ,
                                        verbose=True ,
                                        mode='auto')
        curr_model = None
        if len (pruned_models) == 0 :
            curr_model = baseline_model
        else :
            curr_model = pruned_models [-1]

        pruned_model = create_sparse_model (model=curr_model ,
                                            input_dim=NVARIABLES ,
                                            output_dim=2 ,
                                            k_sparsity=sparsity ,
                                            pruning_type=pruning_type,
                                            bn_epsilon=eps ,
                                            bn_momentum=momentum ,
                                            l1_reg=l1_reg ,
                                            l2_reg=l1_reg ,
                                            kernel_initializer="glorot_uniform",
                                            optimizer=adam)

        pruned_model , history = train_model (model=pruned_model ,
                                              x=X_train ,
                                              y=np.column_stack ((y_train , dxy_train)) ,
                                              epochs=retrain_epochs ,
                                              batch_size=retrain_batch_size ,
                                              callbacks=[lr_decay ,
                                                         early_stopping ,
                                                         terminate_on_nan] ,
                                              verbose=True ,
                                              validation_split=0.1)

        k_fold_validation (model=pruned_model ,
                           x=X_test ,
                           y=y_test ,
                           dxy=dxy_test ,
                           folds=cv_folds ,
                           metric_type="MAE")
        k_fold_validation (model=pruned_model ,
                           x=X_test ,
                           y=y_test ,
                           dxy=dxy_test ,
                           folds=cv_folds ,
                           metric_type="RMSE")
        if isinstance(plot_colors, list):
            plot_color = plot_colors[i]
        else:
            plot_color = plot_colors
        __generate_delta_plots__ (model=pruned_model ,
                                  x=X_test ,
                                  y=y_test ,
                                  dxy=dxy_test ,
                                  color=plot_color)

        pruned_models.append (pruned_model)
        training_historys.append (history)

        # training ends
        i += 1
        init_sparsity += pruning_fraction_step

    return pruned_models , training_historys


if __name__ == "__main__" :
    x_train_displ , x_test_displ , y_train_displ , y_test_displ , dxy_train_displ , dxy_test_displ = muon_data_split (
        filename=DATAFILEPATH ,
        reg_pt_scale=REG_PT_SCALE ,
        reg_dxy_scale=REG_DXY_SCALE ,
        test_size=TEST_SIZE ,
        nentries=NENTRIES ,
        nvariables=NVARIABLES
    )

    print("------------------------------- LOADING THE BASELINE TO BE PRUNED -------------------------------")
    baseline = loading_trained_model (filepath= MODELFILEPATH,
                                      model_filename=BASELINEFILENAME)
    baseline.summary ()

    print ("------------------------------- PERFORMANCE METRICS FOR BASELINE -------------------------------")

    k_fold_validation (model=baseline ,
                       x=x_test_displ ,
                       y=y_test_displ ,
                       dxy=dxy_test_displ ,
                       folds= FOLDS ,
                       metric_type="MAE")
    k_fold_validation (model=baseline ,
                       x=x_test_displ ,
                       y=y_test_displ ,
                       dxy=dxy_test_displ ,
                       folds= FOLDS ,
                       metric_type="RMSE")

    __generate_delta_plots__ (model=baseline ,
                              x=x_test_displ ,
                              y=y_test_displ ,
                              dxy=dxy_test_displ ,
                              color= "slateblue")

    print("------------------------------- STARTING TO PRUNE TILL {} -------------------------------".format(TARGET_SPARSITY))

    plot_colors = ["red","orange","turquoise","purple","green","moccasin", "royalblue"]
    pruned_models , training_historys = run_iterative_pruning_v2 ( x_train_displ ,
                                                                   y_train_displ ,
                                                                   dxy_train_displ ,
                                                                   x_test_displ , y_test_displ , dxy_test_displ ,
                                                                   baseline_model=baseline ,
                                                                   init_sparsity=INIT_SPARSITY ,
                                                                   target_sparsity=TARGET_SPARSITY ,
                                                                   pruning_fraction_step=PRUNING_STEP ,
                                                                   pruning_type= PRUNING_TYPE,
                                                                   training_params=FT_PARAMS ,
                                                                   cv_folds=FOLDS ,
                                                                   plot_colors=plot_colors
                                                                   )
