from nn_logging import getLogger

logger = getLogger( )
import sys

NVARIABLES = 23
DATAFILEPATH = "./data/NN_input_params_FlatXYZ.npz"
NENTRIES = 100000000
REG_PT_SCALE = 100.
REG_DXY_SCALE = 1.
TEST_SIZE = 0.315
MODELFILEPATH = "./models"  # sys.path[-1] + "/models"#
BASELINEFILENAME = "model"
FOLDS = 1
TRAINING_LOGS = "./training_logs"

INIT_SPARSITY = 0.1
TARGET_SPARSITY = 0.7
PRUNING_STEP = 0.1
PRUNING_TYPE = "unstructured"

INPUTFEATURES = [ "dphi_1" , "dphi_2" , "dphi_3" , "dphi_4" , "dphi_5" , "dphi_6" ,
                  "dtheta_1" , "dtheta_2" , "dtheta_3" , "dtheta_4" , "dtheta_5" , "dtheta_6" ,
                  "bend_1" , "bend_2" , "bend_3" , "bend_4" ,
                  "FR" , "track theta" , "ME11" ,
                  "RPC_1" , "RPC_2" , "RPC_3" , "RPC_4" ]

FT_PARAMS = [
    {
        "lr" : 1e-3 ,  # 5e-4
        "clipnorm" : 100. ,
        "eps" : 1e-4 ,
        "momentum" : 0.9 ,
        "epochs" : 250 ,
        "batch_size" : 2000 ,  # 1000
        "l1_reg" : 0.0 ,
        "l2_reg" : 0.0
    } ,

    {
        "lr" : 1.5e-3 ,  # 6e-4
        "clipnorm" : 100. ,
        "eps" : 1e-4 ,
        "momentum" : 0.9 ,
        "epochs" : 250 ,
        "batch_size" : 2000 ,
        "l1_reg" : 0.0 ,
        "l2_reg" : 0.0
    } ,

    {
        "lr" : 2.5e-3 ,  # 7.5e-4
        "clipnorm" : 100. ,
        "eps" : 1e-4 ,
        "momentum" : 0.9 ,
        "epochs" : 250 ,
        "batch_size" : 1000 ,
        "l1_reg" : 0.0 ,
        "l2_reg" : 0.0
    } ,

    {
        "lr" : 5.0e-3 ,  # 1e-3
        "clipnorm" : 100. ,
        "eps" : 1e-4 ,
        "momentum" : 0.9 ,
        "epochs" : 250 ,
        "batch_size" : 1000 ,
        "l1_reg" : 0.0 ,
        "l2_reg" : 0.0
    } ,

    {
        "lr" : 7.5e-3 ,  # 3e-3 for unstructured
        "clipnorm" : 100. ,
        "eps" : 1e-4 ,
        "momentum" : 0.9 ,
        "epochs" : 250 ,
        "batch_size" : 1000 ,
        "l1_reg" : 0.0 ,
        "l2_reg" : 0.0
    } ,

    {
        "lr" : 1e-2 ,  # 7.5e-3
        "clipnorm" : 100. ,
        "eps" : 1e-4 ,
        "momentum" : 0.9 ,
        "epochs" : 250 ,
        "batch_size" : 750 ,
        "l1_reg" : 0.0 ,
        "l2_reg" : 0.0
    } ,

    {
        "lr" : 1e-2 ,
        "clipnorm" : 100. ,
        "eps" : 1e-4 ,
        "momentum" : 0.9 ,
        "epochs" : 250 ,
        "batch_size" : 1000 ,
        "l1_reg" : 0.0 ,
        "l2_reg" : 0.0
    }
]
