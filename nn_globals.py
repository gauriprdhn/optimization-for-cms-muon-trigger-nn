from nn_logging import getLogger
logger = getLogger ( )

NVARIABLES = 23
DATAFILEPATH =  "./data/NN_input_params_FlatXYZ.npz"
NENTRIES = 100000000
REG_PT_SCALE = 100.
REG_DXY_SCALE = 1.
TEST_SIZE = 0.315
MODELFILEPATH =  "./models/"
BASELINEFILENAME = "model"
FOLDS = 1

# PARAMS FOR PRUNING ALGO
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
        "lr" : 3e-4 ,
        "clipnorm" : 100. ,
        "eps" : 1e-4 ,
        "momentum" : 0.9 ,
        "epochs" : 100 ,
        "batch_size" : 1000 ,
        "l1_reg" : 0.0 ,
        "l2_reg" : 0.0
    } ,

    {
        "lr" : 5e-4 ,
        "clipnorm" : 100. ,
        "eps" : 1e-4 ,
        "momentum" : 0.9 ,
        "epochs" : 125 ,
        "batch_size" : 1000 ,
        "l1_reg" : 0.0 ,
        "l2_reg" : 0.0
    } ,

    {
        "lr" : 7e-4 ,
        "clipnorm" : 50. ,
        "eps" : 1e-4 ,
        "momentum" : 0.9 ,
        "epochs" : 150 ,
        "batch_size" : 1000 ,
        "l1_reg" : 0.0 ,
        "l2_reg" : 0.0
    } ,

    {
        "lr" : 1e-3 ,
        "clipnorm" : 50. ,
        "eps" : 1e-4 ,
        "momentum" : 0.9 ,
        "epochs" : 175 ,
        "batch_size" : 1000 ,
        "l1_reg" : 0.0 ,
        "l2_reg" : 0.0
    } ,

    {
        "lr" : 3e-3 ,
        "clipnorm" : 50. ,
        "eps" : 1e-4 ,
        "momentum" : 0.9 ,
        "epochs" : 200 ,
        "batch_size" : 1000 ,
        "l1_reg" : 0.0 ,
        "l2_reg" : 0.0
    } ,

    {
        "lr" : 2.5e-3 ,
        "clipnorm" : 10. ,
        "eps" : 1e-4 ,
        "momentum" : 0.9 ,
        "epochs" : 300 ,
        "batch_size" : 750 ,
        "l1_reg" : 0.0 ,
        "l2_reg" : 0.0
    } ,

    {
        "lr" : 5e-3 ,
        "clipnorm" : 0. ,
        "eps" : 1e-4 ,
        "momentum" : 0.9 ,
        "epochs" : 400 ,
        "batch_size" : 750 ,
        "l1_reg" : 0.0 ,
        "l2_reg" : 0.0
    } ,
    {
        "lr" : 7.5e-3 ,
        "clipnorm" : 0. ,
        "eps" : 1e-4 ,
        "momentum" : 0.9 ,
        "epochs" : 500 ,
        "batch_size" : 750 ,
        "l1_reg" : 0.0 ,
        "l2_reg" : 0.0
    }
]
