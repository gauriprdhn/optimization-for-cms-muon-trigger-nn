## Model Compression Using Iterative Pruning:

***
*The current version of the library as depicted by the code in this branch of the repository, is NOT our final version. It is subject to change later if required to improve the present pipeline.*

This library can be used to perform iterative pruning on a TF/Keras model using custom keras layer.
The underlying notebooks can be used to study how the package can be used to perform iterative pruning on a neural network used for regression.

<pre> To install the library run the following steps from your terminal: 

1. Pull the package from git via `git clone https://github.com/gauriprdhn/optimization-for-cms-muon-trigger-nn.git`
2. `cd` to the directory where `setup.py` is present on the local system.
3. Run `python3 setup.py install` to install the necessary backups and scripts as a package. </pre>
***

<pre> Present folders hold files pertaining to the project:

 1. models : The `.json` and `.h5py` files for trained [and pruned] models for model config and weights respectively.
 2. baseline_checkpoints: Checkpoints to assess model weights during training. Used to store the checkpoints while emulating the training of the baseline. </pre>
