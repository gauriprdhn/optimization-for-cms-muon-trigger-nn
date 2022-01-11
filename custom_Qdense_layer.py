import warnings
import tensorflow as tf
from qkeras.quantizers import *
from qkeras.quantizers import get_quantizer
from keras.layers import Dense
from qkeras import Clip , get_auto_range_constraint_initializer , QInitializer
from keras import initializers, regularizers
from tensorflow_model_optimization.python.core.sparsity.keras.prunable_layer import PrunableLayer
import tensorflow.keras.backend as K
from keras import activations, constraints

class CustomQDense (Dense , PrunableLayer) :
    """Implements a quantized Dense layer with masking."""

    # Following layer is a modification of the original QDense layer.
    # Base code inherited from [https://github.com/google/qkeras/blob/master/qkeras/qlayers.py#L544]
    # Most of these parameters follow the implementation of Dense in
    # Keras, with the exception of kernel_range, bias_range,
    # kernel_quantizer, bias_quantizer, and kernel_initializer.
    # kernel_quantizer: quantizer function/class for kernel
    # bias_quantizer: quantizer function/class for bias
    # kernel_range/bias_ranger: for quantizer functions whose values
    #   can go over [-1,+1], these values are used to set the clipping
    #   value of kernels and biases, respectively, instead of using the
    #   constraints specified by the user.
    # we refer the reader to the documentation of Dense in Keras for the
    # other parameters.

    def __init__ ( self ,
                   units ,
                   mask ,
                   activation=None ,
                   use_bias=False ,
                   kernel_initializer="he_normal" ,
                   bias_initializer="zeros" ,
                   kernel_regularizer=None ,
                   bias_regularizer=None ,
                   activity_regularizer=None ,
                   kernel_constraint=None ,
                   bias_constraint=None ,
                   kernel_quantizer=None ,
                   bias_quantizer=None ,
                   kernel_range=None ,
                   bias_range=None ,
                   **kwargs ) :

        self.layer_mask = K.constant (mask , dtype = K.floatx ( ))
        self.use_bias = use_bias
        if kernel_range is not None :
            warnings.warn ("kernel_range is deprecated in QDense layer.")

        if bias_range is not None :
            warnings.warn ("bias_range is deprecated in QDense layer.")

        self.kernel_range = kernel_range
        self.bias_range = bias_range

        self.kernel_quantizer = kernel_quantizer
        self.bias_quantizer = bias_quantizer

        self.kernel_quantizer_internal = get_quantizer (self.kernel_quantizer)
        self.bias_quantizer_internal = get_quantizer (self.bias_quantizer)

        # optimize parameter set to "auto" scaling mode if possible
        if hasattr (self.kernel_quantizer_internal , "_set_trainable_parameter") :
            self.kernel_quantizer_internal._set_trainable_parameter ( )

        self.quantizers = [
            self.kernel_quantizer_internal , self.bias_quantizer_internal
        ]

        self.kernel_constraint , self.kernel_initializer = (
            get_auto_range_constraint_initializer (self.kernel_quantizer_internal ,
                                                   kernel_constraint ,
                                                   kernel_initializer))

        if self.use_bias :
            self.bias_constraint , self.bias_initializer = (
                get_auto_range_constraint_initializer (self.bias_quantizer_internal ,
                                                       bias_constraint ,
                                                       bias_initializer))
        if activation is not None :
            activation = get_quantizer (activation)

        super (CustomQDense , self).__init__ (
            units = units ,
            activation = activation ,
            use_bias = use_bias ,
            kernel_initializer = kernel_initializer ,
            bias_initializer = bias_initializer ,
            kernel_regularizer = kernel_regularizer ,
            bias_regularizer = bias_regularizer ,
            activity_regularizer = activity_regularizer ,
            kernel_constraint = kernel_constraint ,
            bias_constraint = bias_constraint ,
            **kwargs)

    def build ( self , input_shape ) :

        last_dim = int (input_shape [-1])
        self.kernel = self.add_weight ("kernel" ,
                                       shape = (last_dim , self.units) ,
                                       initializer = self.kernel_initializer ,
                                       regularizer = self.kernel_regularizer ,
                                       constraint = self.kernel_constraint ,
                                       dtype = K.floatx ( ) ,
                                       trainable = True)
        if self.use_bias :
            self.bias = self.add_weight ("bias" ,
                                         shape = (self.units ,) ,
                                         initializer = self.bias_initializer ,
                                         regularizer = self.bias_regularizer ,
                                         constraint = self.bias_constraint ,
                                         dtype = K.floatx ( ) ,
                                         trainable = True)

        super (CustomQDense , self).build (input_shape)
        self.built = True

    def call ( self , inputs ) :
        if self.kernel_quantizer :
            quantized_kernel = self.kernel_quantizer_internal (self.kernel)
        else :
            quantized_kernel = self.kernel
        output = tf.keras.backend.dot (inputs , quantized_kernel * self.layer_mask)
        if self.use_bias :
            if self.bias_quantizer :
                quantized_bias = self.bias_quantizer_internal (self.bias)
            else :
                quantized_bias = self.bias
            output = tf.keras.backend.bias_add (output , quantized_bias ,
                                                data_format = "channels_last")
        if self.activation is not None :
            output = self.activation (output)
        return output

    def compute_output_shape ( self , input_shape ) :
        assert input_shape and len (input_shape) >= 2
        assert input_shape [-1]
        output_shape = list (input_shape)
        output_shape [-1] = self.units
        return tuple (output_shape)

    def get_config ( self ) :
        config = {
            "units" : self.units ,
            "activation" : activations.serialize (self.activation) ,
            "use_bias" : self.use_bias ,
            "kernel_quantizer" :
                constraints.serialize (self.kernel_quantizer_internal) ,
            "bias_quantizer" :
                constraints.serialize (self.bias_quantizer_internal) ,
            "kernel_initializer" :
                initializers.serialize (self.kernel_initializer) ,
            "bias_initializer" :
                initializers.serialize (self.bias_initializer) ,
            "kernel_regularizer" :
                regularizers.serialize (self.kernel_regularizer) ,
            "bias_regularizer" :
                regularizers.serialize (self.bias_regularizer) ,
            "activity_regularizer" :
                regularizers.serialize (self.activity_regularizer) ,
            "kernel_constraint" :
                constraints.serialize (self.kernel_constraint) ,
            "bias_constraint" :
                constraints.serialize (self.bias_constraint) ,
            "kernel_range" : self.kernel_range ,
            "bias_range" : self.bias_range ,
            "mask" : self.layer_mask.numpy ( )
        }
        base_config = super (CustomQDense , self).get_config ( )
        return dict (list (base_config.items ( )) + list (config.items ( )))

    def get_quantization_config ( self ) :
        return {
            "kernel_quantizer" :
                str (self.kernel_quantizer_internal) ,
            "bias_quantizer" :
                str (self.bias_quantizer_internal) ,
            "activation" :
                str (self.activation) ,
            "units" : str (self.units)
        }

    def get_quantizers ( self ) :
        return self.quantizers

    def get_prunable_weights ( self ) :
        return [self.kernel]

    def get_constraint ( identifier , quantizer ) :
        """Gets the initializer.
        Args:
        identifier: A constraint, which could be dict, string, or callable function.
        quantizer: A quantizer class or quantization function
        Returns:
        A constraint class
        """
        if identifier :
            if isinstance (identifier , dict) and identifier ['class_name'] == 'Clip' :
                return Clip.from_config (identifier ['config'])
            else :
                return constraints.get (identifier)
        else :
            max_value = max (1 , quantizer.max ( )) if hasattr (quantizer , "max") else 1.0
            return Clip (-max_value , max_value , identifier , quantizer)

    def get_initializer ( identifier ) :
        """Gets the initializer.
        Args:
        identifier: An initializer, which could be dict, string, or callable function.
        Returns:
        A initializer class
        Raises:
        ValueError: An error occurred when quantizer cannot be interpreted.
        """
        if identifier is None :
            return None
        if isinstance (identifier , dict) :
            if identifier ['class_name'] == 'QInitializer' :
                return QInitializer.from_config (identifier ['config'])
            else :
                return initializers.get (identifier)
        elif isinstance (identifier , six.string_types) :
            return initializers.get (identifier)
        elif callable (identifier) :
            return identifier
        else :
            raise ValueError ("Could not interpret initializer identifier: " +
                              str (identifier))

    def get_masked_weights ( self ) :
        """ Gets the quantized kernel weights with insignificant indices marked as 0
        """
        return K.batch_get_value ([self.kernel_quantizer_internal (self.kernel) * self.layer_mask])