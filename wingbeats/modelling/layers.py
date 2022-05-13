"""Library for custom layers"""



# Import libraries
from tensorflow.keras.layers import Layer, Conv1D, Conv2D, MaxPool1D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, Dropout, Dense, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras.applications import MobileNet, EfficientNetB0
from tensorflow.math import l2_normalize
from tensorflow import identity



class ConvBlock1d(Layer):
    """Class for a simple one-dimensional convolutional block. The block consists of \
    ``Conv1D(num_filters, kernel_size = 3, strides = 1) + BatchNorm + Relu + MaxPool1D(pool_size = ", strides = 2)``.
    
    :param num_filters: Number of convolutional filters.
    :type num_filters: int
    """
    
    def __init__(self, num_filters, **kwargs):
        """Constructor method"""
        
        super().__init__(**kwargs)
        self.num_filters = num_filters
      
        self.conv1d = Conv1D(num_filters, kernel_size = 3, strides = 1, padding = 'same', 
                             name = 'conv1d_' + str(num_filters), trainable = self.trainable)
        self.bnorm = BatchNormalization(name = 'batch_norm_' + str(num_filters), trainable = self.trainable)
        # Note: trainable = whether to freeze weights in Conv1D and BatchNorm
        self.activation = Activation('relu', name = 'relu_' + str(num_filters))
        self.maxpool1d = MaxPool1D(pool_size = 2, strides = 2, name = 'maxpool1d_' + str(num_filters))

    def call(self, inputs, training = None):
        """Apply forward propagation to **inputs**.
        
        :param inputs: Tensor batch of shape *(batch_size, len_signal, 1)*.
        :type inputs: Tensor
        :param training: Whether to run in training mode (relevant for Batch Normalization). Defaults to *None*.
        :type training: bool, optional
        :return: Transformed inputs. 
        """
        
        x = self.conv1d(inputs)
        x = self.bnorm(x, training = training) 
        # Note: in training mode, bnorm normalizes its inputs using mean and variance of current batch
        #       in inference mode, it normalizes using mean and variance learned during training
        x = self.activation(x)
        x = self.maxpool1d(x)
        
        return x

    def get_config(self):
        """Get configuration (needed for serialization when extra arguments are provided in ``__init__()``).
        
        :return: Configuration dictionary.
        :rtype: dict
        """
        
        config = super().get_config()
        config.update({"num_filters": self.num_filters})

        return config

#############################################################################################

class ConvBlock2d(Layer):
    """Class for a simple two-dimensional convolutional block. The block consists of \
    ``Conv2D(num_filters, kernel_size = 3, strides = 1) + BatchNorm + Relu + MaxPool1D(pool_size = ", strides = 2)``.
    
    :param num_filters: Number of convolutional filters.
    :type num_filters: int
    """
    
    def __init__(self, num_filters, **kwargs):
        """Constructor method"""
        
        super().__init__(**kwargs)
        self.num_filters = num_filters
      
        self.conv2d = Conv2D(num_filters, kernel_size = 3, strides = 1, padding = 'same', 
                             name = 'conv2d_' + str(num_filters), trainable = self.trainable)
        self.bnorm = BatchNormalization(name = 'batch_norm_' + str(num_filters), trainable = self.trainable)
        # Note: trainable = whether to freeze weights in Conv2D and BatchNorm
        self.activation = Activation('relu', name = 'relu_' + str(num_filters))
        self.maxpool2d = MaxPool2D(pool_size = 2, strides = 2, name = 'maxpool2d_' + str(num_filters))

    def call(self, inputs, training = None):
        """Apply forward propagation to **inputs**.
        
        :param inputs: Tensor batch of shape *(batch_size, height, width, 3)*.
        :type inputs: Tensor
        :param training: Whether to run in training mode (relevant for Batch Normalization). Defaults to *None*.
        :type training: bool, optional
        :return: Transformed inputs. 
        """
        
        x = self.conv2d(inputs)
        x = self.bnorm(x, training = training) 
        # Note: in training mode, bnorm normalizes its inputs using mean and variance of current batch
        #       in inference mode, it normalizes using mean and variance learned during training
        x = self.activation(x)
        x = self.maxpool2d(x)
        
        return x

    def get_config(self):
        """Get configuration (needed for serialization when extra arguments are provided in ``__init__()``).
        
        :return: Configuration dictionary.
        :rtype: dict
        """
        
        config = super().get_config()
        config.update({"num_filters": self.num_filters})

        return config

#############################################################################################

class CNN1D(Layer):
    """Class for a CNN made up of 5 **ConvBlock1d**'s of increasing filter sizes.
    
    :param drop_rate: Dropout rate (between 0 and 1).
    :type drop_rate: float
    :param mcdrop: Whether to apply Monte-Carlo Dropout. Defaults to *False*.
    :type mcdrop: bool, optional
    """
    
    def __init__(self, drop_rate, mcdrop = False, **kwargs):
        """Constructor method"""
        
        super().__init__(**kwargs)
        
        self.drop_rate = drop_rate
        self.mcdrop = mcdrop
        
        self.bnorm_0 = BatchNormalization(name = 'batch_norm_0', trainable = self.trainable)
        self.block_1 = ConvBlock1d(16,  name = 'conv_block_16',  trainable = self.trainable)
        self.block_2 = ConvBlock1d(32,  name = 'conv_block_32',  trainable = self.trainable)
        self.block_3 = ConvBlock1d(64,  name = 'conv_block_64',  trainable = self.trainable)
        self.block_4 = ConvBlock1d(128, name = 'conv_block_128', trainable = self.trainable)
        self.block_5 = ConvBlock1d(256, name = 'conv_block_256', trainable = self.trainable)
        
    def call(self, inputs, training = None):
        """Apply forward propagation to **inputs**.
        
        :param inputs: Tensor batch of shape *(batch_size, len_signal, 1)*.
        :type inputs: Tensor
        :param training: Whether to run in training mode (relevant for Batch Normalization). Defaults to *None*.
        :type training: bool, optional
        :return: Transformed inputs. 
        """
        
        # Normalize each batch instead of rescaling the whole data before training
        x = self.bnorm_0(inputs, training = training)
        x = self.block_1(x, training)
        x = self.block_2(x, training)
        x = self.block_3(x, training)
        x = self.block_4(x, training)
        x = self.block_5(x, training)
        x = Flatten()(x)
        if self.mcdrop:
            x = Dropout(self.drop_rate, name = 'dropout')(x, training = True)
        else:
            x = Dropout(self.drop_rate, name = 'dropout')(x, training)
            
        return x
    
    def get_config(self):
        """Get configuration (needed for serialization when extra arguments are provided in ``__init__()``).
        
        :return: Configuration dictionary.
        :rtype: dict
        """
        
        config = super().get_config()
        config.update({"drop_rate": self.drop_rate, "monte_carlo_dropout": self.mcdrop})

        return config
    
#############################################################################################
    
class CNN2D(Layer):
    """Class for a CNN made up of 5 **ConvBlock2d**'s of increasing filter sizes.
    
    :param drop_rate: Dropout rate (between 0 and 1).
    :type drop_rate: float
    :param mcdrop: Whether to apply Monte-Carlo Dropout. Defaults to *False*.
    :type mcdrop: bool, optional
    """
    
    def __init__(self, drop_rate, mcdrop = False, **kwargs):
        """Constructor method"""
        
        super().__init__(**kwargs)
        
        self.drop_rate = drop_rate
        self.mcdrop = mcdrop
        
        self.bnorm_0 = BatchNormalization(name = 'batch_norm_0', trainable = self.trainable)
        self.block_1 = ConvBlock2d(16,  name = 'conv_block_16',  trainable = self.trainable)
        self.block_2 = ConvBlock2d(32,  name = 'conv_block_32',  trainable = self.trainable)
        self.block_3 = ConvBlock2d(64,  name = 'conv_block_64',  trainable = self.trainable)
        self.block_4 = ConvBlock2d(128, name = 'conv_block_128', trainable = self.trainable)
        self.block_5 = ConvBlock2d(256, name = 'conv_block_256', trainable = self.trainable)
        
    def call(self, inputs, training = None):
        """Apply forward propagation to **inputs**.
        
        :param inputs: Tensor batch of shape *(batch_size, height, width, 3)*.
        :type inputs: Tensor
        :param training: Whether to run in training mode (relevant for Batch Normalization). Defaults to *None*.
        :type training: bool, optional
        :return: Transformed inputs. 
        """
        
        # Normalize each batch instead of rescaling the whole data before training
        x = self.bnorm_0(inputs, training = training)
        x = self.block_1(x, training)
        x = self.block_2(x, training)
        x = self.block_3(x, training)
        x = self.block_4(x, training)
        x = self.block_5(x, training)
        x = GlobalAveragePooling2D()(x)
        if self.mcdrop:
            x = Dropout(self.drop_rate, name = 'dropout')(x, training = True)
        else:
            x = Dropout(self.drop_rate, name = 'dropout')(x, training)
        
        return x
    
    def get_config(self):
        """Get configuration (needed for serialization when extra arguments are provided in ``__init__()``).
        
        :return: Configuration dictionary.
        :rtype: dict
        """
        
        config = super().get_config()
        config.update({"drop_rate": self.drop_rate, "monte_carlo_dropout": self.mcdrop})

        return config
    
#############################################################################################
    
class CNN_Mobile(Layer):
    """Class for a CNN made up of the feature extractor of a **MobileNet**.
    
    :param in_shape: Input shape.
    :type in_shape: tuple
    :param alpha: Network width parameter. If **alpha** < 1.0, the number of filters proportionally \
    decreases in each layer. For **alpha** > 1.0, it increases.
    :type alpha: float
    :param drop_rate: Dropout rate (between 0 and 1).
    :type drop_rate: float
    :param mcdrop: Whether to apply Monte-Carlo Dropout. Defaults to *False*.
    :type mcdrop: bool, optional
    """
    
    def __init__(self, in_shape, alpha, drop_rate, mcdrop = False, **kwargs):
        """Constructor method"""
        
        super().__init__(**kwargs)
        
        self.in_shape = in_shape
        self.alpha = alpha
        self.drop_rate = drop_rate
        self.mcdrop = mcdrop
        
        self.mobile = MobileNet(include_top = False, weights = 'imagenet', 
                                alpha = alpha, input_shape = in_shape)
        self.global_avg = GlobalAveragePooling2D()
        
    def call(self, inputs, training = None):
        """Apply forward propagation to **inputs**.
        
        :param inputs: Tensor batch of shape *(batch_size, height, width, 3)*.
        :type inputs: Tensor
        :param training: Whether to run in training mode (relevant for Batch Normalization). Defaults to *None*.
        :type training: bool, optional
        :return: Transformed inputs. 
        """
        
        x = self.mobile(inputs, training)
        x = self.global_avg(x)
        if self.mcdrop:
            x = Dropout(self.drop_rate, name = 'dropout')(x, training = True)
        else:
            x = Dropout(self.drop_rate, name = 'dropout')(x, training)
        
        return x
    
    def get_config(self):
        """Get configuration (needed for serialization when extra arguments are provided in ``__init__()``).
        
        :return: Configuration dictionary.
        :rtype: dict
        """
        
        config = super().get_config()
        config.update({"in_shape": self.in_shape, "alpha": self.alpha, 
                       "drop_rate": self.drop_rate, "monte_carlo_dropout": self.mcdrop})
        
        return config

#############################################################################################

class CNN_Efficient(Layer):
    """Class for a CNN made up of the feature extractor of an **EfficientNet**.
    
    :param in_shape: Input shape.
    :type in_shape: tuple
    :param drop_rate: Dropout rate (between 0 and 1).
    :type drop_rate: float
    :param mcdrop: Whether to apply Monte-Carlo Dropout. Defaults to *False*.
    :type mcdrop: bool, optional
    """
    
    def __init__(self, in_shape, drop_rate, mcdrop = False, **kwargs):
        """Constructor method"""
        
        super().__init__(**kwargs)
        
        self.in_shape = in_shape
        self.drop_rate = drop_rate
        self.mcdrop = mcdrop
        
        self.efficient = EfficientNetB0(include_top = False, weights = 'imagenet', 
                                        input_shape = in_shape)
        self.global_avg = GlobalAveragePooling2D()
        
    def call(self, inputs, training = None):
        """Apply forward propagation to **inputs**.
        
        :param inputs: Tensor batch of shape *(batch_size, height, width, 3)*.
        :type inputs: Tensor
        :param training: Whether to run in training mode (relevant for Batch Normalization). Defaults to *None*.
        :type training: bool, optional
        :return: Transformed inputs. 
        """
        
        x = self.efficient(inputs, training)
        x = self.global_avg(x)
        if self.mcdrop:
            x = Dropout(self.drop_rate, name = 'dropout')(x, training = True)
        else:
            x = Dropout(self.drop_rate, name = 'dropout')(x, training)
        
        return x
    
    def get_config(self):
        """Get configuration (needed for serialization when extra arguments are provided in ``__init__()``).
        
        :return: Configuration dictionary.
        :rtype: dict
        """
        
        config = super().get_config()
        config.update({"in_shape": self.in_shape,  
                       "drop_rate": self.drop_rate, "monte_carlo_dropout": self.mcdrop})
        
        return config
    
#############################################################################################    
    
class DenseBlock(Layer):
    """Class for a simple Dense block. The block consists of \
    ``BatchNorm + Relu + Dense(units)``.
    
    :param units: Number of nodes in the Dense layer.
    :type units: int
    :param reg_param: L2-Regularization factor.
    :type reg_param: float
    """

    def __init__(self, units, reg_param, **kwargs):
        """Constructor method"""
        
        super().__init__(**kwargs)
        self.units = units
        self.reg_param = reg_param
        
        self.bnorm = BatchNormalization()
        self.activation = Activation('relu')
        self.dense = Dense(units, dtype = 'float32', 
                           kernel_regularizer = regularizers.l2(reg_param))
        
    def call(self, inputs, training = None):
        """Apply forward propagation to **inputs**.
        
        :param inputs: Tensor batch.
        :type inputs: Tensor
        :param training: Whether to run in training mode (relevant for Batch Normalization). Defaults to *None*.
        :type training: bool, optional
        :return: Transformed inputs. 
        """
        
        x = self.bnorm(inputs, training = training)
        x = self.activation(x)
        x = self.dense(x)
        
        return x

    def get_config(self):
        """Get configuration (needed for serialization when extra arguments are provided in ``__init__()``).
        
        :return: Configuration dictionary.
        :rtype: dict
        """
        
        config = super().get_config()
        config.update({"units": self.units, "reg_param": self.reg_param})
        
        return config
 
#############################################################################################
        
class L2_Norm(Layer):
    """L2-Normalization layer (meant for hierarchical embedding output).
    """

    def __init__(self, **kwargs):   
        """Constructor method"""
        
        super().__init__(**kwargs)
        self.l2_norm = Lambda(lambda x: l2_normalize(x, axis=-1))
        
    def call(self, inputs):
        """Apply L2-Normalization on **inputs**.
        
        :param inputs: Tensor batch.
        :type inputs: Tensor
        :return: Transformed inputs. 
        """
        
        x = self.l2_norm(inputs)
        
        return x
    
#############################################################################################    
    
class Identity(Layer):
    """ Identity layer (just returns input unmodified)
    """

    def __init__(self, **kwargs):  
        """Constructor method"""
        
        super().__init__(**kwargs)
        self.id = Lambda(lambda x: identity(x))
        
    def call(self, inputs):
        """Apply identity function on **inputs**.
        
        :param inputs: Tensor batch.
        :type inputs: Tensor
        :return: The same inputs. 
        """
        
        x = self.id(inputs)
        
        return x   
    
    
