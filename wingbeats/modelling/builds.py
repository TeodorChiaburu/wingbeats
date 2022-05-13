"""Library for factory functions to define model architectures within the Functional API"""
  


# Import libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, Softmax
from tensorflow.keras import Input, regularizers

from wingbeats.modelling.layers import DenseBlock, L2_Norm, Identity



def build_embedder(in_shape, out_shape, f_extractor, reg_param = 1e-3, 
                   input_name = "input_signal", model_name = "Embedder",
                   training = None):
    """Build model for learning hierarchical class embeddings.
    
    Architectures 
        f_extractor(Layer) + Dense + L2 
    
    Outputs
        predicted embedding
    
    :param in_shape: Input shape. No need to specify batch dimension.
    :type in_shape: tuple
    :param out_shape: Model output shape (here equal to the size of embedded taxonomic level).
    :type out_shape: tuple
    :param f_extractor: Feature extractor. 
    :type f_extractor: tf.Layer
    :param reg_param: Regularization parameter for the weights in the Dense layer. Defaults to 0.001.
    :type reg_param: float
    :param input_name: Name of the input. Defaults to 'input_signal'.
    :type input_name: str
    :param model_name: Name of the architecture. Defaults to 'Embedder'.
    :type model_name: str
    :param training: Whether to run model in training mode. Particularly relevant for layers such as \
    Batch Normalization and Dropout. Defaults to *None*.
    :type training: bool, optional
    :return: Embedder
    """

    # Define input
    sig = Input((in_shape), name = input_name)
    
    # Extract features
    features = f_extractor(sig, training = training)
    
    # Predict embedding
    pred_emb = embed(features, out_shape, reg_param, apply_l2 = True)
    
    # Build model
    model = Model(inputs = sig, outputs = pred_emb, name = model_name)
    
    return model
    
#############################################################################################

def build_simple_classifier(in_shape, out_shape, f_extractor, reg_param = 1e-3,
                            taxonomic_levels = ['species'],
                            input_name = "input_signal", model_name = "Simple_Classifier",
                            training = None):
    """Build model for classifying signals according to only one taxonomic level i.e. genus or species. \  
    It is possible to extend the loss function to penalize the model for getting wrong \
    higher hierarchies, as well (just add them to the tax_levels list).
    
    Architectures 
        f_extractor(Layer) + Dense + Softmax 
    
    Outputs 
        predicted class probabilities
    
    :param in_shape: Input shape. No need to specify batch dimension.
    :type in_shape: tuple
    :param out_shape: Model output shape (here equal to the size of embedded taxonomic level).
    :type out_shape: tuple
    :param f_extractor: Feature extractor. 
    :type f_extractor: tf.Layer
    :param reg_param: Regularization parameter for the weights in the Dense layer. Defaults to 0.001.
    :type reg_param: float
    :param taxonomic_levels: Taxonomic levels to include in the loss function. The model only predicts one \
    but multiple superior levels can be inferred from the predicted one and penalized in the loss. Defaults to ['species'].
    :type taxonomic_levels: list
    :param input_name: Name of the input. Defaults to 'input_signal'.
    :type input_name: str
    :param model_name: Name of the architecture. Defaults to 'Simple_Classifier'.
    :type model_name: str
    :param training: Whether to run model in training mode. Particularly relevant for layers such as \
    Batch Normalization and Dropout. Defaults to *None*.
    :type training: bool, optional
    :return: Simple Classifier
    """
    
    # Define input
    sig = Input((in_shape), name = input_name)
    
    # Extract features
    features = f_extractor(sig, training = training)       
    
    # Predict normalized probabilities
    outputs = []
    pred_prob = predict_prob(features, out_shape, reg_param, taxonomic_levels[0],
                             add_softmax = True, as_block = False)
    outputs.append(pred_prob)
    
    # Add penalty terms to the loss for other taxonomic levels as well
    if len(taxonomic_levels) > 1:
        for i in range(len(taxonomic_levels)-1):
            pred_prob_id = Lambda(lambda x: tf.identity(x), name = taxonomic_levels[i+1])(pred_prob)
            outputs.append(pred_prob_id)
    
    # Build model
    model = Model(inputs = sig, outputs = outputs, name = model_name)
    
    return model
    
#############################################################################################

def build_simple_embedder_classifier(in_shape, out_shapes, f_extractor, reg_param = 1e-3,
                                     taxonomic_levels = ['species'],
                                     input_name = "input_signal", model_name = "Simple_Embedder_Classifier",
                                     training = None):
    """Build model for learning the embedding of one taxonomic level \
    and classifying signals according to only one taxonomic level (does not have to coincide to the embedding). \    
    It is possible to extend the loss function to penalize the model for getting wrong \
    higher hierarchies, as well (just add them to the tax_levels list).
    
    Architectures (layers in brackets are branched out) 
        f_extractor(Layer) + Dense(+ L2) + DenseBlock + Softmax
                                             
    Outputs 
        predicted embedding and class probabilities
        
    :param in_shape: Input shape. No need to specify batch dimension.
    :type in_shape: tuple
    :param out_shape: Model output shape (here equal to the size of embedded taxonomic level).
    :type out_shape: tuple
    :param f_extractor: Feature extractor. 
    :type f_extractor: tf.Layer
    :param reg_param: Regularization parameter for the weights in the Dense layer. Defaults to 0.001.
    :type reg_param: float
    :param taxonomic_levels: Taxonomic levels to include in the loss function. The model only predicts one \
    but multiple superior levels can be inferred from the predicted one and penalized in the loss. Defaults to ['species'].
    :type taxonomic_levels: list
    :param input_name: Name of the input. Defaults to 'input_signal'.
    :type input_name: str
    :param model_name: Name of the architecture. Defaults to 'Simple_Embedder_Classifier'.
    :type model_name: str
    :param training: Whether to run model in training mode. Particularly relevant for layers such as \
    Batch Normalization and Dropout. Defaults to *None*.
    :type training: bool, optional
    :return: Simple Embedder Classifier 
    """
    
    # Define input
    sig = Input((in_shape), name = input_name)
    
    # Extract features
    features = f_extractor(sig, training = training)       
    
    # Predict embedding
    outputs = []
    pred_emb = embed(features, out_shapes[0], reg_param, apply_l2 = False)
    # Note: identity function necessary in order to build a 2nd branch 
    #       if apply_l2 is True
    #pred_emb_out = Identity(name = 'embedding')(pred_emb)
    pred_emb_out = L2_Norm(name = 'embedding')(pred_emb)
    outputs.append(pred_emb_out)
    
    # Predict normalized probabilities
    pred_prob = predict_prob(pred_emb, out_shapes[1], reg_param, taxonomic_levels[0],
                             add_softmax = True, as_block = True)
    outputs.append(pred_prob)
    
    # Add penalty terms to the loss for other taxonomic levels as well
    if len(taxonomic_levels) > 1:
        for i in range(len(taxonomic_levels)-1):
            pred_prob_id = Identity(name = taxonomic_levels[i+1])(pred_prob)
            outputs.append(pred_prob_id)
    
    # Build model
    model = Model(inputs = sig, outputs = outputs, name = model_name)
    
    return model

#############################################################################################

def build_hiera_classifier(in_shape, out_shapes, f_extractor, reg_param = 1e-3,
                           taxonomic_levels = ['genus', 'species'], parallel = True,
                           input_name = "input_signal", model_name = "Hiera_Classifier",
                           training = None):
    """Build model for classifying signals according to more than one taxonomic level.
    
    Architectures (layers in brackets are branched out) 
        - (series)   f_extractor(Layer) + Dense(+ Softmax) + DenseBlock(+ Softmax) ... + DenseBlock + Softmax
        - (parallel) f_extractor(Layer) (+ Dense + Softmax) (+ Dense + Softmax) ...
                                       
    Outputs 
        predicted class probabilities
    
    :param in_shape: Input shape. No need to specify batch dimension.
    :type in_shape: tuple
    :param out_shape: Model output shape (here equal to the size of embedded taxonomic level).
    :type out_shape: tuple
    :param f_extractor: Feature extractor. 
    :type f_extractor: tf.Layer
    :param reg_param: Regularization parameter for the weights in the Dense layer. Defaults to 0.001.
    :type reg_param: float
    :param taxonomic_levels: Taxonomic levels to include in the loss function. The model only predicts one \
    but multiple superior levels can be inferred from the predicted one and penalized in the loss. Defaults to ['genus', 'species'].
    :type taxonomic_levels: list
    :param parallel: Whether to attach parallel Dense layers for every prediction. Otherwise, they are connected one after another. Defaults to *True*.
    :type parallel: bool
    :param input_name: Name of the input. Defaults to 'input_signal'.
    :type input_name: str
    :param model_name: Name of the architecture. Defaults to 'Hiera_Classifier'.
    :type model_name: str
    :param training: Whether to run model in training mode. Particularly relevant for layers such as \
    Batch Normalization and Dropout. Defaults to *None*.
    :type training: bool, optional
    :return: Hierarchical Classifier 
    """
    
    # Define input
    sig = Input((in_shape), name = input_name)
    
    # Extract features
    features = f_extractor(sig, training = training)       
    
    # Predict normalized probabilities
    outputs = []
    if parallel:
        for tax, out_shape in zip(taxonomic_levels, out_shapes):
            pred_prob = predict_prob(features, out_shape, reg_param, tax,
                                     add_softmax = True, as_block = False)
            outputs.append(pred_prob)
    else: # in series
        pred_prob = features
        as_block = False # only the first iteration needs a simple Dense layer,
                         # the following ones need a DenseBlock
        for tax, out_shape in zip(taxonomic_levels, out_shapes):
            pred_prob = predict_prob(pred_prob, out_shape, reg_param, tax,
                                     add_softmax = False, as_block = as_block)
            pred_prob_out = Softmax(name = tax)(pred_prob) # branch out
            # Add the softmax probabilities to the outputs to be given to the loss,
            # but feed forward in the network the unnormalized ones
            outputs.append(pred_prob_out)
            as_block = True 
               
    # Build model
    model = Model(inputs = sig, outputs = outputs, name = model_name)
    
    return model

#############################################################################################

def build_hiera_embedder_classifier(in_shape, out_shapes, f_extractor, reg_param = 1e-3,
                                    taxonomic_levels = ['genus', 'species'], parallel = True,
                                    input_name = "input_signal", model_name = "Hiera_Embedder_Classifier",
                                    training = None):
    """Build model for learning embeddings and classifying signals according to more than one taxonomic level.
    
    Architectures (layers in brackets are branched out) 
        - (series) f_extractor(Layer) + Dense (+ L2) + DenseBlock(+ Softmax) + ... + DenseBlock + Softmax
        - (parallel) f_extractor(Layer) + Dense (+ L2) (+ DenseBlock + Softmax) ...
                                              
    Outputs 
        predicted embedding and class probabilities
    
    :param in_shape: Input shape. No need to specify batch dimension.
    :type in_shape: tuple
    :param out_shape: Model output shape (here equal to the size of embedded taxonomic level).
    :type out_shape: tuple
    :param f_extractor: Feature extractor. 
    :type f_extractor: tf.Layer
    :param reg_param: Regularization parameter for the weights in the Dense layer. Defaults to 0.001.
    :type reg_param: float
    :param taxonomic_levels: Taxonomic levels to include in the loss function. The model only predicts one \
    but multiple superior levels can be inferred from the predicted one and penalized in the loss. Defaults to ['genus', 'species'].
    :type taxonomic_levels: list
    :param parallel: Whether to attach parallel Dense layers for every prediction. Otherwise, they are connected one after another. Defaults to *True*.
    :type parallel: bool
    :param input_name: Name of the input. Defaults to 'input_signal'.
    :type input_name: str
    :param model_name: Name of the architecture. Defaults to 'Hiera_Embedder_Classifier'.
    :type model_name: str
    :param training: Whether to run model in training mode. Particularly relevant for layers such as \
    Batch Normalization and Dropout. Defaults to *None*.
    :type training: bool, optional
    :return: Hierarchical Embedder Classifier
    """
    
    # Define input
    sig = Input((in_shape), name = input_name)
    
    # Extract features
    features = f_extractor(sig, training = training)       
    
    # Predict embedding
    pred_emb = embed(features, out_shapes[0], reg_param, apply_l2 = False)
    # Note: identity function necessary in order to build a 2nd branch 
    #       if apply_l2 is True
    #pred_emb_out = Identity(name = 'embedding')(pred_emb)
    pred_emb_out = L2_Norm(name = 'embedding')(pred_emb)
    
    # Predict normalized probabilities
    outputs = [pred_emb_out] # embedded vector is the first output
    if parallel:
        for tax, out_shape in zip(taxonomic_levels, out_shapes[1:]): # first out_shape was for embedding
            pred_prob = predict_prob(pred_emb, out_shape, reg_param, tax,
                                     add_softmax = True, as_block = True)
            outputs.append(pred_prob)
    else: # in series
        pred_prob = pred_emb
        for tax, out_shape in zip(taxonomic_levels, out_shapes[1:]):
            pred_prob = predict_prob(pred_prob, out_shape, reg_param, tax,
                                     add_softmax = False, as_block = True)
            pred_prob_out = Softmax(name = tax)(pred_prob) # branch out
            # Add the softmax probabilities to the outputs to be given to the loss,
            # but feed forward in the network the unnormalized ones
            outputs.append(pred_prob_out)
               
    # Build model
    model = Model(inputs = sig, outputs = outputs, name = model_name)
    
    return model

#############################################################################################  
#############################################################################################
"""Auxiliary functions called inside the *build*-functions"""

def embed(x, out_shape, reg_param = 1e-3, apply_l2 = False):
    """Compute embedding of a vector by passing it through a Dense layer \
    and l2-normalizing it (meant to be used as module in models with embedding layers).
    
    :param x: Input features.
    :type x: Tensor
    :param out_shape: Dense layer output shape.
    :type out_shape: int
    :param reg_param: Regularization parameter for the weights in the Dense layer. Defaults to 0.001.
    :type reg_param: float
    :param apply_l2: Whether to l2-normalize the features output by the Dense layer. Defaults to *False*.
    :type apply_l2: bool
    :return: Predicted embedding.
    """
    
    pred_emb = Dense(out_shape, dtype = 'float32', name = 'dense_emb', 
                     kernel_regularizer = regularizers.l2(reg_param))(x)
    if apply_l2:
        pred_emb = L2_Norm(name = 'embedding')(pred_emb)
    
    return pred_emb

#############################################################################################

def predict_prob(x, out_shape, reg_param = 1e-3, taxonomic_level = 'species',
                 add_softmax = True, as_block = False):
    """Compute (normalized) probabilities of signal belonging to different classes of specified taxonomic level.
    
    :param x: Input features.
    :type x: Tensor
    :param out_shape: Dense layer output shape.
    :type out_shape: int
    :param reg_param: Regularization parameter for the weights in the Dense layer. Defaults to 0.001.
    :type reg_param: float
    :param taxonomic_level: Predicted taxonomic level. Defaults to 'species'.
    :type taxonomic_level: str
    :param add_softmax: Whether to normalize the probabilities with a Softmax layer. Defaults to *True*.
    :type add_softmax: bool
    :param as_block: Whether to pass **x** through a simple Dense layer or a Dense block. See wingbeats.modelling.layers. Defaults to *False*.
    :type as_block: bool
    :return: Predicted probability vector.
    """
    
    # Pass vector through simple Dense layer or through Dense block
    if as_block:
        pred_prob = DenseBlock(out_shape, reg_param, name = 'dense_'+taxonomic_level)(x)
    else:
        pred_prob = Dense(out_shape, dtype = 'float32', name = 'dense_'+taxonomic_level, 
                          kernel_regularizer = regularizers.l2(reg_param))(x)
        
    if add_softmax:
        pred_prob = Softmax(name = taxonomic_level)(pred_prob)
    
    return pred_prob