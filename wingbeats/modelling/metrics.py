"""Library for custom metrics and losses"""



# Import libraries
import tensorflow as tf
import numpy as np



def embedding_similarity(emb_matrix):
    """Compute embedding similarity of current batch.
    
    :param emb_matrix: Matrix of hierarchical embeddings.
    :type emb_matrix: array
    :return: Function that computes similarity between true and predicted embeddings.
    """
    
    def similarity(y_true, pred_emb):
        """Compute similarity between true and predicted embeddings.
        
        :param y_true: True class indices
        :type y_true: list or array
        :param pred_emb: Output of the embedder (after l2-normalization).
        :type pred_emb: list or array
        :return: Dot products between true and predicted embeddings.
        """

        b_size = tf.shape(y_true)[0] # batch size (may vary at the end of a training step)

        # Get equivalent embeddings of the true labels (vectorized version)
        true_emb = tf.gather(emb_matrix, tf.cast(y_true, tf.int32), axis=0)
        true_emb = tf.squeeze(true_emb)

        # Embedding similarity as sum of dot products divided by batch size
        # Note: All variables need to be float32 in order to support MixedPrecision on GPU
        dot_products = tf.cast( tf.reduce_sum(pred_emb*true_emb), dtype = tf.float32 ) / tf.cast(b_size, dtype=tf.float32)
        
        return dot_products
    return similarity

#############################################################################################

def embedding_loss(emb_matrix):
    """Compute embedding loss of current batch as *1.0 - embedding_similarity*.
        
    :param emb_matrix: Matrix of hierarchical embeddings.
    :type emb_matrix: array
    :return: Function that computes the embedding loss w.r.t. true and predicted embeddings.
    """
    
    def emb_loss(y_true, pred_emb):

        return tf.constant(1.0, dtype=tf.float32) - embedding_similarity(emb_matrix)(y_true, pred_emb)
    return emb_loss
    
#############################################################################################

def focal_loss(gamma = 2.0):
    """Compute focal loss as modified cross entropy loss. The goal is to penalize hard examples harsher.
    
    :param gamma: Penalty exponent. If **gamma** is 0.0, the focal loss becomes the standard cross entropy loss. Defaults to 2.0.
    :type gamma: float
    :return: Focal loss function w.r.t. true and predicted probabilities.
    """
    
    def loss(y_true, y_pred):

        b_size = tf.shape(y_true)[0] # batch size (may vary at the end of a training step)

        # Get probabilities of true predictions
        true_prob = tf.gather(y_pred, tf.cast(y_true, tf.int32), axis=-1, batch_dims=1)
        
        # Note: All variables need to be float32 in order to support MixedPrecision on GPU       
        return -(1-true_prob)**gamma * tf.math.log(true_prob) / tf.cast(b_size, dtype=tf.float32)
    return loss

#############################################################################################

def predict_gen_spec(model, X, model_name, genus_mapping, emb_matrix):
    """Make genus and species predictions on dataset **X** according to model architecture.
    
    :param model: Pretrained classifier.
    :type model: tf.Model
    :param X: Matrix of signals.
    :type X: tf.Dataset
    :param model_name: Name of the architecture. Currently only allowed: *SimpleCls*, *Emb*, *SimpleEmbCls*, \
    *HieraCls*, *HieraEmbCls*.
    :type model_name: str
    :param genus_mapping: List that maps the index of the species to the index of the genus. 
    :type genus_mapping: list
    :param emb_matrix: Matrix of hierarchical embeddings. Only needed for *Emb*.
    :type emb_matrix: array
    :return: Predicted genus and species
    :rtype: tuple
    """
    
    if model_name == 'SimpleCls':
        pred_specs = np.argmax(model.predict(X, verbose = 0), axis = -1)
        pred_gens  = genus_mapping[pred_specs]
        
    elif model_name == 'Emb':
        pred_embs = model.predict(X, verbose = 0)
        pred_specs = get_species_from_embeddings(pred_embs, genus_mapping, emb_matrix)
        pred_gens  = genus_mapping[pred_specs]
    
    elif model_name == 'SimpleEmbCls':
        _, pred_specs = model.predict(X, verbose = 0)
        pred_specs = np.argmax(pred_specs, axis = -1)
        pred_gens  = genus_mapping[pred_specs]
        
    elif model_name == 'HieraCls':
        pred_gens, pred_specs = model.predict(X, verbose = 0)
        pred_gens  = np.argmax(pred_gens,  axis = -1)
        pred_specs = np.argmax(pred_specs, axis = -1)
        
    elif model_name == 'HieraEmbCls':
        _, pred_gens, pred_specs = model.predict(X, verbose = 0)
        pred_gens  = np.argmax(pred_gens,  axis = -1)
        pred_specs = np.argmax(pred_specs, axis = -1)    
    
    return pred_gens, pred_specs

#############################################################################################

def get_species_from_embeddings(pred_embs, genus_mapping, emb_matrix):
    """Infer predicted species from predicted embeddings.
    
    :param pred_embs: Predicted embeddings to be compared via *nearest neighbor* to the true embeddings.
    :type pred_embs: array
    :param genus_mapping: List that maps the index of the species to the index of the genus. 
    :type genus_mapping: list
    :param emb_matrix: Matrix of hierarchical embeddings. Only needed for *Emb*.
    :type emb_matrix: array
    :return: Predicted species.
    :rtype: list
    """
        
    pred_specs = []
    for pred_emb in pred_embs:

        # Distances to all fixed embeddings
        pred_emb_distances = []
        for emb in emb_matrix:
            pred_emb_distances.append( tf.linalg.norm(pred_emb - emb) )

        # Predicted species index = index of the smallest distance(pred_emb, emb_list)
        pred_spec = pred_emb_distances.index(min(pred_emb_distances))
        pred_specs.append(pred_spec)
        
    return pred_specs


