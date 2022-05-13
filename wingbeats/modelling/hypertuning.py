"""Library for hyperparameter optimization functions"""



# Import libraries
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
#from uncertainties import unumpy

from wingbeats.modelling.builds import *
from wingbeats.modelling.metrics import embedding_similarity, embedding_loss
from wingbeats.modelling.metrics import predict_gen_spec

from wingbeats.processing.preprocessing import preprocess_dataset
from wingbeats.processing.postprocessing import compute_mean_conf_mat



def kfold_cv(X, y, models, genus_mapping, emb_matrix = None, n_splits = 4, epochs = 30, batch_size = 64, 
             sm = None, model_callbacks = None, sampling_rate = 16000, 
             window = None, nperseg = None, noverlap = None, cutoff = None):
    """Execute KFold Cross-Validation on dataset **(X, y)** which is to be split into **n_splits** stratified folds. \       
    User should provide **n_splits** models to be trained on each fold. \
    Each model name should follow the pattern *architecture_inputFormat* (e.g. *HieraCls_spectro*). \
    The **n_splits-1** folds held for training are also augmented using SMOTE, if **sm** is not *None*. 
    
    :param X: Matrix of signals.
    :type X: list
    :param y: Label vector.
    :type y: list
    :param genus_mapping: List containing genus indexes ef every species.
    :type genus_mapping: list
    :param emb_matrix: Matrix of hierarchical embeddings. Defaults to *None*. Only needed for *Embedder* models.
    :type emb_matrix: array
    :param n_splits: Number of folds. Defaults to 4.
    :type n_splits: int
    :param epochs: Number of epochs to train models each fold. Defaults to 30.
    :type epochs: int
    :param batch_size: Size of one signal batch in tf.Dataset. Defaults to 64.
    :type batch_size: int
    :param sm: SMOTE object to augment data. Defaults to *None*.
    :type sm: imblearn.smote, optional
    :param model_callbacks: Callbacks for model training (e.g. Early Stopping, Model Checkpoint, Learning Rate Schedules). Defaults to *None*.
    :type model_callbacks: list
    :param sampling_rate: Sampling frequency. Defaults to 16000.
    :type sampling_rate: int
    :param window: Window-function to multiply each segment with i.e. 'hann' (for psd) or ``tf.signal.hann_window`` (for spectrograms). Defaults to *None*.
    :type window: str (for psd) or function pointer (for spectrograms)
    :param nperseg: Length of a segment for applying the Welch-Transform or STFT. Defaults to *None*.
    :type nperseg: int
    :param noverlap: Lenth of overlapping region between segments. Defaults to *None*.
    :type noverlap: int
    :param cutoff: How many PSD frequencies should be kept. Defaults to *None*.
    :type cutoff: int, optional
    :return: Mean and std. dev. confusion matrices over all folds for genus and species and list of training histories for every fold.
    """   

    # Create object for splitting dataset into folds
    skf = StratifiedKFold(n_splits = n_splits)

    gen_accuracies, spec_accuracies = [], [] # list of accuracies for every fold
    gen_conf_mats,  spec_conf_mats  = [], [] # list of confusion matrices per fold
    fold_histories = [] # list of histories for every fold
    fold_ind = 1 # fold counter
    
    # Induce model name and input format
    full_model_name = models[0].name
    substr_ind = full_model_name.find('_')
    model_name, input_format = full_model_name[:substr_ind], full_model_name[substr_ind+1:]
    if input_format not in ['raw', 'psd', 'spectro']:
        print('Input format unknown!')
        return None

    # Train and evaluate each fold
    for train_index, cv_index in skf.split(X, np.asarray(y)[:,1]):

        print('\nFOLD ' + str(fold_ind))
        X_train_fold, X_cv_fold = [X[i] for i in train_index], [X[i] for i in cv_index]
        y_train_fold = np.asarray( [y[i] for i in train_index] ) 
        y_cv_fold    = np.asarray( [y[i] for i in cv_index] )
        
    
        # Apply SMOTE (outputs will be Numpy in Colab and lists in Kaggle)
        if sm is not None:
            # Note: you can only input one-column y-vectors into sm; 
            #       if y_train has more columns, you need to reconstruct it after applying sm
            X_train_fold, y_train_smote = sm.fit_resample(X_train_fold, y_train_fold[:,1])
            y_train_fold = []
            for spec in y_train_smote:
                y_train_fold.append([genus_mapping[spec], spec])

        # Convert datasets into the right formats
        train_set = preprocess_dataset(X_train_fold, y_train_fold, model_name, input_format, 
                                       sampling_rate, batch_size, window, nperseg, noverlap, cutoff, 
                                       shuffle = True, cache = False)       
        cv_set    = preprocess_dataset(X_cv_fold, y_cv_fold, model_name, input_format, 
                                       sampling_rate, batch_size, window, nperseg, noverlap, cutoff, 
                                       shuffle = False, cache = False)

        # Model fitting
        model = models[fold_ind-1]
        history = model.fit(train_set, epochs = epochs, validation_data = cv_set,
                            callbacks = model_callbacks, verbose = 0)
        fold_histories.append(history.history)

        # Predictions + metrics evaluation
        pred_gens, pred_specs = predict_gen_spec(model, cv_set, model_name, genus_mapping, emb_matrix)
        
        correct_gens  = sum(pred_gens  == y_cv_fold[:, 0])
        correct_specs = sum(pred_specs == y_cv_fold[:, 1])
        
        gen_val_acc  = correct_gens / len(pred_gens)
        spec_val_acc = correct_specs / len(pred_specs)
        
        gen_accuracies.append(gen_val_acc)
        spec_accuracies.append(spec_val_acc)        

        # Build the confusion matrix
        # For genus
        conf_mat = np.round( confusion_matrix(y_cv_fold[:, 0], pred_gens, normalize = 'true'), 2 )
        gen_conf_mats.append(conf_mat)
        # For species
        conf_mat = np.round( confusion_matrix(y_cv_fold[:, 1], pred_specs, normalize = 'true'), 2 )
        spec_conf_mats.append(conf_mat)

        # Increase fold index
        fold_ind += 1

    print('*********************************************') 
    mean_acc, std_acc = np.round( np.mean(gen_accuracies), 4 ), np.round( np.std(gen_accuracies), 4 )
    print('MEAN GENUS   VAL_ACC: ' + str(mean_acc) + ' +/- ' + str(std_acc))   
    mean_acc, std_acc = np.round( np.mean(spec_accuracies), 4 ), np.round( np.std(spec_accuracies), 4 )
    print('MEAN SPECIES VAL_ACC: ' + str(mean_acc) + ' +/- ' + str(std_acc))   
    
    # Compute mean confusion matrices
    gen_mean_mat,  gen_std_mat  = compute_mean_conf_mat(gen_conf_mats)
    spec_mean_mat, spec_std_mat = compute_mean_conf_mat(spec_conf_mats)
    
    # Note: unumpy.uarray(mean_mat, std_mat) gives you the confusion matrix in the form mean +/- std
    return gen_mean_mat, gen_std_mat, spec_mean_mat, spec_std_mat, fold_histories
    
#############################################################################################

def build_hyper_simple_classifier(hp, in_shape, out_shape, f_extractor, lr_values, reg_values,
                                  taxonomic_levels = ['species'], input_name = "input_signal", 
                                  model_name = "Hyper_Simple_Classifier", strategy = None):
    """Build Simple Classifier for Hyperband-optimization.
    
    :param hp: Hyperband object.
    :type hp: kerastuner.hyperband
    :param in_shape: Input shape. No need to specify batch dimension.
    :type in_shape: tuple
    :param out_shape: Model output shape (here equal to the size of embedded taxonomic level).
    :type out_shape: tuple
    :param f_extractor: Feature extractor. 
    :type f_extractor: tf.Layer
    :param lr_values: Discrete learning rate values.
    :type lr_values: list
    :param reg_values: Discrete regularization parameter values.
    :type reg_values: list
    :param taxonomic_levels: Taxonomic levels to include in the loss function. Defaults to ['species'].
    :type taxonomic_levels: list
    :param input_name: Name of the input. Defaults to 'input_signal'.
    :type input_name: str
    :param model_name: Name of the architecture. Defaults to 'Hyper_Simple_Classifier'.
    :type model_name: str
    :param strategy: Distribution strategy (CPU, GPU, TPU). Defaults to *None* (CPU).
    :type strategy: Strategy from tf.distribute, optional
    :return: Hyperband-optimizable Simple Classifier
    """
    
    if strategy is None:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
      
    # Hyperparameters
    hp_lr  = hp.Choice('learning_rate', values = lr_values)
    hp_reg = hp.Choice('reg_param',     values = reg_values)

    with strategy.scope():
        model = build_simple_classifier(in_shape, out_shape, f_extractor, hp_reg, 
                                        taxonomic_levels, input_name, model_name)

        model.compile(optimizer = Adam(hp_lr),
                      loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    return model

#############################################################################################
    
def build_hyper_embedder(hp, in_shape, out_shape, f_extractor, lr_values, reg_values, emb_matrix,
                         input_name = "input_signal", model_name = "Hyper_Embedder", strategy = None):
    """Build Embedder for Hyperband-optimization.
    
    :param hp: Hyperband object.
    :type hp: kerastuner.hyperband
    :param in_shape: Input shape. No need to specify batch dimension.
    :type in_shape: tuple
    :param out_shape: Model output shape (here equal to the size of embedded taxonomic level).
    :type out_shape: tuple
    :param f_extractor: Feature extractor. 
    :type f_extractor: tf.Layer
    :param lr_values: Discrete learning rate values.
    :type lr_values: list
    :param reg_values: Discrete regularization parameter values.
    :type reg_values: list
    :param emb_matrix: Matrix of hierarchical embeddings. Defaults to *None*. 
    :type emb_matrix: array
    :param input_name: Name of the input. Defaults to 'input_signal'.
    :type input_name: str
    :param model_name: Name of the architecture. Defaults to 'Hyper_Embedder'.
    :type model_name: str
    :param strategy: Distribution strategy (CPU, GPU, TPU). Defaults to *None* (CPU).
    :type strategy: Strategy from tf.distribute, optional
    :return: Hyperband-optimizable Embedder
    """
    
    if strategy is None:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
      
    # Hyperparameters
    hp_lr  = hp.Choice('learning_rate', values = lr_values)
    hp_reg = hp.Choice('reg_param',     values = reg_values)

    with strategy.scope():
        model = build_embedder(in_shape, out_shape, f_extractor, hp_reg, input_name, model_name) 
        
        model.compile(optimizer = Adam(hp_lr),
                      loss = embedding_loss(emb_matrix),
                      metrics = embedding_similarity(emb_matrix))

    return model

#############################################################################################
    
def build_hyper_simple_embedder_classifier(hp, in_shape, out_shape, f_extractor, lr_values, reg_values, 
                                           emb_matrix, taxonomic_levels = ['species'], input_name = "input_signal", 
                                           model_name = "Hyper_Simple_Embedder_Classifier", strategy = None):
    """Build Simple Embedder Classifier for Hyperband-optimization.
    
    :param hp: Hyperband object.
    :type hp: kerastuner.hyperband
    :param in_shape: Input shape. No need to specify batch dimension.
    :type in_shape: tuple
    :param out_shape: Model output shape (here equal to the size of embedded taxonomic level).
    :type out_shape: tuple
    :param f_extractor: Feature extractor. 
    :type f_extractor: tf.Layer
    :param lr_values: Discrete learning rate values.
    :type lr_values: list
    :param reg_values: Discrete regularization parameter values.
    :type reg_values: list
    :param emb_matrix: Matrix of hierarchical embeddings. Defaults to *None*. 
    :type emb_matrix: array
    :param input_name: Name of the input. Defaults to 'input_signal'.
    :type input_name: str
    :param model_name: Name of the architecture. Defaults to 'Hyper_Simple_Embedder_Classifier'.
    :type model_name: str
    :param strategy: Distribution strategy (CPU, GPU, TPU). Defaults to *None* (CPU).
    :type strategy: Strategy from tf.distribute, optional
    :return: Hyperband-optimizable Simple Embedder Classifier
    """
    
    if strategy is None:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
       
    # Hyperparameters
    hp_lr  = hp.Choice('learning_rate', values = lr_values)
    hp_reg = hp.Choice('reg_param',     values = reg_values)

    with strategy.scope():
        model = build_simple_embedder_classifier(in_shape, out_shape, f_extractor, hp_reg, 
                                                 taxonomic_levels, input_name, model_name)
        
        model.compile(
            optimizer = Adam(hp_lr),
            loss = {
                "embedding": embedding_loss(emb_matrix), 
                "species": 'sparse_categorical_crossentropy'
            },
            loss_weights = {"embedding": 1.0, "species": 1.0},
            metrics = {"embedding": embedding_similarity(emb_matrix), "species": 'accuracy'})

    return model

#############################################################################################
    
def build_hyper_hiera_classifier(hp, in_shape, out_shape, f_extractor, lr_values, reg_values,
                                 taxonomic_levels = ['genus', 'species'], parallel = True, input_name = "input_signal", 
                                 model_name = "Hyper_Hiera_Classifier", strategy = None):
    """Build Hierarchical Classifier for Hyperband-optimization.
    
    :param hp: Hyperband object.
    :type hp: kerastuner.hyperband
    :param in_shape: Input shape. No need to specify batch dimension.
    :type in_shape: tuple
    :param out_shape: Model output shape (here equal to the size of embedded taxonomic level).
    :type out_shape: tuple
    :param f_extractor: Feature extractor. 
    :type f_extractor: tf.Layer
    :param lr_values: Discrete learning rate values.
    :type lr_values: list
    :param reg_values: Discrete regularization parameter values.
    :type reg_values: list
    :param taxonomic_levels: Taxonomic levels to include in the loss function. Defaults to ['genus', 'species'].
    :type taxonomic_levels: list
    :param parallel: Whether to attach parallel Dense layers for every prediction. Otherwise, they are connected one after another. Defaults to *True*.
    :type parallel: bool
    :param input_name: Name of the input. Defaults to 'input_signal'.
    :type input_name: str
    :param model_name: Name of the architecture. Defaults to 'Hyper_Hiera_Classifier'.
    :type model_name: str
    :param strategy: Distribution strategy (CPU, GPU, TPU). Defaults to *None* (CPU).
    :type strategy: Strategy from tf.distribute, optional
    :return: Hyperband-optimizable Hierarchical Classifier
    """
    
    if strategy is None:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
      
    # Hyperparameters
    hp_lr  = hp.Choice('learning_rate', values = lr_values)
    hp_reg = hp.Choice('reg_param',     values = reg_values)

    with strategy.scope():
        model = build_hiera_classifier(in_shape, out_shape, f_extractor, hp_reg, 
                                       taxonomic_levels, parallel, input_name, model_name)
        
        model.compile(
            optimizer = Adam(hp_lr),
            loss = {
                "genus":   'sparse_categorical_crossentropy', 
                "species": 'sparse_categorical_crossentropy',
            },
            loss_weights = {"genus": 1.0, "species": 1.0},
            metrics = {"genus": 'accuracy', "species": 'accuracy'})

    return model

#############################################################################################
    
def build_hyper_hiera_embedder_classifier(hp, in_shape, out_shape, f_extractor, lr_values, reg_values, 
                                          emb_matrix, taxonomic_levels = ['genus', 'species'], 
                                          parallel = True, input_name = "input_signal", 
                                          model_name = "Hyper_Hiera_Embedder_Classifier", strategy = None):
    """Build Hierarchical Embedder-Classifier for Hyperband-optimization.
    
    :param hp: Hyperband object.
    :type hp: kerastuner.hyperband
    :param in_shape: Input shape. No need to specify batch dimension.
    :type in_shape: tuple
    :param out_shape: Model output shape (here equal to the size of embedded taxonomic level).
    :type out_shape: tuple
    :param f_extractor: Feature extractor. 
    :type f_extractor: tf.Layer
    :param lr_values: Discrete learning rate values.
    :type lr_values: list
    :param reg_values: Discrete regularization parameter values.
    :type reg_values: list
    :param emb_matrix: Matrix of hierarchical embeddings. Defaults to *None*. 
    :type emb_matrix: array
    :param taxonomic_levels: Taxonomic levels to include in the loss function. Defaults to ['genus', 'species'].
    :type taxonomic_levels: list
    :param parallel: Whether to attach parallel Dense layers for every prediction. Otherwise, they are connected one after another. Defaults to *True*.
    :type parallel: bool
    :param input_name: Name of the input. Defaults to 'input_signal'.
    :type input_name: str
    :param model_name: Name of the architecture. Defaults to 'Hyper_Hiera_Embedder_Classifier'.
    :type model_name: str
    :param strategy: Distribution strategy (CPU, GPU, TPU). Defaults to *None* (CPU).
    :type strategy: Strategy from tf.distribute, optional
    :return: Hyperband-optimizable Hierarchical Embedder-Classifier
    """
    
    if strategy is None:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    
    # Hyperparameters
    hp_lr  = hp.Choice('learning_rate', values = lr_values)
    hp_reg = hp.Choice('reg_param',     values = reg_values)

    with strategy.scope():
        model = build_hiera_embedder_classifier(in_shape, out_shape, f_extractor, hp_reg, 
                                                taxonomic_levels, parallel, input_name, model_name)
        
        model.compile(
            optimizer = Adam(hp_lr),
            loss = {
                "embedding": embedding_loss(emb_matrix), 
                "genus": 'sparse_categorical_crossentropy',
                "species": 'sparse_categorical_crossentropy'
            },
            loss_weights = {"embedding": 1.0, "genus": 1.0, "species": 1.0},
            metrics = {"embedding": embedding_similarity(emb_matrix), "genus": 'accuracy', "species": 'accuracy'})

    return model


