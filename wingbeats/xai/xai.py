"""Library for Explainable AI"""



# Import libraries
import numpy as np
import tensorflow as tf
# Download from https://github.com/albermax/innvestigate
#from innvestigate.innvestigate.analyzer import gradient_based as G # for attention methods such as GBP



def compute_gradients(model, X, target_ind = None):
    """Compute the gradients of **model** output at index **target_ind** w.r.t. input(s) **X**. \
    If target_ind not provided, the highest activation is considered.
    
    :param model: Model to compute gradients on.
    :type model: tf.Model
    :param X: Signal or batch of signals.
    :type X: list or array
    :param target_ind: Class index w.r.t. which to compute the gradients. Defaults to *None* (highest activation).
    :type target_ind: int, optional
    :return: Gradients and also the prediction(s).
    """
    
    with tf.GradientTape() as tape:
        tape.watch(X)
        pred = model(X, training=False) 
        # Note: For MC-Dropout, the training parameter has already been set to True in layers.py.
        #       For the predictions here keep training = False, otherwise BatchNorm will behave differently
        #       and the model will always predict the dominant class.

        if target_ind is None: 
            pred_ind = np.argmax(pred.numpy().flatten())
            loss = pred[0][pred_ind] # highest prediction
        else:
            loss = pred[0][target_ind] # probability of target class 

    return tape.gradient(loss, X), pred    
    
#############################################################################################
  
def saliency_map(model, X, target_ind = None):
    """Compute the saliency map(s) of **model** output(s) w.r.t. input(s) **X**.
    
    :param model: Model to compute gradients on.
    :type model: tf.Model
    :param X: Signal or batch of signals.
    :type X: list or array
    :param target_ind: Class index w.r.t. which to compute the gradients. Defaults to *None* (highest activation).
    :type target_ind: int, optional
    :return: Heatmap(s) with the same dimensions as the input sample and the corresponding predictions.
    """

    # Compute gradients w.r.t. X
    heatmap, pred = compute_gradients(model, X, target_ind)
    heatmap = tf.math.abs(heatmap)
    heatmap = np.max(heatmap, axis=-1)[0] # max along each RGB channel

    # Normalize to range between 0 and 1
    arr_min, arr_max  = np.min(heatmap), np.max(heatmap)
    heatmap = (heatmap - arr_min) / (arr_max - arr_min + 1e-16)

    return heatmap, pred

#############################################################################################

def guided_back_prop(analyzer, model, X, target_ind = None):
    """Compute the GBP map(s) of **model** output(s) w.r.t. input(s) **X**.
    
    :param analyzer: Gradient based analyzer, i.e. **from innvestigate.innvestigate.analyzer import gradient_based as G; analyzer = G.GuidedBackprop(model)**.
    :type analyzer: Class created by factory method from *innvestigate*.
    :param model: Model to compute gradients on.
    :type model: tf.Model
    :param X: Signal or batch of signals.
    :type X: list or array
    :param target_ind: Class index w.r.t. which to compute the gradients. Defaults to *None* (highest activation).
    :type target_ind: int, optional
    :return: Heatmap(s) with the same dimensions as the input sample and the corresponding predictions.
    """
    
    # Create GBP analyzer
    #analyzer = G.GuidedBackprop(model)

    if target_ind is None: # max activation
        analysis = analyzer.analyze(X)
    else:  
        analysis = analyzer.analyze(X, neuron_selection = target_ind)

    heatmap = list(analysis.values())[0] # heatmap is the first and only value of the analysis dictionary
    heatmap = tf.math.abs(heatmap)
    heatmap = np.max(heatmap, axis=-1)[0] # max along each RGB channel

    # Normalize to range between 0 and 1
    arr_min, arr_max  = np.min(heatmap), np.max(heatmap)
    heatmap = (heatmap - arr_min) / (arr_max - arr_min + 1e-16)

    # Compute prediction vector (to be in accordance with the signature of the saliency map function)
    pred = model(X, training=False)

    return heatmap, pred

#############################################################################################
    
def interpolate_images(baseline, image, alphas):
    """Compute a scale of images of increasing intensity between **baseline** and original **image**.
    
    :param baseline: Starting image i.e. zero image.
    :type baseline: array
    :param image: Original (target) image.
    :type image: array
    :param alphas: Incremental intensity steps.
    :type: list
    :return: Images of increasing intensities.
    :rtype: array
    """
    
    # Resize the alphas to the same dimensions of the original image + one batch axis
    alphas_x   = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis] 
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta  = input_x - baseline_x # intensity difference between baseline and original
    images = baseline_x + alphas_x * delta
    
    return images    
    
############################################################################################# 
  
def integral_approximation(gradients):
    """Approximate integral with the trapezoid rule.
    
    :param gradients: Discrete gradient values to integrate.
    :type gradients: array
    :return: Integral approximation
    """
    
    # Riemann trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    
    return integrated_gradients
    
############################################################################################# 
    
def integrated_gradients(model, x, baseline, steps = 50, target_ind = None):
    """Compute attention map through the Integrated Gradients method.
    
    :param model: Model to compute gradients on.
    :type model: tf.Model
    :param x: Signal or batch of signals.
    :type x: list or array
    :param baseline: Starting image i.e. zero image for interpolation.
    :type baseline: array
    :param steps: Number of discrete steps (= number of interpolated images). Defaults to 50.
    :type steps: int
    :param target_ind: Class index w.r.t. which to compute the gradients. Defaults to *None* (highest activation).
    :type target_ind: int, optional
    """
    
    # Generate steps intervals for integral_approximation()
    alphas = tf.linspace(start = 0.0, stop = 1.0, num = steps + 1) 

    # Compute in-between images of different intensities betwee baseline and original image
    interpolated_images = interpolate_images(baseline, x, alphas)

    # Get gradients for all in-between images
    path_gradients, _ = compute_gradients(model, interpolated_images, target_ind) 

    # Integrate the gradients
    ig = integral_approximation(gradients = path_gradients)

    # Sum of the attributions across color channels for visualization
    # The attribution mask shape is a grayscale image with height and width equal to the original image
    return tf.reduce_sum(tf.math.abs(ig), axis=-1) 

#############################################################################################
    
def flip_pixels(model, image, target_ind, means, std_devs, 
                percent_to_flip = 0.25, percent_step = 0.005, heatmap = None):
    """Apply Pixel Flipping to an image of dimensions 1 x m x n x 3 of Tensor type. \
    **Percent_to_flip** % of the pixels are chosen in **percent_step** % batches to be replaced with \
    a randomly chosen value from the normal distribution with params. **means** and **std_devs** (one for every pixel). \
    Parameter **heatmap** specifies the attention map type that is applied (saliency_map, guided_back_prop or None for random flipping). \
    The model needs to be stripped of its softmax layer.
        
    :param model: Model to make predictions.
    :type model: tf.Model
    :param image: Input image to flip pixels on.
    :type image: array or Tensor
    :param target_ind: Class index w.r.t. which to compute the gradients. 
    :type target_ind: int
    :param means: Matrix of mean values.
    :type means: array
    :param std_devs: Matrix of standard deviations.
    :type std_devs: array
    :param percent_to_flip: Percent of pixels to flip in total. Defaults to 0.25.
    :type percent_to_flip: float
    :param percent_step: Percentual increment of flipped pixels. Defaults to 0.005.
    :type percent_step: float
    :param heatmap: Attention map according to which to flip the pixels. Defaults to *None* (random flipping).
    :type heatmap: array. 
    :return: Prediction scores w.r.t. every species and every flipping step, indexes of the flipped pixels (as 1D array) \
    and output image after flipping the pixels
    """

    preds = [] # list for predictions after every flipping step
    num_rows, num_cols = image.shape[1], image.shape[2] # image dimensions 
    # Note: along the 3rd dim. the same value will be inserted

    total_k = int( num_cols*num_rows * percent_to_flip ) # max. number of pixels to deactivate 
    step_k  = int( num_cols*num_rows * percent_step )    # batch size of flipped pixels 

    # Either compute attention map and flip pixels according to gradients
    # or just flip them randomly
    if heatmap is None:
        top_ind = np.random.choice(num_cols*num_rows, total_k) # random pixels

    else:
        # Negate heatmap  -> highest gradients are sorted at the beginning
        # Flatten heatmap -> 1D indexes in top_ind can be applied
        neg_flat_heatmap = -heatmap.flatten()

        #top_ind = np.argpartition(neg_flat_heatmap, total_k)[ : total_k] # first total_k pixels with highest gradients (unsorted)
        #top_ind = np.argsort(neg_flat_heatmap[top_ind]) # ... (sorted)  
        top_ind = np.argsort(neg_flat_heatmap)[ : total_k]

    # First prediction before flipping
    preds.append( model(image, training=False)[0] ) # whole prediction vector

    # Convert Tensor to Numpy to allow assignment
    image = image.numpy() 

    # Apply pixel flipping
    for k in range(0, total_k - total_k%step_k, step_k): # loop through pixel batches
        for i in range(k, k + step_k): # loop through current batch
            # Set pixels to a random value drawn from the normal distribution at position (ind_i, ind_j)
            ind_i = top_ind[i] // num_cols # y coord. from flat indexes
            ind_j = top_ind[i] % num_cols  # x coord.
            image[0, ind_i, ind_j, :] = np.random.normal(means[ind_i, ind_j], std_devs[ind_i, ind_j])

        preds.append( model(image, training=False)[0] ) # whole prediction vector

    return preds, top_ind, image[0, :, :, :]
     
#############################################################################################
    
def quantile_maps(heatmaps, quantiles, nbins = 50):
    """Compute quantile maps out of the pixel distributions from a list of heatmaps.
    
    :param heatmaps: Distribution of heatmaps to compute statistics on.
    :type heatmaps: list
    :param quantiles: Quantiles to be computed i.e. 0.25, 0.5, 0.75.
    :type quantiles: list
    :param nbins: Number of bins in histograms. Defaults to 50.
    :type nbins: int
    :return: Quantile heatmaps.
    """
    
    num_rows, num_cols = heatmaps[0].shape

    # Compute distributions for every pixel over the heatmaps
    heatmaps = np.reshape(heatmaps, (len(heatmaps), num_rows, num_cols))
    pixel_bins = np.empty((nbins, num_rows, num_cols))
    pixel_freq = np.empty((nbins, num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            n, bins = np.histogram(heatmaps[:, i, j], bins = nbins) 
            bin_width = bins[1] - bins[0] 
            pixel_bins[:, i, j] = bins[:-1] + bin_width/2 # store the midpoints of the bins
            pixel_freq[:, i, j] = n

    # Define array of quantile maps
    len_q = len(quantiles)
    q_heatmaps = np.empty((len_q, num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            unfolded = []
            for k in range(nbins): # unfold the histogram into a list of multiple occurences
                unfolded += [pixel_bins[k, i, j]] * int(pixel_freq[k, i, j])

            # For every quantile store the corresponding pixel
            q_heatmaps[ : len_q, i, j] = np.quantile(unfolded, quantiles[ : len_q])

    return q_heatmaps  
  