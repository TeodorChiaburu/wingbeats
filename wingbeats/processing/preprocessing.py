"""Library for preprocessing functions"""



# Import libraries
import numpy as np
import tensorflow as tf
import scipy.io.wavfile
import librosa
import glob
import os
from scipy import signal



def load_signals(load_path, species, X, y, genus_mapping = None, with_scipy = True):
    """Load *wav* signals from **load_path** into matrix **X**.
    
    The subfolder structure in **load_path** should follow **species**.
    A vector of species labels is also created at the same time;
    if **genus_mapping** is provided, such that ``genus_mapping[i]`` gives the
    genus index of the i-th species, then **y** will also store the genus index
    as a separate column. You can also choose whether to load the signals
    with ``scipy.io.wavfile.read`` or with ``librosa.load``. Beware that the latter
    converts the signal to *float* automatically by mapping it into [-1, 1].
    The same can be achieved with ``scipy`` by dividing by ``np.iinfo(np.int16)``.
    
    :param load_path: Path to load the *wav* files from.
    :type load_path: str
    :param species: List with species names as strings.
    :type species: list
    :param X: Empty list to append the signals to.
    :type X: list
    :param y: Empty list to append the labels to.
    :type y: list
    :param genus_mapping: List containing genus indexes ef every species. Defaults to *None*.
    :type genus_mapping: list, optional
    :param with_scipy: Boolean to control whether to load the signals using the ``scipy`` library \
        or ``librosa``. Defaults to *True*.
    :type with_scipy: bool, optional
    :returns: List of species lengths.
    """
    
    len_spec = [] # list of number of samples per species
    for i in range(len(species)):
        # Read audios per species
        audio_paths = glob.glob(load_path + species[i] + '/*')
        num_paths = len(audio_paths)
        len_spec.append(num_paths)

        # Build label vector
        if genus_mapping is None:
            y += [i] * num_paths # vector of labels for species
        else:
            y += [[genus_mapping[i], i]] * num_paths # vector of labels for genus and species
        
        # Note: librosa.load runs very slow in Colab. Use scipy.wavfile instead and normalize signals afterwards.
        #       You may use librosa in Kaggle, but beware that they apply a different normalization than simply
        #       dividing rows by their maximums
        if with_scipy:
            for j in range(num_paths):
                X.append(scipy.io.wavfile.read(audio_paths[j])[1])
        else:
            for j in range(num_paths):
                X.append(librosa.load(audio_paths[j], sr = None, res_type = None)[0])
          
        print(species[i] + ': ' + str(num_paths))
    print('\nTotal: ' + str(len(X)))
    
    return len_spec

#############################################################################################

def create_model_folders(models_path):
    """Create folder structure for each model architecture and each input format.
    
    *Example*: The folder **models** has the subfolder **Hierarchical Classifier**, \
    which in turn has two subfolders **psd** and **spectro**, each with subfolders **histories** \
    and **logs**.
    
    :param models_path: Path with model folders.
    :type models_path: str
    """

    os.chdir(models_path)
    model_folders = glob.glob('*')

    for fold in model_folders:
        os.chdir(fold)
        os.mkdir('psd')
        os.mkdir('spectro')
        os.chdir(models_path)

        model_subfolders = glob.glob(fold + '/*')
        for subfold in model_subfolders:
            os.chdir(subfold)
            os.mkdir('histories')
            os.mkdir('logs')
            os.chdir(models_path)

#############################################################################################

def shift_data(signals, labels, shift_axis, shift_max, shift_min = 0):
    """Shift **signals** along time axis.
    
    Every row in the batch **signals** is randomly shifted to the right or left
    by a random amount between **shift_min** and **shift_max** % of the original length of the signal.
    The whole batch is shifted by the same amount.
    The **labels** are not modified, but are requested by the mapping format of ``tf.Dataset``.
    Note: When inputing single signals, choose shift_axis = 0; in case of 2D batches, choose shift_axis = 1.
          
    :param signals: Batch of signals to shift in time.
    :type signals: array
    :param labels: Signal labels.
    :type labels: list
    :param shift_axis: Axis along which the signals are shifted.
    :type shift_axis: int
    :param shift_max: Maximal shifting percent.
    :type shift_max: float
    :param shift_min: Minimal shifting percent. Defaults to 0.
    :type shift_min: float, optional
    :return: Shifted signals and labels.
    :rtype: tuple
    """

    len_signal  = tf.shape(signals)[shift_axis]

    # Choose a random number from closed interval [shift_min, shift_max] with a random sign
    shift_percent = np.random.choice([1, -1]) * np.random.randint(shift_min, shift_max+1)
    shift = int(len_signal * shift_percent / 100) # shifting amount

    # Replace current signals with shifted signals
    signals = tf.roll(signals, shift, axis = shift_axis)
  
    return signals, labels

#############################################################################################

def add_noise(sig, label, snr = 10):  
    """Add random noise to a signal **sig** from a ``tf.Dataset``.
    
    The **label** is not modified but requested by the mapping format of ``tf.Dataset``.
    
    :param sig: Signals to inject noise into.
    :type sig: ndarray or Tensor
    :param label: Signal label.
    :type label: int    
    :param snr: Signal to Noise Ratio (in dB). Defaults to 10.
    :type snr: float
    :return: Noisy signals and their labels.
    :rtype: tuple
    """

    # Power of input signal as squared of its amplitudes (in dB)  
    signal_power = 10*tf.math.log( tf.math.reduce_mean(tf.math.multiply(sig, sig)) ) / tf.math.log(10.)
    sigma_noise  = 10**((signal_power - snr)/20) 
    signal_noise = tf.random.normal([len(sig)], 0, sigma_noise)
    signal_noise = tf.expand_dims(signal_noise, axis = -1) # Add dim. to match dimensions of original signal

    return tf.math.add(sig, signal_noise), label

#############################################################################################

def add_noise_to_list(sig, snr = 10):
    """Add random noise to a signal **sig** callable as a list..
    
    :param sig: Signals to inject noise into.
    :type sig: list 
    :param snr: Signal to Noise Ratio (in dB). Defaults to 10.
    :type snr: float
    :return: Noisy signal.
    :rtype: list
    """

    signal_power    = 10*np.log10( np.mean( sig**2 ) )
    sigma_noise     = 10**((signal_power - snr)/20) 
    signal_noise    = np.random.normal(0, sigma_noise, len(sig))

    return sig + signal_noise

#############################################################################################

def convert_to_fourier(X):
    """Convert a matrix **X** of raw amplitudes into a 3d tensor of complex Fourier coefficients.
    
    Each row of *n* amplitude values will be mapped to 2 new neighbouring rows of *n/2* 
    Fourier coefficients (real part on the 1st row, imaginary part on the 2nd row).
    The matrix grows into a two-slice tensor (all real parts on the 1st slice, all imaginary parts on the 2nd).
    Note that for real valued signals, the Fourier coeffiecients are symmetric (real part even, imaginary part odd).
    
    :param X: Matrix of signals.
    :type X: list or array
    """

    # Matrix dimensions will change during loop, so save the current ones
    num_rows = len(X)
    num_cols = len(X[0])

    for i in range(num_rows):

        fourier_coeff = np.fft.fft(X[i])[0:num_cols//2] # half of the complex Fourier coefficients

        # Each row of amplitudes gets replaced by a new row of real parts
        # and gets extended in depth with a new row of corresponding imaginary parts
        X[i] = np.stack([np.real(fourier_coeff), np.imag(fourier_coeff)], axis = 1)

        # Track of progress
        if i % 10000 == 0:
            print(i)
    print('Converted to Fourier. New shape: ' + str(np.shape(X)))

#############################################################################################

def convert_to_psd(X, fs, window, nperseg, noverlap, cutoff = None):
    """Convert a matrix of raw amplitudes into a matrix of PSD values through the Welch-Transform.
    
    :param X: Matrix of raw signals.
    :type X: list or array
    :param fs: Sampling frequency.
    :type fs: int
    :param window: Window-function to multiply each segment with i.e. 'hann'.
    :type window: str
    :param nperseg: Length of a segment.
    :type nperseg: int
    :param noverlap: Lenth of overlapping region between segments.
    :type noverlap: int
    :param cutoff: How many PSD frequencies should be kept. Defaults to *None*.
    :type cutoff: int, optional
    """

    num_rows = len(X) 
    for i in range(num_rows):
        _, psd_sig = 10*np.log10(signal.welch(X[i], fs = fs, window = window, 
                                              nperseg = nperseg, noverlap = noverlap))
        
        # In case not all frequencies should be kept
        if cutoff is None:
            X[i] = psd_sig
        else:
            X[i] = psd_sig[:cutoff]
        
        # Track of progress
        if i % 10000 == 0:
            print(i)
    print('Converted to PSD. New shape: ' + str(np.shape(X)))

#############################################################################################

@tf.function
def convert_to_spectro(sig, labels, fft_length, frame_step, 
                       window = tf.signal.hann_window, cutoff = None):   
    """Generate spectrograms out of raw signals (converted into ``tf.Dataset``) through Short Time Fourier Transform.
    
    :param sig: Batch of signals to be transformed into spectrograms.
    :type sig: Tensor
    :param labels: Labels of the signals.
    :type labels: list
    :param fft_length: Length of the frames to be Fourier transformed.
    :type fft_length: int
    :param frame_step: Hopping length between frames.
    :type frame_step: int
    :param window: Window-function to multiply each segment with. Defaults to ``tf.signal.hann_window``.
    :type window: function pointer
    :param cutoff: How many frequencies should be kept. Defaults to *None*.
    :type cutoff: int, optional
    :return: Spectrograms with labels
    :rtype: tuple
    """

    # Apply Short Time Fourier Transform to compute the spectrogram
    sig = tf.signal.stft(sig, frame_length = fft_length, frame_step = frame_step, 
                         fft_length = fft_length, window_fn = window)
    
    # Convert datatype from complex to float
    sig = tf.cast(sig, dtype = tf.float32) 
    
    # Don't keep all frequencies
    if cutoff is not None:
        sig = tf.gather(sig, tf.cast( np.linspace(0, cutoff-1, cutoff), dtype = tf.int32 ), 
                        axis=-1)
    
    # Convert to decibels (directly applying np.log10 results in an error after converting to tf.Dataset)
    # Use log-rule to get log10, since tf only has natural log
    sig_dB = 10 * tf.math.log(tf.math.abs(sig)) / tf.math.log(10.) # Important! Write 10 as float, since tf.log does not accept integers

    # Sometimes infinity values come out of the logarithm
    # Replace them with 0, otherwise loss functions will be NaN during training!
    sig = tf.where(tf.math.is_inf(sig_dB), 0.0, sig_dB)

    # Swap axes and flip matrix in order to have the time domain on the x-axis and the frequencies on the y-axis
    sig = tf.transpose(tf.abs(sig))
    sig = tf.reverse(sig, axis = [0])
    
    # Intensity image only has one channel -> make 3 copies for RGB format required by CNNs such as DenseNet, MobileNet...
    sig = tf.stack((sig,) * 3, axis = -1)
    
    # Standard image input format (scales images to [-1, 1])
    sig = tf.keras.applications.mobilenet.preprocess_input(sig)

    # Labels were not modified, but still need to be returned
    return sig, labels

#############################################################################################

@tf.function
def convert_to_wavelet(sig, labels, wavelet = signal.ricker, 
                       hop_length = 50, n_scales = 64, scales_step = 2):
    """Compute the discrete/continuous wavelet transform from raw signals (converted into tf.Dataset).
    
    :param sig: Batch of signals to be transformed into spectrograms.
    :type sig: Tensor
    :param labels: Labels of the signals.
    :type labels: list
    :param wavelet: Function to apply wavelet transform. Defaults to *ricker* ('mexican hat').
    :type wavelet: function pointer
    :param hop_length: Step for moving wavelet window (if 1, all WT coefficients are returned). Defaults to 50.
    :type hop_length: int, optional
    :param n_scales: Number of wavelet scales to compute (usually 16, 32, 64, 128). Defaults to 64.
    :type n_scales: int, optional
    :param scales_step: Step to pick computed scales (if 1, all **n_scales** scales are returned). Defaults to 2.
    :type scales_step: int, optional
    :return: Wavelet signals and labels.
    :rtype: tuple
    """
   
    # Create discrete scales
    scales = np.arange(1, n_scales + 1, scales_step)
    
    # Compute a (discretized) continuous WT
    sig = signal.cwt(sig, wavelet, scales)[:, ::hop_length]
    
    # Intensity image only has one channel -> make 3 copies for RGB format required by CNNs such as DenseNet, MobileNet...
    sig = tf.stack((sig,) * 3, axis = -1)
    
    # Standard image input format (scales images to [-1, 1])
    sig = tf.keras.applications.mobilenet.preprocess_input(sig)
    
    # Labels were not modified, but still need to be returned
    return sig, labels
    
#############################################################################################   
    
def convert_to_tf_dataset(X, y, model_name):
    """Convert dataset **(X, y)** to ``tf.Dataset`` according to model architecture.
    
    :param X: Matrix of signals.
    :type X: ndarray
    :param y: Label vector.
    :type y: ndarray
    :param model_name: Name of model architecture. Currently allowed: *SimpleCls*, *Emb*, *SimpleEmbCls*, \
    *HieraCls*, *HieraEmbCls*.
    :type model_name: str
    :return: Tensorflow dataset
    """
    
    dataset = []
    if model_name in ['SimpleCls', 'Emb', 'SimpleEmbCls']:
        dataset = tf.data.Dataset.from_tensor_slices((X, y[:,1]))
            
    elif model_name == 'HieraCls':        
        dataset = tf.data.Dataset.from_tensor_slices((X, {'genus':   y[:,0],
                                                          'species': y[:,1]})) 
    elif model_name == 'HieraEmbCls':
        dataset = tf.data.Dataset.from_tensor_slices((X, {'embedding': y[:,1],
                                                          'genus':     y[:,0],
                                                          'species':   y[:,1]}))   
    else:
        print('Model unknown!')
    
    return dataset

#############################################################################################

def preprocess_dataset(X, y, model_name, input_format, sampling_rate = 16000, batch_size = 64,
                       window = None, nperseg = None, noverlap = None, cutoff = None, 
                       shuffle = False, prefetch = True, cache = True, drop_uneven_batch = False):
    """Apply preprocessing steps to signal matrix **X** such as conversion to PSD/Spectrogram, conversion to ``tf.Dataset``,
    expand dimensions, shuffle, split into batches and integrate prefetching.
    
    :param X: Matrix of signals.
    :type X: ndarray
    :param y: Label vector.
    :type y: ndarray
    :param model_name: Name of model architecture. Currently allowed: *SimpleCls*, *Emb*, *SimpleEmbCls*, \
    *HieraCls*, *HieraEmbCls*.
    :type model_name: str
    :param input_format: Type of input signal. Currently allowed: *raw*, *psd*, *spectro*.
    :type inupt_format: str
    :param sampling_rate: Sampling frequency. Defaults to 16 kHz.
    :type sampling_rate: int
    :param batch_size: Size of batches to split the dataset into. Defaults to 64.
    :type batch_size: int
    :param window: Window-function to multiply each segment with i.e. 'hann' (for psd) or ``tf.signal.hann_window`` (for spectrograms). Defaults to *None*.
    :type window: str (for psd) or function pointer (for spectrograms)
    :param nperseg: Length of a segment for applying the Welch-Transform or STFT. Defaults to *None*.
    :type nperseg: int
    :param noverlap: Lenth of overlapping region between segments. Defaults to *None*.
    :type noverlap: int
    :param cutoff: How many PSD frequencies should be kept. Defaults to *None*.
    :type cutoff: int, optional
    :param shuffle: Whether to shuffle the signals in the dataset. Note that you need to shuffle the training set before splitting it into batches. Defaults to False.
    :type shuffle: bool
    :param prefetch: Whether to integrate prefetching strategy into the dataset. \
    Note that batching and caching are only applied, if **prefectch** is True. Defaults to True.
    :type prefetch: bool
    :param cache: Whether to cache the dataset so that it is read only once before training. Make sure that there is enough memory. Defaults to True.
    :type cache: bool
    :param drop_uneven_batch: Whether to drop the final natch if it is smaller than the rest. Defaults to False.
    :type drop_uneven_batch: bool
    :return: Processed Tensorflow dataset
    """

    # Convert to PSD
    if input_format == 'psd':
        X = list(X) # SMOTE changes arguments to Numpy in Colab, but convert_to_psd requires lists
        convert_to_psd(X, sampling_rate, window, nperseg, noverlap, cutoff)

    # Add one extra dimension to the matrices, if 1D signals are used (to fit training format) 
    # Convert signals and labels to Numpy arrays, making sure matrix storage does not exceed float32
    if input_format == 'spectro':
        X = np.asarray(X, dtype = np.float32)
    else:
        X = np.expand_dims(X, axis = -1).astype(np.float32)
        
    # Convert label vectors to Numpy
    y = np.asarray(y)

    # Conversion to tf.Dataset        
    tf_dataset = convert_to_tf_dataset(X, y, model_name)   
    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size = len(X), reshuffle_each_iteration = True)

    # Convert to spectrograms (only for 2d models)
    if input_format == 'spectro': 
        tf_dataset = tf_dataset.map(lambda X, y: convert_to_spectro(X, y, nperseg, nperseg - noverlap, window, cutoff))

    # Prefetching + Batching
    if prefetch:
        if cache:
            tf_dataset = tf_dataset.cache().batch(batch_size, drop_remainder = drop_uneven_batch).prefetch(tf.data.experimental.AUTOTUNE)
        else:
            tf_dataset = tf_dataset.batch(batch_size, drop_remainder = drop_uneven_batch).prefetch(tf.data.experimental.AUTOTUNE)
    
    return tf_dataset

#############################################################################################

def get_uniform_random_samples(X, y, sample_size, indexes, return_numpy = True):
    """Build uniform random samples from matrix **X** and label vector **y**.
    
    Parameter indices defines the subintervalls from which the samples are selected: 
        
    - **indexes** = [ind_0, ind_1, ind_2, ..., ind_n]
    - *subsample_1* will be selected from **X** by the half closed index subintervall *[ind_0, ind_1)* ...
    - *subsample_n* will be selected from **X** by the half closed index subintervall *[ind_{n-1}, ind_n)*
    
    Each class is equally represented, if indexes are selected accordingly. \
    Note that the signals in **X** should be ordered according to their class.
    
    :param X: Matrix of signals.
    :type X: ndarray
    :param y: Label vector.
    :type y: ndarray
    :param sample_size: Size of samples from each class.
    :type sample_size: int
    :param indexes: List of starting indexes for each class. Final index gives the length of the dataset.
    :type indexes: list
    :param return_numpy: Whether to return the sampled signals and labels as a Numpy array. Otherwise as list. Defaults to True.
    :type return_numpy: bool
    :return: Sampled signals and labels.
    :rtype: tuple
    """
   
    # Initialize empty lists for the samples
    X_sample = []
    y_sample = []
    for i in range(len(indexes)-1):

        max_subset_size = indexes[i+1] - indexes[i]
        if max_subset_size > sample_size: # Make sure sample size does not exceed index subintervall
            # Choose sample_size random indices
            rand_ind = np.random.choice(range(indexes[i], indexes[i+1]), sample_size, replace = False)
            for ind in rand_ind:
                X_sample.append(X[ind])
                y_sample.append(y[ind])
        else:
            print(str(i+1) + '. set does not have enough data. Max ' + str(max_subset_size))
            break

    if return_numpy:
        X_sample = np.asarray(X_sample)
        y_sample = np.asarray(y_sample)

    return X_sample, y_sample

#############################################################################################
    
def rms_frame(frame):
    """Compute the Root Mean Square value of the amplitudes within **frame**. 
    
    :param frame: Portion of the signal for which to compute RMS.
    :type frame: array
    :return: RMS
    """

    return np.linalg.norm(frame) / np.sqrt(len(frame))

#############################################################################################

def cut_off_rms(X, min_len, nperseg, hop_len):
    """Cut off signals so that part of the uninteresting amplitudes are discarded.
    
    The RMS is computed within multiple frames (**nperseg** long) of signals in **X**, overlapping each other by **nperseg - hop_len**.
    The signal is cut around the frame with the greatest RMS, so that it gets shortened to length **min_len**.
    
    :param X: Matrix of signals.
    :type X: array or list
    :param min_len: length of the cropped signal.
    :type min_len: int
    :param nperseg: Length of the frame for which to compute the RMS.
    :type nperseg: int
    :param hop_len: Step size from frame to frame.
    :type hop_len: int
    """

    for i in range(len(X)): 

        # Derived parameters
        sig_len = len(X[i]) 
        nsegs = (sig_len - nperseg) // hop_len + 1 # number of frames/segments to divide the whole signal into
        max_rms_index = 0 # starting index of the frame with the greatest RMS
        max_rms = 0.0 # variable to store the greatest rms

        # Compute RMS within each frame
        for j in range(nsegs):

            curr_frame = X[i][j*hop_len : j*hop_len + nperseg]
            curr_rms = rms_frame(curr_frame)

            if curr_rms > max_rms:
                max_rms = curr_rms
                max_rms_index = j*hop_len

        # Define limits of the cut-off signal
        whisker = (min_len - nperseg) // 2 # amount by which to extend the chosen frame to left and right
        start_index = max_rms_index - whisker
        end_index = max_rms_index + nperseg + whisker

        # When rounding, if may occur that the length of the cut-off signal is 1 shorter than min_len
        if nperseg + 2*whisker != min_len:
            end_index += 1

        # If start_index gets negative, set it to 0 and add excess to the end_index (keeps the cut-off length fixed)
        if start_index < 0:
            end_index -= start_index
            start_index = 0
        # If end_index stretches above length of original index, 
        # set it to maximum possible index and subtract excess from the start_index (keeps the cut-off length fixed)
        if end_index > sig_len - 1:
            start_index -= (end_index - sig_len + 1)
            end_index = sig_len - 1

        # Replace original signal with cut-off signal
        X[i] = X[i][start_index : end_index]
        
#############################################################################################        
        
def low_pass(sig, fs, cutoff, threshold, window = 'hann'):
    """Custom low pass filter. 
    
    Amplitudes of frequencies after **cutoff** (Hz) are reduced
    to **threshold** if greater than **threshold**. Signal **sig** can be first multiplied with a window function.
    Sampling rate **fs** is needed in order to convert **cutoff** from Hz to array index of the amplitude spectrum.

    :param sig: Raw signal.
    :type sig: list or array
    :param fs: Frequency sample.
    :type fs: int
    :param cutoff: Minimal frequency (Hz) to consider for filtering.
    :type cutoff: flloat
    :param threshold: Maximal allowed amplitude value for frequencies higher than **cutoff** Hz.
    :type threshold: float
    :param window: Window-function to multiply the signal with, before applying FFT. Defaults to 'hann'.
    :type window: str, optional
    :return: Filtered signal.
    """
    
    len_sig = len(sig)
    freq_resol = fs / len_sig # frequency resolution
    cutoff = int(np.ceil(cutoff / freq_resol)) # cutoff index in amplitude spectrum
    
    if window == 'hann':
        sig = sig * signal.hann(len_sig)

    # DFT
    four = np.fft.fft(sig)
    four_amp = np.abs(four)
    four_ang = np.angle(four)
    exp_ang = np.exp(1.0j * four_ang) # needed for the inversion

    # Reduce high frequencies 
    for i in range(cutoff, len_sig-cutoff):
        if four_amp[i] > threshold:
            four_amp[i] = threshold

    # IDFT 
    new_amps = np.fft.ifft(four_amp * exp_ang)
    
    return new_amps.real # no imaginary part, only real part is stored

#############################################################################################


def unitsphere_embedding(class_sim):
    """Find an embedding of n classes on a unit sphere in n-dimensional space, so that their dot products correspond \
    to the pre-defined similarities in **class_sim**.
    
    Code taken from https://github.com/cvjena/semantic-embeddings/blob/master/compute_class_embedding.py
    
    :param class_sim: A n-by-n matrix specifying the desired similarity between each pair of classes.
    :type class_sim: array
    :return: A n-by-n matrix with rows being the locations of the corresponding classes in the embedding space.     
    """
    
    # Check arguments
    if (class_sim.ndim != 2) or (class_sim.shape[0] != class_sim.shape[1]):
        raise ValueError('Given class_sim has invalid shape. Expected: (n, n). Got: {}'.format(class_sim.shape))
    if (class_sim.shape[0] == 0):
        raise ValueError('Empty class_sim given.')
    
    # Place first class
    nc = class_sim.shape[0]
    embeddings = np.zeros((nc, nc))
    embeddings[0,0] = 1.
    
    # Iteratively place all remaining classes
    for c in range(1, nc):
        embeddings[c, :c] = np.linalg.solve(embeddings[:c, :c], class_sim[c, :c])
        embeddings[c, c]  = np.sqrt(1. - np.sum(embeddings[c, :c] ** 2))
    
    return embeddings
