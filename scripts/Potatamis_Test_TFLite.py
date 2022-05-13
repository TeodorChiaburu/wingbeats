""" Potatamis_Test_TFLite.py
    Teodor Chiaburu
    Script to test classifiers on Raspberry Pi 4
"""

# Import built-in Python libraries
import tensorflow as tf
import numpy as np
import pickle
import time
import sys
import glob
import os
from scipy import signal

print('*******************************')
print('Running Potatamis_Test_TFLite.py')
print('TF Version ' + str(tf.__version__))
print('*******************************')


""" Genus dict. and embedding matrix """

# Dictionary of genera with the included species
genus_species_dict = {
    'Aedes': ['Ae_aegypti', 'Ae_albopictus'],
    'Anopheles': ['An_arabiensis', 'An_gambiae'],
    'Culex': ['Cu_pipiens', 'Cu_quinquefasciatus']
}

# Store species from the dictionary values (list flattening)
species = [s for l in list(genus_species_dict.values()) for s in l]
print(species)

# Store genus from the keys
genus = list(genus_species_dict.keys())
print(genus)

# Store genus that each species belongs to as a Look-Up Table
genus_mapping = []
for val, gen_ind in zip(genus_species_dict.values(), range(len(genus))):
  genus_mapping += [gen_ind] * len(val)
print(genus_mapping)
genus_mapping = np.asarray(genus_mapping)

# Embedding matrix
emb_matrix = [[1.,    0.,    0.,    0.,    0.,    0.   ],
              [0.5,   0.866, 0.,    0.,    0.,    0.   ],
              [0.,    0.,    1.,    0.,    0.,    0.   ],
              [0.,    0.,    0.5,   0.866, 0.,    0.   ],
              [0.,    0.,    0.,    0.,    1.,    0.   ],
              [0.,    0.,    0.,    0.,    0.5,   0.866]]
emb_matrix = np.asarray(emb_matrix)
print('Embedding matrix:')
print(emb_matrix)


""" Load data """
# User needs to provide model directory (tflite files), test data and test labels (both as pickle files)
if len(sys.argv) == 4:   
    
    # Load TFLite model and allocate tensors
    model_dir = sys.argv[1]
        
    # Load test amplitudes
    with open(sys.argv[2], 'rb') as f:
        X_test = pickle.load(f)
    print('Test amplitudes loaded')
    
    # Load test labels
    y_test = []
    with open(sys.argv[3], 'rb') as f:
        temp = pickle.load(f)
        for lab in temp:
            y_test.append([genus_mapping[lab], lab])
    print('Test labels loaded')
    num_samples = len(X_test)
    print(str(num_samples) + ' test samples')
    
    # Normalize amplitudes to [-1, 1]
    for i in range(num_samples):
        X_test[i] = X_test[i] / np.max(np.abs(X_test[i]))
    print('Test set normalized\n')
    
else:
    print("Please provide model directory (tflite files), test data and test labels (both as pickle files)")
    sys.exit()

""" Predictions """
os.chdir(model_dir)
model_files = glob.glob('*.tflite')
for model_file in model_files:
    
    print('\n*************************')
    print('Running ' + model_file)
    
    # Get model name from file name
    substr_ind = model_file.find('_')
    model_name = model_file[:substr_ind]
    
    # Deduce the input format
    if 'psd' in model_file:
        input_format = 'psd'
    elif 'spectro' in model_file:
        input_format = 'spectro'
        
    # Create interpreter
    interpreter = tf.lite.Interpreter(model_path = model_file)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    gen_acc, spec_acc = 0.0, 0.0 # accuracies
    count = 0 # for tracking progress
    prog_mark = int(num_samples/10) # samples in 10% of data
    nperseg  = 256

    # Test the TensorFlow Lite model on the test data
    print('Inference started...')
    start_time = time.time()
    for sig, lab in zip(X_test, y_test):

        ### PSD ###
        if input_format == 'psd':
                
            noverlap = 192
            _, sig = 10*np.log10(signal.welch(sig, fs = 8000, window = 'hanning', 
                                              nperseg = nperseg, noverlap = noverlap),
                                 dtype = np.float32) # Welch expands dtype to float64 -> reduce it back by hand     
        
            # Expand dimensions (to fit Interpreter requirements) and convert to Tensor
            sig = tf.expand_dims(sig, axis = 0)  # one extra dim at the front
            sig = tf.expand_dims(sig, axis = -1) # one extra dim at the end

        ### Spectrograms ###
        elif input_format == 'spectro':
            
            noverlap = 256-256//6
            sig = np.asarray(sig, dtype = np.float32)
            
            # Apply Short Time Fourier Transform to compute the spectrogram
            sig = tf.signal.stft(sig, frame_length = nperseg, frame_step = nperseg - noverlap, fft_length = nperseg)
            
            # Convert datatype from complex to float
            sig = tf.cast(sig, dtype = tf.float32) 
            
            # Convert to decibels (directly applying np.log10 results in an erros after converting to tf.Dataset)
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
            sig = tf.expand_dims(sig, axis = 0)  # one extra dim at the front
            
            # Standard image input format (scales images)
            sig = tf.keras.applications.mobilenet.preprocess_input(sig)
            
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], sig) 

        # Make inference
        interpreter.invoke()

        # Get predicted genus and species
        if model_name == 'SimpleCls':
            spec = interpreter.get_tensor(output_details[0]['index'])
            spec = np.argmax(spec)
            gen  = genus_mapping[spec]
        elif model_name == 'SimpleEmbCls':
            spec = interpreter.get_tensor(output_details[1]['index'])
            spec = np.argmax(spec)
            gen  = genus_mapping[spec]
        elif model_name == 'HieraCls':
            gen  = interpreter.get_tensor(output_details[0]['index'])
            gen = np.argmax(gen)
            spec = interpreter.get_tensor(output_details[1]['index'])
            spec = np.argmax(spec)
        elif model_name == 'HieraEmbCls':
            gen  = interpreter.get_tensor(output_details[1]['index'])
            gen = np.argmax(gen)
            spec = interpreter.get_tensor(output_details[2]['index'])
            spec = np.argmax(spec)
            
        # Compute accuracies
        if gen == lab[0]:
            gen_acc += 1
        if spec == lab[1]:
            spec_acc += 1
            
        # Print progress
        count += 1
        if count % prog_mark == 0:
            print(str(count // prog_mark * 10) + '%')

    # Statistics
    total_time = time.time() - start_time
    time_per_sample = total_time / num_samples
    print('Inference done')
    print('%.4f seconds / sample' % (time_per_sample))
    print('Genus   acc.: %.4f' % (gen_acc/num_samples))
    print('Species acc.: %.4f' % (spec_acc/num_samples))
