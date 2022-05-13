""" Teodor Chiaburu 
    Module for creating the necessary folders in Colab for the Potatamis dataset, 
    copying zipped folders from Google Drive and unzipping them on Colab.
"""

import os
import time
import zipfile
import sys
from shutil import copy

print("Running Potatamis_Prepare_Environment.py")

# User has the possibility to run script with second argument use_test,
# which decides whether the test set should be considered or not
if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    print("Please provide second argument data_path.")
    sys.exit()

use_test = False
if len(sys.argv) == 3:
    if sys.argv[2] == "True":
        use_test = True
        print("Loading Train, CV and Test.")
    elif sys.argv[2] == "False":
        use_test = False
        print("Loading Train and CV.")
    else:
        print("Argument use_test can only be True or False!")
        sys.exit()
else:
    print("Only two arguments possible, but more were provided.")
    sys.exit()  

print("###################################\n")
      

# Create temporary local directory to copy data from Drive 
os.chdir('/content')
if not os.path.exists('KInsekten'):
    os.mkdir('KInsekten')
if not os.path.exists('KInsekten/Potatamis'):
    os.mkdir('KInsekten/Potatamis')
if not os.path.exists('KInsekten/Potatamis/Train'): 
    os.mkdir('KInsekten/Potatamis/Train')
if not os.path.exists('KInsekten/Potatamis/CV'): 
    os.mkdir('KInsekten/Potatamis/CV')
if not os.path.exists('KInsekten/Potatamis/Test'): 
    os.mkdir('KInsekten/Potatamis/Test')
if not os.path.exists('KInsekten/Potatamis_zip'):
    os.mkdir('KInsekten/Potatamis_zip')
time.sleep(10) # wait until Colab creates the new directories
######################################################################################


# Copy files from Drive (zipped amplitudes and label vectors)
start_time = time.time()
copy(data_path + 'Train_amplitudes.zip', 'KInsekten/Potatamis_zip')
print('Train_amplitudes.zip copied')

copy(data_path + 'CV_amplitudes.zip', 'KInsekten/Potatamis_zip')
print('CV_amplitudes.zip copied')
copy(data_path + 'CV_labels.pickle', 'KInsekten/Potatamis/CV')
print('CV_labels.pickle copied')

if use_test:
    copy(data_path + 'Test_amplitudes.zip', 'KInsekten/Potatamis_zip')
    print('Test_amplitudes.zip copied')
    copy(data_path + 'Test_labels.pickle', 'KInsekten/Potatamis/Test')
    print('Test_labels.pickle copied')

print("%s seconds\n" % int(time.time() - start_time))
######################################################################################


# Unzip the files locally
start_time = time.time()
with zipfile.ZipFile('/content/KInsekten/Potatamis_zip/Train_amplitudes.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/KInsekten/Potatamis/Train')    
print('Train unzipped')
with zipfile.ZipFile('/content/KInsekten/Potatamis_zip/CV_amplitudes.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/KInsekten/Potatamis/CV')
print('CV unzipped')
if use_test:
    with zipfile.ZipFile('/content/KInsekten/Potatamis_zip/Test_amplitudes.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/KInsekten/Potatamis/Test')  
    print('Test unzipped')
print("%s seconds\n" % int(time.time() - start_time))
######################################################################################

