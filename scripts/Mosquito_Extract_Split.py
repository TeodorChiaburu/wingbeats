""" KInsekten - Mosquito
    Teodor Chiaburu
    Script to extract audio files out of their intermediary date subfolders ...
        - old path: species/date/audio.wav
        - new path: species/audio.wav
    
    ... and split the data into training, cross validation and test sets 
"""


### FILE EXTRACTION ###

import glob
import os

# Local path to the wingbeats folder
data_path = "C:/Users/teo_c/OneDrive/Documents/Beuth/Master/4. Semester/KInsekten/Mosquito/Wingbeats/"

# List of species names
species = ['Ae_aegypti', 'Ae_albopictus', 'An_arabiensis', 
           'An_gambiae', 'Cu_pipiens', 'Cu_quinquefasciatus']

for spec in species:
    
    # Enter current species folder
    os.chdir(data_path + spec) 
    
    # All date subfolders of current species folder (they all begin with 'D')
    list_dates = glob.glob('D*') 

    # Counter for the date folders
    folder_index = 1
    for date in list_dates:
        
        # Enter current date subfolder
        os.chdir(data_path + spec + '/' + date) 
        
        # All audio files in current date subfolder
        files = glob.glob('*.wav') 

        # Move audio files from date subfolder directly into the species folder
        # This makes it easier to access and split data for training later
        for f in files:
            os.replace(data_path + spec + '/' + date + '/' + f, # old path
                       data_path + spec + '/' + f) # new path

    # Delete now empty date subfolder
    try:
        os.rmdir(data_path + spec + '/' + date)
        print(str(folder_index))
        folder_index += 1
    except:
        print(data_path + spec + '/' + date + ' could not be removed.')
      

### SPLIT 60 - 20 - 20 ###
    
from sklearn.model_selection import train_test_split

for spec in species:
    
    # Enter current species folder
    os.chdir(data_path + spec) 
    
    # All audio files in current species folder
    files = glob.glob('*.wav')
  
    # Split files into Training, CV and Test sets
    X_Train_CV, X_Test = train_test_split(files, test_size = 0.2, random_state = 123)
    X_Train, X_CV = train_test_split(X_Train_CV, test_size = 0.25, random_state = 123)
  
    # Counter to keep track of moved files
    file_index = 1
    
    # Move audio files into new folders according to their purpose:
    # Train/species/audio.wav, CV/species/audio.wav, Test/species/audio.wav
    
    print('X_Train: ' + str(len(X_Train)))  
    for f in X_Train:
        os.replace(data_path + spec + '/' + f, 
                   data_path + 'Train/' + spec + '/' + f)
        if file_index % 1000 == 0:
            print(str(file_index))
        file_index += 1
               
    print('X_CV: ' + str(len(X_CV)))
    for f in X_CV:
        os.replace(data_path + spec + '/' + f, 
                   data_path + 'CV/' + spec + '/' + f)
        if file_index % 1000 == 0:
            print(str(file_index))
        file_index += 1
      
    print('X_Test: ' + str(len(X_Test)))    
    for f in X_Test:
        os.replace(data_path + spec + '/' + f, 
                   data_path + 'Test/' + spec + '/' + f)
        if file_index % 1000 == 0:
            print(str(file_index))
        file_index += 1
      

