def main():
    pass

# -------------------
# IMPORTS - NEED TO CONDENSE
# -------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Rescaling
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall, TrueNegatives, TruePositives, FalsePositives, FalseNegatives

# NOTE: do we use these?
import os
import seaborn as sns
from functools import partial
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# do we need this when we have all the "from tf import _"?
import tensorflow as tf 

# -------------------
# PLOTTING VARIABLES
# -------------------
# Font Settings
title_font = {
    'size': 25,
    'weight': 'bold'
}

label_font = {
    'fontsize': 15
}

axis_title_font = {
    'size': 15,
    'weight': 'bold'
}

# Plot Settings
# ?

# Figure Sizes
default_fig_size = (7, 7)
subplot_fig_size = (14, 7)
subplot2_fig_size = (21, 7)

# Color Settings
alpha_value = 0.5
palette = 'colorblind'
cmap = 'colorblind'



# -------------------
# MODEL VARIABLES
# -------------------
random_seed_value = 42
standard_metrics = [
    BinaryAccuracy(), 
    AUC(), 
    Precision(), 
    Recall(), 
    TrueNegatives(), 
    TruePositives(), 
    FalsePositives(), 
    FalseNegatives()
]



# -------------------
# SAVE AND LOAD MODELS
# -------------------
def save_trained_model(model, name=None, save_in_working_directory=True, directory_path=None):
    '''
    Saves trained keras model as .json file and corresponsding weights as .h5 file.

    model -> tensorflow.keras.model object
        A trained keras model to be saved

    name -> str, default = None
        String value to assign to the keras model.name attribute
    
    save_in_working_directory -> bool, default = True
        If False, a different directory path must be specified
    
    directory_path -> str, default=None
        Path to directory where the model and weights will be saved if 
        save_in_working_directory is True.
        
    Returns
        None

    Example
        save_trained_model(my_model, name="my_trained_model",
                           save_in_working_directory=False, 
                           directory_path='../project/models/')
    ''' 

    # rename model (optional)
    if name:
        model._name = name
    
    # set path where model will be saved (default to pwd)
    if not save_in_working_directory:
        path = os.path.join(directory_path, model.name)
    else:
        path = "./" + model.name
    
    # save weights as .h5 file
    model.save_weights(path + "_weights.h5")
    
    # save model as .json file
    model_json = model.to_json()
    with open(f"{path}.json", "w") as json_file:
        json_file.write(model_json)
    
    print(f"""
    {model.name} model saved at {path}.json
    {model.name} weights saved at {path}_weights.h5
    """)

    return



def load_trained_model(model_path, weights_path, compile_model=True):
    '''
    Returns a keras model object with saved weights (compiled by default).

    model_path -> str (.json)
        The path to the model architecture saved as a .json file

    weights_path -> str (.h5)
        The path to the model weights saved as a .h5 file
    
    compile_model -> bool(optional), default = True
    if True, the loaded model will be compiled using standard_metrics,
        the adam optimizer, and binary cross entropy as the loss funciton
    
    Returns
        tensorflow.keras.model object

    Example
        load_trained_model("../project/models/model.json", 
                           "..project/models/model_weights.h5)
    '''  

    # read .json model file
    model_file = open(model_path, 'r')
    loaded_model_file = model_file.read()
    model_file.close()
    
    # load model and weights
    loaded_model = tf.keras.models.model_from_json(loaded_model_file)
    loaded_model.load_weights(weights_path)
    
    if compile_model:
        loaded_model.compile(loss='bce', optimizer='adam', metrics=standard_metrics)
    
    return loaded_model



# -------------------
# HELPER FUNCTIONS
# -------------------
def graph_me(model, list_of_metrics=[]):
    '''
    Returns a subplots graphing history of indicated metrics (min default is loss).

    model -> NN model
        The NN model from which its history data can be extracted for plotting  

    list_of_metrics -> list of str
        The list of metric names to include in the plotting. Eg. ["acc"]
    
    Returns
        None

    Example
        graph_me(res, ["acc"])
    '''
    # Configure figure and plots
    total_graphs = 1 + len(list_of_metrics)
    fig, ax = plt.subplots(1, total_graphs, figsize=subplot2_fig_size)

    # Create data lists for graphing to loop over
    titles = ["Loss History"]
    data = ["loss"]
    val_data = ["val_loss"]

    for metric in list_of_metrics:
        titles.append(f"{metric.title()} History")
        data.append(metric)
        val_data.append(f"val_{metric}")

    # Graphs the data
    for i in range(len(ax)):
        ax[i].set_title(titles[i], **title_font)
        ax[i].set_xlabel("Epoch", **axis_title_font)
        ax[i].set_ylabel("Score", **axis_title_font)
        ax[i].plot(model.history[data[i]], label="train")
        ax[i].plot(model.history[val_data[i]], label="test")
        ax[i].legend(loc='best', **label_font)
    plt.tight_layout()
    plt.show()
    return



def get_true_and_pred_labels(model, validation_dataset, return_class_names=False):
    ''' 
    Summary:
    Retrieves a list of true labels and predicted labels from a keras Dataset object.

    model -> tensorflow.keras.models object
        Trained model used to generate predictions.

    validation_dataset -> tensorflow.keras.Dataset object
        A keras Dataset object that contains data and labels used to make predictions.

    return_class_names -> bool, default=False
        Optional, if True, the list of true and predicted labels will be class names as
        strings from the validation_dataset instead of integer values.

    Returns
        two lists, one of true and one of predicted labels
    
    Example
        get_true_and_pred_labels(my_model, val_ds, return_class_names=True)
    '''
    # Get dataset as array
    dataset_as_array = list(validation_dataset.as_numpy_iterator())
    
    # Separate Image and Label Arrays
    label_batches = [dataset_as_array[i][1] for i in range(len(dataset_as_array))]
    image_batches = [dataset_as_array[i][0] for i in range(len(dataset_as_array))]
    
    # Unpack Image and Label Batches into Single Array
    unpacked_label_batches = np.vstack(label_batches)
    unpacked_image_batches = np.vstack(image_batches)
    
    # Get labels
    true_labels = np.argmax(unpacked_label_batches, axis=1)
    pred_probs = model.predict(unpacked_image_batches)
    predicted_labels = np.argmax(pred_probs, axis=1)
    
    if return_class_names:
        
        # Map Labels to Class Names
        true_class_names = [validation_dataset.class_names[x] for x in true_labels]
        predicted_class_names = [validation_dataset.class_names[x] for x in predicted_labels]
        
        return true_class_names, predicted_class_names
    
    else:
        return true_labels, predicted_labels
    
    
    
def get_class_distributions(directory):
    ''' 
    Summary:
    Calculates the distribution of classes based on numbers of samples in each sub-directory.

    directory -> str
        Parent directory holding sub-directories for each class.

    Returns -> pandas.Series object
        A series with each class as the index and number of observations as the value.
    
    Example
        get_class_distributions(_)
    '''
    class_dist = {}

    for category in os.listdir(directory):
        if 'DS_Store' not in category:
            category_dir = os.path.join(directory, category)
            obs = len(os.listdir(category_dir))

            class_dist.update({category: obs})
    
    return pd.Series(class_dist)



def get_sample_images(directory):
    ''' 
    Summary:
    Retrieves a random image from each sub-directory of a given directory.

    directory -> 
        parent directory with sub-directories for each class.

    Returns -> dict
        A dictionary with one random sample from each subdirectory (each class)
    
    Example
        get_sample_images("../Data/Train/")
    '''
    samples = {}
    
    for category in os.listdir(directory):
        if 'DS_Store' not in category:
            
            # get class directory path
            category_dir = os.path.join(directory, category)
            
            # get random sample image path
            first_image = os.listdir(category_dir)[np.random.randint(0, 1000)]
            first_image_path = os.path.join(category_dir, first_image)
            
            # save image in dictionary as array with class name as key
            sample = img_to_array(load_img(first_image_path)) * 1./255
            samples.update({category: sample})
            
    return samples

if __name__ == "__main__":
    main()