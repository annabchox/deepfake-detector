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
standard_metrics = [BinaryAccuracy(), AUC(), Precision(), Recall(), TrueNegatives(), TruePositives(), FalsePositives(), FalseNegatives()]

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

    model -> 
        desc

    validation_dataset -> 
        desc

    return_class_names -> 
        desc

    Returns
        None
    
    Example
        get_true_and_pred_labels(_)
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

    directory -> 
        desc

    Returns
        None
    
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

    directory -> 
        desc

    Returns
        None
    
    Example
        get_sample_images(_)
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