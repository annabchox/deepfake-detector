def main():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

# Font Settings
title_font = {
    'size': 25,
    'weight': 'bold'
}

label_font = {
    'size': 20,
    'weight': 'bold'
}

axis_title_font = {
    'size': 15,
    'weight': 'medium'
}

# Plot Settings

# Figure Sizes
default_fig_size = (7, 7)
subplot_fig_size = (14, 7)

# Color Settings
alpha_value = 0.5
palette = 'colorblind'
cmap = 'colorblind'



# Helper Functions

def get_true_and_pred_labels(model, validation_dataset, return_class_names=False):


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

    class_dist = {}

    for category in os.listdir(directory):
        if 'DS_Store' not in category:
            category_dir = os.path.join(directory, category)
            obs = len(os.listdir(category_dir))

            class_dist.update({category: obs})
    
    return pd.Series(class_dist)



def get_sample_images(directory):

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