# Deepfake-Detector
Python deep learning project (CNN) that trains 5 separate models to test the detection power against real and fake images. Dataset taken from [OpenForensics](https://zenodo.org/record/5528418#.ZGaehnbMKHv) (dataset is not uploaded in this repo due to size.)

----
## Executive Summary
In an effort to combat a rising risks associated with accessible generative AI known as deepfakes, this project seeks to create a strong deepfake detector using 11 convolutional neural nets (CNNs). Data was taken from OpenForensics (an open-source dataset of labeled real and fake images), preprocessed, and fitted to 2 types of architectures of CNNs: Sequential models and EfficientNet models. The end result of these models peaked at a validation accuracy of 0.965 and precision of 0.992, with the strongest recommended model being the EfficientNet_v2B0 (located in our pre-trained models folder). Thus, we recommended using the EfficientNet_v2B0 model for detecting the difference between deepfakes and real photographs.

This project provides pre-trained models as out-of-the-box solutions for business needs, saving users the time and compute! 

----
## Problem Statement

The rapid evolution of generative artificial intelligence (GPAI, LLMs) has rapidly increased the public’s access to powerful, deceptive tools. One such concern is the increasing prevalence of deepfake images, which pose a significant threat to public trust and undermines the epistemic integrity of visual media. These manipulated images can be utilized to spread false information, manipulate public opinion, and polarize communities, which can have serious consequences for both social and political discourse. 

In this project, we aim to combat the spread of AI risks by developing a deep learning model that can detect differences between deepfakes and real images to combat the spread of manipulated visual media and protect the integrity of social discourse. We will use the precision score 

----
## Folder Directory
|Folder Name|File Name|File Description|
|---        |---      |---             |
|Code|| This folder contains all code for the project
|Code/Helper|| This folder contains all the helper scripts.
|Code/Helper|`config.py`| This files contains all the helper functions and variables used in the `-NN.ipynb` files
|Code/Helper|`intial_csv_export_script.ipynb`| This script is used to create the first instance of the `model_scores.csv`; don't run again
|Code/PretrainedModel|| This folder contains all the pre-trained models.
|Code/PretrainedModel|`deepfake_model_tutorial.ipynb`| This file contains the tutorial on how to use pre-trained models.
|Code/PretrainedModel|`dffnetv2B0.json`| This file contains the efficient_net architecture.
|Code/PretrainedModel|`dffnetv2B0_weights.h5`| This file contains the efficient_net weights.
|Code/PretrainedModel|`cnn_model4.json`| This file contains the benchmark cnn architecture.
|Code/PretrainedModel|`cnn_model4_weights.h5`| This file contains the benchmark cnn weights.
|Code/PretrainedModel/Example Images|| This folder contains sample images tested.
|Code/Main|| This folder contains all the main modeling files.
|Code/Main/Testing|| This folders contains modeling work related to testing
|Code/Main/Testing|`Best_Models_NN.ipynb`| This file contains the best trained model applied to testing data + graphing efforts 
|Code/Main/Training|| This folders contains modeling work related to training and validation
|Code/Main/Training|`Anna_NN.ipynb`| This file contains selected modeling work from Anna.
|Code/Main/Training|`Chris_NN.ipynb`| This file contains selected modeling work from Chris.
|Code/Main/Training|`Reid_NN.ipynb`| This file contains selected modeling work from Reid.
|Code/Results|| This folder contains the summary results from the models.
|Code/Results|`model_eval.csv`| This file contains the summary of model results.
|Images|| This folder contains all the images used in the presentation.
|Data|| This folder is empty because data is not uploaded; but is where relative filepaths should be

----
## Data Collection and Preprocessing

Data is taken from [OpenForensics](https://zenodo.org/record/5528418#.ZGaehnbMKHv), which is an open-source dataset used in the paper "Multi-Face Forgery Detection And Segmentation In-The-Wild Dataset" by Le, Trung-Nghia et al. Not much cleaning was needed besides `train_test_split`.

Preprocessing for modeling included image formatting to (256, 256) and scaling (done within the layers). For certain models, we also applied data augmentation (rotation, flipping, etc.), which were also included in the layers.

----
## Modeling

In total, we used 11 different iterations of CNNs across 2 architectures (custom Sequential and EfficientNet). Compute time took roughly 35 hours.

----
## Conclusion and Recommendations 

Based on our findings, we recommend using our EfficientNet_v2B0 model, which is our largest model. This out-of-the-box solution would provide the highest scores of validation accuracy of 0.965 and validation precision of 0.992.

----
## Uncertainties

Training data did not include further alterations besides the deepfake (eg. the training data set did not include color tints, high contrast, blurred, etc. images), while the testing data did. Thus, we can achieve a higher accuracy with our models by including that in our data augmentation. Likewise, increased time horizons and stronger compute can lead to more complex models.