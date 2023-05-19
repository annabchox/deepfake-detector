# Deepfake-Detector
Python deep learning project (CNN) that trains 5 separate models to test the detection power against real and fake images. Dataset taken from [OpenForensics](https://zenodo.org/record/5528418#.ZGaehnbMKHv) (dataset is not uploaded in this repo due to size.)

----
## Executive Summary
In an effort to combat a rising risks associated with accessible generative AI known as deepfakes, this project seeks to create a strong deepfake detector using 5 convolutional neural nets (CNNs). Data was taken from OpenForensics (an open-source dataset of labeled real and fake images), preprocessed, and fitted to 3 types of CNNs: 3 Sequential models, 1 _ model, and 1 ElasticNet model. The end result of these models range from _ to _, with the strongest recommended model being _ for its highest validation scores (lowest loss and highest accuracy). Thus, we recommended using the _ model for detecting the difference between deepfakes and real photographs. 

----
## Problem Statement

The rapid evolution of generative artificial intelligence (GPAI, LLMs) has rapidly increased the public’s access to powerful, deceptive tools. One such concern is the increasing prevalence of deepfake images, which pose a significant threat to public trust and undermines the epistemic integrity of visual media. These manipulated images can be utilized to spread false information, manipulate public opinion, and polarize communities, which can have serious consequences for both social and political discourse. 

In this project, we aim to combat the spread of AI risks by developing a deep learning model that can detect differences between deepfakes and real images to combat the spread of manipulated visual media and protect the integrity of social discourse.

----
## Folder Directory
|Folder Name|File Name|File Description|
|---        |---      |---             |
|Images|| This folder contains all the images used in the presentation
|Data|| This folder is empty because data is not uploaded; but is where relative filepaths should be
|Code|| This folder contains all code for the project
|Images|`_.png`| Image: 
|Code/Helper|| This folder contains all the helper scripts.
|Code/Main|| This folder contains all the main modeling files.
|Code/Results|| This folder contains the summary results from the models.
|Code/Helper|`config.py`| This files contains all the helper functions and variables used in the `-NN.ipynb` files
|Code/Helper|`intial_csv_export_script.ipynb`| This script is used to create the first instance of the `model_scores.csv`; don't run again
|Code/Main|`Anna-NN.ipynb`| This file contains modeling work from Anna
|Code/Main|`Chris-NN.ipynb`| This file contains modeling work from Chris
|Code/Main|`Reid-NN.ipynb`| This file contains modeling work from Reid
|Code/Results|`model_scores.csv`| This file contains the summary of model results

----
## Data Collection and Preprocessing

Data is taken from [OpenForensics](https://zenodo.org/record/5528418#.ZGaehnbMKHv), which is an open-source dataset used in the paper "Multi-Face Forgery Detection And Segmentation In-The-Wild Dataset" by Le, Trung-Nghia et al. Not much cleaning was needed besides `train_test_split`.

Preprocessing for modeling included 

----
## Modeling

We used 3 types of CNNs (3 Sequential models, 1 _, and 1 ElasticNet), totaling 5 distincti iterations of models.

----
## Conclusion and Recommendations 

-
----
## Uncertainties
