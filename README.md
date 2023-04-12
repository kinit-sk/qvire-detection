# CulturalAI - QVIRE project

This repository contains the code and data to train specific models capable of detecting quiremarks on medieval manuscripts. This codebase was created by KInIT, Kempelen Institute of Intelligent Technologies.

The final models we created for this specific task are located on the following link: https://zenodo.org/record/7813911

If you want to use this repository to train the models yourself, you can freely do so and we further below provide a guide on how to setup the environment and start the training process.

## Setup the environment

We highly recommend using conda environments for easier package installations.
1. Create a new conda environment whose parameters are defined in the `environment.yml` file
``` 
    conda env create -f environment.yml
```
2. Activate newly created environment `conda activate qvire-env`
3. Separately install PyTorch. This specific PyTorch installation expects you to have Nvidia graphics card with CUDA support, precisely with cudatoolkit of version >=11.7. To download the cudatoolkit, visit the following link: https://developer.nvidia.com/cuda-downloads
```
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

If you don't have a GPU card supporting these aforementioned requirements, you can download PyTorch which supports CPU computations only.
```
    pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
```


## Data Acquistion

Since the data is rather large, we stored it separately on the following link: https://zenodo.org/record/7813108

The dataset constists of high-resolution images of 59 medieval manuscripts. The annotations of the quiremarks are stored in a JSON file called `annot.json`. To make minimal changes to our predefined training loops, we suggest you to save the dataset into the folder called `data/images`.

## Data preprocessing

In order to carry out the training process of either model, you need to preprocess the data using the `data_preprocessing.py` script. In that file you can choose which preprocessing pipeline you wish to perform.

If you want to train a detection model (Faster RCNN architecture), you need to downsample the original images so that they can fit into the model. On the other hand, if you want to classify the individual patches of the original images whether they contain quiremarks or not, you need to firstly calculate so called distance masks, and subsequently divide the images into overlapping patches.

## Model training 

After having preprocessed the dataset for the specific task you want to perform, you can train the model either on all 59 manuscripts (`final_all_train.py`) or perform a 5-fold cross validation on all of those manuscripts (`final_cv_train.py`).

If you wish to further play with the configuration of each model, you can do so by modifying the values in JSON config files (`final_classifier_vit_config.json`, `final_rcnn_config.json`).
