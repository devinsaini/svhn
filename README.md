# Street View House Numbers
### Udacity MLND Deep Learning Capstone Project

This repository contains project code, notebooks and report for Street View House Numbers solution.

## Prerequisites
* python 2.7
* Keras 1.2.0 (preferably with Tensorflow backend)
* numpy
* scipy
* scikit-learn
* matplotlib 
* Jupyter
* A reasonably fast machine, GPU highly recommended.

## Setup
1. Clone this repository on your system
2. Download and extract **Format 1** train.tar.gz and test.tar.gz from http://ufldl.stanford.edu/housenumbers/ in inputs folder. There should be train and test folders directly under inputs folder as shown below.


    .
    +-checkpoints
    +-jupyter
    +-inputs
        +-train
        |   +-1.png
        |   +-2.png
        |   +-...
        |
        +-test
        |   +-1.png
        |   +-2.png
        |   +-...
        |
        +-custom
            +-customimage.png
            
## Training
1. You might want to backup the files in checkpoints folder since they've been trained for a long time on AWS p2.xlarge instance. If not, then delete them.
2. Start jupyter, open svhntrain.ipynb and run all sections.
3. Training notebook creates a model.yaml file with the neural network structure and a model.hdf5 file containing network weights.

## Testing
1. Make sure that model.yaml and model.hdf5 files exist in ckecpoints directory. If not, then reset the repository or train as instructed above.
2. Open svhntest.ipynb notebook and run all sections.
3. Test metrics are shown with description in the notebook.

## Prediction on custom images
1. Make sure that model.yaml and model.hdf5 files exist in checkpoints directory. If not, then reset the repository or train as instructed above.
2. Copy your custom images in inputs/custom folder. Only jpg and png files are supported.
3. Open customimages.ipynb notebook and run all sections.
3. Prediction results are shown graphically in the notebook. 

##  File description
* Notebooks
	* jupyter/dataexploration.ipynb - Exploratory analysis on SVHN dataset
	* jupyter/svhntrain.ipynb - Training notebook
	* jupyter/svhntest.ipynb - Testing notebook
	* jupyter/customimages.ipynb - for prediction on your own images
* Supporting code
	* jupyter/graphics.py - utilities for displaying samples and convolutions
	* jupyter/preprocessing.py - extension of Keras ImageDataGenerator
	* jupyter/svhn.py - utilities to load SVHN mat files and process data
	* jupyter/keras_utils.py - DynamicPlot Keras callback to display live training plots. Used for monitoring and hyperparameter adjustment. Not used in final training.
