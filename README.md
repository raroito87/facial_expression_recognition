# facial_expression_recognition
This project has been carried out as the final project of the Machine Learning bootcamp at neuefische GmbH (https://www.neuefische.de/weiterbildung/data-science)
The purpose of this project is to apply the concepts learnt along the course implementing a personal project of our own choice.

# Description
The project uses a dataset provided by Kaggle: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/

The goal is to determine anyone's emotional state by taking a picture (using the laptop's Webcam for example). This will be achieved by building an image classification system in which face pictures are analyzed and categorized accoring to their facial expression.

# Motivation

Facial expression recognition is not a new topic. Nevertheless it doesnt make it less interesting. Developing a model that can read a face and properly determine the emotion behind it sounds quite exciting for someone like me who just enter the whole of Machine Learning and that is why I choose this as the topic for my first big project.

# Code

Code is structured as:

data (not included in the repository): contains the data after preprocessing
models/best: contains the computed model with the best accuracy of the test dataset. This model is used in predict_captured_image.py script
Notebooks: contains several notebooks mainly used to analyze preprocess data and  visualize results using histograms or confusion matrices. 
Scripts: contains several script for dirretent uses: data processing, predicttion of emotions, training and testing of the different models.
src: source code.
  - dataset: contains the original dataset before preprocessing
  - image_utils: different classes with utilities for images.
  - models: teh different architectures of CNN that I have work with in the project
  - train: in these classes the train loop is implemented. train_classifier1 is the first version. train_classifier2 is an improved version and optimized to be used with GPU. trainclassifier3&4 are not completed and has not been used.
  - utils: diferent classes with utilities.
  - test: intended to be for testing the utilities. Only two test for image_converter are impletented so far.
  

## Setup

```sh
git clone https://github.com/raroito87/facial_expression_recognition/
```

```sh
conda env create --file environment.yml
source activate my_voice_predictor
```

## Usage

```sh
conda env update --file environment.yml
conda activate facial_expression_recognition
python scripts/predict/predict_captured_image.py # Webcam video is displayed. The face should centered in the red rectangle. Fill the rectangle as much as possible (the face should be as big as posible in the image). Press 'Space'.
```

## Todo

This is a working version with a quite nice accuracy (would be Top5 in the Klaggle compettition) but it still an be improved.
things to be done still:
- use a pretrained CNN
- increase the resolution on training images using an upscaler.
