# # Object Localization with TensorFlow and COCO Dataset

This project implements an object localization model using TensorFlow and the COCO dataset. The model is designed to detect and localize cars and trucks within images. It utilizes Convolutional Neural Networks (CNN) to predict bounding boxes around the specified objects.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)

## Features
- Loads and preprocesses images and annotations from the COCO dataset.
- Constructs a CNN model with Batch Normalization and Dropout layers to improve performance.
- Implements EarlyStopping to prevent overfitting during training.
- Visualizes predictions alongside ground truth bounding boxes.

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- pycocotools

You can install the required packages using pip:

```bash
pip install tensorflow numpy opencv-python matplotlib pycocotools
```

## Installation
Clone this repository:

```bash
git clone https://github.com/bhuminChodvadiya/Object_Localization.git
```

cd object_localization
- Download the COCO dataset and place the _annotations.coco.json file in the train directory.
