# Homework 2 - Digit Recognition 🔍

Student ID: 313551073

Name: 顏琦恩

## Introduction

Homework 2 focuses on digit recognition in photographs, where a model learns to detect and classify numerical digits in real-world conditions. The task involves data preprocessing, designing a Fast R-CNN-based model, and implementing training, validation, and evaluation procedures. The dataset is provided by the TAs, and further details are provided below.

## Requirements

The pretrained model used in this homework is from pytorch, just make sure to have a python environment with a proper pytorch version.

## Getting started

### Training

- To train and validate the model, simply run the command `python main.py`.
- Please checkout the dataset path and the checkpoint path before executing. You can find the line for the paths by searching __CHECK HERE__ in script `main.py`.
- The testing process will be executed automatically if lines 190 and 191 in the main function are not commented out.
- The parameter `resume` in the main function can only be used for resuming the training process.

### Inference

- To perform model testing, run python `main.py` with lines 187 and 188 in the main function commented out.