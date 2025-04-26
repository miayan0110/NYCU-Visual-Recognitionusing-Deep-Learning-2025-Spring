# Homework 3 - Instance Segmentation 🧩

Student ID: 313551073

Name: 顏琦恩

## Introduction

Homework 3 focuses on instance segmentation, where a model learns to detect and delineate individual object instances in real-world images. The task involves data preprocessing, designing a Mask R-CNN-based model, and implementing training, validation, and evaluation procedures. The dataset is provided by the TAs, and further details are provided below.

## Requirements

The pretrained model used in this homework is from pytorch, make sure to have a python environment with a proper pytorch version. The pycocotools is also used in this homework, install it using the command `pip install pycocotools`.

## Getting started

### Training

- To train and validate the model, simply run the command `python main.py`.
- The parameter `resume` in the main function is used for resuming the training process, specify your checkpoint path using `--resume_path your_path`.

### Inference

- To perform model testing, run the command `python main.py --mode test`, remember to specify the weight you want to inference using `--resume_path your_path`.