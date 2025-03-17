# Homework 1 - Image classification 🖼️

Student ID: 313551073

Name: 顏琦恩

## Introduction

Homework 1 focuses on image classification, where a model learns to categorize images into predefined classes. The task involves data preprocessing, designing a ResNet-based model, and implementing training, validation, and evaluation. The dataset is provided by the TAs. Requirements and other details are decribed below.

## Requirements

The pretrained model used in this homework is from pytorch, just make sure to have a python environment with a proper pytorch version.

## Getting started

### Training

To train the model, run `bash run_train.sh` or execute the following command.
```
python main.py \
--gpu_id 1 \
--mode train \
--ckpt_root ./ckpt \
--save_per_epoch 2 \
--batch_size 64 \
--num_epochs 200 \
# --resume \
```
The parameter `resume` can only be used for resuming the training process.

### Inference

To train the model, run `bash run_test.sh` or execute the following command.
```
python main.py \
--gpu_id 1 \
--mode test \
--ckpt_root ./ckpt \
--result_path prediction.csv \
# --ckpt_path ./your_ckpt_path
```
The parameter `ckpt_path` is used for specify the checkpoint you want to evaluate.