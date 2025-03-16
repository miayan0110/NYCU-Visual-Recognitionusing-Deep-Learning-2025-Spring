# Homework 1 - Image classification

Student ID: 313551073

Name: 顏琦恩

## Introduction

The task of homework 1 is image classification. I use ResNeXt101 as base model and change the output class number to 100 to fit our task.

## Requirements



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