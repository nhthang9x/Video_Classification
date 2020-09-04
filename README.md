# [KERAS] Character-level Convolutional Networks for Text Classification

## Introduction

Here is my keras implementation of the model of Video Classification. 


## Training vs Testing
Step 0: Download raw data from link (https://drive.google.com/file/d/1n8SbC3_mVO8_xk9SFZ5AUX6TKRyXN9Ct/view?usp=sharing) and put it into this repo

Step 1: ```python preprocess.py -f 1 -s 101 ```

Step 2: ```python data_augmentation.py ```

Step 3: ```python generate_data.py -d train_data -l train_label ```

Step 4: ```python preprocess.py -f 101 -s 116 ```

Step 5: ```python data_augmentation.py ```

Step 6: ```python generate_data.py -d test_data -l test_label ```

Step 7: ```python train.py ```


