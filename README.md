### 1. Introduction

This is the source code of "Efficient Parameter-free Adaptive Hashing for Large-Scale Cross-Modal Retrieval".


### 2. Requirements

- python 3.8.13
- pytorch 1.7.1+cuda11.0
- ...


### 3. Dataset Preparation

The NUS-WIDE dataset is available at: https://pan.baidu.com/s/1tizJc3VZIBE0RghRByahNA; password is: 7qim.

The IAPRR TC-12 dataset is available at: https://pan.baidu.com/s/1cYlypEM91g31rnAH0TUrmA; password is: hvjo.

You should generate the following files for each dataset. The structure of directory `./dataset` should be:
```
    dataset
    ├── NUS-WIDE
    │   ├── images
    │   ├── database_ind.txt
    │   ├── images_ind.txt
    │   ├── images_name.txt
    │   ├── label_hot.txt
    │   ├── test_ind.txt
    │   ├── train_ind.txt
    │   └── y_vector.txt
    └── IAPRTC-12
    │   ├── images
    │   ├── database_ind.txt
    │   ├── images_ind.txt
    │   ├── images_name.txt
    │   ├── label_hot.txt
    │   ├── test_ind.txt
    │   ├── train_ind.txt
    │   └── y_vector.txt
```


### 4. Train and Test

#### 4.1 NUS-WIDE
``` 
run EPAH_NUS-WIDE
```

