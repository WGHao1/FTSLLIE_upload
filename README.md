# Multi Task Low Light Image Enhancement Based on Frequency Domain Mapping
### 1. Dependencies

* Python3
* PyTorch>=1.0
* OpenCV- Python, TensorboardX
* NVIDIA GPU+CUDA
  
### 3. Data Preparation

#### 3.1. Training dataset

* 485 low/high-light image pairs from our485 of [LOL dataset]
* (https://daooshee.github.io/BMVC2018website/).
  
#### 3.2. Tesing dataset

* 15 low/high-light image pairs from eval15 of [LOL dataset](https://daooshee.github.io/BMVC2018website/).

* 100 low-light images from VE-LOL.
* 20 low-light images from LSRW-Huawei.
* 30 low-light images from LSRW-Nikon.
* 64 low-light images from DICM.
* 10 low-light images from LIME.
* 17 low-light images from MEF.
* 85 low-light images from NPE.
* 24 low-light images from VV.

### 4. Usage

#### 4.1. Testing 

* Evaluation: ```python eval.py```
  
### 5. Reference

* This code implementation is based on the algorithm described in the paper "Low-light Image Enhancement via Breaking Down the Darkness"

