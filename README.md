# BaMSGAN: self-attention GAN with Blur and Memory for anime face generation

## Introduction of paper
BaMSGAN (Self-Attention Generative Adversarial Network with Blur and Memory) is a novel network architecture designed for generating high-quality anime faces. It introduces unique features like edge blur preprocessing and a memory repository to enhance image clarity and prevent mode collapse and catastrophic forgetting in the training process.

- Paper link: <https://doi.org/10.3390/math11204401>

## Features of model
- **Self-Attention Mechanism:** Improves clarity and detail in generated anime faces of GAN.
- **Edge Blur Preprocessing:** Enhances edge definition in generated images.
- **Memory Repository:** Stores generated images to prevent forgetting and improve training stability.

## Introduction of project structure
- Directories: 
   
   data -- where you put your dataset
   
   data_blur -- where blurred pitctures are put
   
   data_his -- where history fake pitctures are put
   
   img -- the result(generated fake examples)
- Python Files: 
   
   blur.py -- code to blur original data
   
   other python files -- code of BaMSGAN model and training

## Dataset
- Dataset Link: <https://www.kaggle.com/datasets/splcher/animefacedataset>

This dataset has 63,632 high-quality anime faces.

## Usage
1. requirement: basic pytorch environment and cv2 + skimage
```bash
pip install torch
pip install opencv-python
pip install scikit-image
```
2. put dataset in the 'data' directory
   
3. blur part of dataset, and it has 2 parameters, size of dataset and blur ratio 
(0-1, 0.1 means 10% of picture in dataset will be blurred). 
the default parameter is fit for our dataset.
```bash
python blur.py --options
```
Then the blurred pitcture will be put in the data_blur/Blur_images.
   
4. start the training process, set parameters.
```bash
python train.py --options
```
you can see the generated pictures in the 'img' directory.

## Tips
- If you met the problem about the '.ipynb_checkpoints', directly remove the file of '.ipynb_checkpoints' in the mentioned directory.
```bash
ls -a
rm -r .ipynb_checkpoints
```
- If you want to continue training with past model, please modify codes in line 110 of the 'train.py', set the start_epoch, then note off the line 111 and 112, set the path of D and G that you want to use.

