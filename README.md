# BaMSGAN: self-attention GAN with Blur and Memory for anime face generation

PS: This repository was created nearly 1 years ago, and was reconfigured on 12-15-2024.

## Introduction of paper
BaMSGAN (Self-Attention Generative Adversarial Network with Blur and Memory) is a novel network architecture designed for generating high-quality anime faces. It introduces unique features like edge blur preprocessing and a memory repository to enhance image clarity and prevent mode collapse and catastrophic forgetting in the training process.

- Paper link: <https://doi.org/10.3390/math11204401>

## Features of model
- **Self-Attention Mechanism:** Improves clarity and detail in generated anime faces of GAN.
- **Edge Blur Preprocessing:** Enhances edge definition in generated images.
- **Memory Repository:** Stores generated images to prevent forgetting and improve training stability.

## Project structure
```
# New structure organized for easier reuse and update
BaMSGAN_GitHub/
├── data/                 
│   ├── data_original/     # Original training data
│   ├── data_blur/         # Blurred data produced by Canny Algorithm
│   ├── data_his/          # Historical data used as inputs for further training
├── logs/                  
│   └── log.txt            # Log files
├── checkpoints/           
├── outputs/               
│   └── fake images        # fake images generated for each epoch
├── src/                   
│   ├── modules
│   │   ├── utils.py       # Utility functions
│   │   ├── losses.py      # Loss function definitions
│   │   └── layers.py      # Custom layer definitions
│   ├── model.py           # Model architecture definition
│   ├── train.py           # Training script
│   └── blur.py            # Data processing script
└── requirements.txt       
```
## Dataset
- Dataset Link: <https://www.kaggle.com/datasets/splcher/animefacedataset>

This dataset has 63,632 high-quality 96 X 96 anime faces.

## Usage
### Reproduction
1. Requirement: basic pytorch environment and cv2 + skimage
```bash
pip install -r requirement.txt
```
Or install manually by
```bash
# After installing Python, PyTorch, NumPy ...
pip install opencv-python == 4.10.0.84
pip install scikit-image == 0.25.0
```

2. Prepare data: use the same dataset as ours
- Orinigal training data: Download above-mentioned dataset and put dataset in the 'data/data_original' directory

Attention: For the convenience of using ImageFolder, we set another directory -- 'class' in each kind of dataset to save images.

- Blurred data:
```bash
# Read the configuration and just use default parameters if you just want to reproduce our experiments.
python blur.py
```
```
Arguments for the Blurred Dataset Script:
-----------------------------------------
--datasize     : int    [default=63560]  Total size of the dataset.
--blurratio    : float  [default=0.1]   Proportion of the dataset to blur (0-1).
--data_path    : str    [default='/root/BaMSGAN_GitHub/data/data_original/class']  
                          Path to the original image dataset.
--save_path    : str    [default='/root/BaMSGAN_GitHub/data/data_blur/class']  
                          Path to save the blurred images.
--t1           : int    [default=200]   Lower threshold for Canny edge detection.
--t2           : int    [default=500]   Upper threshold for Canny edge detection.
```
Then blurred data will be saved in the 'data/data_blur' directory.

3. Train the model: Run the training script to start training BaMSGAN
```bash
# Read the configuration and just use default parameters if you just want to reproduce our experiments.
python train.py
```
```
Arguments for the GAN Training Script:
--------------------------------------
--batchSize      : int    [default=128]         Batch size for training.
--blurratio      : float  [default=0.1]        Blurred data ratio.
--imageSize      : int    [default=64]         Image size for training.
--nz             : int    [default=100]        Size of the latent z vector.
--ngf            : int    [default=64]         Number of generator feature maps.
--ndf            : int    [default=64]         Number of discriminator feature maps.
--epoch          : int    [default=100]        Number of training epochs.
--start_epoch    : int    [default=1]          Epoch to start training from.
--lrd            : float  [default=2e-4]       Learning rate for the discriminator.
--lrg            : float  [default=2e-4]       Learning rate for the generator.
--data_path      : str    [default='/root/BaMSGAN_GitHub/data/data_original']  
                                               Folder for training data.
--output         : str    [default='/root/BaMSGAN_GitHub/output']  
                                               Output folder for generated images.
--logs           : str    [default='/root/BaMSGAN_GitHub/logs']  
                                               Folder for generated logs.
--checkpoints    : str    [default='/root/BaMSGAN_GitHub/checkpoints']  
                                               Output folder for checkpoints.
--blur_path      : str    [default='/root/BaMSGAN_GitHub/data/data_blur']  
                                               Folder for blurred data.
--his_path       : str    [default='/root/BaMSGAN_GitHub/data/data_his/class']  
                                               Folder for historical data.
--checkpointG    : str    [default=None]       Path to generator checkpoint.
--checkpointD    : str    [default=None]       Path to discriminator checkpoint.
```
4. Monitor training process: through the Terminal or log.txt in the 'logs' dictionary
```
Important information in log.txt:
Parameters, Loss of Generator and Discriminator for each epoch, Epoch duration, Path of saved models ... 
```

5. Get generated images and saved model weights after training
- Generated images for each epoch are saved in 'output' dictionary
- Model weights are saved every 5 epoch in 'checkpoints' dictionary

PS: As example, we training BaMSGAN using aboved-mentioned dataset and default parameters for 50 epochs and some of results are saved in both dictionaries.

This is the fake_samples_epoch_50:

![fake_samples_epoch_050](https://github.com/user-attachments/assets/de989ea3-7371-488c-8cbd-4237ed3f763c)

6. Resume training: the training could be continued from a specific checkpoint
```bash
# resume the training from epoch 51
python train.py --start_epoch 51 --checkpointG checkpoints/netG_0050.pth --checkpointD checkpoints/netD_0050.pth
```
### Custom setting
1.	Custom Dataset:
- Replace the dataset in the data/data_original directory with your custom dataset.
- Ensure the dataset format matches the required format (e.g., images in .jpg or .png).

2. Training Parameters:
- Modify training parameters in the script or pass them as command-line arguments.
```bash
# just for example and the parameter lists of blur.py and train.py are shown above.
python train.py --batchSize 64 --lrg 0.0001 --lrd 0.0001
```

3.	Memory Repository:
- The memory repository stores historical samples to improve stability and prevent forgetting.
- ~~Adjust the memory size or disable it in train.py.~~

PS: Oops!!! After submitting to the github, I just realized that I forgot to set the start epoch of Saving Memory as the Parameter, it is actually a catastrophic forgetting. 
I will add it in the next version if I have time. Please modify it manually now.

## Citaion
If you use BaMSGAN in your research, please cite the original paper:
```
@article{BaMSGAN,
  author    = {Li, X.; Li, B.; Fang, M.; Huang, R.; Huang, X},
  title     = {BaMSGAN: Self-Attention GAN with Blur and Memory for Anime Face Generation},
  journal   = {Mathematics},
  volume    = {11},
  number    = {20},
  year      = {2023},
  doi       = {10.3390/math11204401}
}
```
