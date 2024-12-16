import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import shutil
import logging
from datetime import datetime

from model import Generator, Discriminator  
from modules.losses import Wasserstein, Hinge, Hinge1  
from modules.utils import samimg, img_repo, rand_del, len_repo


# Set random seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(15)


# Initialize weights for model layers
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


# Get learning rate of the optimizer
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Training script for a GAN model.")
parser.add_argument('--batchSize', type=int, default=128, help="Batch size for training.")
parser.add_argument('--blurratio', type=float, default=0.1, help="Blurred data ratio.")
parser.add_argument('--imageSize', type=int, default=64, help="Image size for training.")
parser.add_argument('--nz', type=int, default=100, help="Size of the latent z vector.")
parser.add_argument('--ngf', type=int, default=64, help="Number of generator feature maps.")
parser.add_argument('--ndf', type=int, default=64, help="Number of discriminator feature maps.")
parser.add_argument('--epoch', type=int, default=100, help="Number of training epochs.")
parser.add_argument('--start_epoch', type=int, default=1, help="Epoch to start training from.")
parser.add_argument('--lrd', type=float, default=2e-4, help="Learning rate for the discriminator.")
parser.add_argument('--lrg', type=float, default=2e-4, help="Learning rate for the generator.")
parser.add_argument('--data_path', default='/root/BaMSGAN_GitHub/data/data_original', help="Folder for training data.")
parser.add_argument('--output', default='/root/BaMSGAN_GitHub/output', help="Output folder for generated images.")
parser.add_argument('--logs', default='/root/BaMSGAN_GitHub/logs', help="Folder for generated logs.")
parser.add_argument('--checkpoints', default='/root/BaMSGAN_GitHub/checkpoints', help="Output folder for checkpoints.")
parser.add_argument('--blur_path', default='/root/BaMSGAN_GitHub/data/data_blur', help="Folder for blurred data.")
parser.add_argument('--his_path', default='/root/BaMSGAN_GitHub/data/data_his/class', help="Folder for historical data.")
parser.add_argument('--checkpointG', default=None, help="Path to generator checkpoint.")
parser.add_argument('--checkpointD', default=None, help="Path to discriminator checkpoint.")
parser.add_argument('--memeory_epoch', default=25, help="Epoch that start to save memory(work when it >= 25) ")
parser.add_argument('--memeory_size', default=120, help="Size of memory repository, and random deletion will be conducted when it is filled")
opt = parser.parse_args()

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Check and create folders if they don't exist
def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        print(f"Creating folder: {folder_path}")
        os.makedirs(folder_path)


ensure_folder_exists(opt.data_path)
ensure_folder_exists(opt.blur_path)
ensure_folder_exists(opt.his_path)
ensure_folder_exists(opt.output)
ensure_folder_exists(opt.checkpoints)
ensure_folder_exists(opt.logs)

log_file = os.path.join(opt.logs, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

logging.basicConfig(
    filename=log_file,  
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'  
)

logging.info("Training script initialized.")
logging.info(f"Logs will be saved to: {log_file}")
logging.info(f"Training parameters: {vars(opt)}")

# Define image transformations
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize([opt.imageSize, opt.imageSize]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Function to clean hidden folders recursively
def clean_hidden_folders(root_dir, target_name=".ipynb_checkpoints"):
    """
    Recursively search and delete specified hidden folders.
    
    Args:
        root_dir (str): Root directory to scan.
        target_name (str): Name of the hidden folder to delete (default: ".ipynb_checkpoints").
    """
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if dir_name == target_name:
                target_path = os.path.join(root, dir_name)
                print(f"Found and removed: {target_path}")
                shutil.rmtree(target_path)


# Clean hidden folders in data and blur paths
clean_hidden_folders(opt.data_path)
clean_hidden_folders(opt.blur_path)

# Load datasets
dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)
dataset_blur = torchvision.datasets.ImageFolder(opt.blur_path, transform=transforms)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
)

dataloader_blur = torch.utils.data.DataLoader(
    dataset=dataset_blur,
    batch_size=round(opt.blurratio * opt.batchSize),
    shuffle=True,
    drop_last=True,
)

# Initialize models
netG = Generator().to(device)
netD = Discriminator().to(device)

# Check if training is resuming from a specific epoch
if opt.start_epoch > 1:
    if not opt.checkpointG or not opt.checkpointD:
        raise ValueError(
            "Checkpoint paths must be provided when resuming training from a specific epoch!"
        )
    print(f"Resuming training from epoch {opt.start_epoch}...")
    print(f"Loading generator checkpoint from {opt.checkpointG}")
    netG.load_state_dict(torch.load(opt.checkpointG, map_location=device))
    print(f"Loading discriminator checkpoint from {opt.checkpointD}")
    netD.load_state_dict(torch.load(opt.checkpointD, map_location=device))
else:
    print("Starting training from scratch...")
    netG.apply(weights_init)
    netD.apply(weights_init)

print('Generator model initialized.')
print(f'Total parameters in Generator: {sum(p.numel() for p in netG.parameters())}')
print('Discriminator model initialized.')
print(f'Total parameters in Discriminator: {sum(p.numel() for p in netD.parameters())}')

# Define loss functions and optimizers
criterionG = Hinge()
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.lrg, weight_decay=0.0001)
optimizerD = torch.optim.RMSprop(netD.parameters(), lr=opt.lrd, weight_decay=0.0001)

# Define learning rate schedulers
lrd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=5, eta_min=5E-5)
lrg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=5, eta_min=5E-5)

criterionD = Hinge()
criterionD1 = Hinge1()

# Training variables
total_lossD = 0.0
total_lossG = 0.0
fixed_noise = torch.randn(opt.batchSize, 100, 1, 1).to(device)

# Training loop
for epoch in range(opt.start_epoch, opt.epoch + 1):
    
    epoch_start_time = datetime.now()
    logging.info(f"Starting Epoch {epoch}/{opt.epoch}...")
    
    with tqdm(total=len(dataloader),
              desc=f'Epoch {epoch}/{opt.epoch}',
              postfix=dict,
              mininterval=0.3) as pbar:
        for i, ((imgs, _), (imgs_blur, _)) in enumerate(zip(dataloader, dataloader_blur)):
            imgs = imgs.to(device)
            imgs_blur = imgs_blur.to(device)

            # Train Discriminator
            optimizerD.zero_grad()
            outputreal = netD(imgs)
            noise = torch.randn(opt.batchSize, opt.nz, 1, 1).to(device)
            fake = netG(noise)
            outputfake = netD(fake.detach())
            outputfake_blur = netD(imgs_blur)
            
            # Use historical data stored after memeory_epoch
            if epoch > opt.memeory_epoch and opt.memeory_epoch >= 25:
                imgs_his = samimg(round(opt.blurratio * opt.batchSize), repo_path=opt.his_path).to(device)
                outputfake_his = netD(imgs_his)
                lossD = criterionD1(outputreal, outputfake, outputfake_blur, outputfake_his)
            else:
                lossD = criterionD(outputreal, outputfake, outputfake_blur)

            total_lossD += lossD.item()
            lossD.backward()
            optimizerD.step()
            lrd_scheduler.step()

            # Train Generator
            optimizerG.zero_grad()
            noise = torch.randn(opt.batchSize, opt.nz, 1, 1).to(device)
            fake = netG(noise)
            output = netD(fake)
            lossG = criterionG(output)
            total_lossG += lossG.item()
            lossG.backward()
            optimizerG.step()
            lrg_scheduler.step()

            # Update progress bar
            pbar.set_postfix(
                **{
                    'total_lossD': total_lossD / (i + 1),
                    'lrd': get_lr(optimizerD),
                    'total_lossG': total_lossG / (i + 1),
                    'lrg': get_lr(optimizerG),
                }
            )
            pbar.update(1)

        # Ensure progress bar reaches 100%
        pbar.n = pbar.total
        pbar.last_print_n = pbar.total
        pbar.update(0)

    # Store historical data after epoch memeory_epoch - 5
    if epoch > opt.memeory_epoch - 5 and opt.memeory_epoch >= 25:
        for num in range(round(opt.batchSize * opt.blurratio)):
            n = torch.randn((1, opt.nz, 1, 1)).to(device)
            f = netG(n)
            img_repo(f, epoch, num, repo_path=opt.his_path)  # Use opt.his_path

    # Control memory repository size
    current_size = len_repo(opt.his_path)  # Get the current number of images
    excess_images = current_size - opt.memeory_size  # Calculate excess images
    
    if excess_images > 0:  # Delete extra images if memory exceeds the limit
        logging.info(f"Repository exceeds memory size. Deleting {excess_images} images...")
        rand_del(excess_images, repo_path=opt.his_path)

    # Save generated images for current epoch
    fake = netG(fixed_noise)
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
    vutils.save_image(
        fake.data,
        f'{opt.output}/fake_samples_epoch_{epoch:03d}.png',
        normalize=True
    )

    epoch_end_time = datetime.now()
    epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
    logging.info(f"Epoch {epoch}/{opt.epoch} completed. "
             f"Total D Loss: {total_lossD:.4f}, Total G Loss: {total_lossG:.4f}, Duration: {epoch_duration:.2f}s")
    # Reset loss trackers
    total_lossG = 0.0
    total_lossD = 0.0

    # Save model checkpoints every 5 epochs
    if epoch % 5 == 0:
        generator_path = os.path.join(opt.checkpoints, f'netG_{epoch:04d}.pth')
        discriminator_path = os.path.join(opt.checkpoints, f'netD_{epoch:04d}.pth')

        torch.save(netG.state_dict(), generator_path)
        torch.save(netD.state_dict(), discriminator_path)

        logging.info(f"Model checkpoints saved: Generator -> {generator_path}, Discriminator -> {discriminator_path}")
        
   

    