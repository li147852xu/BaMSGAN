"""
Script for managing image repositories with functionalities for saving, deleting, and sampling images.
"""

# Import Modules
import os
import cv2
import torch
import random
import shutil
from pathlib import Path
from torchvision.utils import save_image


# Save an image to the repository
def img_repo(img, epoch, num, repo_path='data_his'):
    """
    Save an image to the repository with a specific epoch and number in its name.

    Args:
        img (Tensor): Image tensor to save.
        epoch (int): Epoch number for naming.
        num (int): Image index for naming.
        repo_path (str): Path to the repository folder. Defaults to 'data_his'.
    """
    Path(repo_path).mkdir(parents=True, exist_ok=True)  # Ensure the folder exists
    save_image(img, os.path.join(repo_path, f'epoch{epoch}_num{num}.png'))


# Get the number of images in the repository
def len_repo(repo_path='data_his'):
    """
    Get the total number of images in the repository.

    Args:
        repo_path (str): Path to the repository folder. Defaults to 'data_his'.

    Returns:
        int: Number of files in the repository.
    """
    files = [f for f in os.listdir(repo_path) if os.path.isfile(os.path.join(repo_path, f))]
    return len(files)


# Delete an image from the repository
def del_img(index, repo_path='data_his'):
    """
    Delete an image by its index from the repository.

    Args:
        index (int): Index of the image to delete.
        repo_path (str): Path to the repository folder. Defaults to 'data_his'.
    """
    files = os.listdir(repo_path)
    if index < len(files):
        os.remove(os.path.join(repo_path, files[index]))


# Randomly delete a specified number of images from the repository
def rand_del(num_to_delete, repo_path='data_his', max_images=400):
    """
    Randomly delete a specified number of images from the repository if the count exceeds a threshold.

    Args:
        num_to_delete (int): Number of images to delete.
        repo_path (str): Path to the repository folder. Defaults to 'data_his'.
        max_images (int): Maximum allowed images in the repository. Defaults to 400.
    """
    total_images = len_repo(repo_path)
    if total_images > max_images:
        indices_to_delete = random.sample(range(total_images), num_to_delete)
        indices_to_delete.sort(reverse=True)  # Sort in descending order for safe deletion
        for idx in indices_to_delete:
            del_img(idx, repo_path)


# Sample a specified number of images from the repository
def samimg(num_to_sample, repo_path='data_his'):
    """
    Sample a specified number of images from the repository.

    Args:
        num_to_sample (int): Number of images to sample.
        repo_path (str): Path to the repository folder. Defaults to 'data_his'.

    Returns:
        Tensor: A batch of sampled image tensors.
    """
    images = []
    if os.path.exists(os.path.join(repo_path, '.ipynb_checkpoints')):
        shutil.rmtree(os.path.join(repo_path, '.ipynb_checkpoints'), ignore_errors=True)  # Remove hidden checkpoints folder

    total_images = len_repo(repo_path)  # Get total available images
    num_to_sample = min(num_to_sample, total_images)  # Ensure num_to_sample is not larger than total_images

    if total_images == 0:  # No images available
        raise ValueError("No images available in the repository to sample.")

    sampled_indices = random.sample(range(total_images), num_to_sample)

    for idx in sampled_indices:
        files = os.listdir(repo_path)
        image_path = os.path.join(repo_path, files[idx])
        image = cv2.imread(image_path)
        if image is not None:
            tensor_image = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)  # Convert to CHW format
            images.append(tensor_image)

    return torch.cat(images, dim=0) if images else torch.empty(0)