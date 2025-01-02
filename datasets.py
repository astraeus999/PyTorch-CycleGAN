import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2
import torch

# class ImageDataset(Dataset):
#     def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
#         self.transform = transforms.Compose(transforms_)
#         self.unaligned = unaligned

#         self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
#         self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

#     def __getitem__(self, index):
#         item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

#         if self.unaligned:
#             item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
#         else:
#             item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

#         return {'A': item_A, 'B': item_B}

#     def __len__(self):
#         return max(len(self.files_A), len(self.files_B))

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transforms_ = transforms_  # Store transformations
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.tif'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.tif'))

    def __getitem__(self, index):
        # Load image with cv2 and resize to 256x256
        img_A = cv2.imread(self.files_A[index % len(self.files_A)], cv2.IMREAD_UNCHANGED)
        img_A = cv2.resize(img_A, (256, 256))  # Resize to 256x256

        # Handle grayscale or RGB images
        if len(img_A.shape) == 2:  # Grayscale
            img_A = cv2.cvtColor(img_A, cv2.COLOR_GRAY2RGB)
        else:  # Convert BGR to RGB
            img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor
        item_A = torch.from_numpy(img_A).float().permute(2, 0, 1) / 255.0

        if self.unaligned:
            # print('Process the Unpaired data...')
            img_B = cv2.imread(self.files_B[random.randint(0, len(self.files_B) - 1)], cv2.IMREAD_UNCHANGED)
        else:
            img_B = cv2.imread(self.files_B[index % len(self.files_B)], cv2.IMREAD_UNCHANGED)

        img_B = cv2.resize(img_B, (256, 256))  # Resize to 256x256

        # Handle grayscale or RGB images
        if len(img_B.shape) == 2:  # Grayscale
            img_B = cv2.cvtColor(img_B, cv2.COLOR_GRAY2RGB)
        else:  # Convert BGR to RGB
            img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor
        item_B = torch.from_numpy(img_B).float().permute(2, 0, 1) / 255.0

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))