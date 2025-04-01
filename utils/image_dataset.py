import os
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torch.utils.data import Dataset



class OptimizedImageDataset(Dataset):
    """
    Improved PyTorch Dataset for loading images with optional augmentations and error handling.

    Args:
        dataFrame (DataFrame): DataFrame containing image information (merged.csv).
        image_dir (str): Path to the directory where images are stored.
        transform (callable, optional): Optional transform to apply to images.
        augment (bool, optional): Whether to apply random augmentations (default: False).
        image_size (tuple, optional): Resize target size (default: (200, 200)).

    Methods:
        __getitem__(self, idx): Retrieves and transforms the image at the specified index.
        __len__(self): Returns the total number of images in the dataset.
    """

    def __init__(self, dataFrame, image_dir="./data/images", augment=False, image_size=(200, 200)):
        self.dataFrame = dataFrame
        self.image_dir = image_dir
        self.augment = augment
        self.image_size = image_size

        # Basic transform (resize + normalize)
        base_transforms = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

        # Add augmentations if enabled
        if self.augment:
            augmentation_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ]
            self.transform = transforms.Compose(augmentation_transforms + base_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)

    def __getitem__(self, idx):
        """
        Loads and transforms an image from the dataset.

        Args:
            idx (int): Index of the image.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        try:
            image_path = os.path.join(self.image_dir, str(self.dataFrame.iloc[idx]["objectid"]) + ".jpg")
            image = Image.open(image_path).convert("RGB")

            # Apply transformations
            image = self.transform(image)
            return image

        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Error loading image at {image_path}: {e}")
            # Return a zero tensor if image is corrupt/missing
            return torch.zeros(3, *self.image_size)

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.dataFrame)