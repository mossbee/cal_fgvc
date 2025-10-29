""" NDTWIN-2009-2010 Dataset"""
import os
from PIL import Image
from torch.utils.data import Dataset
from utils import get_transform_ndtwin
import torch
from pathlib import Path

# DATAPATH = '/kaggle/input/celeba-448'
DATAPATH = '/kaggle/input/nd-twin-448'
LANDMARKS_MASK_PATH = '/kaggle/input/landmarks_masks/landmarks_masks'  # Kaggle dataset path

image_path = {}
image_label = {}


class NDTwinDataset(Dataset):
    def __init__(self, phase='train', resize=(448, 448), use_landmarks=True):
        self.phase = phase
        self.image_id = []
        self.num_classes = 347 #6213 #347 
        self.use_landmarks = use_landmarks
        self.landmarks_mask_dir = Path(LANDMARKS_MASK_PATH)

        # get image path from images.txt
        with open(os.path.join(DATAPATH, 'images.txt')) as f:
            for line in f.readlines():
                id, path = line.strip().split(' ')
                image_path[id] = path
                if self.phase == 'train':
                    self.image_id.append(id)

        # get image label from image_class_labels.txt
        with open(os.path.join(DATAPATH, 'image_class_labels.txt')) as f:
            for line in f.readlines():
                id, label = line.strip().split(' ')
                image_label[id] = int(label)

        # transform
        self.transform = get_transform_ndtwin()

    def _load_landmark_mask(self, image_id):
        """Load precomputed landmark mask for given image_id"""
        # Extract filename without extension from image path
        img_path = image_path[image_id]
        filename = Path(img_path).stem  # e.g., '90003d13'
        
        mask_path = self.landmarks_mask_dir / f"{filename}.pt"
        
        if mask_path.exists():
            mask = torch.load(mask_path)  # Shape: (1, H, W)
            return mask
        else:
            # Return uniform mask if landmark file doesn't exist
            # This ensures training doesn't break for missing landmarks
            print(f"Warning: Landmark mask not found for {filename}, using uniform mask")
            return torch.ones(1, 448, 448) * 0.5

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]

        # image
        image = Image.open(os.path.join(DATAPATH, 'ND_TWIN_448', image_path[image_id])).convert('RGB')
        image = self.transform(image)

        # Load landmark mask if using guided attention and in training phase
        if self.use_landmarks and self.phase == 'train':
            landmark_mask = self._load_landmark_mask(image_id)
            return image, image_label[image_id] - 1, landmark_mask
        else:
            # For validation or when not using landmarks
            return image, image_label[image_id] - 1

    def __len__(self):
        return len(self.image_id)


class NDTwinVerificationDataset(Dataset):
    def __init__(self, resize=(448, 448)):
        self.data_path = DATAPATH
        self.pairs = []
        self.unique_images = set()
        
        # Read pairs file
        with open(os.path.join(DATAPATH, 'test_pairs.txt'), 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 3:
                    img1_path, img2_path, label = parts
                    self.pairs.append((img1_path, img2_path, int(label)))
                    self.unique_images.add(img1_path)
                    self.unique_images.add(img2_path)
        
        # Transform for test images (no augmentation for testing)
        self.transform = get_transform_ndtwin()
        
        # Cache for loaded images to avoid duplicate loading
        self.image_cache = {}
    
    def _load_image(self, img_path):
        """Load image with caching to handle duplicates efficiently"""
        if img_path not in self.image_cache:
            full_path = os.path.join(self.data_path, 'ND_TWIN_448', img_path)
            # full_path = os.path.join(self.data_path, 'CelebA_HQ_448_Final', 'CelebA_HQ_448_Final', img_path)
            image = Image.open(full_path).convert('RGB')
            self.image_cache[img_path] = self.transform(image)
        return self.image_cache[img_path]
    
    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        return img1, img2, label
    
    def __len__(self):
        return len(self.pairs)


if __name__ == '__main__':
    # Test without landmarks
    print("Testing dataset WITHOUT landmarks:")
    ds = NDTwinDataset('train', use_landmarks=False)
    print(f"Dataset size: {len(ds)}")
    for i in range(0, 3):
        data = ds[i]
        if len(data) == 2:
            image, label = data
            print(f"Sample {i}: image shape={image.shape}, label={label}")
        else:
            image, label, mask = data
            print(f"Sample {i}: image shape={image.shape}, label={label}, mask shape={mask.shape}")
    
    # Test with landmarks
    print("\nTesting dataset WITH landmarks:")
    ds_landmarks = NDTwinDataset('train', use_landmarks=True)
    print(f"Dataset size: {len(ds_landmarks)}")
    for i in range(0, 3):
        data = ds_landmarks[i]
        if len(data) == 3:
            image, label, mask = data
            print(f"Sample {i}: image shape={image.shape}, label={label}, mask shape={mask.shape}")
            print(f"  Mask stats: min={mask.min():.3f}, max={mask.max():.3f}, mean={mask.mean():.3f}")
        else:
            image, label = data
            print(f"Sample {i}: image shape={image.shape}, label={label} (no mask)")
