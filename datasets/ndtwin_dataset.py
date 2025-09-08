""" NDTWIN-2009-2010 Dataset"""
import os
from PIL import Image
from torch.utils.data import Dataset
from utils import get_transform_ndtwin

DATAPATH = '/kaggle/input/celeba-448'
# DATAPATH = '/kaggle/input/nd-twin-448'
image_path = {}
image_label = {}


class NDTwinDataset(Dataset):
    def __init__(self, phase='train'):
        self.phase = phase
        self.image_id = []
        self.num_classes = 6213 # 347

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

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]

        # image
        image = Image.open(os.path.join(DATAPATH, 'CelebA_HQ_448_Final', 'CelebA_HQ_448_Final', image_path[image_id])).convert('RGB')  # (C, H, W)
        # image = Image.open(os.path.join(DATAPATH, 'ND_TWIN_448', image_path[image_id])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        # return image and label
        return image, image_label[image_id] - 1  # count begin from zero

    def __len__(self):
        return len(self.image_id)

class NDTwinVerificationDataset(Dataset):
    def __init__(self):
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
            # full_path = os.path.join(self.data_path, 'ND_TWIN_448', img_path)
            full_path = os.path.join(self.data_path, 'CelebA_HQ_448_Final', 'CelebA_HQ_448_Final', img_path)
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
    ds = NDTwinDataset('train')
    print(len(ds))
    for i in range(0, 10):
        image, label = ds[i]
        print(image.shape, label)
