# landmarks/preprocess_landmarks.py
import os
import json
import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import gaussian_filter
import cv2
from tqdm import tqdm

# Mediapipe landmark indices for important facial regions
LANDMARK_INDICES = {
    'eyes': [
        # Right eye
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        # Left eye  
        # 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466
    ],
    'nose': [
        # Nose bridge and tip
        # 1, 220, 4, 5, 19, 94, 168, 6, 197, 195, 5, 440, 79, 309
        122, 196, 3, 51, 45,
        351, 419, 248, 281, 275,
        1, 220, 440, 79, 309, 6, 115, 344, 218, 438, 166, 392
    ],
    'mouth': [
        # Outer mouth
        # 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 306,
        # Inner mouth
        # 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
        76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306,
        # Additional important points
        # 17, 84, 314, 87, 317
    ],
}

def create_landmark_mask(landmarks_data, image_size=(448, 448), radius=15, use_gaussian=True):
    """
    Create a soft mask from landmark coordinates
    
    Args:
        landmarks_data: dict with 'faces' key containing landmarks
        image_size: tuple (H, W)
        radius: pixel radius around each landmark
        use_gaussian: if True, apply Gaussian smoothing for soft masks
    
    Returns:
        mask: numpy array of shape (H, W) with values in [0, 1]
    """
    H, W = image_size
    mask = np.zeros((H, W), dtype=np.float32)
    
    # Check if landmarks exist
    if not landmarks_data.get('faces') or len(landmarks_data['faces']) == 0:
        print(f"Warning: No faces found in landmarks data")
        return mask
    
    # Get first face landmarks (assuming single face per image)
    face_landmarks = landmarks_data['faces'][0]['landmarks']
    
    # Collect important landmark coordinates
    important_coords = []
    for region_name, indices in LANDMARK_INDICES.items():
        for idx in indices:
            if idx < len(face_landmarks):
                lm = face_landmarks[idx]
                x, y = int(lm['x']), int(lm['y'])
                # Ensure coordinates are within image bounds
                if 0 <= x < W and 0 <= y < H:
                    important_coords.append((x, y))
    
    # Create binary mask by drawing circles around landmarks
    for x, y in important_coords:
        cv2.circle(mask, (x, y), radius, 1.0, -1)
    
    # Apply Gaussian smoothing for soft transitions
    if use_gaussian:
        mask = gaussian_filter(mask, sigma=radius/3)
        # Normalize to [0, 1]
        if mask.max() > 0:
            mask = mask / mask.max()
    
    return mask

def preprocess_all_landmarks(
    landmarks_dir='./landmarks',
    output_dir='./landmarks_masks',
    image_size=(448, 448),
    radius=15
):
    """
    Preprocess all JSON landmark files into tensor masks
    """
    landmarks_dir = Path(landmarks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find all JSON files
    json_files = list(landmarks_dir.glob('*.json'))
    print(f"Found {len(json_files)} landmark JSON files")
    
    # Process each file
    for json_path in tqdm(json_files, desc="Creating landmark masks"):
        # Load JSON
        with open(json_path, 'r') as f:
            landmarks_data = json.load(f)
        
        # Create mask
        mask = create_landmark_mask(
            landmarks_data, 
            image_size=image_size, 
            radius=radius,
            use_gaussian=True
        )
        
        # Convert to tensor and save
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # Add channel dim: (1, H, W)
        
        # Save as .pt file (much faster loading than JSON)
        output_path = output_dir / f"{json_path.stem}.pt"
        torch.save(mask_tensor, output_path)
    
    print(f"Saved {len(json_files)} mask files to {output_dir}")
    
    # Create a visualization for sanity check
    if len(json_files) > 0:
        sample_json = json_files[0]
        with open(sample_json, 'r') as f:
            sample_data = json.load(f)
        sample_mask = create_landmark_mask(sample_data, image_size, radius)
        
        # Save visualization
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.imshow(sample_mask, cmap='hot')
        plt.title(f'Sample Landmark Mask: {sample_json.stem}')
        plt.colorbar()
        plt.savefig(output_dir / 'sample_mask_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved sample visualization to {output_dir / 'sample_mask_visualization.png'}")

if __name__ == '__main__':
    # Run preprocessing
    preprocess_all_landmarks(
        landmarks_dir='./landmarks',
        output_dir='./landmarks_masks',
        image_size=(448, 448),
        radius=18  # Adjust based on your preference
    )