import json
import cv2
import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mask_generation import create_landmark_mask

def visualize_mask_on_image(
    json_path,
    image_path=None,
    radius=18,
    alpha=0.5,
    save_path=None
):
    """
    Visualize landmark mask overlaid on the original image
    
    Args:
        json_path: path to landmark JSON file
        image_path: path to face image (if None, extracts from JSON)
        radius: radius for landmark circles
        alpha: transparency of mask overlay (0=transparent, 1=opaque)
        save_path: where to save visualization (if None, only displays)
    """
    # Load landmarks JSON
    with open(json_path, 'r') as f:
        landmarks_data = json.load(f)
    
    # Get image path from JSON if not provided
    if image_path is None:
        original_path = landmarks_data['image_path']
        relative_path = '/'.join(original_path.split('/')[-2:])
        image_path = f"/home/mossbee/Work/Dataset/ND_TWIN_448/{relative_path}"
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get image size
    H, W = image.shape[:2]
    
    # Create mask
    mask = create_landmark_mask(
        landmarks_data,
        image_size=(H, W),
        radius=radius,
        use_gaussian=True
    )
    
    # Create colored overlay (red heatmap)
    mask_colored = np.zeros((H, W, 3), dtype=np.float32)
    mask_colored[:, :, 0] = mask  # Red channel
    
    # Blend image with mask
    blended = image_rgb.astype(np.float32) / 255.0
    blended = (1 - alpha) * blended + alpha * mask_colored
    blended = np.clip(blended, 0, 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Mask only
    im = axes[1].imshow(mask, cmap='hot')
    axes[1].set_title(f'Landmark Mask (radius={radius})', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(blended)
    axes[2].set_title(f'Overlay (alpha={alpha})', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    plt.close(fig)  # Close instead of show
    
    # Also draw landmarks as green dots
    image_with_landmarks = image_rgb.copy()
    if landmarks_data.get('faces') and len(landmarks_data['faces']) > 0:
        face_landmarks = landmarks_data['faces'][0]['landmarks']
        for lm in face_landmarks:
            x, y = int(lm['x']), int(lm['y'])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(image_with_landmarks, (x, y), 2, (0, 255, 0), -1)
    
    # Show landmarks
    fig2 = plt.figure(figsize=(8, 8))
    plt.imshow(image_with_landmarks)
    plt.title('All Landmarks (green dots)', fontsize=14)
    plt.axis('off')
    
    if save_path:
        landmarks_save_path = Path(save_path).parent / f"{Path(save_path).stem}_landmarks.png"
        plt.savefig(landmarks_save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved landmarks visualization to {landmarks_save_path}")
    
    plt.close(fig2)  # Close instead of show

def visualize_from_mask_file(
    mask_path,
    image_path,
    alpha=0.5,
    save_path=None
):
    """
    Visualize pre-computed mask tensor on image
    
    Args:
        mask_path: path to .pt mask file
        image_path: path to face image
        alpha: transparency of mask overlay
        save_path: where to save visualization
    """
    # Load mask tensor
    mask_tensor = torch.load(mask_path)
    mask = mask_tensor.squeeze().numpy()  # Remove channel dim
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize mask if needed
    H, W = image_rgb.shape[:2]
    if mask.shape != (H, W):
        mask = cv2.resize(mask, (W, H))
    
    # Create colored overlay
    mask_colored = np.zeros((H, W, 3), dtype=np.float32)
    mask_colored[:, :, 0] = mask  # Red channel
    
    # Blend
    blended = image_rgb.astype(np.float32) / 255.0
    blended = (1 - alpha) * blended + alpha * mask_colored
    blended = np.clip(blended, 0, 1)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    im = axes[1].imshow(mask, cmap='hot')
    axes[1].set_title('Landmark Mask', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    axes[2].imshow(blended)
    axes[2].set_title(f'Overlay (alpha={alpha})', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    plt.close(fig)

if __name__ == '__main__':
    # Example 1: Visualize from JSON landmarks
    json_path = "landmarks/90004d13.json"
    output_path = "mask_visualization.png"
    
    print("Creating mask visualization...")
    visualize_mask_on_image(
        json_path=json_path,
        radius=18,
        alpha=0.4,
        save_path=output_path
    )
    print("\nDone! Check the output files.")
    
    # Example 2: Visualize from pre-computed mask
    # mask_path = "landmarks_masks/90003d13.pt"
    # image_path = "/home/mossbee/Work/Dataset/ND_TWIN_448/90003/90003d13.jpg"
    # visualize_from_mask_file(
    #     mask_path=mask_path,
    #     image_path=image_path,
    #     alpha=0.4,
    #     save_path="mask_from_tensor.png"
    # )