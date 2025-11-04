### **Method 1: Landmark-Weighted Attention Loss**
*Simplest and most direct approach*

**Concept**: Add an auxiliary loss that encourages attention maps to have higher values at landmark regions.

**Implementation**:
```python
# Attention regularization loss
def landmark_attention_loss(attention_maps, landmark_masks):
    """
    attention_maps: (B, M, H, W) - learned attention maps
    landmark_masks: (B, 1, H, W) - binary/soft masks for important regions
    """
    # Upsample attention to image size if needed
    B, M, AH, AW = attention_maps.shape
    _, _, MH, MW = landmark_masks.shape
    
    if AH != MH or AW != MW:
        attention_maps = F.interpolate(attention_maps, size=(MH, MW), mode='bilinear')
    
    # Compute attention concentration on landmarks
    # Average across all attention heads
    attention_avg = attention_maps.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    
    # L2 loss to encourage attention on landmarks
    landmark_region_attention = attention_avg * landmark_masks
    non_landmark_region_attention = attention_avg * (1 - landmark_masks)
    
    # Maximize attention on landmarks, minimize on non-landmarks
    loss = -landmark_region_attention.mean() + 0.5 * non_landmark_region_attention.mean()
    
    return loss
```

**Data Preparation**:
- **Input**: 478 Mediapipe landmarks → Select key regions:
  - Eyes: landmarks 33, 133, 159, 145, 362, 385, 380, 386 (eye contours)
  - Nose: landmarks 1, 2, 98, 327 (nose bridge and tip)
  - Mouth: landmarks 61, 291, 0, 17, 84, 314 (mouth contours)
- **Create binary mask**: Set pixels within radius (e.g., 10-20 pixels) of these landmarks to 1
- **Store**: `landmark_masks` as (B, 1, H, W) tensor alongside images

**Pros**: Simple, interpretable, low computational cost  
**Cons**: Treats all attention heads equally, may be too rigid