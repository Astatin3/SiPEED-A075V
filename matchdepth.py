import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def align_depth_maps(tof_depth, img_depth, tof_mask):
    """
    Aligns the scale and bounds of image-to-depth output to match the TOF depth map
    
    Parameters:
    -----------
    tof_depth : numpy.ndarray
        Depth map from TOF sensor (incomplete but spatially accurate)
    img_depth : numpy.ndarray
        Depth map from image-to-depth model (complete but incorrectly scaled)
    tof_mask : numpy.ndarray
        Binary mask where 1 indicates valid TOF measurements
    
    Returns:
    --------
    numpy.ndarray
        Aligned image depth map with scale and bounds matching TOF depth map
    """
    # Extract valid TOF depth values and corresponding image depth values
    valid_mask = tof_mask > 0
    tof_valid = tof_depth
    img_valid = img_depth*tof_mask
    
    if len(tof_valid) < 10:
        raise ValueError("Not enough valid TOF points for reliable alignment")
    
    # Reshape for sklearn
    tof_valid = tof_valid.reshape(-1, 1)
    img_valid = img_valid.reshape(-1, 1)
    
    # Fit linear regression to find scale and offset
    # We're solving for: tof_depth = a * img_depth + b
    model = LinearRegression()
    model.fit(img_valid, tof_valid)
    
    scale = model.coef_[0][0]
    offset = model.intercept_[0]
    
    print(f"Fitted scale: {scale:.4f}")
    print(f"Fitted offset: {offset:.4f}")
    
    # Apply transformation to the entire image depth map
    aligned_depth = scale * img_depth + offset
    
    return aligned_depth

def evaluate_alignment(tof_depth, aligned_depth, tof_mask):
    """
    Evaluate how well the aligned depth map matches the TOF depth map
    """
    valid_mask = tof_mask > 0
    tof_valid = tof_depth[valid_mask]
    aligned_valid = aligned_depth[valid_mask]
    
    abs_diff = np.abs(tof_valid - aligned_valid)
    mean_abs_error = np.mean(abs_diff)
    max_abs_error = np.max(abs_diff)
    
    print(f"Mean Absolute Error: {mean_abs_error:.4f}")
    print(f"Max Absolute Error: {max_abs_error:.4f}")
    
    return mean_abs_error, max_abs_error

def visualize_results(tof_depth, img_depth, aligned_depth, tof_mask):
    """
    Visualize original and aligned depth maps
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original TOF depth map
    im0 = axes[0, 0].imshow(tof_depth)
    axes[0, 0].set_title("TOF Depth Map")
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Original image depth map
    im1 = axes[0, 1].imshow(img_depth)
    axes[0, 1].set_title("Image-to-Depth Map (Original)")
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Aligned image depth map
    im2 = axes[1, 0].imshow(aligned_depth)
    axes[1, 0].set_title("Image-to-Depth Map (Aligned)")
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Difference where TOF is valid
    diff = np.zeros_like(tof_depth)
    diff[tof_mask > 0] = np.abs(tof_depth[tof_mask > 0] - aligned_depth[tof_mask > 0])
    im3 = axes[1, 1].imshow(diff)
    axes[1, 1].set_title("Absolute Difference (where TOF is valid)")
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()