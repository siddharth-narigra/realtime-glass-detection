"""
Image Processing Utilities

Edge detection and region analysis helpers for glasses detection.
"""

import cv2
import numpy as np
from typing import Tuple


def compute_edge_map(image: np.ndarray, blur_kernel: Tuple[int, int] = (11, 11)) -> np.ndarray:
    """
    Compute edge map using Gaussian blur and Sobel filter.
    
    Args:
        image: Grayscale input image.
        blur_kernel: Kernel size for Gaussian blur.
        
    Returns:
        Binary edge map after Otsu thresholding.
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, blur_kernel, 0)
    
    # Detect vertical edges using Sobel filter
    vertical_edges = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=-1)
    edge_map = cv2.convertScaleAbs(vertical_edges)
    
    # Apply Otsu's thresholding for binary mask
    _, binary_mask = cv2.threshold(
        edge_map, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return binary_mask


def calculate_edge_density(binary_mask: np.ndarray) -> float:
    """
    Calculate the density of white pixels in a binary mask.
    
    Args:
        binary_mask: Binary image (0 or 255 values).
        
    Returns:
        Density value between 0.0 and 1.0.
    """
    if binary_mask.size == 0:
        return 0.0
    
    white_pixels = np.sum(binary_mask / 255)
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    
    return white_pixels / total_pixels


def extract_glasses_regions(
    binary_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the regions of interest for glasses detection.
    
    The detection uses two key regions:
    1. Area above the nose (frame edges)
    2. Cheek areas (lens shadows)
    
    Args:
        binary_mask: Binary edge map of the aligned face.
        
    Returns:
        Tuple of (nose_region, left_cheek_region, right_cheek_region).
    """
    img_size = len(binary_mask) * 0.5
    
    # Region 1: Above nose (frame edges)
    region1_x = np.int32(img_size * 6 / 7)
    region1_y = np.int32(img_size * 3 / 4)
    region1_w = np.int32(img_size * 2 / 7)
    region1_h = np.int32(img_size * 2 / 4)
    
    # Region 2: Cheek areas (lens shadows)
    region2_x1 = np.int32(img_size * 1 / 4)
    region2_x2 = np.int32(img_size * 5 / 4)
    region2_w = np.int32(img_size * 1 / 2)
    region2_y = np.int32(img_size * 8 / 7)
    region2_h = np.int32(img_size * 1 / 2)

    # Extract regions
    nose_region = binary_mask[region1_y:region1_y + region1_h, region1_x:region1_x + region1_w]
    left_cheek = binary_mask[region2_y:region2_y + region2_h, region2_x1:region2_x1 + region2_w]
    right_cheek = binary_mask[region2_y:region2_y + region2_h, region2_x2:region2_x2 + region2_w]
    
    return nose_region, left_cheek, right_cheek
