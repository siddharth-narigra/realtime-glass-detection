"""
Face Alignment Utilities

Functions for extracting eye positions and aligning faces for consistent analysis.
"""

import cv2
import numpy as np
from typing import Tuple, Dict


def extract_eye_positions(facial_landmarks: Dict[str, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract left and right eye positions from MTCNN facial landmarks.
    
    Args:
        facial_landmarks: Dictionary containing 'left_eye' and 'right_eye' coordinates.
        
    Returns:
        Tuple of (left_eye_position, right_eye_position) as numpy arrays.
    """
    left_eye_pos = np.array(facial_landmarks['left_eye'])
    right_eye_pos = np.array(facial_landmarks['right_eye'])
    return left_eye_pos, right_eye_pos


def align_face(
    image: np.ndarray, 
    left_eye_coord: np.ndarray, 
    right_eye_coord: np.ndarray,
    target_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    Align a face image so that the eyes are level and centered.
    
    This function rotates and scales the image based on eye positions,
    making further analysis (like glasses detection) more accurate.
    
    Args:
        image: Input image (BGR or RGB format).
        left_eye_coord: Left eye (x, y) coordinates.
        right_eye_coord: Right eye (x, y) coordinates.
        target_size: Output image size as (width, height).
        
    Returns:
        Aligned face image of the specified target size.
    """
    target_width, target_height = target_size
    eye_distance = target_width * 0.5

    # Calculate center point between eyes
    center_x = (left_eye_coord[0] + right_eye_coord[0]) * 0.5
    center_y = (left_eye_coord[1] + right_eye_coord[1]) * 0.5
    
    # Calculate rotation and scaling parameters
    delta_x = right_eye_coord[0] - left_eye_coord[0]
    delta_y = right_eye_coord[1] - left_eye_coord[1]
    distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)
    scaling_factor = eye_distance / distance
    rotation_angle = np.degrees(np.arctan2(delta_y, delta_x))
    
    # Create transformation matrix
    transformation_matrix = cv2.getRotationMatrix2D(
        (center_x, center_y), 
        rotation_angle, 
        scaling_factor
    )
    
    # Adjust for centering
    offset_x = target_width * 0.5
    offset_y = target_height * 0.5
    transformation_matrix[0, 2] += (offset_x - center_x)
    transformation_matrix[1, 2] += (offset_y - center_y)

    # Apply transformation
    aligned_face = cv2.warpAffine(
        image, 
        transformation_matrix, 
        (target_width, target_height)
    )
    
    return aligned_face
