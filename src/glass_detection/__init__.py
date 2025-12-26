"""
Glass Detection Package

A lightweight real-time system to detect eyeglasses using classical 
computer vision techniques with MTCNN for face detection.
"""

from .detector import EyeglassDetector

__version__ = "1.0.0"
__all__ = ["EyeglassDetector"]
