"""
Eyeglass Detector

Main class for detecting eyeglasses in images and video streams.
Uses MTCNN for face detection and edge density analysis for glasses detection.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from mtcnn.mtcnn import MTCNN

from .face_alignment import extract_eye_positions, align_face
from .utils import compute_edge_map, calculate_edge_density, extract_glasses_regions


class EyeglassDetector:
    """
    Real-time eyeglass detector using classical computer vision techniques.
    
    This detector uses MTCNN for face and keypoint detection, followed by
    edge density analysis in specific facial regions to determine if a
    person is wearing glasses.
    
    Attributes:
        threshold: Score threshold for glasses detection (default: 0.15).
        face_detector: MTCNN instance for face detection.
        
    Example:
        >>> detector = EyeglassDetector(threshold=0.15)
        >>> detector.run_webcam(camera_id=0, show_debug=True)
    """
    
    def __init__(self, threshold: float = 0.15):
        """
        Initialize the eyeglass detector.
        
        Args:
            threshold: Detection score threshold. Higher values require
                      stronger evidence of glasses. Default is 0.15.
        """
        self.threshold = threshold
        self.face_detector = MTCNN()
        
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image using MTCNN.
        
        Args:
            image: Input image in RGB format.
            
        Returns:
            List of face dictionaries containing 'box' and 'keypoints'.
        """
        return self.face_detector.detect_faces(image)
    
    def check_glasses(
        self, 
        grayscale_face: np.ndarray,
        show_debug: bool = False
    ) -> Tuple[bool, float]:
        """
        Check if a face image shows glasses.
        
        Args:
            grayscale_face: Grayscale aligned face image.
            show_debug: If True, display debug windows showing regions.
            
        Returns:
            Tuple of (is_wearing_glasses, detection_score).
        """
        # Compute edge map
        binary_mask = compute_edge_map(grayscale_face)
        
        # Extract regions of interest
        nose_region, left_cheek, right_cheek = extract_glasses_regions(binary_mask)
        cheek_region = np.hstack([left_cheek, right_cheek])
        
        # Calculate edge densities
        density1 = calculate_edge_density(nose_region)
        density2 = calculate_edge_density(cheek_region)
        
        # Combined weighted score
        score = density1 * 0.3 + density2 * 0.7
        
        if show_debug:
            cv2.imshow('Region_1_Nose', nose_region)
            cv2.imshow('Region_2_Cheeks', cheek_region)
            print(f"Detection score: {score:.4f}")
        
        is_wearing_glasses = score > self.threshold
        return is_wearing_glasses, score
    
    def process_frame(
        self, 
        frame: np.ndarray,
        show_debug: bool = False
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame and detect glasses on all faces.
        
        Args:
            frame: Input frame in BGR format (from OpenCV).
            show_debug: If True, display debug windows.
            
        Returns:
            Tuple of (annotated_frame, results) where results is a list
            of dicts containing face info and glasses status.
        """
        # Convert to RGB for MTCNN
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detected_faces = self.detect_faces(rgb_image)
        results = []
        
        for idx, face_data in enumerate(detected_faces):
            bbox = face_data['box']
            landmarks = face_data['keypoints']
            face_x, face_y, face_w, face_h = bbox
            
            # Draw face bounding box
            cv2.rectangle(
                frame, 
                (face_x, face_y), 
                (face_x + face_w, face_y + face_h), 
                (255, 0, 0), 2
            )
            cv2.putText(
                frame, 
                f"Person {idx + 1}", 
                (face_x - 10, face_y - 10),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2
            )
            
            # Extract and draw eye positions
            left_eye, right_eye = extract_eye_positions(landmarks)
            cv2.circle(frame, tuple(left_eye), 4, (255, 255, 0), -1)
            cv2.circle(frame, tuple(right_eye), 4, (255, 255, 0), -1)
            
            # Align face
            aligned_face = align_face(rgb_image, left_eye, right_eye)
            
            if show_debug:
                cv2.imshow(f"Face_{idx + 1}_Aligned", aligned_face)
            
            # Convert to grayscale and check for glasses
            grayscale_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)
            wearing_glasses, score = self.check_glasses(grayscale_face, show_debug)
            
            # Draw result
            status_text = "Wearing Glasses" if wearing_glasses else "No Glasses"
            text_color = (0, 255, 255) if wearing_glasses else (0, 0, 255)
            cv2.putText(
                frame, 
                status_text, 
                (face_x + 120, face_y - 10),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, text_color, 2
            )
            
            results.append({
                'person_id': idx + 1,
                'bbox': bbox,
                'wearing_glasses': wearing_glasses,
                'score': score
            })
            
            if show_debug:
                print(f"Person {idx + 1}: {status_text} (score: {score:.4f})")
        
        return frame, results
    
    def run_webcam(
        self, 
        camera_id: int = 0, 
        show_debug: bool = False,
        window_name: str = "Live Detection"
    ) -> None:
        """
        Run real-time glasses detection on webcam feed.
        
        Args:
            camera_id: Camera device ID (default: 0 for primary webcam).
            show_debug: If True, show debug windows with detection regions.
            window_name: Name for the main display window.
        """
        video_stream = cv2.VideoCapture(camera_id)
        
        if not video_stream.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        print(f"Starting webcam detection (camera {camera_id})...")
        print("Press 'q' to quit.")
        
        try:
            while video_stream.isOpened():
                success, current_frame = video_stream.read()
                if not success:
                    print("Failed to read frame from webcam.")
                    break
                
                # Process frame
                annotated_frame, _ = self.process_frame(current_frame, show_debug)
                
                # Display result
                cv2.imshow(window_name, annotated_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting...")
                    break
                    
        finally:
            video_stream.release()
            cv2.destroyAllWindows()
