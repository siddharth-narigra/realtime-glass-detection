import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

def extract_eye_positions(facial_landmarks):
    left_eye_pos = np.array(facial_landmarks['left_eye'])
    right_eye_pos = np.array(facial_landmarks['right_eye'])
    return left_eye_pos, right_eye_pos

def align_face_image(image, left_eye_coord, right_eye_coord):
    target_width = 256
    target_height = 256
    eye_distance = target_width * 0.5

    center_x = (left_eye_coord[0] + right_eye_coord[0]) * 0.5
    center_y = (left_eye_coord[1] + right_eye_coord[1]) * 0.5
    
    delta_x = right_eye_coord[0] - left_eye_coord[0]
    delta_y = right_eye_coord[1] - left_eye_coord[1]
    distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)
    scaling_factor = eye_distance / distance
    rotation_angle = np.degrees(np.arctan2(delta_y, delta_x))
    
    transformation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, scaling_factor)
    
    offset_x = target_width * 0.5
    offset_y = target_height * 0.5
    transformation_matrix[0, 2] += (offset_x - center_x)
    transformation_matrix[1, 2] += (offset_y - center_y)

    warped_face = cv2.warpAffine(image, transformation_matrix, (target_width, target_height))
    return warped_face

def detect_spectacles(face_img):
    blurred = cv2.GaussianBlur(face_img, (11, 11), 0)
    vertical_edges = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=-1)
    edge_map = cv2.convertScaleAbs(vertical_edges)

    _, binary_mask = cv2.threshold(edge_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_size = len(binary_mask) * 0.5
    region1_x = np.int32(img_size * 6 / 7)
    region1_y = np.int32(img_size * 3 / 4)
    region1_w = np.int32(img_size * 2 / 7)
    region1_h = np.int32(img_size * 2 / 4)
    
    region2_x1 = np.int32(img_size * 1 / 4)
    region2_x2 = np.int32(img_size * 5 / 4)
    region2_w = np.int32(img_size * 1 / 2)
    region2_y = np.int32(img_size * 8 / 7)
    region2_h = np.int32(img_size * 1 / 2)

    area1 = binary_mask[region1_y:region1_y + region1_h, region1_x:region1_x + region1_w]
    area2_left = binary_mask[region2_y:region2_y + region2_h, region2_x1:region2_x1 + region2_w]
    area2_right = binary_mask[region2_y:region2_y + region2_h, region2_x2:region2_x2 + region2_w]
    area2 = np.hstack([area2_left, area2_right])

    density1 = np.sum(area1 / 255) / (area1.shape[0] * area1.shape[1])
    density2 = np.sum(area2 / 255) / (area2.shape[0] * area2.shape[1])
    combined_measure = density1 * 0.3 + density2 * 0.7

    cv2.imshow('Region_1', area1)
    cv2.imshow('Region_2', area2)
    print(f"Detection score: {combined_measure}")

    glasses_present = combined_measure > 0.15
    print(f"Glasses status: {glasses_present}")
    return glasses_present

face_detector = MTCNN()
video_stream = cv2.VideoCapture(0)

while video_stream.isOpened():
    success, current_frame = video_stream.read()
    if not success:
        break

    rgb_image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    detected_faces = face_detector.detect_faces(rgb_image)

    for idx, face_data in enumerate(detected_faces):
        bbox = face_data['box']
        landmarks = face_data['keypoints']
        face_x, face_y, face_w, face_h = bbox

        cv2.rectangle(current_frame, (face_x, face_y), (face_x + face_w, face_y + face_h), (255, 0, 0), 2)
        cv2.putText(current_frame, f"Person {idx + 1}", (face_x - 10, face_y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)

        left_eye_point, right_eye_point = extract_eye_positions(landmarks)
        cv2.circle(current_frame, tuple(left_eye_point), 4, (255, 255, 0), -1)
        cv2.circle(current_frame, tuple(right_eye_point), 4, (255, 255, 0), -1)

        aligned_face_img = align_face_image(rgb_image, left_eye_point, right_eye_point)
        cv2.imshow(f"Face_{idx + 1}_Aligned", aligned_face_img)

        grayscale_face = cv2.cvtColor(aligned_face_img, cv2.COLOR_RGB2GRAY)
        wearing_glasses = detect_spectacles(grayscale_face)
        
        status_text = "Wearing Glasses" if wearing_glasses else "No Glasses"
        text_color = (0, 255, 255) if wearing_glasses else (0, 0, 255)
        cv2.putText(current_frame, status_text, (face_x + 120, face_y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, text_color, 2)

    cv2.imshow("Live Detection", current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.release()
cv2.destroyAllWindows()