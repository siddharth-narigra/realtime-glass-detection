# Real-Time Eyeglass Detection System

## Abstract

This project presents a real-time eyeglass detection system utilizing computer vision techniques. The model integrates face detection through the Multi-task Cascaded Convolutional Networks (MTCNN), alignment based on eye keypoints, and feature extraction using edge-based analysis. The goal is to detect eyeglasses without relying on pre-trained deep models by instead leveraging traditional image processing techniques. Gaussian blur and the Sobel operator were applied for edge detection, followed by Otsu's thresholding to identify prominent facial edges. The system then examines specific Regions of Interest (ROIs) on the face to calculate edge density, which serves as a proxy for the presence of glasses. Experiments conducted in real-time using a webcam stream demonstrated promising accuracy with an adjustable detection threshold. The proposed approach is lightweight and explainable compared to deep learning models. Limitations include sensitivity to lighting and head pose, but the method shows strong potential for applications in accessibility tools and user authentication systems. This paper outlines the methodology, implementation, evaluation, and potential areas for future enhancement.

## Introduction

Eyeglass detection is a crucial subtask within the broader field of computer vision, particularly in facial analysis systems. Its significance is underscored by its impact on various applications such as enhancing facial recognition accuracy, improving accessibility for visually impaired users, enabling immersive experiences in augmented reality (AR), and aiding in certain medical diagnostic procedures.

Modern approaches typically rely on deep learning architectures such as Convolutional Neural Networks (CNNs), which often demand high computational resources, large annotated datasets, and long training times. This project offers an alternative, emphasizing interpretability, efficiency, and real-time applicability by leveraging classical computer vision techniques—such as edge detection and facial landmark-based geometric reasoning—combined with modern face detection tools like MTCNN.

### Problem Statement

Despite advancements in machine learning, achieving fast and reliable eyeglass detection without heavy computational overhead remains a challenge. Many existing solutions are dependent on GPU-accelerated inference pipelines, which are not feasible in all deployment scenarios. This project addresses how to detect eyeglasses on faces accurately and in real time using simple, transparent, and explainable image processing techniques—bypassing the need for complex deep neural networks.

### Objective

This project aims to achieve the following:

*   To detect eyeglasses on human faces using a combination of facial landmark localization and edge-density analysis in specific facial regions.
*   To implement a real-time eyeglass detection pipeline using live webcam feed, powered by Multi-task Cascaded Convolutional Networks (MTCNN) for face detection and alignment.
*   To provide a modular baseline system that is lightweight, interpretable, and can be easily extended or upgraded with deep learning models in the future if needed.

## Technical Stack

*   **Python 3.8**
*   **OpenCV 4.6:** For image processing tasks.
*   **TensorFlow 2.12:** To support the MTCNN face detection model.
*   **MTCNN:** For face detection and keypoint extraction.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/siddharth-narigra/realtime-glass-detection.git
    cd realtime-glass-detection
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install opencv-python tensorflow mtcnn
    ```
    *(Note: You might need to install `dlib` separately for `shape_predictor_5_face_landmarks.dat` if it's used directly in the code, though the description implies MTCNN handles landmarks.)*

4.  **Run the eyeglass detection script:**
    ```bash
    python eyeglass_detector.py
    ```
    This will likely open a webcam feed and display the real-time eyeglass detection.

## Project Structure

```
project/
├── docs/
│   ├── group7_projectreport.docx
│   └── group7.pptx
├── realtime-glasses-detection/
│   ├── data/
│   │   └── shape_predictor_5_face_landmarks.dat
│   ├── img/
│   │   ├── example_1.gif
│   │   └── schematic.PNG
│   ├── eyeglass_detector.py
│   └── extra.py
└── .gitignore
```

## Screenshots

Here are some visual examples of the system in action:

![Example GIF](realtime-glasses-detection/img/example_1.gif)

*A schematic diagram illustrating the system's workflow:*

![System Schematic](realtime-glasses-detection/img/schematic.PNG)

## Methodology

This project follows a real-time image processing pipeline involving:

1.  **Face Detection and Keypoint Extraction:** Performed with MTCNN, which predicts both face bounding boxes and five facial keypoints: the left eye, right eye, nose tip, and two mouth corners.
2.  **Face Alignment:** Using the detected eye coordinates, each face is rotated and scaled via an affine transformation (implemented with OpenCV’s `getRotationMatrix2D` and `warpAffine` functions) to normalize interocular distance and orientation.
3.  **Edge Detection:** The aligned face region undergoes Gaussian smoothing to suppress noise, followed by Sobel filtering to compute vertical gradients, which highlight eyeglass frame edges.
4.  **Thresholding:** The gradient magnitude map is binarized using Otsu’s method, which automatically selects the optimal threshold by minimizing intra-class variance between foreground (edges) and background.
5.  **Eyeglass Classification (ROI Analysis):** Two Regions of Interest (ROIs)—one above the nose bridge (targeting frame edges) and another over the lower cheeks (capturing lens shadows)—are extracted for edge density analysis. A composite eyeglass score is computed as `0.3 × EdgeDensity₁ + 0.7 × EdgeDensity₂`, with an empirically determined threshold (~0.15) deciding the presence of glasses.

### Algorithmic Concept

*   **Face Detection and Keypoint Extraction:** Input: RGB video frame. Method: Apply MTCNN to detect face bounding boxes and extract five keypoints per face (left eye, right eye, nose tip, mouth corners).
*   **Face Alignment:** Based on the geometric midpoint of the eyes, the image is rotated and scaled using affine transformation. The rotation angle `θ` is calculated as `arctan((y_R - y_L) / (x_R - x_L))`, where `(x_L, y_L)` and `(x_R, y_R)` are the coordinates of the left and right eyes, respectively.
*   **Edge Detection:** Gaussian Blur is applied to reduce noise, followed by Sobel Filtering to compute the vertical gradient (`S_y = ∂I/∂y`) by convolving with the Sobel–Feldman vertical kernel, emphasizing frame edges.
*   **Thresholding:** Otsu’s method is applied to the gradient magnitude image to binarize strong edge responses, selecting the threshold that minimizes intra-class variance.
*   **Eyeglass Classification (ROI Analysis):** Two ROIs are examined: one over the nose bridge (ROI 1, targeting eyeglass frames) and another over the lower cheeks (ROI 2, capturing lens shadows/edges). Edge density is computed as the ratio of edge pixels to total pixels in each ROI. A combined score is calculated as `Score = 0.3 * EdgeDensity_ROI1 + 0.7 * EdgeDensity_ROI2`. Glasses are classified as present if `Score ≥ 0.15`.

## Real-World System Implementation

The system was implemented on a personal computer equipped with an Intel Core i7-9700K CPU operating at 3.6 GHz, 16 GB of RAM, and running Windows 10. A Logitech C920 Pro HD webcam, capable of capturing 1080p video at 30 frames per second, served as the input device. The software environment comprised Python 3.8, OpenCV 4.6, and TensorFlow 2.12.

### System Workflow

The real-time pipeline operates as follows: continuous capture of video frames from the webcam; utilization of the MTCNN model to detect faces and extract key facial landmarks; calculation of the rotation angle and scaling factor to align the face horizontally; conversion of the aligned face image to grayscale, followed by Gaussian blurring; application of the Sobel operator to highlight potential eyeglass frames; use of Otsu's method to binarize the edge-detected image; definition of specific facial regions to compute edge density metrics; calculation of a weighted sum of edge densities to determine the presence of eyeglasses, with a threshold value of 0.15; and overlay of the classification result on the original video frame for real-time feedback.

### Performance Evaluation

The system achieves a processing speed of approximately 28–32 frames per second when detecting a single face and maintains a rate of 22–26 frames per second for up to four concurrent face detections, indicating its capability for real-time operation without the need for GPU acceleration.

### Environmental Considerations

Testing under various lighting conditions revealed that uneven illumination could lead to false positives, particularly due to shadows or hair strands within the ROIs. To mitigate these effects, techniques such as Contrast Limited Adaptive Histogram Equalization (CLAHE) can be integrated to enhance contrast and improve detection accuracy under challenging lighting scenarios.

## Results and Discussion

Our real-time eyeglass detection pipeline, evaluated on live webcam feeds under varied head poses, skin tones, and frame styles, achieved peak performance at an edge density threshold of 0.15, yielding 84.2% accuracy with 0.87 precision and 0.82 recall. Lower thresholds increased false positives (higher recall, lower precision), while higher thresholds missed more eyeglasses (higher precision, lower recall), illustrating the classic sensitivity-specificity trade-off central to ROC analysis.

### Dataset and Evaluation Metrics

This system was tested exclusively on real-time webcam input, consisting of volunteers with diverse head orientations, lighting conditions, skin tones, and eyeglass frame geometries. No pre-labeled corpus was used; instead, evaluation relied on manual verification of edge responses and frame-wise presence/absence labeling.

Performance was measured using four standard binary classification metrics: accuracy, precision, recall, and F1 score.

*   **Precision** = TP / (TP + FP)
*   **Recall** = TP / (TP + FN)
*   **F1-Score** = 2 * (Precision * Recall) / (Precision + Recall)

*(TP = True Positives, FP = False Positives, FN = False Negatives)*

### Performance Table (Varying Thresholds)

| Threshold | Accuracy (%) | Precision | Recall | F1-score |
| :-------- | :----------- | :-------- | :----- | :------- |
| 0.10      | 78.5         | 0.81      | 0.75   | 0.78     |
| **0.15**  | **84.2**     | **0.87**  | **0.82** | **0.84** |
| 0.20      | 71.0         | 0.73      | 0.69   | 0.71     |

At 0.15, the system achieves the highest accuracy (84.2%) and balanced precision/recall, reflecting optimal threshold selection via intra-class variance minimization.

### Group Portrait Evaluation: Multi-Face Eyeglass Detection

To assess the scalability and robustness of our eyeglass detection system, we extended our evaluation to group portrait images containing multiple individuals with diverse facial features, orientations, and eyewear types. The existing pipeline was applied to each detected face within the group images, involving face detection and landmark extraction, individual face alignment, edge detection and ROI analysis, and eyeglass classification.

## Discussion and Future Directions

These results demonstrate that a classical edge-based approach—when combined with precise facial alignment—can rival lightweight deep models in real-time eyeglass detection scenarios. Key limitations include sensitivity to variable illumination and partial occlusions (e.g., hair or hats), suggesting improvements via adaptive thresholding or illumination normalization preprocessing.

Future directions include implementing temporal smoothing across consecutive frames to mitigate label flicker and improve user experience in video streams. Additionally, enriching this pipeline with a shallow neural classifier trained on extracted edge density features may further boost accuracy while preserving interpretability—bridging classical and deep learning paradigms.

## Credits and Acknowledgments

This project builds upon foundational work in computer vision and leverages open-source tools and research. We acknowledge the contributions of the authors of the MTCNN framework, OpenCV, and the various researchers whose work on edge detection, image processing, and facial analysis has paved the way for this system.

## References

*   K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, “Joint face detection and alignment using multitask cascaded convolutional networks,” IEEE Signal Processing Letters, 2016.
*   J. Redmon and A. Farhadi, “YOLOv3: An incremental improvement,” arXiv preprint, 2018.
*   D. Marr, “Vision: A Computational Investigation into the Human Representation and Processing of Visual Information,” MIT Press, 1982.
*   OpenCV Documentation, [https://docs.opencv.org](https://docs.opencv.org)
*   S. Suzuki and K. Be, “Topological structural analysis of digitized binary images by border following,” Computer Vision, Graphics, and Image Processing, 1985.
*   A. Otsu, “A threshold selection method from gray-level histograms,” IEEE Trans. Sys., Man., Cyber., 1979.
*   R. Gonzalez and R. Woods, “Digital Image Processing,” 4th ed., Pearson, 2018.
*   MTCNN GitHub Repo, [https://github.com/ipazc/mtcnn](https://github.com/ipazc/mtcnn)
*   Y. LeCun et al., “Gradient-based learning applied to document recognition,” IEEE, 1998.
*   Z. Lin, J. Lu, “Real-time Glasses Detection with Convolutional Networks,” Pattern Recognition, 2021.
*   A. K. Jain, “Fundamentals of Digital Image Processing,” Prentice Hall, 1989.
*   B. Horn, “Robot Vision,” MIT Press, 1986.
*   H. Bay, A. Ess, T. Tuytelaars, and L. Van Gool, “Speeded-Up Robust Features (SURF),” Computer Vision and Image Understanding, 2008.
*   X. Zhu and D. Ramanan, “Face detection, pose estimation, and landmark localization in the wild,” CVPR, 2012.
*   R. Szeliski, Computer Vision: Algorithms and Applications, Springer, 2010.
*   D. Ziou and S. Tabbone, “Edge Detection Techniques – An Overview,” Pattern Recognition and Image Analysis, vol. 8, no. 4, pp. 537–559, 1998.
*   J. S. Owotogbe, T. S. Ibiyemi, and B. A. Adu, “Edge Detection Techniques on Digital Images – A Review,” International Journal of Innovative Science and Research Technology, vol. 4, no. 11, pp. 234–239, 2019.
*   C. Ghosh et al., “Different Edge Detection Techniques: A Review,” in Proceedings of the International Conference on Electronic Systems and Intelligent Computing (ESIC 2020), Lecture Notes in Electrical Engineering, vol. 686, pp. 893–902, 2020.
*   R. V. Ramana, T. V. Rathnam, and A. Sankar Reddy, “A Review on Edge Detection Algorithms in Digital Image Processing Applications,” International Journal on Recent and Innovation Trends in Computing and Communication, vol. 5, no. 10, pp. 69–73, 2017.
*   K. Bala Krishnan, S. P. Ranga, and N. Guptha, “A Survey on Different Edge Detection Techniques for Image Segmentation,” Indian Journal of Science and Technology, vol. 10, no. 4, pp. 1–8, 2017.


