# Real-Time Eyeglass Detection System

## Overview

This project is a lightweight real-time system to detect eyeglasses using classical computer vision techniques. Instead of relying on large deep learning models, it uses edge detection, facial landmarks, and simple mathematical heuristics to identify the presence of glasses in a webcam feed.

It leverages:

* **MTCNN** for face and keypoint detection
* **Edge detection + alignment** for analyzing key facial regions
* **ROI-based logic** to estimate edge density, which indicates glasses

The system runs efficiently on CPUs and is ideal for explainable, fast, and resource-friendly deployment.

## Objectives

* Detect eyeglasses using face landmarks and edge density
* Run the detection pipeline in real time using webcam input
* Create an interpretable and modular system with easy extensibility

## Tech Stack

* **Python 3.8**
* **OpenCV 4.6** – image processing
* **TensorFlow 2.12** – for MTCNN model support
* **MTCNN** – for face detection and keypoint localization

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/siddharth-narigra/realtime-glass-detection.git
   cd realtime-glass-detection
   ```

2. **Set up a virtual environment (recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install opencv-python tensorflow mtcnn
   ```

4. **Run the script:**

   ```bash
   python eyeglass_detector.py
   ```

## Project Structure

```
realtime-glass-detection/
├── README.md
└── realtime-glasses-detection/
    ├── data/
    │   └── shape_predictor_5_face_landmarks.dat
    └── eyeglass_detector.py
```

## How It Works

1. **Face Detection & Landmarks**
   MTCNN finds faces and extracts key points like eyes and mouth.

2. **Face Alignment**
   The face is rotated so the eyes are level, making further analysis more accurate.

3. **Edge Detection**
   Gaussian blur reduces noise, and Sobel filters highlight edges (especially glasses frames).

4. **Thresholding**
   Otsu’s method finds the best threshold to isolate strong edges from noise.

5. **Eyeglass Detection**
   The system looks at two small areas on the face:

   * Above the nose (frame edges)
   * Cheek area (lens shadows)

   It calculates edge density in both and uses a weighted score:

   ```
   Score = 0.3 × EdgeDensity1 + 0.7 × EdgeDensity2
   ```

   If the score is above `0.15`, glasses are detected.

## Real-Time System Setup

* **System:** Intel i7-9700K, 16GB RAM, Windows 10
* **Webcam:** Logitech C920 HD
* **Performance:**

  * \~28–32 FPS (1 face)
  * \~22–26 FPS (up to 4 faces)

The system runs efficiently without requiring a GPU.

## Screenshots

Example of detection in real-time:

![Example GIF](realtime-glasses-detection/img/example_1.gif)

System flow diagram:

![System Schematic](realtime-glasses-detection/img/schematic.PNG)

## Results

Tested on live webcam feeds with different faces, poses, and lighting:

| Threshold | Accuracy (%) | Precision |   Recall | F1 Score |
| --------- | -----------: | --------: | -------: | -------: |
| 0.10      |         78.5 |      0.81 |     0.75 |     0.78 |
| **0.15**  |     **84.2** |  **0.87** | **0.82** | **0.84** |
| 0.20      |         71.0 |      0.73 |     0.69 |     0.71 |

**Best performance** was at a threshold of 0.15 with balanced precision and recall.

### Metrics

* **Accuracy**
* **Precision** = TP / (TP + FP)
* **Recall** = TP / (TP + FN)
* **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)

## Environmental Notes

Performance was consistent, but lighting variations sometimes caused false positives. Adding methods like CLAHE (adaptive histogram equalization) can further improve robustness under challenging conditions.

## Future Improvements

* Add CLAHE to improve contrast in poor lighting
* Explore hybrid methods with light neural networks
* Include datasets for benchmarking and reproducibility
