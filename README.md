# Computer Vision and Digital Image Processing (CCA 3)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13.0-5C3EE8.svg)](https://opencv.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8n-00FFFF.svg)](https://docs.ultralytics.com/)
[![scikit-image](https://img.shields.io/badge/scikit--image-0.26.0-orange.svg)](https://scikit-image.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.3.4-013243.svg)](https://numpy.org/)

---

## 📌 Project Overview

This repository contains the complete experimental implementation, comparative evaluations, and analytical conclusions for **Continuous Comprehensive Assessment 3 (CCA 3): Computer Vision and Digital Image Processing (Units 3, 4, and 5)**.

The project demonstrates practical implementations across four fundamental computer vision domains:
1. **Image Restoration & Morphological Segmentation**: Noise modeling (Gaussian vs. Salt-and-Pepper), spatial restoration filtering (Gaussian vs. Median), quantitative benchmarking ($\text{PSNR}$ & $\text{SSIM}$), morphological operations, Canny edge detection, and marker-controlled Watershed segmentation.
2. **Feature Extraction & Image Matching**: Dense gradient descriptors (HOG), scale-invariant keypoint detection and 128-D descriptor generation (SIFT), KNN descriptor matching with Lowe's ratio test, and geometric rotation invariance testing ($45^\circ$ affine rotation).
3. **Object Recognition (Viola–Jones vs. YOLOv8)**: Real-time Haar cascade facial detection vs. state-of-the-art deep convolutional single-shot object detection (YOLOv8 Nano), including bounding box regression, multi-class confidence analysis, and detector failure-case diagnostics.
4. **Video Motion Estimation**: Temporal frame differencing, adaptive Gaussian Mixture Model (GMM / MOG2) background subtraction, and pyramidal Lucas–Kanade sparse optical flow tracking with motion vector visualization.

---

## 📂 Repository Structure

```text
CV-Assignment/
├── CCA3_Computer_Vision.ipynb          # Master Jupyter Notebook with all 4 tasks & visualizations
├── README.md                           # Comprehensive project documentation
├── haarcascade_frontalface_default.xml # Pretrained Haar Cascade model for frontal face detection
├── yolov8n.pt                          # Pretrained Ultralytics YOLOv8 Nano weights (~6.5 MB)
├── data/
│   ├── images/
│   │   ├── original.jpg                # Primary input test image (1600x720, 3-channel RGB)
│   │   └── second.jpg                  # Secondary test image for feature matching experiments
│   └── videos/
│       ├── v1.mp4                      # Primary video sequence for motion estimation (60 FPS, 1725 frames)
│       └── 1.  Alpha Community Update_.mp4 # Supplementary test video
└── outputs/
    └── task1_image_restoration/        # Generated visual and quantitative outputs
        ├── original.png
        ├── gaussian_noise.png
        ├── salt_pepper_noise.png
        ├── gaussian_filtered.png
        ├── gaussian_filtered_salt_pepper.png
        ├── median_filtered_gaussian.png
        ├── median_filtered_salt_pepper.png
        ├── binary_image.png
        ├── erosion.png
        ├── dilation.png
        ├── opening.png
        ├── closing.png
        ├── canny_edges.png
        ├── watershed_segmentation.png
        ├── restoration_metrics.csv     # Quantitative metrics (PSNR & SSIM)
        └── restoration_comparison.csv  # Extended restoration comparisons
```

---

## ⚙️ Environment Setup & Installation

### 1. Prerequisites
- **Python**: 3.10 or higher
- **Virtual Environment Tool**: `venv` or `conda`

### 2. Setup Virtual Environment
Run the following commands in PowerShell or your terminal from the project root:

```powershell
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

### 3. Install Dependencies
```powershell
pip install opencv-python numpy matplotlib scikit-image ultralytics pandas pillow ipykernel
```

### 4. Verified Dependency Versions
| Package | Verified Version | Purpose |
| :--- | :--- | :--- |
| **Python** | `3.10+` | Core execution environment |
| **OpenCV (`cv2`)** | `4.13.0` | Spatial filtering, morphology, SIFT, Haar cascades, optical flow |
| **NumPy** | `2.3.4` | Matrix computations, noise modeling, array slicing |
| **scikit-image** | `0.26.0` | Image metrics (`psnr`, `ssim`), HOG feature extraction |
| **Ultralytics** | `8.x` | Pretrained YOLOv8 Nano object detection |
| **Matplotlib** | `3.8+` | Figure layouts, image rendering, subplots |
| **Pandas** | `2.x` | Quantitative metric tables and formatting |

---

## 🔬 Detailed Task Implementations & Technical Results

---

### Task 1: Image Restoration and Morphological Segmentation

#### Part 1A: Noise Modeling and Spatial Filtering
Two stochastic noise profiles were simulated on `data/images/original.jpg`:
- **Additive Gaussian Noise**: $\mathcal{N}(\mu=0, \sigma=20)$, zero-mean independent Gaussian perturbation applied across all channels and clipped to $[0, 255]$.
- **Impulse (Salt-and-Pepper) Noise**: Independent random binary corruption replacing pixels with extreme values ($255$ for salt, $0$ for pepper) with noise ratio $p = 0.05$.

Restoration filters evaluated:
1. **Gaussian Filter**: Linear smoothing filter utilizing a $5 \times 5$ spatial kernel with standard deviation $\sigma = 1.0$.
2. **Median Filter**: Non-linear order-statistic filter replacing each pixel with the median value within a $5 \times 5$ neighborhood.

#### Quantitative Metric Evaluation:
Restoration fidelity was measured using **Peak Signal-to-Noise Ratio (PSNR)** and **Structural Similarity Index (SSIM)**:
$$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right)$$

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

| Image Condition / Filter | PSNR (dB) | SSIM | Observation & Analysis |
| :--- | :---: | :---: | :--- |
| **Gaussian Noisy (Degraded)** | `22.20` | `0.3795` | Low signal-to-noise ratio; fine textural detail corrupted. |
| **Gaussian Filtered (on Gaussian Noise)** | **`30.32`** | **`0.7857`** | **Optimal for Gaussian noise**: Smooths normal variations effectively; minor edge blur. |
| **Median Filtered (on Gaussian Noise)** | `28.85` | `0.7551` | Moderately effective, but smooth gradients are slightly stepped. |
| **Salt-and-Pepper Noisy (Degraded)** | `19.37` | `0.4105` | Severe structural destruction from extreme impulse outliers. |
| **Gaussian Filtered (on Salt-and-Pepper)** | `28.35` | `0.7172` | **Sub-optimal**: Blurs impulses into surrounding pixels without eliminating them. |
| **Median Filtered (on Salt-and-Pepper)** | **`30.53`** | **`0.8790`** | **Optimal for Salt-and-Pepper**: Outlier rejection completely removes impulses with sharp edge preservation. |

---

#### Part 1B: Morphological Processing and Segmentation
1. **Binary Thresholding**: Grayscale conversion followed by fixed thresholding ($T = 127, \text{maxVal} = 255$) separating foreground regions from background.
2. **Morphological Operations** ($5 \times 5$ rectangular structuring element $\mathbf{B}$):
   - **Erosion** ($A \ominus B$): Strips boundary layers, eliminates isolated foreground artifacts.
   - **Dilation** ($A \oplus B$): Expands foreground boundaries, bridges narrow gaps.
   - **Opening** ($A \circ B = (A \ominus B) \oplus B$): Smooths contours, eliminates small foreground noise without altering overall area.
   - **Closing** ($A \bullet B = (A \oplus B) \ominus B$): Fills holes and micro-cavities while preserving object shapes.
3. **Canny Edge Detection**:
   - Multi-stage boundary extraction using Gaussian pre-smoothing, Sobel gradient magnitude/orientation, non-maximum suppression (NMS), and dual-threshold hysteresis ($T_{\text{lower}} = 100, T_{\text{upper}} = 200$). Accurately traces dominant structural contours.
4. **Marker-Controlled Watershed Segmentation**:
   - Preprocessing with Otsu's thresholding (`THRESH_BINARY_INV + THRESH_OTSU`).
   - Morphological opening ($3 \times 3$ kernel, 2 iterations) for noise cleanup.
   - Sure background calculation via morphological dilation (3 iterations).
   - Sure foreground calculation via Euclidean Distance Transform (`DIST_L2`, mask size 5) thresholded at $40\%$ of maximum distance:
     $$\text{dist\_transform} = \min_{y \in \text{bg}} \|x - y\|_2$$
   - Unknown boundary region extraction: $\text{unknown} = \text{sure\_bg} - \text{sure\_fg}$.
   - Connected component labeling and topological flooding with dam-building (`markers == -1` highlighted in red `[255, 0, 0]`).

---

### Task 2: Feature Extraction and Image Matching

#### 2.1 Histogram of Oriented Gradients (HOG)
- **Parameters**: 9 orientation bins ($0^\circ - 180^\circ$ unsigned), $8 \times 8$ pixel cells, $2 \times 2$ cells per block with L2-Hys block normalization.
- **Output**: 1D feature vector of dimension **$637,596$**, capturing local intensity gradients and edge directionality across spatial cells.

#### 2.2 SIFT Keypoint Detection & Descriptors
- **Algorithm**: Scale-Invariant Feature Transform (Lowe, 2004).
- **Keypoints Identified**: **$6,198$** keypoints localized across multi-scale Difference-of-Gaussians (DoG) octaves.
- **Descriptor Array**: Shape **$(6198, 128)$**, describing gradient orientation histograms invariant to scale, orientation, and illumination changes.

#### 2.3 Feature Matching via k-NN & Lowe's Ratio Test
- **Matching Metric**: Euclidean distance in 128-D descriptor space via `cv2.BFMatcher`.
- **Ratio Test Formulation**:
  $$\frac{\|D_{1, i} - D_{2, j}^{(1)}\|_2}{\|D_{1, i} - D_{2, j}^{(2)}\|_2} < 0.6$$
- Only match pairs with distance ratio $< 0.6$ against the second-nearest neighbor were retained, eliminating ambiguous background matches. The top 50 strongest matches were isolated and visualized.

#### 2.4 Transformation & Invariance Evaluation
- **Experiment**: The original image was rotated by **$45^\circ$** around its centroid using an affine rotation matrix:
  $$M = \begin{bmatrix} \alpha & \beta & (1-\alpha)\cdot c_x - \beta\cdot c_y \\ -\beta & \alpha & \beta\cdot c_x + (1-\alpha)\cdot c_y \end{bmatrix}, \quad \alpha = \cos(45^\circ), \beta = \sin(45^\circ)$$
- **Results**:
  - Original keypoints: $6,198$
  - Rotated image keypoints: $3,753$
  - Successful matches after rotation: Top 50 robust correspondences correctly identified across dominant canonical orientations.
- **Conclusion**: Demonstrates SIFT's intrinsic scale and rotational invariance. Keypoint drop is primarily attributed to border clipping and bilinear interpolation resamplings during affine warping.

---

### Task 3: Object Recognition (Viola–Jones vs. YOLOv8)

#### 3.1 Viola–Jones (Haar Cascade Classifier)
- **Model**: `haarcascade_frontalface_default.xml`
- **Detection Mechanism**: Integral image computation, Haar-like rectangular feature evaluation, AdaBoost cascaded rejection filters.
- **Tuning Parameters**: `scaleFactor = 1.1`, `minNeighbors = 5`, `minSize = (30, 30)`.
- **Results**: Detected **2 faces** in `data/images/original.jpg`.

#### 3.2 YOLOv8 Nano Object Detection
- **Model Architecture**: Ultralytics YOLOv8n (`yolov8n.pt`, PyTorch single-stage anchor-free CNN).
- **Execution Config**: Resolution `imgsz = 320`, device `cpu`, confidence threshold `conf = 0.25`.
- **Results**:
  - Class: `person`
  - Confidence: **`92.15%`**
  - Bounding Box: `[x1=66, y1=580, x2=560, y2=1598]`

#### 3.3 Comparative Evaluation

| Evaluation Metric | Viola–Jones (Haar Cascade) | Ultralytics YOLOv8 Nano |
| :--- | :--- | :--- |
| **Model Paradigm** | Traditional feature engineering (Adaboost + Haar) | Deep Learning (Anchor-free CNN) |
| **Inference Speed** | Very fast on CPU; minimal computation required | Highly optimized (~15–30ms on modern CPU/GPU) |
| **Target Flexibility** | Single dedicated class per cascade (e.g., frontal faces) | Multi-class detection (80 COCO classes simultaneously) |
| **Output Representation** | Bounding box coordinates | Bounding boxes + Class label + Confidence probability |
| **Robustness to Pose/Angle**| Fragile to in-plane/out-of-plane rotation, tilt, occlusion | Robust across extreme poses, viewpoints, and scales |
| **False Positive Rate** | Higher (vulnerable to high-contrast textured backgrounds) | Low (leverages deep contextual representations) |

#### 3.4 Failure Analysis & Edge Cases:
- **Haar Cascade**: Produced 2 detections — one corresponding to the actual human face and a secondary false positive triggered by clothing texture and shadow contrasts satisfying the two-rectangle Haar intensity differential.
- **YOLOv8**: Confidently identified the entire `person` entity ($92.15\%$) without detecting individual facial sub-components (as face is not a discrete COCO class). Extreme downsampling (`imgsz=320`) can cause smaller distant objects to fall below the detection threshold.

---

### Task 4: Video Motion Estimation and Analysis

Video Input: `data/videos/v1.mp4` (Duration: $28.74$s, Total Frames: $1,725$, Frame Rate: $60.01$ FPS).

#### 4.1 Frame Differencing
- **Methodology**: Computes absolute pixel-wise difference between frames separated by $\Delta t = 10$ frames:
  $$D(x, y) = |I_{t+\Delta t}(x, y) - I_t(x, y)|$$
  Followed by binary thresholding: $M(x, y) = 255 \text{ if } D(x, y) > 25 \text{ else } 0$.
- **Strengths**: Computationally trivial, near-zero latency, immediate response to motion.
- **Weaknesses**: Leaves "hollow" interiors in uniformly textured objects; exhibits double-contour ghosting; hypersensitive to sensor noise and camera vibration.

#### 4.2 GMM Background Subtraction (MOG2)
- **Methodology**: Models the history of each pixel over $500$ frames as a mixture of $K$ Gaussians (`cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)`):
  $$P(X_t) = \sum_{i=1}^K \omega_{i, t} \cdot \eta(X_t, \mu_{i, t}, \Sigma_{i, t})$$
- **Strengths**: Adapts dynamically to gradual illumination changes and swaying background elements; provides solid foreground masks.
- **Weaknesses**: Requires a learning/warm-up period; stationary foreground objects eventually become absorbed into the background model.

#### 4.3 Lucas–Kanade Sparse Optical Flow
- **Feature Detection**: Shi–Tomasi corner detector (`goodFeaturesToTrack` with `maxCorners=100`, `qualityLevel=0.3`, `minDistance=7`, `blockSize=7`).
- **Flow Estimation**: Iterative Lucas–Kanade with image pyramids (`calcOpticalFlowPyrLK` with `winSize=(15, 15)`, `maxLevel=2`, termination criteria $\epsilon=0.03, \text{iterations}=10$).
- **Output**: Tracked **100 feature points** with motion vectors drawn as directed green arrows $(\vec{u}, \vec{v})$ denoting instantaneous velocity and displacement direction.
- **Strengths**: Provides quantitative velocity vectors and directional dynamics rather than unoriented masks.
- **Weaknesses**: Assumes brightness constancy and small displacements; fails on featureless surfaces or severe motion blur.

#### 4.4 Motion Estimation Comparison Matrix

| Feature / Criteria | Frame Differencing | GMM Background Subtraction | Lucas–Kanade Optical Flow |
| :--- | :--- | :--- | :--- |
| **Output Type** | Binary change mask | Segmented foreground silhouette | Vector field (magnitude & orientation) |
| **Computational Cost** | $\mathcal{O}(N)$ (Minimal) | $\mathcal{O}(K \cdot N)$ (Moderate) | $\mathcal{O}(M \cdot W^2)$ (Low to moderate for sparse) |
| **Background Adaptation**| None | Adaptive statistical model | Not applicable (tracks local corners) |
| **Directional Motion** | ❌ No | ❌ No | ✅ Yes |
| **Best Used For** | Fast trigger / intrusion detection | Static-camera surveillance | Trajectory tracking, action recognition |

---

## 🚀 How to Run the Notebook

1. Ensure the virtual environment is activated:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Launch Jupyter Lab or VS Code Notebook:
   ```powershell
   jupyter notebook CCA3_Computer_Vision.ipynb
   ```
3. Run all cells sequentially (`Cell` -> `Run All`).
4. Output figures and CSV metric tables will be automatically generated and saved in:
   ```text
   outputs/task1_image_restoration/
   ```

---

## 📊 Summary of Generated Output Artifacts

| Filename | Description | Source Task |
| :--- | :--- | :--- |
| `original.png` | Saved baseline test image | Task 1 |
| `gaussian_noise.png` | Image with additive Gaussian noise ($\sigma=20$) | Task 1A |
| `salt_pepper_noise.png` | Image with salt-and-pepper impulse noise ($p=0.05$) | Task 1A |
| `gaussian_filtered.png` | Gaussian filter applied to Gaussian noisy image | Task 1A |
| `gaussian_filtered_salt_pepper.png` | Gaussian filter applied to salt-and-pepper noisy image | Task 1A |
| `median_filtered_gaussian.png` | Median filter applied to Gaussian noisy image | Task 1A |
| `median_filtered_salt_pepper.png` | Median filter applied to salt-and-pepper noisy image | Task 1A |
| `binary_image.png` | Binary thresholded mask ($T=127$) | Task 1B |
| `erosion.png` / `dilation.png` | Morphological erosion and dilation results | Task 1B |
| `opening.png` / `closing.png` | Morphological opening and closing results | Task 1B |
| `canny_edges.png` | Canny edge map ($T_{\text{low}}=100, T_{\text{high}}=200$) | Task 1B |
| `watershed_segmentation.png` | Marker-controlled watershed segmentation boundaries | Task 1B |
| `restoration_metrics.csv` | PSNR and SSIM quantitative evaluations | Task 1A |

---

## 👥 Authors & Academic Context
- **Course**: Computer Vision and Digital Image Processing
- **Assignment**: Continuous Comprehensive Assessment 3 (CCA 3)
- **Topics Covered**: Image Filtering, Mathematical Morphology, Feature Detectors (HOG/SIFT), Classical and Deep Object Detection (Viola–Jones & YOLOv8), Video Motion Analysis (Differencing, GMM, Optical Flow).
