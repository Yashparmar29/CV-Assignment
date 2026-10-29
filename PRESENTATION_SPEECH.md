# 🎙️ 15-Minute Project Presentation Speech Script
## Continuous Comprehensive Assessment 3 (CCA 3): Computer Vision & Digital Image Processing (Units 3, 4 & 5)

> **Speaker Profile:** Presenting student / researcher  
> **Target Duration:** 15 Minutes (~1,950 words at an optimal pace of 130–140 words/min)  
> **Format:** Spoken script with timestamp markers, slide/visual cues, speaker delivery tips, and anticipated Q&A defense.

---

### ⏱️ Presentation Timing Breakdown

| Section | Topic Covered | Target Time | Cumulative Time |
| :--- | :--- | :---: | :---: |
| **Section 1** | Title, Context, Objectives & Input Dataset | 2.0 min | 00:00 – 02:00 |
| **Section 2** | Task 1A: Noise Modeling & Image Restoration (PSNR/SSIM) | 3.5 min | 02:00 – 05:30 |
| **Section 3** | Task 1B: Morphological Processing, Canny & Watershed | 2.5 min | 05:30 – 08:00 |
| **Section 4** | Task 2: Feature Extraction (HOG) & Matching (SIFT Invariance) | 2.5 min | 08:00 – 10:30 |
| **Section 5** | Task 3: Object Recognition (Viola–Jones vs. YOLOv8) | 2.0 min | 10:30 – 12:30 |
| **Section 6** | Task 4: Video Motion Estimation (Differencing, GMM, Optical Flow) | 2.0 min | 12:30 – 14:30 |
| **Section 7** | Synthesis, Practical Takeaways & Q&A Transition | 0.5 min | 14:30 – 15:00 |

---

## 🗣️ Verbatim Presentation Script

---

### SECTION 1: Introduction, Objectives & Experimental Data
**⏱️ Time: 00:00 – 02:00 (2 Minutes)**  
**Visual Cue:** *Slide 1 (Title, Student Details & GitHub Repo) & Slide 2 (Input Images & Video Specifications)*

**[Spoken Script]**

> "Good morning, respected professors, external evaluators, and colleagues.
>
> Today, I am proud to present my comprehensive implementation and analytical evaluation for **Continuous Comprehensive Assessment 3: Computer Vision and Digital Image Processing**, spanning Units 3, 4, and 5 of our curriculum.
>
> In real-world computer vision, visual systems must operate across a full operational hierarchy: starting from raw sensor-level noise and degradation, moving through low-level morphological segmentation, advancing to invariant geometric feature representation, and culminating in high-level object recognition and temporal video motion estimation.
>
> The core objective of this project is not merely to execute standard API calls, but to rigorously benchmark **classical, analytical algorithms against modern, deep-learning paradigms** under identical empirical conditions.
>
> Let us review our dataset:
> 1. Our primary experimental image is `original.jpg`, an uncompressed portrait of $1600 \times 720$ resolution across 3 color channels, featuring a subject against an intricate background of foliage and diamond-pattern fencing. This provides rich textural diversity and complex edge geometries.
> 2. Our secondary image, `second.jpg`, shares this scene for dual-image descriptor matching.
> 3. For video motion estimation, I recorded a high-definition real-world video, `v1.mp4`, containing $1,725$ frames captured at $60\text{ frames per second}$ over a duration of $28.74\text{ seconds}$, capturing a dynamic outdoor crowd scene.
>
> All experiments were implemented in Python using OpenCV 4.13, NumPy, scikit-image, Matplotlib, and Ultralytics YOLOv8. Let us begin with Unit 3: Image Restoration."

---

### SECTION 2: Task 1A — Image Restoration & Quantitative Metrics
**⏱️ Time: 02:00 – 05:30 (3.5 Minutes)**  
**Visual Cue:** *Slide 3 (Noisy vs. Restored Image 3x3 Grid) & Slide 4 (PSNR & SSIM Comparison Table)*

**[Spoken Script]**

> "In Task 1A, we investigate the fundamental problem of image degradation and spatial reconstruction. We modeled two distinct mathematical noise distributions:
> 1. First, **Additive Gaussian Noise**, generated with zero mean and standard deviation $\sigma = 20$. This affects every pixel continuously according to a Gaussian probability distribution, simulating sensor electronic thermal noise.
> 2. Second, **Impulse Salt-and-Pepper Noise**, where $4\%$ of the total pixels were randomly and independently corrupted, with $50\%$ converted to maximum intensity ($255$, salt) and $50\%$ converted to zero intensity ($0$, pepper).
>
> To restore these images, we applied two distinct classes of spatial filters using identical $5 \times 5$ neighborhoods:
> - A linear **Gaussian Filter** with $\sigma = 1.0$, which computes a 2D distance-weighted kernel average.
> - A non-linear **Median Filter**, which sorts local intensities and replaces the central pixel with the statistical median rank.
>
> Rather than relying on subjective visual inspection, we evaluated the fidelity quantitatively using **Peak Signal-to-Noise Ratio (PSNR)** in decibels and the **Structural Similarity Index (SSIM)**.
>
> Let us examine the exact empirical numbers:
> - For **Gaussian Noise**, the degraded image started at a PSNR of $22.20\text{ dB}$ and SSIM of $0.3795$. Applying the **Gaussian filter** boosted PSNR to **$30.32\text{ dB}$** and SSIM to **$0.7857$**. The median filter achieved $28.85\text{ dB}$ and $0.7551$. Here, the Gaussian filter was the undisputed winner because Gaussian convolution is mathematically optimal for averaging out zero-mean, normally distributed perturbations.
> - However, look at what happens under **Salt-and-Pepper Noise**. The raw corrupted image dropped to a severe $19.37\text{ dB}$ and $0.4105$ SSIM. When we applied the Gaussian filter, it failed to resolve the spikes; instead, it averaged the extreme $0$ and $255$ outliers into surrounding pixels, creating smeared gray blotches, only reaching $28.35\text{ dB}$ and $0.7172$ SSIM.
> - In contrast, the **Median filter** produced an exceptional recovery: a PSNR of **$30.53\text{ dB}$** and a remarkable SSIM of **$0.8790$**.
>
> The theoretical justification is profound: the median filter is an order-statistic rank operator. Because impulse noise creates extreme outliers at the tails of the local distribution, the median completely ignores both $0$ and $255$, selecting a genuine neighborhood pixel and preserving structural edge transitions without blurring. This confirms that filter selection must strictly match the underlying noise distribution."

---

### SECTION 3: Task 1B — Morphological Processing, Canny & Watershed
**⏱️ Time: 05:30 – 08:00 (2.5 Minutes)**  
**Visual Cue:** *Slide 5 (Thresholding & Morphological Operators 2x3 Grid) & Slide 6 (Canny Edges & Marker-Controlled Watershed Output)*

**[Spoken Script]**

> "Moving to Task 1B, we explore binary morphology and object segmentation.
>
> We began by converting our image to grayscale and applying a global binary threshold at $T = 127$. As seen in the output, global thresholding successfully extracted the high-reflectance white shirt, but because the outdoor background foliage had non-uniform ambient illumination, it produced fragmented background noise.
>
> To structure this binary mask, we implemented four fundamental morphological operators using a $5 \times 5$ square structuring element:
> - **Erosion** stripped away single-pixel boundary layers, successfully eliminating isolated background speckles.
> - **Dilation** expanded foreground boundaries and closed small fractures.
> - **Morphological Opening**—erosion followed by dilation—cleanly excised background clutter without altering the overall shape or area of the subject.
> - **Morphological Closing**—dilation followed by erosion—bridged small dark gaps and micro-cavities within the subject's clothing.
>
> Next, we applied the **Canny Edge Detector** with dual hysteresis thresholds of $T_{\text{lower}} = 100$ and $T_{\text{upper}} = 200$. Canny's 4-stage pipeline—Gaussian presmoothing, Sobel gradient magnitude and direction, non-maximum suppression (NMS) along gradient vectors, and hysteresis tracking—produced crisp, 1-pixel-wide topological contours outlining the subject, collar, and fence struts, while completely suppressing flat regions.
>
> Finally, we implemented **Marker-Controlled Watershed Segmentation**. We performed Otsu's inverted thresholding, cleaned noise via morphological opening, computed sure background via dilation, and critically derived sure foreground using the **Euclidean Distance Transform (`DIST_L2`)** thresholded at $40\%$ of its maximum distance. Connected components formed the marker basins, and the watershed algorithm flooded the topographic gradient surface.
>
> In our result, the red watershed dams cleanly isolate the overall human silhouette against the boundary. This demonstrates both the power of marker-guided flooding and its inherent sensitivity in natural outdoor scenes compared to synthetic multi-object benchmarks."

---

### SECTION 4: Task 2 — Feature Extraction & Geometric Image Matching
**⏱️ Time: 08:00 – 10:30 (2.5 Minutes)**  
**Visual Cue:** *Slide 7 (HOG Feature Visualization) & Slide 8 (SIFT Keypoints, 50 Horizontal Matches vs. 50 Rotated 45° Matches)*

**[Spoken Script]**

> "In Unit 4, we transitioned from pixel-level masks to invariant feature descriptors.
>
> First, we extracted **Histogram of Oriented Gradients (HOG)** features using 9 unsigned orientation bins, $8 \times 8$ pixel cells, and $2 \times 2$ block normalization. For our $1600 \times 720$ image, this generated an extensive feature vector of **$637,596$ dimensions**. The resulting HOG visualization clearly maps high-magnitude gradient vectors around the silhouette contours, proving why HOG remains a premier descriptor for dense pedestrian silhouette classification.
>
> Next, we implemented David Lowe's **Scale-Invariant Feature Transform (SIFT)**:
> - SIFT searches for scale-space extrema across Difference-of-Gaussians (DoG) octaves.
> - It identified **$6,198$ keypoints**, generating a $(6198, 128)$ descriptor matrix. In the visualization, keypoints densely cluster around textured high-entropy zones—the foliage, fence intersections, and striped shirt pattern.
>
> We then performed feature matching between `original.jpg` and `second.jpg` using a Brute-Force Matcher with $k$-Nearest Neighbors ($k=2$) and **Lowe's Ratio Test** set to a strict threshold of $0.6$:
> $$\frac{d(\text{best})}{d(\text{second-best})} < 0.6$$
> Visualizing the top 50 matches produced perfectly horizontal correspondence lines, confirming identical spatial framing and flawless feature repeatability.
>
> To test **geometric invariance**, we subjected the image to a **$45^\circ$ affine rotation** around its centroid using `cv2.getRotationMatrix2D`. Because rotation alters the bounding canvas and introduces black clipped borders, detected keypoints dropped to $3,753$.
>
> Yet, when we matched the rotated descriptor set, SIFT successfully linked 50 robust correspondences with slanted matching vectors. Because SIFT computes canonical dominant orientations from local gradient distributions, the descriptors rotate in synchrony with the patch, empirically verifying SIFT's genuine rotation invariance."

---

### SECTION 5: Task 3 — Object Recognition: Viola–Jones vs. YOLOv8
**⏱️ Time: 10:30 – 12:30 (2.0 Minutes)**  
**Visual Cue:** *Slide 9 (Haar vs. YOLOv8 Side-by-Side Visual Detections) & Slide 10 (Comparison Table & False Positive Analysis)*

**[Spoken Script]**

> "In Unit 5, Task 3 examines one of the most critical evolutions in computer vision: classical feature cascades versus modern deep learning object detectors.
>
> On one side, we deployed the classical **Viola–Jones detector** using the OpenCV pretrained frontal face Haar cascade (`scaleFactor = 1.1`, `minNeighbors = 5`). Viola–Jones relies on integral images and an AdaBoost cascade of weak classifiers.
>
> On the other side, we deployed **Ultralytics YOLOv8 Nano (`yolov8n.pt`)**, a state-of-the-art single-stage anchor-free convolutional neural network trained on the 80-class COCO benchmark, executed with an input size of $320$.
>
> The experimental results offer fascinating comparative insights:
> 1. **Viola–Jones** successfully localized the subject's face. However, it also generated a **false positive** detection on the background fence and foliage to the right. Why did this happen? The alternating light and dark wooden lattice paired with leaf shadows mimicked the exact horizontal and vertical intensity differences of human eyes and nose bridges, tricking the rigid Haar filter thresholds. Furthermore, Viola–Jones is constrained strictly to frontal faces and cannot identify generic objects.
> 2. **YOLOv8**, in contrast, detected the subject as a `person` with a remarkable **$92.15\%$ confidence score**, generating a tight bounding box from $y=580$ to $y=1598$. It produced **zero false positives** in the background. Because YOLOv8 learns hierarchical deep contextual representations, it understands semantic object boundaries even against heavy background clutter.
>
> While Viola–Jones was a monumental breakthrough in 2001 for low-power edge face detection, YOLOv8 demonstrates the overwhelming superiority of deep features for multi-class robustness and background discrimination."

---

### SECTION 6: Task 4 — Video Motion Estimation
**⏱️ Time: 12:30 – 14:30 (2.0 Minutes)**  
**Visual Cue:** *Slide 11 (Video Properties & Sample Frames) & Slide 12 (Differencing vs. GMM vs. Lucas–Kanade 2x2 Grid)*

**[Spoken Script]**

> "Our final task tackled dynamic scene analysis through video motion estimation on our self-recorded video `v1.mp4`—$1,725$ frames of a vibrant outdoor gathering.
>
> We evaluated three distinct motion estimation methodologies across the temporal baseline between Frame 0 and Frame 10:
>
> 1. **Temporal Frame Differencing** ($\Delta t = 10$, $T = 25$): We subtracted consecutive frames and thresholded the absolute residual. While computationally instantaneous ($\mathcal{O}(N)$), the resulting mask is noisy and exhibits severe 'aperture' and ghosting artifacts—only moving object edges light up, leaving interior regions hollow.
>
> 2. **GMM Background Subtraction (MOG2)**: Using `cv2.createBackgroundSubtractorMOG2` with a 500-frame history and variance threshold of 40, we modeled each pixel's history as an adaptive Gaussian mixture over all $1,725$ frames. This produced a much cleaner, coherent silhouette of foreground movement, successfully adapting to gradual environmental lighting shifts.
>
> 3. **Lucas–Kanade Sparse Optical Flow**: We detected 100 optimal Shi–Tomasi corner points (`goodFeaturesToTrack`) and tracked their motion vectors across frames using a 2-level pyramidal Lucas–Kanade solver (`calcOpticalFlowPyrLK`, $15 \times 15$ window). All 100 feature points were successfully tracked, visualized as green directional vectors.
>
> Crucially, our empirical analysis uncovered a vital real-world insight: because this video was recorded handheld, subtle camera ego-motion caused background vectors to shift uniformly alongside pedestrian motion. Optical flow was the **only method capable of disambiguating this**, because it provides explicit 2D motion velocity and direction $(\vec{u}, \vec{v})$, whereas frame differencing and GMM can only register scalar pixel change masks."

---

### SECTION 7: Conclusion & Transition to Defense
**⏱️ Time: 14:30 – 15:00 (30 Seconds)**  
**Visual Cue:** *Slide 13 (Summary Conclusion Matrix & Thank You)*

**[Spoken Script]**

> "To conclude:
> - In **Restoration**, non-linear median filtering is essential for impulse noise ($0.8790$ SSIM), while linear Gaussian filtering excels for Gaussian degradation ($30.32\text{ dB}$).
> - In **Feature Matching**, SIFT guarantees true rotational and scale invariance where global descriptors fail.
> - In **Object Recognition**, deep anchor-free detectors like YOLOv8 eliminate the textural false positives inherent to classical Haar cascades.
> - In **Motion Estimation**, optical flow provides directional velocity dynamics that scalar background subtraction cannot capture.
>
> This project has bridged the gap between theoretical image processing foundations and robust computer vision engineering.
>
> Thank you for your time and kind attention. I now welcome your questions."

---

## 🛡️ Anticipated Viva / Defense Questions & Strategic Answers

Be prepared to answer these specific technical questions if posed by the evaluators:

### Q1: Why did the Median filter achieve a higher SSIM than the Gaussian filter on Salt-and-Pepper noise?
> **Answer:** "Gaussian filtering calculates a linear weighted average. When an impulse noise pixel ($0$ or $255$) is within the kernel window, its extreme value heavily skews the average, diffusing the corrupted intensity into neighboring pixels and degrading edge sharpness. In contrast, median filtering is an order-statistic rank filter: it sorts the 25 neighborhood pixels and picks the 13th value. Since impulse noise pixels sit at the extreme ends ($0$ or $255$), they are mathematically excluded from the median position, completely eliminating the noise while leaving edge boundaries untouched."

### Q2: Why did Viola–Jones detect a face in the background fence, and how would you fix it?
> **Answer:** "Viola–Jones evaluates rectangular Haar-like features: sums of pixels in adjacent dark and light rectangles. The lattice fencing interwoven with foliage cast sharp horizontal and vertical shadow bands that satisfied the cascaded classifier's threshold for eye-socket and nose bridge differentials. To eliminate this false positive, we can:
> 1. Increase the `minNeighbors` parameter from $5$ to $7$ or $8$, requiring stronger candidate cluster overlap.
> 2. Increase the `minSize` constraint to avoid small background false alarms.
> 3. Implement skin-tone color filtering (in YCbCr or HSV space) as an early candidate gating step."

### Q3: What is the significance of Lowe's Ratio Test in SIFT matching?
> **Answer:** "In high-dimensional descriptor spaces (such as SIFT's 128 dimensions), random noise or repeated background textures (like repetitive foliage) will still yield a nearest neighbor with a finite Euclidean distance. Lowe's ratio test compares the distance of the best match ($d_1$) to the second-best match ($d_2$). If $d_1 / d_2 < 0.6$, it proves that the best match is significantly and uniquely distinct. If the ratio is close to $1.0$, the feature is ambiguous or repetitive and is discarded, eliminating over $90\%$ of false matches."

### Q4: Why did camera motion affect GMM and Frame Differencing more severely than Optical Flow?
> **Answer:** "Frame differencing and GMM assume a stationary camera platform where pixel coordinate $(x, y)$ corresponds to the exact same physical world point across time. Handheld camera ego-motion causes the entire scene coordinate frame to translate, causing static background pixels to exceed the variance threshold. Optical flow, however, computes explicit vector displacements $(\Delta x, \Delta y)$ for discrete corner features. By analyzing the global vector field, we can estimate the homography or affine transformation of the camera and subtract camera ego-motion from genuine object movement."

---

## 💡 Practical Speaker Delivery Tips
1. **Pacing:** Speak at a steady pace of ~130–140 words per minute. Pause for 2 seconds after presenting each quantitative result ($30.53\text{ dB}$, $6,198$ keypoints, $92.15\%$ confidence).
2. **Slide Synchronization:** Advance your slides during the designated *Visual Cues* indicated at the beginning of each section.
3. **Emphasis:** Verbally stress key contrasting concepts: *linear vs. non-linear*, *scalar change masks vs. directional vector fields*, and *classical handcrafted features vs. deep hierarchical representations*.
