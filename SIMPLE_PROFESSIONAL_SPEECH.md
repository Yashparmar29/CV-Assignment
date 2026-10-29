# 🎓 Simple & Professional 15-Minute Presentation Speech
## Continuous Comprehensive Assessment 3: Computer Vision & Digital Image Processing
### Units 3, 4, and 5 | Python, OpenCV, and YOLOv8

> **How to use this script:**
> This speech is written in **clean, simple, spoken English**. It avoids unnecessary tongue-twisters while keeping all the technical facts, numbers, and logical explanations completely sound so your professors understand exactly what you built, why each algorithm behaves the way it does, and what your conclusions are.

---

### ⏱️ Time & Section Tracker (Total: 15 Minutes)

* **00:00 – 01:30 (1.5 min):** Introduction, Project Goal & Data Used
* **01:30 – 05:00 (3.5 min):** Task 1A — Noise & Image Restoration (Gaussian vs. Median Filtering)
* **05:00 – 07:30 (2.5 min):** Task 1B — Binary Thresholding, Morphology, Canny & Watershed
* **07:30 – 10:00 (2.5 min):** Task 2 — Feature Extraction (HOG) & Matching (SIFT Rotation Invariance)
* **10:00 – 12:15 (2.25 min):** Task 3 — Face & Object Detection (Viola–Jones vs. YOLOv8)
* **12:15 – 14:15 (2.0 min):** Task 4 — Video Motion Estimation (Frame Differencing, GMM, Optical Flow)
* **14:15 – 15:00 (45 sec):** Final Summary & Ready for Questions

---

## 🗣️ Full Verbatim Speech Script

---

### Part 1: Introduction & Project Overview
**⏱️ Time: 00:00 – 01:30 | Slide 1 & 2**

"Good morning, respected professors and evaluators. 

Today, I am presenting my Computer Vision and Digital Image Processing project for Continuous Assessment 3, covering Units 3, 4, and 5.

In this project, our goal is to understand how computer vision algorithms solve practical visual problems step by step. We go through four main stages:
1. **First**, restoring an image when it gets corrupted by noise, and segmenting its shapes.
2. **Second**, finding key features in an image and matching them even if the image is rotated.
3. **Third**, comparing traditional machine learning face detection against modern deep learning object detection.
4. **And fourth**, tracking movement inside a video using three different motion estimation techniques.

Let me introduce the data we used:
- For our still-image experiments, we used an original portrait photograph with a resolution of **1600 by 720 pixels**. It shows a person standing outdoors in front of a tree and a diamond-pattern fence. This gives us both smooth areas like the shirt, and busy, textured areas in the background.
- We also used a second photograph of the same scene for image matching.
- For our motion experiments, we used a video named `v1.mp4`. It has **1,725 frames**, runs at **60 frames per second**, and is about **28.7 seconds long**, showing people moving in an outdoor street gathering.

We implemented everything in Python using OpenCV, NumPy, scikit-image, and Ultralytics YOLOv8. 

Let us start with Unit 3: Image Restoration."

---

### Part 2: Task 1A — Noise Modeling & Image Restoration
**⏱️ Time: 01:30 – 05:00 | Slide 3 & 4**

"In Task 1A, we wanted to see how different filters clean up different kinds of image noise.

We created two types of noisy images from our clean photo:
1. **Gaussian noise**: This is random, smooth noise that follows a normal bell curve. It simulates sensor noise that you often see in low-light photography. It makes the whole picture look grainy.
2. **Salt-and-pepper noise**: This is impulse noise. Here, we randomly corrupted 4% of the pixels in the image. Half of them became pure white dots—which is the salt—and half became pure black dots—which is the pepper.

Then, we tested two classic spatial filters:
- A **Gaussian Filter** with a 5 by 5 window, which smooths the image by calculating a weighted average of neighboring pixels.
- A **Median Filter**, also with a 5 by 5 window, which looks at the 25 neighboring pixels, sorts them from lowest to highest, and picks the middle value.

To measure image quality scientifically, we calculated two standard metrics:
- **PSNR (Peak Signal-to-Noise Ratio)**: Measured in decibels, where higher means lower error.
- **SSIM (Structural Similarity Index)**: Measured from 0 to 1, where 1 means the structure of the image looks identical to the original clean photo.

Now, let us look at the actual results, because the numbers teach us a very clear lesson:

**First, for Gaussian noise:**
- The noisy image had a PSNR of **22.20 dB** and an SSIM of **0.3795**.
- When we applied the **Gaussian filter**, the PSNR jumped up to **30.32 dB** and the SSIM reached **0.7857**.
- The median filter only achieved **28.85 dB**.
- So, **the Gaussian filter is clearly the winner for Gaussian noise**. Because the noise is a random normal distribution, averaging nearby pixels cancels out the positive and negative variations very well.

**Next, look at Salt-and-Pepper noise:**
- The corrupted image dropped down to **19.37 dB** and an SSIM of **0.4105**.
- When we applied the Gaussian filter, the result was disappointing: it only reached **28.35 dB** and **0.7172** SSIM. Visually, the Gaussian filter did not remove the black and white dots; it just smeared them into blurry gray spots!
- But when we applied the **Median filter**, the score shot up to **30.53 dB**, and the SSIM reached **0.8790**—the highest score in the entire experiment!

**Why does this happen?**
The reason is simple:
The Gaussian filter uses averaging. If an extreme white pixel value of 255 enters the calculation, it pulls the average way off.
The Median filter, on the other hand, sorts the numbers. The extreme white dots (255) and extreme black dots (0) are pushed to the very ends of the sorted list. The filter picks the number right in the middle, which is almost always a real, uncorrupted pixel.

So our conclusion for Task 1A is: **Gaussian filtering is best for Gaussian noise, and Median filtering is by far the best for Salt-and-Pepper noise.**"

---

### Part 3: Task 1B — Morphological Operations, Canny & Watershed
**⏱️ Time: 05:00 – 07:30 | Slide 5 & 6**

"Next, in Task 1B, we focused on binary processing, edge detection, and segmentation.

We started by converting our image to grayscale and applying a **binary threshold of 127**. 
As you can see in the output, the light-colored shirt turned white, and the dark trousers turned black. However, because the background leaves had uneven lighting, parts of the background also turned into messy white spots. A single threshold was not enough to separate the person cleanly.

To fix and clean binary shapes, we applied four basic **morphological operations** using a 5 by 5 square kernel:
1. **Erosion**: This shrinks white regions by peeling away border pixels. It successfully wiped out isolated speckles in the background.
2. **Dilation**: This grows white regions. It filled small holes inside the shirt and connected broken parts.
3. **Opening** (which is erosion followed by dilation): This cleaned away small background noise while keeping the main shape of the person intact.
4. **Closing** (which is dilation followed by erosion): This bridged small dark cracks and closed gaps without enlarging the object.

After morphology, we applied **Canny Edge Detection** using lower and upper thresholds of **100 and 200**. 
Canny works in four steps: it smooths out noise, finds gradient directions, suppresses non-maximum pixels to make edges thin, and uses hysteresis thresholding to keep strong, connected edges. The result is a clean, sharp outline of the person, the shirt stripes, and the fence, while flat regions stay clean and black.

Finally, we performed **Marker-Controlled Watershed Segmentation**.
Watershed treats an image like a landscape where high pixel values are hills and low values are valleys, and it floods them with water from chosen markers.
To create reliable markers:
- We found sure background by dilating our binary mask.
- We found sure foreground using the **Euclidean Distance Transform**, picking areas that are at least 40% of the maximum distance from the edge.
- The watershed algorithm flooded the surface and marked the meeting boundaries in red.

In our result, the red line wraps around the person and along the image borders. Because our photo is a natural portrait with a detailed background rather than separate round objects like coins, it separates the main silhouette, showing how marker control guides the watershed process."

---

### Part 4: Task 2 — Feature Extraction & SIFT Image Matching
**⏱️ Time: 07:30 – 10:00 | Slide 7 & 8**

"Moving to Unit 4, we examined how to extract features from an image and match them between two pictures.

First, we extracted **HOG (Histogram of Oriented Gradients)** features.
HOG breaks an image into small 8 by 8 pixel cells, counts how edge gradients point across 9 orientation directions, and normalizes them over 2 by 2 blocks. For our image, this created a large feature vector of **637,596 values**. The HOG visualization highlights the outer silhouette of the person and the strong geometric lines of the fence. This shows why HOG is widely used for pedestrian detection.

Next, we used **SIFT (Scale-Invariant Feature Transform)**:
- SIFT is designed to find points that stay recognizable even if an image is scaled, zoomed, or rotated.
- SIFT detected **6,198 keypoints** on our clean image. Each keypoint has a **128-dimensional descriptor** that describes the texture around that point.
- Most keypoints clustered in high-detail areas: the leaves, the fence joints, and the stripes on the shirt.

We then performed feature matching between our original photo and the second photo using a Brute-Force Matcher and **Lowe's Ratio Test** with a strict threshold of **0.6**.
Lowe's test ensures that a match is accepted only if the closest match is clearly better than the second-closest match. This eliminates false matches from repetitive background patterns.
When we displayed the **top 50 matches**, the connecting lines were almost perfectly horizontal. This proves that both images had identical framing and that SIFT matched the exact same physical spots on the person and the fence.

**Now came the critical test: Rotation Invariance.**
We took the image and rotated it by **45 degrees** around its center using an affine rotation matrix.
Because rotating an image cuts off the corners and leaves black borders, the number of detected keypoints dropped from 6,198 to **3,753**.

However, when we matched the original image with the 45-degree rotated image:
- SIFT successfully found **50 correct matches**.
- The match lines are now slanted at an angle, pointing directly to the corresponding rotated spots on the body, shirt, and background.

This is a key takeaway: SIFT computes a dominant orientation for every keypoint. When the image rotates, the descriptor rotates with it. That is why SIFT maintains its matching accuracy under rotation."

---

### Part 5: Task 3 — Object Recognition: Viola–Jones vs. YOLOv8
**⏱️ Time: 10:00 – 12:15 | Slide 9 & 10**

"In Unit 5, Task 3, we compared traditional computer vision against modern deep learning for object recognition.

We tested two detectors on the exact same photo:
1. **The classical Viola–Jones detector**, using OpenCV's pretrained Haar cascade for frontal faces.
2. **Ultralytics YOLOv8 Nano**, which is a modern, lightweight deep learning object detector trained on the 80 classes of the COCO dataset.

Here are the results, which clearly highlight the strengths and weaknesses of each:

**Viola–Jones:**
- It is very fast and requires minimal computing power.
- It detected **2 faces**.
- One bounding box correctly detected the face of the person.
- But the second bounding box was a **false positive**! It detected a face in the background fence and leaves to the right of the person.

**Why did Viola–Jones make this mistake?**
Viola–Jones relies on simple Haar features—rectangles of light and dark pixels. The wooden fence slats and leaf shadows happened to create a pattern of light and dark bands that matched the mathematical ratio of eyes, nose, and cheeks. Because the algorithm has no deeper understanding of what a face actually is, it got fooled by the texture. Furthermore, Viola–Jones can only detect frontal faces; it cannot detect people, clothes, or other objects.

**Now look at YOLOv8 Nano:**
- YOLO processed the image in a single neural network pass.
- It detected **1 object: a person**, with a high confidence score of **92.15%**.
- The bounding box accurately enclosed the entire standing figure from head to shoe.
- Most importantly, YOLO produced **zero false positives** on the fence or foliage.

Because YOLO uses a deep convolutional network, it looks at the entire context of the scene—arms, legs, torso, and proportions—rather than just local light-dark rectangles.

So, while Viola–Jones was historically groundbreaking for fast face detection, modern YOLO models are far more reliable, versatile, and resistant to background clutter."

---

### Part 6: Task 4 — Video Motion Estimation
**⏱️ Time: 12:15 – 14:15 | Slide 11 & 12**

"Our final task was motion estimation using our self-recorded video, `v1.mp4`. 
The video has 1,725 frames recorded at 60 frames per second. We compared three different motion detection techniques between Frame 0 and Frame 10:

1. **Frame Differencing**:
   - We subtracted Frame 0 from Frame 10 and applied a difference threshold of 25.
   - It is extremely fast to compute.
   - However, the output is noisy. It only highlights the edges of moving objects, leaving the middle of moving people hollow.

2. **GMM Background Subtraction (MOG2)**:
   - This method models every pixel as a mixture of Gaussians over 500 frames to learn what belongs to the static background.
   - It produced a much cleaner foreground silhouette than simple differencing, and it can adapt to gradual lighting changes over time.

3. **Lucas–Kanade Sparse Optical Flow**:
   - First, we found the 100 strongest corner points using the Shi–Tomasi algorithm (`goodFeaturesToTrack`).
   - Then, we tracked those exact points to Frame 10 using a 2-level pyramidal Lucas–Kanade algorithm.
   - All **100 feature points were tracked successfully**, and we drew them as green arrows showing the direction and distance each point moved.

**An Important Real-World Observation:**
Because this video was recorded using a handheld camera, the camera shook and moved slightly while filming.
- In Frame Differencing and GMM, the moving camera made large parts of the static background light up as if they were moving.
- But **Lucas–Kanade Optical Flow** revealed what was really happening! Because it gives us directional vectors—arrows with magnitude and angle—we could clearly see that the background vectors all pointed together in the direction of the camera movement, while moving people had distinct individual vectors.

This proves that while background subtraction is great for fixed security cameras, optical flow is necessary when you need to know the actual direction and speed of motion, especially if the camera itself is moving."

---

### Part 7: Conclusion & Thank You
**⏱️ Time: 14:15 – 15:00 | Slide 13**

"To summarize what we learned across all four tasks:
1. **For Image Restoration**: The best filter depends entirely on the noise. Gaussian filtering fixes Gaussian noise best (30.32 dB), while Median filtering is essential for impulse noise (30.53 dB, 0.8790 SSIM).
2. **For Morphology and Edges**: Morphological opening and closing clean up thresholded noise, and Canny produces clean, 1-pixel-wide boundaries.
3. **For Features**: SIFT successfully maintains feature correspondences even under a 45-degree rotation, proving true rotation invariance.
4. **For Object Detection**: YOLOv8's deep learning features eliminate the false detections that classical Haar cascades produce on textured backgrounds.
5. **And for Video Motion**: Optical flow gives us true directional motion vectors, which provides far more insight than simple change masks.

All source code, outputs, and documentation are available in the repository.

Thank you very much for your time and attention. I am now happy to answer any questions you may have."

---

## 💡 Quick Cheat-Sheet for Professor Questions (Viva Defense)

If a professor asks you a question after your speech, here are the short, confident answers:

* **"Why did Median filter do so much better on Salt-and-Pepper noise?"**
  * *Answer:* "Because salt and pepper noise is extreme values—0 or 255. A Gaussian filter averages those numbers, which spreads the noise out into gray blur. The Median filter sorts the numbers, so 0 and 255 sit at the ends, and the filter selects the clean median value in the middle."

* **"Why did Viola-Jones detect a face in the background fence?"**
  * *Answer:* "Viola-Jones looks for rectangular Haar features of light and dark pixels. The fence wood and leaf shadows created a light-dark checkerboard pattern that accidentally matched the mathematical ratio of a face. YOLO did not make this mistake because it uses deep learning to look at the whole context of a person."

* **"What is the advantage of Optical Flow over Background Subtraction?"**
  * *Answer:* "Background subtraction only tells you *which* pixels changed, like a black-and-white silhouette. Optical flow gives you *vectors*—meaning it tells you both the speed and the direction of movement (e.g., whether someone is moving left, right, or toward the camera)."

* **"Why did keypoints drop from 6,198 to 3,753 after rotating the image?"**
  * *Answer:* "When you rotate a rectangular image by 45 degrees inside the same canvas size, the outer corners get cut off and the new corners are filled with black space. Because image content is lost at the boundaries, fewer keypoints can be detected, but the surviving points still matched accurately."
