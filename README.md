![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Ultralytics](https://img.shields.io/badge/ultralytics-006BD3?style=for-the-badge&logo=ultralytics&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

# 🖨️ 3D Printing Monitoring & Failure Detection system

A multi-model Deep Learning pipeline designed to monitor 3D printing processes in real-time, detect "Spaghetti" failures, track the toolhead, and identify bed adhesion issues.

---

## 📖 Project Overview
3D printing is a time-intensive process where failures can lead to significant material waste and hardware damage. This project provides an automated solution using three specialized models to ensure the printing process remains within nominal parameters.

## 🎯 Project Objective
The primary goal of this system is to **optimize resource management and operational efficiency** for 3D printer users. 

By providing real-time monitoring and automated failure detection:
* **Waste Reduction:** Early detection of "Spaghetti" or bed detachment allows users to terminate failed prints immediately, saving expensive filaments and materials.
* **Time Efficiency:** Users receive instant notifications upon failure detection, preventing hours of useless printing on a failed build.
* **Remote Monitoring:** Reduces the need for constant physical supervision, allowing the system to act as a "digital eye" that alerts the user only when intervention is required.

## 🧠 The Models

| Model | Task | Description |
| :--- | :--- | :--- |
| **Spaghetti Detection** | Classification | Identifies if the print has failed and turned into a "spaghetti" mess. |
| **Toolhead Detection** | Object Detection | Tracks the real-time position of the printer's extruder (Toolhead). |
| **Bed Adhesion** | Attention Model | Uses spatial attention to detect if the print is lifting or shifting from the bed. |

---

## 📷 Dataset & Data Engineering
A core contribution of this project is the creation of a custom dataset, as no comprehensive public datasets were available for these specific tasks.

### Classification (Spaghetti)
To build a robust classifier, we performed manual data curation:
* **Sourcing:** We identified and downloaded hundreds of 3D printing time-lapse videos from YouTube.
* **Refinement:** Instead of using raw footage, we used **video editing software** to manually isolate and trim specific segments where failures occurred vs. successful print intervals. This ensured high-quality, noise-free training labels.

<img src="explaination_data/Classification_data/0_good.jpg" width="200" height="300" alt="Alt Text">
<img src="explaination_data/Classification_data/0_spaghetti.jpg" width="200" height="300" alt="Alt Text">

### Detection (Toolhead)
For the object detection task (YOLO-based), we created a targeted dataset:
* **Sourcing:** We captured manual **screenshots** from various YouTube videos to represent a wide array of printer models, toolhead designs, and lighting conditions.
* **Annotation:** All bounding boxes were manually drawn using **AnyLabeling**, providing the model with precise ground-truth data for the toolhead position.

<img src="explaination_data\detection_data\Figure_1.png" width="200" height="300" alt="Alt Text">
<img src="explaination_data\detection_data\1_toolhead.jpg" width="200" height="300" alt="Alt Text">

* **Resolution Scaling:** All captured images and video frames were resized to a standardized resolution of **224x224 pixels**.
* **Rationale:** This resolution provides an optimal balance between computational efficiency and preserving enough spatial detail for both failure classification and toolhead localization.

### MobileNet-GRU (Print detachment from Print bed classification)
For the task of classiflying wether or not a print had detached from the print bed using a GRU we had to collect sequences of printing images.

* **Sourcing:** We curated a dataset by scraping YouTube videos of failed 3D prints that detached mid-print.
* **Refinement:** To optimize computational efficiency, we downsampled the video to 1 FPS(frame per second) and focused on 16-second sequences, rather than processing every frame of the entire video.
* **Final Dataset Creation:** After isolating the specific "detachment frame" in our raw footage, we extracted a buffer of 38 frames per video:
    * **Frames 1-25:** Clean printing (normal operation).
    * **Frame 26:** The Detachment Event (moment of failure).
    * **Frames 27-38:** Post-detachment (failure state).

    **Sliding Window Generation:**
    We applied a sliding window of size 16 (representing 16 seconds) over these 38 frames. This method generated a balanced dataset for each video source:
    * **10 Clean Sequences** (Windows ending before frame 26)
    * **13 Detachment Sequences** (Windows containing the detachment event or post-detachment frames)

<img src="explaination_data\transfer_GRU_data\1_pre.png" width="244" height="244" alt="Alt Text">
<img src="explaination_data\transfer_GRU_data\1_detach.png" width="244" height="244" alt="Alt Text">
<img src="explaination_data\transfer_GRU_data\1_post.png" width="244" height="244" alt="Alt Text">

## 📊 Exploratory Data Analysis (EDA)

### Classification (Spaghetti)
We performed a targeted EDA to ensure our classification model (Spaghetti vs. Good Print) learns actual visual features rather than dataset artifacts:

 #### Class Distribution & Class Balance: We showcase the distribution between "Good" and "Spaghetti" samples.

<img src="explaination_data\Classification_data\Class_balance.png" width="600" height="400" alt="Alt Text">

#### Structural Consistency ("Ghost" Images): We generated average images for both classes. "Good" prints showed a structured, consistent shape, while "Spaghetti" resulted in a chaotic blur, confirming distinct structural features for the model to learn.

<img src="explaination_data\Classification_data\image_analysis.png" width="600" height="400" alt="Alt Text">

#### Environmental Bias (Lighting): We analyzed brightness distribution across all samples to ensure the model learns based on texture and geometry rather than being biased by lighting conditions (e.g., night vs. day shots).

<img src="explaination_data\Classification_data\brightness_distribution.png" width="600" height="400" alt="Alt Text">

### Detection (Toolhead)

For the Object Detection model (YOLO), we performed a specialized EDA to ensure the model effectively learns to locate the toolhead across different environments:

* Ground Truth Verification: We visualized random samples of our manual annotations to ensure the bounding boxes created in AnyLabeling perfectly align with the toolhead in the screenshots.

* Box Aspect Ratio Analysis: We analyzed the height-to-width ratios of our boxes. This ensures our toolhead dimensions are compatible with YOLO's anchor boxes, helping the model predict shapes accurately rather than struggling with unusual proportions.

#### Class Balance

<img src="explaination_data\detection_data\class_balance.png" width="600" height="400" alt="Alt Text">

#### Object Size Distribution:
Distance & Scale Invariance for Toolhead Tracking
The Toolhead Detection model is optimized to maintain a precise lock on the extruder assembly regardless of its proximity to the lens or position in the build volume:

* Adaptive Scale Training: Employs Scale and Random Cropping augmentations to identify toolhead features at various resolutions, ensuring stability across different camera mounting distances.

* Small Object Precision: Prioritizes high-resolution feature extraction to accurately detect the toolhead even when its pixel footprint decreases at the furthest corners of the print bed.

<img src="explaination_data\detection_data\Object size.png" width="400" height="400" alt="Alt Text">

 #### Location Heatmap: 
 We mapped the spatial distribution of the toolhead across all images. This allowed us to verify that the toolhead appears in various positions (corners, edges, center), preventing the model from becoming biased toward a single "hotspot" in the frame.

<img src="explaination_data\detection_data\Heatmap.png" width="400" height="400" alt="Alt Text">



### MobileNet-GRU (Print detachment from Print bed classification)

To validate the feasibility of a GRU-based approach, we analyzed the class distribution and temporal characteristics of our video dataset.

### 1. Class Balance
Training on raw video footage often leads to severe class imbalance (e.g., 10 hours of "clean" printing vs. 5 seconds of failure). To address this, we implemented a smart sliding window strategy that extracts fixed-length sequences (16 frames) from each video.

As shown below, this method downsamples the "Clean" majority class while preserving all "Detachment" instances, resulting in a training set that is effectively balanced.

<img src="explaination_data\transfer_GRU_data\eda_class_balance.png" width="500" height="500" alt="Alt Text">

* **Clean Sequences:** Windows ending before the failure occurs.
* **Detachment Sequences:** Windows containing the detachment event or post-failure chaos.
* **Result:** The model is not biased toward predicting "Normal," improving its sensitivity to actual failures.

### 2. Temporal Motion Analysis
We hypothesized that a Recurrent Neural Network (GRU) would be effective because print failures exhibit a distinct temporal signature compared to normal printing.

To verify this, we calculated the **Frame-to-Frame Pixel Difference (MSE)** across a printing timeline.

<img src="explaination_data\transfer_GRU_data\eda_temporal_motion.png" width="500" height="300" alt="Alt Text">

**Key Observations:**
* **Stable Phase (Blue):** During normal printing, the pixel variance is low and consistent. The print head moves, but the object remains stationary relative to the bed.
* **The Spike (Red):** The moment of detachment creates a massive spike in motion energy. This confirms that there is a strong, sudden "motion signal" that the GRU can learn to detect.
* **Chaotic Phase:** Post-detachment, the variance remains high and unpredictable (the "spaghetti" effect), distinguishing it clearly from the stable phase.

## 🛠️ Data Augmentations

### Classification (Spaghetti)

| Preprocessing Step | Description | Rationale | Metric |
| :--- | :--- | :--- | :--- |
| **Resolution Scaling** | All images were resized from their original resolution to **224x224** pixels. | Ensures computational efficiency and prevents the model from being overburdened by high-dimensional input. | **224 × 224** |

### Detection (Toolhead)

| Augmentation | Description | Rationale | Metric |
| :--- | :--- | :--- | :--- |
| **Mosaic** | Combines four different training images into a single image in varying ratios. | Improves the model's ability to detect small objects and enriches spatial context without increasing the batch size. | 1.0 |
| **Mixup** | Overlays two different training images and their labels using a weighted linear combination. | Regularizes the model to favor simple linear behavior between classes, reducing overconfidence and improving generalization. | 0.1 |
| **Degrees** | Randomly rotates the input image within a specified range of degrees. | Enhances robustness to variations in object orientation, ensuring the model recognizes objects that are not perfectly aligned. | 15° |
| **Translate** | Shifts the image horizontally or vertically by a fraction of the total width/height. | Helps the model become invariant to the position of the object within the frame, simulating cases where the object is partially off-center. | 0.1 |
| **Scale** | Randomly zooms in or out of the image by a specified gain factor. | Teaches the model to recognize objects at various distances and sizes, improving performance on both near and far subjects. | 0.5 |
| **Fliplr** | Flips the image horizontally (left to right) with a given probability. | Doubles the diversity of the dataset by simulating different perspectives, assuming the horizontal orientation does not change the object's class. | 0.5 |

### MobileNet-GRU (Print detachment from Print bed classification)
To prevent overfitting and ensure the model generalizes to different environments (e.g., dark rooms, tilted webcams, out-of-focus lenses), we applied the following augmentations during training.

| Augmentation | Explanation | Rationale | Metric |
| :--- | :--- | :--- | :--- |
| **Resize** | Scales all input images to a fixed square dimension. | Standardizes input size for the MobileNetV3 architecture. | **224 × 224 pixels** |
| **Temporal Consistency** | Applies the *exact same* random augmentation to all 16 frames in a window. | **Crucial:** Prevents the "strobe light" effect. If frames were augmented individually, the GRU would interpret random brightness changes as rapid motion/failure. | **Consistent per Window** |
| **Random Rotation** | Rotates the image slightly by a random angle. | Simulates imperfect camera mounting or slight vibrations. | **± 10 Degrees** |
| **Gaussian Blur** | Blurs the image to reduce sharpness. | Simulates cheap webcams or out-of-focus lenses, forcing the model to look for large features (spaghetti) rather than fine texture. | **p=0.2, Kernel=5** |
| **Random Horizontal Flip** | Flips the image left-to-right (mirror effect). | Doubles data variance. A print failure is valid whether it "flows" left or right. | **p=0.5** |
| **Color Jitter** | Randomly adjusts brightness, contrast, saturation, and hue. | Simulates extreme lighting conditions (e.g., night printing, harsh LED strips) to ensure the model isn't dependent on specific exposure. | **Bright=0.3, Contrast=0.3, Sat=0.1, Hue=0.05** |
| **Normalization** | Scales pixel values to standard deviations. | Aligns input data with the pre-trained ImageNet statistics required for the MobileNetV3 weights. | **ImageNet Standard** |

> *Note: Augmentations are only applied during the **Training** phase. Validation and Inference use only Resize and Normalization to ensure consistent evaluation.*

## 🤖 Model Selection and Training

### Classification (Spaghetti)

#### Training Performance:

* "While the model reached near 100% accuracy, we specifically aimed for high Recall to ensure that no print failure goes undetected. Our current results show a perfect recall rate, successfully catching all 'spaghetti' anomalies during validation to protect the hardware and save material."

* Rapid & Stable Convergence: Both training and validation loss minimized sharply by the 5th epoch and remained stable thereafter, indicating a well-regularized training process without overfitting.

<img src="explaination_data\Classification_data\classification_training.png" width="600" height="400" alt="Alt Text">
<img src="explaination_data\Classification_data\c_matrix.png" width="600" height="400" alt="Alt Text">


### Detection (Toolhead)

#### Training Performance:

* The "Click" Moment: If you look at the accuracy charts, there is a massive jump right around Epoch 10. This is where the model "figured out" the core features of the toolhead and started tracking it reliably across the build plate.

* Solid Reliability: We hit a 90% mAP@50 score, which in plain English means the model is extremely dependable at spotting the toolhead in almost every frame, even during high-speed movements.

* Precision and Detail: The mAP@50-95 score of 0.6 tells us that the bounding boxes aren't just "close"—they are tightly hugging the actual edges of the toolhead. This high level of precision is exactly what we need to calculate accurate $(x, y)$ coordinates.

* Stable Learning: While the loss curves show some minor "wiggles" (which is normal as the model tries to handle different lighting and perspectives), the overall downward trend confirms that the model is genuinely learning features rather than just memorizing images.

<img src="explaination_data\detection_data\Box_regression_loss.png" width="1200" height="600" alt="Alt Text">



### MobileNet-GRU (Print detachment from Print bed classification)
To solve the problem of real-time print detachment detection, we designed a custom hybrid architecture named **SpaghettiNet**. This model combines the spatial feature extraction capabilities of a Convolutional Neural Network (CNN) with the temporal sequence processing of a Recurrent Neural Network (RNN).

Our architecture is a **Time-Distributed CNN + GRU** pipeline. We process video as a sequence of frames rather than individual images, allowing the model to understand the *motion* of failure (e.g., spaghetti forming over time) rather than just static shapes.

| Component | Layer Type | Rationale |
| :--- | :--- | :--- |
| **Feature Extractor** | **MobileNetV3-Small** (Pre-trained) | A lightweight, low-latency CNN optimized for edge devices. We use it to convert each raw video frame (224x224) into a compact feature vector (size 576). |
| **Temporal Logic** | **GRU** (Gated Recurrent Unit) | A recurrent layer that remembers the context of previous frames. It analyzes the *sequence* of feature vectors to detect changes over time. |
| **Classifier** | **Linear + Sigmoid** | A simple fully connected layer that outputs a single probability score (0.0 - 1.0) indicating the likelihood of failure. |

**Why GRU instead of LSTM?**
We chose a GRU (Gated Recurrent Unit) over an LSTM because it has fewer parameters and offers faster inference speeds on embedded devices (like a Raspberry Pi) while maintaining comparable accuracy for short sequences.

### Training Strategy
The model was trained using a transfer learning approach, where the CNN weights were frozen to retain their ImageNet knowledge, and only the GRU and classification heads were trained.

* **Loss Function:** `BCEWithLogitsLoss` (Binary Cross Entropy) — Ideal for binary classification (Normal vs. Detached).
* **Optimizer:** Adam (Learning Rate: `1e-4`) — Chosen for stable convergence.
* **Batch Size:** 4 Sequences (Effective batch size: 4 × 16 frames = 64 images per step).
* **Sequence Length:** 16 Frames (representing 16 seconds of print time).

### Hyperparameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Input Size** | 224 × 224 | Standard MobileNet input resolution. |
| **Hidden Size** | 128 | Number of features in the GRU hidden state. |
| **Dropout** | 0.4 | Applied before the final layer to prevent overfitting. |
| **Epochs** | 137 | Total training passes. |
| **Train/Val Split** | 80% / 20% | Random split of video sequences. |

We use a "Stateful" inference approach during live monitoring. The GRU's hidden memory is preserved between frames and only reset after a full window of 16 seconds, simulating a continuous stream of awareness.*

<img src="explaination_data\transfer_GRU_data\training_graph_v2.png" width="1000" height="400" alt="Alt Text">
<img src="explaination_data\transfer_GRU_data\confusion_matrix.png" width="400" height="400" alt="Alt Text">

## 📈 Evaluation

### Classification (Spaghetti)

* Near-Perfect Accuracy: The model achieved near 100% Top-1 validation accuracy within 10 epochs, with qualitative testing showing 99.5%-100% confidence scores on unseen validation data.

* Zero Misclassifications: The confusion matrix confirms 100% precision and recall across 252 validation samples, successfully identifying all 203 clean prints and 49 spaghetti failures with zero false alarms.

* Robust Spatial Coverage: Rapid loss convergence indicates efficient feature learning, while spatial heatmaps verify the model remains accurate regardless of where the failure occurs on the print bed.

* Reliable Real-Time Intervention: These metrics demonstrate a highly dependable system capable of triggering immediate hardware intervention the moment a printing anomaly is detected.

<img src="explaination_data\Classification_data\results.png" width="1200" height="600" alt="Alt Text">


### Detection (Toolhead)

* Learning Curve: The model had a "lightbulb moment" around Epoch 10, where accuracy (mAP) surged as it mastered the toolhead's geometry. It stabilized at a mAP@50 of ~0.9, meaning it is incredibly consistent at finding the toolhead even during fast movements.

* Spatial Coverage: Our Object Location Heatmap confirms that we didn't just train on center-frame images; the model successfully recognizes the toolhead across the entire build plate, ensuring no "blind spots" at the edges of the bed.

* Real-World Confidence: In live tests, the model consistently returns confidence scores between 0.88 and 0.95, proving it can distinguish the toolhead from the printed part and background clutter with high certainty.

<img src="explaination_data\detection_data\toolhead_1.png" width="400" height="400" alt="Alt Text">
<img src="explaination_data\detection_data\toolhead_2.png" width="400" height="400" alt="Alt Text">
<img src="explaination_data\detection_data\toolhead_3.png" width="400" height="400" alt="Alt Text">


### MobileNet-GRU (Print detachment from Print bed classification)
* **Zero False Alarms (Precision Focused):** The confusion matrix reveals a highly precise system with **100% Precision** on the validation set. It correctly identified 8 clean prints and 12 failures with **zero false positives**. This is critical for 3D printing monitoring, as it ensures the system never pauses a successful print unnecessarily.

* **Stable Convergence:** The training graph demonstrates clear learning behavior. The Training Loss (Red) decreases steadily from ~0.7 to <0.1 over 200 epochs, while the Validation Accuracy (Blue) climbs from ~40% to a peak of **~87%**. This confirms the GRU is successfully learning the temporal features of "spaghetti" formation despite the complexity of the video data.

* **Perfect Discriminative Ability:** The Receiver Operating Characteristic (ROC) curve shows an **Area Under Curve (AUC) of 1.00**. This indicates that the model has perfectly separated the probability distributions for "Normal" and "Detached" classes, suggesting that with threshold tuning, the recall (currently 80%) can be further improved without sacrificing precision.

* **Conservative Intervention:** While the model missed 3 detachment events (False Negatives), its perfect False Positive rate makes it an ideal "conservative" guardian—it only triggers when it is absolutely certain of a failure, guaranteeing a frustration-free user experience.

<p align="center">
  <img src="explaination_data/transfer_GRU_data/demo.gif" width="600" alt="SpaghettiNet Live Demo">
  <br>
  <em>Real-time inference running at 1 FPS</em>
</p>

## 💯 End Summary

### 🍝 Spaghetti Classification: The Safety Net

The classification model serves as the system's primary failsafe. By achieving near-perfect recall, the model ensures that catastrophic failures—where filament turns into a tangled mess-are identified instantly.

* Key Achievement: 100% precision and recall on validation sets, meaning zero false alarms for the user and zero missed failures for the hardware.

* Impact: Drastic reduction in material waste and a significant decrease in the risk of "the blob" (molten plastic encasing the hotend). 

### 🏗️ Toolhead Detection (Research Phase)
The toolhead detection model was developed using a YOLO-based architecture to provide real-time spatial tracking of the extruder.
* **Performance:** Reached a solid **90% mAP@50**, successfully mastering the toolhead's geometry across the entire build plate.
* **Outcome:** Although the model performed well technically, it was **not included in the final production system**. The decision was made to omit this model because it was not longer nessecesry for the rest of our system.

### 🎞️ MobileNet-GRU (Print detachment from Print bed classification)
Unlike standard image classifiers that analyze a single static snapshot, this hybrid architecture analyzes the motion of the print over a 16-second window to detect bed adhesion failures.

* **Key Achievement:** Achieved **Zero False Positives (100% Precision)** on the validation set. This confirms the system's ability to act as a "conservative guardian," ensuring that a successful print is never interrupted by a false alarm.
* **Impact:** By combining the lightweight **MobileNetV3** with a memory-based **GRU**, the system successfully distinguishes between the rhythmic, predictable motion of a healthy print and the sudden, chaotic spike of a print detaching from the bed.


## 🛠 Tech Stack

### Core Frameworks
* **Language:** Python 3.x
* **Deep Learning:** PyTorch ( `torch`, `torch.nn`, `torch.optim`)
* **Computer Vision:** * `torchvision` (Pre-trained models & transforms)
    * `OpenCV` (cv2) (Video frame extraction & color conversion)
    * `Pillow` (PIL) (Image manipulation)

### Model Architectures

1.  **MobileNet-GRU (Print detachment from Print bed classification)**
    * **CNN Backbone:** MobileNetV3-Small (Pre-trained on ImageNet, frozen weights).
    * **Temporal Processor:** Gated Recurrent Unit (GRU) with 128 hidden units.
    * **Input Strategy:** Sliding window sequences of 16 frames.
    * **Architecture:** CNN Feature Extractor $\rightarrow$ RNN Sequence Modeling $\rightarrow$ Binary Classification Head.

2.  **Toolhead Detection/Spaghetti Classification**
    * **Framework:** Ultralytics YOLO26.
    * **Usage:** Used for Object Detection and Classification

### Data Processing & Training
* **Data Loading:** Custom `PrinterFrameDataset` with sliding window logic for temporal continuity.
* **Augmentation Pipeline:** * Geometry: Random Horizontal Flip, Rotation ($\pm 10^{\circ}$).
    * Visual: Gaussian Blur (simulating focus issues), Color Jitter (Brightness, Contrast, Saturation).
* **Optimization:** Adam Optimizer (`lr=1e-4`) with Binary Cross Entropy loss.
* **Hardware Support:** Native support for **NVIDIA CUDA**

### Analysis & Tools
* **Visualization:** `Matplotlib` and `Seaborn` for training curves and confusion matrices.
* **Metrics:** `Scikit-Learn` for accuracy and confusion metric calculations.
* **Showcasing:** Jupyter Notebooks (`.ipynb`).

---
