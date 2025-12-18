# Human_Activity_Recognition_Project
# Human Activity Recognition (HAR) — 10-Class Movement Subset  
*Classical ML pipeline with compact video features (10 features per video)*

> This repository implements a Human Activity Recognition system using *classical machine learning* trained on a *small, interpretable feature set* extracted directly from video clips.  
> The pipeline automatically selects *up to 10 “general movement” classes* (e.g., walk/run/jump-like actions) that actually exist in the dataset, extracts *motion + intensity + temporal* features from sampled frames, and produces ready-to-train datasets for ML models (SVM / Random Forest / Logistic Regression, etc.).

---

## Table of Contents
1. [Project Goal](#project-goal)  
2. [What This Code Does](#what-this-code-does)  
3. [Why We Use a 10-Class Subset](#why-we-use-a-10-class-subset)  
4. [Dataset Assumptions](#dataset-assumptions)  
5. [End-to-End Pipeline Overview](#end-to-end-pipeline-overview)  
6. [Automatic Class Selection (Movement Keyword Filter)](#automatic-class-selection-movement-keyword-filter)  
7. [Video Frame Sampling](#video-frame-sampling)  
8. [Feature Extraction (10 Features)](#feature-extraction-10-features)  
9. [Building X and y (Training Data)](#building-x-and-y-training-data)  
10. [Parameter-by-Parameter Justification](#parameter-by-parameter-justification)  
11. [Reproducibility and Fairness Notes](#reproducibility-and-fairness-notes)  
12. [How to Run](#how-to-run)  
13. [Common Errors and Fixes](#common-errors-and-fixes)  
14. [How to Extend This Project](#how-to-extend-this-project)  

---

## Project Goal
The objective is to build a *Human Activity Recognition* system that:
- Recognizes actions from videos (movement-based activities)
- Trains a *classical ML model* using compact, interpretable features
- Avoids high-dimensional handcrafted feature vectors (like 101 features) by using only *10 meaningful features*
- Works reliably inside *Google Colab* with a dataset already loaded into dataframes (train_df, val_df, test_df)

---

## What This Code Does
This project provides a clean pipeline that:

1. *Finds available labels* in the dataset (train_df["label"])
2. *Automatically selects up to 10 movement-related classes* based on keywords (walk/run/jump/etc.)
3. *Filters* train_df, val_df, test_df to only those selected classes
4. For each video:
   - Samples a fixed number of frames (NUM_FRAMES)
   - Converts frames to grayscale and resizes them
   - Computes 10 compact features:
     - 2 intensity statistics
     - 4 motion statistics (optical flow magnitude)
     - 4 temporal statistics (per-frame mean trends)
5. Builds:
   - X_train, y_train
   - X_val, y_val
   - X_test, y_test

These arrays are then ready to be used in classical ML training (e.g., SVM, RandomForest).

---

## Why We Use a 10-Class Subset
Large action datasets (e.g., sports/action datasets) often have *many classes*, and many of them are complex sports-specific actions.

For a general “movement recognition” objective, we restrict to a smaller set because:
- *Faster training and debugging*
- *Less class confusion*
- Better focus on *coarse human movement categories*
- Fewer classes often yields more reliable baselines during development
- Reduces imbalance and complexity when dataset is large

This choice is especially useful when your goal is “walking vs running vs jumping” type recognition rather than fine-grained sports taxonomy.

---

## Dataset Assumptions
The pipeline assumes these objects already exist:

- train_df, val_df, test_df are pandas DataFrames
- They contain at least two important columns:
  - label → activity name / class label
  - clip_path_resolved → full path to the corresponding video file

Example row format:

| clip_path_resolved | label |
|---|---|
| /content/data/v1.mp4 | WalkingWithDog |

If your dataset uses different column names, update the constants:
- LABEL_COL
- PATH_COL

---

## End-to-End Pipeline Overview
The workflow looks like this:

1. *Read unique labels from training set*
2. *Select a 10-class subset*
3. *Filter train/val/test*
4. *Extract features video-by-video*
5. *Build numpy arrays for ML training*

---

## Automatic Class Selection (Movement Keyword Filter)
We do *not hardcode* class names like "Walking Upstairs" because many datasets do not contain those exact labels.

Instead, we:
- Read *all labels* from dataset
- Search labels for movement-related words like:
  - walk, run, jump, climb, sit, stand, etc.
- Select the first 10 matches

### Why keyword-based selection?
Because class naming differs across datasets:
- WalkingWithDog
- JumpingJack
- RunSprint
- CliffDiving

Keyword matching adapts automatically without manual guessing.

---

## Video Frame Sampling
Videos have variable length (different number of frames).  
Classical ML requires a *fixed-size feature vector* per sample, so we need consistent sampling.

### Sampling strategy
- Use cv2.VideoCapture to open video
- Find total frames using:
  - cv2.CAP_PROP_FRAME_COUNT
- Sample indices uniformly using:
  - np.linspace(0, total-1, NUM_FRAMES)

### Why uniform sampling?
Uniform sampling ensures:
- We represent the *whole video*, not only the beginning
- We reduce computation while still capturing motion patterns
- The model sees consistent temporal coverage

### Why grayscale?
We convert frames to grayscale because:
- Activity recognition (movement) often depends more on motion than color
- Grayscale reduces computation
- It reduces noise from lighting/color variations
- Improves speed while keeping motion meaningful

### Why resize frames?
We resize to FRAME_SIZE = (112,112) because:
- Optical flow is expensive at high resolution
- 112×112 is a balanced tradeoff:
  - fast computation
  - enough detail to detect motion direction/magnitude

---

## Feature Extraction (10 Features)
The goal was explicitly to *avoid 101 features* and reduce to a small set.

We compute:

### (A) Intensity Features (2)
1. mean_intensity  
2. std_intensity

*Why?*  
These represent overall brightness/contrast and can correlate with scene characteristics. While not the most discriminative alone, they add useful baseline information at near-zero cost.

---

### (B) Motion Features via Optical Flow Magnitude (4)
We compute dense optical flow between consecutive frames using:

cv2.calcOpticalFlowFarneback

Then we convert flow vectors (dx,dy) to magnitude using:

cv2.cartToPolar

We summarize magnitudes using:
3. mean_motion  
4. std_motion  
5. max_motion  
6. min_motion

*Why optical flow magnitude?*  
Because action recognition is strongly linked to motion intensity:
- Walking → moderate motion
- Running → higher motion
- Standing/Sitting → low motion
- Jumping → bursts of motion (max values)

---

### (C) Temporal Statistics (4)
We compute per-frame mean intensity across time:
- per_frame_means = frames.mean(axis=(1,2))

Then compute:
7. temporal_mean  
8. temporal_std  
9. temporal_max  
10. temporal_min

*Why temporal stats?*  
They capture global variation over time:
- Some actions show stable patterns (standing)
- Some show periodic change (walking/running)
- Some show sudden changes (jumping)

---

## Building X and y (Training Data)
We loop over each row in the dataframe:

- Read video path: row["clip_path_resolved"]
- Check it exists: Path(vp).exists()
- Extract feature vector
- Append to arrays

We skip:
- Missing video files
- Videos that cannot be read
- Videos that produced fewer than 2 frames (not enough for optical flow)

At the end:
- X becomes a 2D numpy array: (N_samples, 10)
- y becomes a 1D array: (N_samples,)

We raise a clear error if X becomes empty:
> "No samples extracted. Check video paths."

This prevents silent failures.

---

## Parameter-by-Parameter Justification

### NUM_FRAMES = 8
- Controls how many frames are sampled per video
- *Why 8?*
  - very fast extraction
  - enough to compute optical flow between consecutive frames (7 flow steps)
  - good trade-off for Colab runtime

If you increase it:
- accuracy may improve slightly
- runtime increases

---

### FRAME_SIZE = (112,112)
- Controls resolution of frames used for optical flow
- *Why 112×112?*
  - faster than 224×224
  - still captures motion patterns
  - optical flow becomes much slower at high resolution

---

### Optical Flow Parameters (Farneback)
```python
cv2.calcOpticalFlowFarneback(prev, cur, None,
    0.5, 3, 15, 3, 5, 1.2, 0
)
