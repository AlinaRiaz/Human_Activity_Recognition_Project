# Human Activity Recognition (HAR) System  
## Feature-Engineered Classical Machine Learning Approach
*Data Set Used* : matthewjansen/ucf101-action-recognition
---

## 1. Problem Statement

Human Activity Recognition (HAR) aims to identify human actions from visual or sensor data. In this project, the task is to classify *general human movement activities* from video clips using *classical machine learning models*.

Instead of using end-to-end deep learning, this project deliberately adopts a *feature-engineering-first approach*, where meaningful motion-based features are extracted from video data and then used to train classical classifiers.

---

## 2. Why Feature Engineering Was Used

### 2.1 Motivation for Feature Engineering

Feature engineering was chosen over direct end-to-end deep learning for the following reasons:

1. *Interpretability*  
   Engineered features allow us to understand what aspects of the video contribute to classification (motion, intensity, temporal variation), which is not easily achievable with deep neural networks.

2. *Computational Efficiency*  
   Deep learning models require large datasets and powerful GPUs. Feature-engineered classical ML models:
   - Train faster
   - Require less memory
   - Are suitable for Google Colab environments

3. *Dataset Constraints*  
   Many action recognition datasets:
   - Have limited samples per class
   - Are imbalanced
   - Contain noisy or diverse actions

   Classical ML models perform more reliably under these conditions when paired with good features.

4. *Academic Clarity*  
   Feature engineering allows clear theoretical justification of design choices, which is important in coursework-based evaluation.

---

## 3. Design Principles Behind Feature Engineering

The feature engineering strategy follows these principles:

- *Compactness:* Avoid large, redundant feature vectors (e.g., 101 handcrafted features)
- *Relevance:* Extract features directly related to human motion
- *Consistency:* Produce fixed-length vectors for variable-length videos
- *Simplicity:* Use well-understood computer vision techniques

The final design results in *exactly 10 features per video*.

---

## 4. Why Video Frame Sampling Is Necessary

Videos differ in length (number of frames). Classical ML models require a *fixed-size input vector*, so temporal normalization is required.

### Sampling Strategy
- Uniformly sample NUM_FRAMES = 8 frames across the entire video
- This ensures:
  - Coverage of the full action duration
  - Temporal consistency across samples
  - Reduced computational load

*Why not use all frames?*
- Redundant information
- Increased noise
- Significantly higher computation without proportional performance gain

---

## 5. Why Grayscale and Spatial Resizing Were Used

### Grayscale Conversion
Human activity recognition is largely driven by *motion patterns*, not color information. Grayscale:
- Reduces sensitivity to lighting conditions
- Simplifies optical flow computation
- Lowers dimensionality

### Spatial Resizing (112×112)
- Optical flow computation is expensive at high resolution
- 112×112 preserves motion structure while remaining efficient
- Ensures stable and fast feature extraction

---

## 6. Why Optical Flow Was Chosen for Motion Representation

Optical flow estimates pixel-wise motion between frames and is a classical, well-established method for motion analysis.

### Why Optical Flow Is Appropriate for HAR
- Human actions are defined by *movement*
- Optical flow captures:
  - Speed of motion
  - Intensity of movement
  - Temporal dynamics

### Why Only Magnitude (Not Direction)?
- Direction varies due to camera viewpoint
- Magnitude is:
  - More robust
  - Rotation-invariant
  - Strongly correlated with action intensity

---

## 7. Explanation of Each Feature (10 Features)

### 7.1 Intensity Features (2)

| Feature | Purpose |
|-------|--------|
| Mean intensity | Global brightness / scene context |
| Std intensity | Contrast and variability |

These provide basic scene-level information at minimal cost.

---

### 7.2 Motion Features (4)

| Feature | Purpose |
|------|--------|
| Mean motion | Average movement strength |
| Std motion | Motion variability |
| Max motion | Peak activity moments |
| Min motion | Low-activity regions |

These are *core discriminators* for actions like walking vs running.

---

### 7.3 Temporal Features (4)

| Feature | Purpose |
|------|--------|
| Temporal mean | Overall activity level |
| Temporal std | Consistency vs variability |
| Temporal max | Action peaks |
| Temporal min | Action inactivity |

These capture how motion evolves over time.

---

## 8. Why We Did NOT Use 101 Features

Using very large handcrafted feature sets often leads to:
- Redundant information
- Curse of dimensionality
- Overfitting
- Poor generalization

By limiting features to *10 meaningful dimensions*, we:
- Reduce noise
- Improve model stability
- Simplify interpretation
- Enable faster hyperparameter tuning

---

## 9. Why Classical Machine Learning Models Were Used

### 9.1 Rationale for Classical ML

Classical ML models are suitable when:
- Feature dimensionality is low
- Features are semantically meaningful
- Dataset size is moderate
- Interpretability is important

This project intentionally aligns feature engineering with classical ML strengths.

---

## 10. Why These Specific Models Were Chosen

### 10.1 Logistic Regression

*Why used:*
- Serves as a baseline
- Tests linear separability of features
- Highly interpretable

*What it tells us:*
- Whether motion features alone are sufficient for linear classification

---

### 10.2 Support Vector Machine (SVM)

*Why used:*
- Strong performance in low-dimensional spaces
- Effective non-linear decision boundaries
- Resistant to overfitting

*How it differs from Logistic Regression:*
- Maximizes margin instead of probability
- Can model non-linear relationships using kernels
- Focuses on support vectors (critical samples)

---

### 10.3 Random Forest

*Why used:*
- Captures complex non-linear interactions
- Robust to noise and outliers
- Does not assume linearity

*How it differs from SVM:*
- Ensemble of decision trees
- Learns feature interactions automatically
- Provides feature importance scores

---

## 11. Model Differences: A Comparative View

| Aspect | Logistic Regression | SVM | Random Forest |
|------|-------------------|-----|--------------|
| Linearity | Linear | Linear / Non-linear | Non-linear |
| Feature scaling | Required | Required | Not required |
| Interpretability | High | Medium | Medium |
| Overfitting risk | Low | Medium | Low |
| Handles interactions | No | Limited | Yes |

---

## 12. Why Feature Engineering + These Models Work Well Together

- Feature engineering extracts *motion-centric signals*
- Logistic Regression checks linear separability
- SVM refines separation using margins/kernels
- Random Forest captures non-linear interactions

This layered approach provides *robust validation of feature quality*.

---

## 13. Evaluation Strategy

Multiple metrics are used:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

This ensures performance is evaluated fairly, especially under class imbalance.

---

## 14. Limitations

- Optical flow is sensitive to camera motion
- Compact features may miss subtle gestures
- Classical ML cannot learn hierarchical spatio-temporal representations

These limitations are accepted given the project scope and objectives.

---

## 15. Conclusion

This project demonstrates that *thoughtful feature engineering, combined with **appropriate classical machine learning models*, can effectively solve Human Activity Recognition tasks.

The approach prioritizes:
- Interpretability
- Efficiency
- Theoretical justification
- Academic rigor

This makes the system suitable for coursework evaluation and as a foundation for future deep learning extensions.

---
