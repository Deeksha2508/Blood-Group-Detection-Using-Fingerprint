# Blood Group Detection Using Fingerprint Analysis

## Project Overview

Blood Group Detection Using Fingerprint Analysis is a Machine Learning and Deep Learning project that predicts an individual's blood group from fingerprint images. The system uses image processing, handcrafted feature extraction, traditional machine learning algorithms, and deep learning models to classify fingerprints into different blood group categories.

This approach provides a non-invasive, fast, and cost-effective alternative to traditional blood group testing methods.

---

## Objectives

- Predict blood groups using fingerprint images.
- Apply image preprocessing techniques to enhance fingerprint quality.
- Extract meaningful texture and ridge-based features.
- Compare multiple Machine Learning models.
- Improve prediction performance using Deep Learning and Ensemble Learning.

---

## Dataset
The fingerprint dataset used in this project was obtained from **Kaggle** and contains fingerprint images categorized into eight blood group classes.


- A+
- A-
- B+
- B-
- AB+
- AB-
- O+
- O-

The dataset is organized into separate folders for each blood group.

---

##  Methodology

### 1. Image Preprocessing

Fingerprint images undergo:

- Grayscale conversion
- Image resizing
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Normalization

These steps improve ridge visibility and image quality.

---

### 2. Feature Extraction

The project extracts texture and ridge features using:

#### GLCM (Gray Level Co-occurrence Matrix)
- Contrast
- Correlation
- Energy
- Homogeneity

#### LBP (Local Binary Pattern)
- Local texture descriptors
- Histogram-based features

#### Gabor Filters
- Ridge orientation analysis
- Multi-scale texture extraction

**Total Features Extracted: 154**

| Feature Type | Number of Features |
|-------------|------------------|
| GLCM | 48 |
| LBP | 26 |
| Gabor | 80 |
| Total | 154 |

---

### 3. Machine Learning Models

The following models were trained and evaluated:

- Support Vector Machine (SVM)
- Random Forest
- Logistic Regression
- XGBoost

Performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

### 4. Deep Learning Model

A pretrained **ResNet18** model was fine-tuned using transfer learning.

Features:

- ImageNet pretrained weights
- Data augmentation
- Class-weighted loss
- Dropout regularization
- Cosine Annealing Learning Rate Scheduler

---

### 5. Ensemble Learning

A Late Fusion Ensemble was implemented by combining:

- SVM Predictions
- Random Forest Predictions
- ResNet18 Predictions

Weighted averaging was used to improve classification performance.

---

##  Technologies Used

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Scikit-Image
- PyTorch
- TorchVision
- XGBoost

---

## Evaluation Metrics

The project evaluates model performance using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Cross Validation

---

##  Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Blood-Group-Detection-Using-Fingerprint.git

cd Blood-Group-Detection-Using-Fingerprint
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

1. Place the fingerprint dataset inside the project directory.
2. Open the Jupyter Notebook:

```bash
jupyter notebook
```

3. Run:

```bash
Blood_Group_Detection_usingFingerPrintAnalysis_AI_Project.ipynb
```

---

##  Results

- Successfully extracted fingerprint texture features.
- Compared multiple machine learning algorithms.
- Fine-tuned ResNet18 for fingerprint classification.
- Improved prediction performance using ensemble learning.
- Demonstrated the feasibility of non-invasive blood group prediction using fingerprints.

---

## Future Improvements

- Collect larger and more diverse datasets.
- Deploy the model as a web application.
- Develop a real-time fingerprint scanner integration.
- Explore advanced CNN architectures such as EfficientNet and Vision Transformers.
- Improve ensemble strategies for higher accuracy.


## Team Contribution

This project was developed collaboratively by a two-member team.

### My Contributions (Deeksha Sharma)

- Collected, organized, and prepared the fingerprint dataset for model training and evaluation.
- Implemented feature extraction techniques including GLCM, LBP, and Gabor Filters to obtain texture and ridge-based fingerprint features.
- Assisted in data preprocessing and feature engineering.
- Created and maintained the GitHub repository, including project documentation, README preparation, and version control.

### Collaborator Contributions

- Developed and trained machine learning and deep learning models.
- Implemented model optimization and ensemble learning techniques.
- Conducted model evaluation, performance analysis, and experimentation.
- Assisted in project integration and final testing.
---

## Author

Deeksha Sharma

GitHub:https://github.com/Deeksha2508/ 



---

