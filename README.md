# 🧠 Deep Learning Projects with Keras & OpenCV

Welcome to a curated collection of deep learning projects developed in Python using Keras, OpenCV, NumPy, and other foundational libraries. These projects showcase practical applications of image classification using both custom and public datasets.

---

## 📁 Project List

### 1. 📸 AI Landmark Classifier (Custom Dataset)
**File:** `AI_CAP_KARTIK_MARWAL.py`  
**Description:**  
A deep learning model built using the VGG19 architecture (from scratch) to classify landmark images from a structured dataset.

#### 🔧 Features:
- Reads structured image data from nested folders using file ID mappings.
- Uses `LabelEncoder` for categorical label transformation.
- Resizes images to 224x224 and normalizes pixel data.
- Custom training loop with `train_on_batch()` for memory efficiency.
- Data visualization with Matplotlib and error analysis post-training.

#### 🧰 Tech Stack:
- Python, Keras, OpenCV, NumPy, Pandas, PIL, Matplotlib
- Model: VGG19 (custom-trained, no pre-trained weights)

#### 📊 Highlights:
- Handles 20,000+ image samples across multiple classes.
- Implements custom batching, image reshaping, and label decoding.
- Includes code for sample visualization and basic prediction check.

---

### 2. 👟 Fashion MNIST Image Classification
**File:** `FASHION_MNIST_CLASSIFICATION.py`  
**Description:**  
Trains a fully connected neural network to classify grayscale images of clothing from the Fashion MNIST dataset.

#### 🔧 Features:
- Preprocessing and normalization of 28x28 images.
- One-hot encoding of labels using `to_categorical()`.
- Simple 3-layer neural network: `Flatten → Dense(128) → Dense(64) → Dense(10)`
- Evaluation and visual display of prediction results.

#### 🧰 Tech Stack:
- Python, Keras, Matplotlib, NumPy

#### 📊 Highlights:
- High accuracy with minimal architecture.
- Includes side-by-side comparison of actual vs predicted labels.
- Uses validation split and visualizes sample input images.

---

## 🚀 Setup & Run

### Prerequisites
Ensure you have the following Python libraries installed:

```bash
pip install numpy pandas opencv-python matplotlib keras pillow





To Run Any Project
1 Clone the repository:
    git clone https://github.com/KartikMarwal25/PROJECTS.git
    cd PROJECTS
2 Open the desired .py file and run it using:
    python filename.py
For the landmark classifier project, ensure the train.csv file and image dataset are correctly structured and paths updated accordingly.
    
