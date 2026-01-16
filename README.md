# Face Mask Detection Using MobileNetV2 (Deep Learning)

This project builds a **binary image classification** model to detect whether a person is wearing a **Mask** or **No Mask**. It uses **Transfer Learning** with **MobileNetV2** (pretrained on ImageNet), freezes the base model, and trains a small custom classifier head on a mask dataset.

ML flow covered:
**Problem Statement → Selection of Data → Collection of Data → EDA/Preprocessing → Train/Test Split → Model Selection → Evaluation Metric**

---

## Problem Statement
Detect if a face image belongs to:
- **With Mask**
- **Without Mask**

**Output:** Predicted class using softmax (2 classes)

---

## Selection of Data
**Dataset Type Used:** Image dataset (2 folders / binary classes)

Expected dataset structure:
- `with_mask/` (label = 1)
- `without_mask/` (label = 0)

---

## Collection of Data
Images are read from local folders using `cv2.imread()` and resized to `128 × 128`.  
Labels are converted to one-hot format using `to_categorical()`.

---

## Preprocessing (EDA + Preparation)
- Resize all images to `IMG_SIZE = 128`
- Normalize pixel values: `img / 255.0`
- Convert labels to categorical: 2 classes

---

## Dividing Training and Testing
Data is split using `train_test_split`:
- `test_size=0.2`
- `shuffle=True`
- `stratify=labels.argmax(axis=1)` to keep class balance in train and test

---

## Model Selection
**Model used:** MobileNetV2 (Transfer Learning)

How it works in this project:
- Load pretrained MobileNetV2 without top layer: `include_top=False`
- Freeze base network: `base.trainable = False`
- Add custom head:
  - GlobalAveragePooling2D
  - Dropout(0.3)
  - Dense(2, softmax)

---

## Evaluation Metric (Used in this Project)
- **Accuracy** is used as the main metric during training and evaluation.
- Final test accuracy is printed using `model.evaluate()`.

---

## Main Libraries Used (and why)

1. `os`  
   - Access folders and build dataset paths.

2. `cv2 (OpenCV)`  
   - Read images, resize images for training.

3. `numpy`  
   - Store images as arrays and normalize pixel values.

4. `tensorflow / keras`  
   - Build, train, evaluate, and save the deep learning model.

5. `sklearn.model_selection.train_test_split`  
   - Split dataset into train and test sets with stratification.

---
## How to Run (Colab)

1. Upload the dataset ZIP:
   ```python
   from google.colab import files
   files.upload()
   
2. Unzip the dataset:
   ```python
   !unzip -q Dataset-example.zip -d /content/mp/mask_project/

3. Check folder structure:

!ls /content/mp/mask_project/Dataset-example
!ls /content/mp/mask_project/Dataset-example/with_mask
!ls /content/mp/mask_project/Dataset-example/without_mask

4. Set the dataset path in code:

dataset_path = "/content/mp/mask_project/Dataset-example"

5. Run the training cells:

- Train the model  
- Evaluate accuracy  
- Model saves as `mask_model_mobilenet.h5`  

---

## Developer
Grishma C.D
