# 🎯 Task A – Gender Classification Using CNN

## 🧠 Objective  
This project addresses **Task A** of the **FACECOM Challenge 2025**, which involves **binary gender classification (Male/Female)** from facial images using a **Convolutional Neural Network (CNN)**.

---

## 📁 Dataset Structure

```
Task_A/
├── train/
│   ├── male/
│   └── female/
└── val/
    ├── male/
    └── female/
```

Each subfolder contains face images labeled by gender.

---

## 🧩 Model Architecture

The model is built using `tf.keras.Sequential` with the following layers:

- `Conv2D` layers with ReLU activation  
- `BatchNormalization`  
- `MaxPooling2D`  
- `Dropout` for regularization  
- `Flatten` → Fully Connected (`Dense`) Layers  
- Final output layer: `Sigmoid` activation for binary classification  

✅ **Input shape:** `(128, 128, 3)`

---

## ⚙️ Training Configuration

- **Data Augmentation:** Rotation, shifting, shearing, zooming, horizontal flipping  
- **Loss Function:** `BinaryCrossentropy`  
- **Optimizer:** `Adam`  
- **Evaluation Metric:** `Accuracy`  
- **EarlyStopping:** Enabled (monitors validation loss)  
- **Class Weights:** Automatically computed for dataset imbalance  

---

## 📈 Model Performance

### ✅ Training Set:

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.9486  |
| Precision  | 0.9877  |
| Recall     | 0.9471  |
| F1-Score   | 0.9670  |

### ✅ Validation Set:

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.9052  |
| Precision  | 0.9397  |
| Recall     | 0.9338  |
| F1-Score   | 0.9367  |

---

### 📊 Classification Report:

```
              precision    recall  f1-score   support

    female       0.80      0.82      0.81       105
      male       0.94      0.93      0.94       317

    accuracy                           0.91       422
   macro avg       0.87      0.88      0.87       422
weighted avg       0.91      0.91      0.91       422
```

---

### 🧮 Confusion Matrix:

```
[[ 86  19]
 [ 21 296]]
```

---

## 📦 Project Files

| File Name                   | Description                                  |
|----------------------------|----------------------------------------------|
| `Task_A_Model.ipynb`        | Notebook for training and evaluation         |
| `gender_classifier_model.h5`| Trained model (download link below)         |
| `test_Task_A.py`            | CLI-based testing script                     |
| `accuracy_loss_plot.png`    | Accuracy/loss training graph                 |
| `taskA_model.png`           | Model architecture diagram                   |
| `README_Task_A.md`          | This documentation file                      |

---

## 🔗 Download Trained Model

Due to GitHub file size limits, the trained model (`gender_classifier_model.h5`) is hosted externally:

📥 [Download gender_classifier_model.h5](https://drive.google.com/file/d/1Z6nHdgoAZ6U2CpxGS83-YnIAQ4ywHI56/view?usp=sharing)

---

## 🚀 How to Run the Test Script

Make sure you're in the `Task_A/` directory and run:

```bash
python test_Task_A.py gender_classifier_model.h5 val
```

### 📌 Ensure:

- `gender_classifier_model.h5` is downloaded and placed in the same directory.
- `val/` directory structure is preserved as shown above.

Expected Output:

- 🎯 Accuracy  
- 📊 Confusion Matrix  
- 📄 Classification Report (Precision / Recall / F1-score)

---

## 🧪 Requirements

Install necessary packages with:

```bash
pip install tensorflow scikit-learn matplotlib
```

---

## 👩‍💻 Authors

- **Srijani Saha**  
- **Shreyanka Roy**  

📝 **Submission for:** FACECOM Challenge 2025