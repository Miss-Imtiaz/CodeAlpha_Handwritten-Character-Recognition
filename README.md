# Handwritten Character Recognition

Identifies handwritten English letters (A-Z) from images using a Convolutional Neural Network (CNN). Built as part of CodeAlpha internship — Task 3.

## What it does

Given a 28×28 grayscale image of a handwritten letter, the model predicts which of 26 alphabets (A-Z) it is — across diverse handwriting styles (cursive, printed, slanted, uppercase, lowercase).

## Dataset

**EMNIST Letters** (Extended MNIST)
- 88,800 training images, 14,800 test images
- 26 classes: A to Z
- 28×28 grayscale images, balanced across all letters
- Download via Kaggle: `kaggle datasets download -d crawford/emnist`

## Model Architecture (CNN)

```
Input (28×28×1)
    ↓
Conv2D (32 filters, 3×3) + ReLU       → learns edges and simple shapes
    ↓
MaxPooling2D (2×2)                     → reduces size, keeps important features
    ↓
Conv2D (64 filters, 3×3) + ReLU       → learns complex letter shapes
    ↓
MaxPooling2D (2×2)
    ↓
Flatten → Dense (128) + ReLU + Dropout(0.3)
    ↓
Dense (26) + Softmax                   → probability for each letter A-Z
```

**Total parameters: 227,098**

## Results

| Metric | Value |
|---|---|
| Training Accuracy (Epoch 10) | 93.58% |
| Validation Accuracy | 93.60% |
| **Test Accuracy** | **92.20%** |

Training and validation accuracy stayed close throughout — no overfitting. Model generalizes well to unseen handwriting.

## Training Progress

| Epoch | Train Accuracy | Val Accuracy |
|---|---|---|
| 1 | 74.4% | 88.7% |
| 3 | 89.4% | 92.1% |
| 5 | 91.2% | 92.7% |
| 10 | 93.6% | 93.6% |

## Key Learnings

- **CNNs work because they scan for local patterns** — edges and curves first, then combine them into letter shapes. A dense (fully connected) network would treat each pixel independently, missing the spatial relationships that make letters recognizable.
- **Dropout prevents overfitting** — randomly disabling 30% of neurons during training forces the network to learn redundant representations, making it more robust on new data.
- **Image normalization matters** — dividing pixel values (0–255) by 255 brings all inputs to 0–1 range, which stabilizes training significantly.
- **EMNIST is harder than MNIST** — 26 classes vs 10 digits, plus letters like I/l, O/0, S/5 look similar in handwriting. 92% on this dataset is genuinely strong.

## Tech Stack

- Python, TensorFlow/Keras, NumPy, Matplotlib, pandas
- Google Colab (free GPU)
- Kaggle API for dataset download

## How to Run

Open in Google Colab (recommended — TensorFlow pre-installed):

```python
# Step 1: Download dataset
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_api_key'
!pip install kaggle -q
!kaggle datasets download -d crawford/emnist
!unzip -q emnist.zip

# Step 2: Run the notebook
# All code is in handwritten_recognition.ipynb
```

## Files

- `handwritten_recognition.ipynb` — full Colab notebook: data loading, CNN training, evaluation
- `training_history.png` — accuracy and loss curves over 10 epochs
- `predictions.png` — sample predictions (green = correct, red = wrong)
- `sample_letters.png` — sample EMNIST letters from the dataset
