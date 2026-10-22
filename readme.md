# IMDB Sentiment Analysis using LSTM & Deep Learning

An end-to-end Natural Language Processing (NLP) pipeline that classifies IMDB movie reviews into positive or negative sentiments using sequential Deep Learning (LSTM) in TensorFlow/Keras.

---

## 📌 Project Overview

This repository contains a complete sentiment analysis workflow trained on a balanced 5,000-review subset (2,500 positive, 2,500 negative) of the famous IMDB 50k Dataset.

The pipeline covers:
- **Data Preprocessing & Cleaning:** Removing HTML breaks, special characters, and converting text to lower case.
- **Tokenization & Sequence Padding:** Integer mapping with vocabulary truncating and pre-padding for sequential stability.
- **Model Architecture:** A GPU-optimized LSTM model with custom Embedding and SpatialDropout layers to mitigate overfitting.
- **Evaluation:** Precision, Recall, F1-Score, Accuracy, and Confusion Matrix visualization.
- **Inference Pipeline:** Demonstration on 5 unseen custom reviews.

---

## 🛠 Model Architecture

The deep learning sequential pipeline consists of the following layers:

| Layer Type | Configuration / Output Shape | Description / Function |
| :--- | :--- | :--- |
| **Input Layer** | `(batch_size, 200)` | Accepts padded integer sequences of fixed length 200. |
| **Embedding** | `input_dim=1000, output_dim=64` | Maps discrete tokens into 64-dimensional dense vectors. |
| **SpatialDropout1D** | `rate=0.4` | Drops entire feature maps to prevent reliance on specific words. |
| **LSTM** | `units=64` | Captures contextual sequential information over sentences. |
| **Dropout** | `rate=0.5` | Regularization layer to prevent overfitting on small datasets. |
| **Dense** | `32 units, activation='relu'` | Non-linear feature extraction. |
| **Dropout** | `rate=0.5` | Regularization layer. |
| **Dense (Output)** | `1 unit, activation='sigmoid'` | Squashes output score to probability $[0.0, 1.0]$. |

- **Optimizer:** `Adam`
- **Loss Function:** `binary_crossentropy`
- **Callbacks:** `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`

---

## 📊 Performance & Evaluation Results

Evaluated on an independent **20% Test Set** (1,000 unseen samples):

- **Accuracy:** `79.70%`
- **Precision:** `78.39%`
- **Recall:** `82.00%`
- **F1-Score:** `80.16%`

### Confusion Matrix

| | Predicted Negative | Predicted Positive |
| :--- | :---: | :---: |
| **Actual Negative** | **397** (TN) | **103** (FP) |
| **Actual Positive** | **100** (FN) | **400** (TP) |

---

## 🔮 Inference on Unseen Reviews

```
Review: Absolutely fantastic movie, loved the brilliant acting and stunning visuals.
Prediction: Positive (Confidence: 0.9114)

Review: Terrible waste of time. The plot was boring and the acting was wooden.
Prediction: Negative (Confidence: 0.0584)

Review: It was okay. Not the best I've seen, but had some funny moments.
Prediction: Positive (Confidence: 0.5544)

Review: I walked out halfway through. The dialogue was incredibly cringeworthy.
Prediction: Negative (Confidence: 0.2928)

Review: A masterful cinematic experience that will stay with me for a long time.
Prediction: Positive (Confidence: 0.6345)
```

---

## 🚀 How to Run

### Prerequisites
Ensure you have the following installed:
- Python 3.10+
- TensorFlow 2.x / Keras 3.x
- Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/imdb-sentiment-lstm.git
   cd imdb-sentiment-lstm
   ```

2. **Download the Dataset:**
   Download the IMDB dataset from Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place `IMDB Dataset.csv` inside your data directory.

3. **Run the Notebook:**
   Open and execute `dl-cat-2-imdb-movie-reivews.ipynb` in Jupyter Notebook, JupyterLab, or upload to Google Colab/Kaggle Notebooks.

---

## ⚙ Key Takeaways & Technical Insights

1. **Pre-padding (`padding='pre'`) vs Post-padding:** Using `'pre'` padding ensures the LSTM cell updates its hidden state on actual word vectors rather than sequence-ending zeros, drastically improving validation accuracy.
2. **GPU Optimization:** Avoiding `recurrent_dropout` inside the LSTM layer keeps the computation on cuDNN kernels, enabling fast GPU training (~1s per epoch on NVIDIA T4).