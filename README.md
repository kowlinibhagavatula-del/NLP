# 🌍 Language Detection using Deep Learning (LSTM & GRU)

## 📌 Project Overview

This project detects the language of a given text using Deep Learning models. It is developed using TensorFlow, Keras, and Streamlit. The project trains both LSTM and GRU models on a multilingual dataset containing 22 different languages and predicts the language of user-entered text.

---

# 🎯 Problem Statement

Language identification is an important Natural Language Processing (NLP) task used in applications such as:

- Machine Translation
- Chatbots
- Search Engines
- Voice Assistants
- Text Analytics
- Social Media Analysis

The objective of this project is to build a deep learning model capable of automatically identifying the language of an input sentence.

---

# 📂 Dataset Information

- Dataset Name: Language Detection Dataset
- Total Samples: 22,000
- Number of Languages: 22

### Dataset Columns

| Column | Description |
|---------|-------------|
| Text | Sentence or paragraph |
| Language | Target language |

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras
- Streamlit
- Joblib
- Matplotlib

---

# 📁 Project Structure

```text
Language_Detection_NLP/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── dataset.csv
│
├── models/
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   ├── lstm_model.keras
│   └── gru_model.keras
│
├── reports/
│
└── src/
    ├── config.py
    ├── utils.py
    ├── preprocess.py
    ├── train_lstm.py
    ├── train_gru.py
    └── predict.py
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project folder:

```bash
cd Language_Detection_NLP
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Step 1: Data Preprocessing

```bash
python src/preprocess.py
```

---

## Step 2: Train LSTM Model

```bash
python src/train_lstm.py
```

---

## Step 3: Train GRU Model

```bash
python src/train_gru.py
```

---

## Step 4: Predict Language

```bash
python src/predict.py
```

---

## Step 5: Launch Streamlit Application

```bash
streamlit run app.py
```

---

# 🧠 Deep Learning Models

## LSTM

Long Short-Term Memory (LSTM) is a Recurrent Neural Network (RNN) architecture designed to learn long-term dependencies in sequential data. It is well suited for text processing tasks.

---

## GRU

Gated Recurrent Unit (GRU) is a simplified version of LSTM. It has fewer parameters, trains faster, and performs efficiently on many sequence modeling tasks.

---

# 🔄 Project Workflow

```text
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Tokenization
    │
    ▼
Label Encoding
    │
    ▼
Padding
    │
    ▼
Train-Test Split
    │
    ▼
LSTM / GRU Training
    │
    ▼
Model Saving
    │
    ▼
Prediction
    │
    ▼
Streamlit Web Application
```

---

# 📊 Features

- Language Detection
- Text Preprocessing
- Tokenization
- Label Encoding
- LSTM Model
- GRU Model
- Model Saving and Loading
- Interactive Streamlit Interface

---

# 📈 Future Enhancements

- Character-level tokenization
- Transformer-based models (BERT/XLM-R)
- Improved preprocessing
- Hyperparameter tuning
- Model comparison dashboard
- Cloud deployment

---

# 👨‍💻 Author

**Bhagavatula Kowlini**

Machine Learning & NLP Project

---

# 📄 License

This project is developed for educational and learning purposes.