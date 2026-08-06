# 📰 Fake News Detection using NLP

A Machine Learning and Natural Language Processing (NLP) based Fake News Detection system that classifies news articles as **Real** or **Fake**.

The model analyzes the textual content of news articles using NLP preprocessing techniques and predicts whether the news is genuine or misleading.

---

## 📌 Features

- Detects whether a news article is **Real** or **Fake**
- NLP-based text preprocessing
- Machine Learning classification model
- Simple and easy-to-use interface
- Fast predictions
- Trained model saved using Joblib

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK (Natural Language Toolkit)
- Joblib
- Jupyter Notebook

---

## 📂 Project Structure

```
Fake-News-Detection/
│
├── app.py                     # Main application
├── NLP_newsprediction_model.joblib
├── news.csv                   # Dataset
├── newsprediction.ipynb        # Training notebook
└── README.md
```

---

## 📊 Dataset

> **Current Dataset:** used **US News Dataset**

This project is currently trained using a **US news dataset** containing authentic and fake news articles from American news sources.

### Important Note

The model has learned patterns from **US-style news articles**, including:

- American politics
- US media writing style
- US current affairs
- US news formatting

Because of this, the model performs best on **US news content**.

Using news articles from other countries (India, UK, etc.) may reduce prediction accuracy.

---

## 🧠 NLP Pipeline

The input text goes through several preprocessing steps:

1. Convert text to lowercase
2. Remove punctuation
3. Remove numbers
4. Tokenization
5. Remove stopwords
6. Stemming/Lemmatization (if applied)
7. TF-IDF Vectorization
8. Model Prediction

---

## 🚀 Installation

Clone the repository

```bash
https://github.com/bakshikul/Fake-News-Detection-Machine-Learning-Model.git
```

Move into the project folder

```bash
cd Fake-News-Detection
```

Run the application

```bash
streamlit run app.py
```

---

## 💻 Example

Input:

```
The White House announced a new economic policy to reduce inflation.
```

Prediction:

```
✅ Real News
```

---

Input:

```
Aliens have taken control of the US Congress according to secret NASA files.
```

Prediction:

```
❌ Fake News
```

## 🔮 Future Improvements

- Support for Indian news datasets
- Multi-country news detection
- Deep Learning (LSTM/BERT)
- Transformer-based NLP models
- Confidence score for predictions
- Explainable AI (highlight important words influencing predictions)
- API deployment
- Web application deployment

---

## 📈 Model Workflow

```
News Article
      │
      ▼
Text Preprocessing
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Trained NLP Model
      │
      ▼
Real / Fake Prediction
```

---

## 🤝 Contributing

Contributions are welcome!

If you find bugs or have ideas for improvements, feel free to open an issue or submit a pull request.

---

## 📜 License

This project is intended for educational and research purposes.

---

## 👨‍💻 Author

Developed as an NLP-based Fake News Detection project using Machine Learning and Natural Language Processing.

⭐ If you found this project useful, consider giving it a star!
