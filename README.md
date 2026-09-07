# 📩 SMS Spam Detection using Machine Learning

<p align="center">
  <b>End-to-End NLP Classification Project | CountVectorizer + Logistic Regression</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/NLP-Text%20Classification-8A2BE2" alt="NLP">
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-2E8B57" alt="Logistic Regression">
  <img src="https://img.shields.io/badge/Accuracy-97.67%25-success" alt="Accuracy">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License">
</p>

<p align="center">
  <a href="https://github.com/AnshuRaj1201/SMS-Spam-detection-bootcamp-project">Repository</a> •
  <a href="https://github.com/AnshuRaj1201/SMS-Spam-detection-bootcamp-project/blob/main/SMS%20Spam%20detection%20bootcamp%20project.ipynb">Notebook</a> •
  <a href="https://github.com/AnshuRaj1201/SMS-Spam-detection-bootcamp-project/blob/main/spam.csv">Dataset</a>
</p>

---

## 📌 Project Overview

**SMS Spam Detection** is an end-to-end Natural Language Processing (NLP) and Machine Learning project that automatically classifies SMS messages as either:

- 🟢 **Ham** — legitimate/normal message
- 🔴 **Spam** — unwanted or promotional message

The project demonstrates a practical text-classification workflow using **Python, Pandas, Matplotlib, Scikit-learn, CountVectorizer, and Logistic Regression**.

Instead of treating SMS messages as raw text, the project transforms them into numerical feature vectors and trains a supervised binary classifier to identify spam patterns.

> **Core pipeline:**  
> `Raw SMS → Data Preparation → Train/Test Split → CountVectorizer → Logistic Regression → Prediction → Evaluation`

---

## 🎯 Problem Statement

Unwanted SMS messages can contain promotions, fraudulent offers, misleading links, or other unsolicited content. Manually filtering every message is inefficient.

The goal of this project is to build a lightweight machine learning classifier that can learn from previously labeled SMS messages and predict whether a **new, unseen message** is Ham or Spam.

This project focuses on a simple, interpretable, and efficient classical NLP approach rather than using a computationally expensive deep-learning model.

---

## ✨ Key Features

- 📂 Loads and explores an SMS classification dataset using Pandas
- 🔎 Performs basic dataset inspection and class-distribution analysis
- 📊 Visualizes Ham vs Spam message counts
- 🔢 Encodes target labels:
  - `Ham → 0`
  - `Spam → 1`
- ✂️ Splits data into **80% training / 20% testing**
- 🧹 Removes common English stopwords during vectorization
- 🔤 Converts SMS text into numerical features using **CountVectorizer**
- 🤖 Trains a **Logistic Regression** classifier
- 🎯 Generates predictions on unseen test messages
- 📈 Evaluates the model using accuracy
- 🧩 Generates a confusion matrix
- 💬 Allows users to enter a custom SMS and receive a Ham/Spam prediction

---

## 🧠 Machine Learning Workflow

```text
                    ┌─────────────────────┐
                    │     SMS Dataset     │
                    │   spam.csv          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Exploration    │
                    │ & Cleaning          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Label Encoding      │
                    │ Ham=0 | Spam=1      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Train/Test Split    │
                    │ 80% / 20%           │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             Training Data          Testing Data
                    │                     │
                    ▼                     │
          ┌─────────────────┐             │
          │ CountVectorizer │             │
          └────────┬────────┘             │
                   │                      │
                   ▼                      │
          ┌─────────────────┐             │
          │ Logistic        │             │
          │ Regression      │             │
          └────────┬────────┘             │
                   │                      │
                   └──────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │   Predictions       │
                    │   Ham / Spam        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    │ Accuracy + CM       │
                    └─────────────────────┘
```

---

## 📊 Dataset

The repository includes the dataset as:

```text
spam.csv
```

After selecting the relevant columns, the data is represented as:

| Column | Description |
|---|---|
| `Category` | Target label: `ham` or `spam` |
| `Message` | Raw SMS message text |

### Dataset Statistics

The notebook reports:

- **Total messages:** 5,572
- **Ham:** 4,825
- **Spam:** 747
- **Columns used:** 2

The class distribution is imbalanced toward Ham messages, which is representative of the dataset used in the project.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib** | Data visualization |
| **Scikit-learn** | Machine learning workflow |
| **CountVectorizer** | Text-to-numeric feature extraction |
| **Logistic Regression** | Binary classification |
| **Confusion Matrix** | Classification evaluation |
| **Jupyter Notebook** | Interactive development and experimentation |

---

## 🔬 Methodology

### 1. Data Loading

The dataset is loaded with Pandas:

```python
df = pd.read_csv("spam.csv", encoding="latin-1")
```

The original CSV contains additional unused columns, so the notebook retains only the relevant label and message fields.

---

### 2. Data Preparation

The project standardizes the dataset to:

```text
Category | Message
```

Target labels are encoded as:

```text
Ham  → 0
Spam → 1
```

The feature and target variables are then separated:

```python
X = df["Message"]
y = df["Category"]
```

---

### 3. Train/Test Split

The dataset is divided using an **80/20 split**:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

This produces:

- **4,457 training messages**
- **1,115 testing messages**

The `stratify=y` parameter preserves the Ham/Spam class distribution across the training and testing sets.

---

### 4. Text Feature Extraction

Machine learning models require numerical input, so raw SMS text is converted into numerical features using:

```python
CountVectorizer(stop_words="english")
```

The vectorizer is fitted only on the training data:

```python
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)
```

The resulting feature matrices contain **7,440 vocabulary features** for this run.

This approach also prevents the vocabulary from being learned from the test set.

---

### 5. Model Training

The classifier used is Logistic Regression:

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectors, y_train)
```

Logistic Regression is well suited to this binary classification problem and provides a strong baseline for sparse text features.

---

### 6. Prediction

Predictions are generated for unseen test messages:

```python
predictions = model.predict(X_test_vectors)
```

The output follows:

```text
0 → Ham
1 → Spam
```

---

## 📈 Model Performance

The current notebook run achieved:

### **97.67% Accuracy**

```text
Accuracy: 0.9766816143497757
Accuracy percentage: 97.66816143497758%
```

The model correctly classified:

### **1,089 out of 1,115 test messages**

The confusion matrix was:

```text
[[966   0]
 [ 26 123]]
```

Interpreted as:

| Actual | Predicted | Count |
|---|---|---:|
| Ham | Ham | 966 |
| Ham | Spam | 0 |
| Spam | Ham | 26 |
| Spam | Spam | 123 |

### What this means

- **966 Ham messages** were correctly identified as Ham.
- **0 Ham messages** were classified as Spam in this test run.
- **26 Spam messages** were incorrectly classified as Ham.
- **123 Spam messages** were correctly identified as Spam.

> **Important:** Accuracy is dependent on the dataset, preprocessing, train/test split, and model configuration. The reported 97.67% is the result recorded by the current notebook run, not a universal guarantee of future performance.

---

## 🧩 Confusion Matrix

The project uses Scikit-learn's `ConfusionMatrixDisplay` to visualize classification results.

```python
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot()
plt.title("Confusion Matrix - SMS Spam Detection")
plt.show()
```

The confusion matrix helps distinguish between:

- **True Negatives:** Ham correctly classified as Ham
- **False Positives:** Ham incorrectly classified as Spam
- **False Negatives:** Spam incorrectly classified as Ham
- **True Positives:** Spam correctly classified as Spam

---

## 💬 Test Your Own SMS

The notebook supports interactive prediction for a new message:

```python
message = input("Enter an SMS message: ")

message_vector = vectorizer.transform([message])

prediction = model.predict(message_vector)[0]

if prediction == 1:
    print("Prediction: SPAM")
else:
    print("Prediction: HAM")
```

Example:

```text
Enter an SMS message: Congratulations! You have won a free prize. Call now!
Prediction: SPAM
```

This demonstrates how a trained NLP model can be applied to previously unseen text.

---

## 📁 Repository Structure

```text
SMS-Spam-detection-bootcamp-project/
│
├── 📓 SMS Spam detection bootcamp project.ipynb
│   └── Complete end-to-end ML/NLP workflow
│
├── 📄 spam.csv
│   └── SMS dataset used for training and evaluation
│
├── 📜 README.md
│   └── Project documentation
│
└── ⚖️ LICENSE
    └── MIT License
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have:

- Python 3.x
- Jupyter Notebook or JupyterLab
- pip

### 1. Clone the repository

```bash
git clone https://github.com/ashishraj-hub/SMS-Spam-detection-bootcamp-project.git
cd SMS-Spam-detection-bootcamp-project
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the notebook

Open:

```text
SMS Spam detection bootcamp project.ipynb
```

### 5. Run the notebook

Execute the cells from top to bottom.

---

## 📦 Minimal Requirements

If you prefer a `requirements.txt`, the project can use:

```text
pandas
numpy
matplotlib
scikit-learn
jupyter
```

---

## 🧪 Example Prediction Flow

```text
User enters SMS
       ↓
CountVectorizer
       ↓
Numerical feature vector
       ↓
Trained Logistic Regression
       ↓
┌───────────────┐
│  Prediction   │
├───────────────┤
│ HAM / SPAM    │
└───────────────┘
```

---

## 💡 Why This Project Matters

Although the implementation is intentionally lightweight, it demonstrates several important concepts used in real-world ML systems:

- Supervised learning
- Binary classification
- Natural Language Processing
- Feature engineering
- Train/test methodology
- Sparse text representations
- Model evaluation
- Error analysis
- User-facing inference

The project is therefore a useful foundation for progressing from **classical machine learning → production NLP → deep learning/transformer-based text classification**.

---

## ⚠️ Limitations

This project is designed as a focused classical NLP machine-learning implementation. It has several limitations:

1. **Bag-of-words representation**  
   CountVectorizer captures word occurrence but does not understand deeper semantic relationships.

2. **Dataset bias**  
   Model performance depends on the messages represented in the training dataset.

3. **Class imbalance**  
   Ham messages significantly outnumber Spam messages.

4. **No advanced text normalization pipeline**  
   The current implementation does not include a comprehensive tokenizer, stemming/lemmatization, spelling normalization, or sophisticated text cleaning pipeline.

5. **No production API/UI**  
   The current repository focuses on the notebook-based ML workflow rather than deployment.

6. **Accuracy alone is not sufficient**  
   For a production spam filter, precision, recall, F1-score, ROC-AUC, and cost of false negatives should also be considered.

---

## 🔮 Future Improvements

This project can be extended into a more production-oriented spam detection system.

### NLP Improvements

- TF-IDF features
- Word and character n-grams
- Better text normalization
- Stemming / lemmatization
- URL and phone-number feature extraction

### Model Improvements

Experiment with:

- Multinomial Naive Bayes
- Linear SVM
- Random Forest
- Gradient Boosting
- Ensemble methods

### Deep Learning / Modern NLP

A future version could explore:

- Word embeddings
- LSTM/GRU
- CNN-based text classification
- Transformer models
- BERT-style fine-tuning

### Production Improvements

- Build a **Streamlit** web interface
- Create a **FastAPI** inference API
- Save the trained vectorizer and model using `joblib`
- Add automated tests
- Add CI/CD with GitHub Actions
- Containerize with Docker
- Add model monitoring and data-drift checks
- Track experiments and model versions

---

## 📚 Learning Outcomes

By completing this project, you practice:

- [x] Python programming
- [x] Data loading with Pandas
- [x] Exploratory Data Analysis
- [x] Data visualization
- [x] Label encoding
- [x] Train/test splitting
- [x] NLP feature extraction
- [x] CountVectorizer
- [x] Logistic Regression
- [x] Binary classification
- [x] Model prediction
- [x] Accuracy evaluation
- [x] Confusion matrix analysis
- [x] Interactive inference

---

## 🧑‍💻 Skills Demonstrated

**Machine Learning:**  
`Supervised Learning` · `Binary Classification` · `Logistic Regression`

**NLP:**  
`Text Classification` · `Bag-of-Words` · `CountVectorizer` · `Stopword Removal`

**Data Science:**  
`Pandas` · `NumPy` · `EDA` · `Matplotlib`

**Model Evaluation:**  
`Accuracy` · `Confusion Matrix` · `Error Analysis`

**Development:**  
`Python` · `Jupyter Notebook` · `Git` · `GitHub`

---

## 📌 Project Highlights for Recruiters

> **Built an end-to-end SMS spam classification pipeline using Python and Scikit-learn, transforming raw text into 7,440 numerical features with CountVectorizer and training a Logistic Regression classifier that achieved 97.67% accuracy on a held-out 20% test set.**

The project also demonstrates:

- Proper train/test separation
- Sparse text feature engineering
- Reproducible splitting with `random_state=42`
- Class-distribution analysis
- Confusion-matrix-based evaluation
- Interactive inference on new SMS messages

---

## 📖 References & Documentation

- [Scikit-learn — CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [Scikit-learn — Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Scikit-learn — train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [Scikit-learn — Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/)

---

## ⭐ Support the Project

If you find this project useful:

- ⭐ Star the repository
- 🍴 Fork the project
- 💡 Open an issue with suggestions
- 🔧 Submit a pull request
- 📢 Share it with other learners and developers

---

## 📄 License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for details.

---

## 👤 Author

**Anshu Raj**

GitHub: [@anshuraj-hub](https://github.com/AnshuRaj1201)

---

<p align="center">
  <b>Built with Python • Scikit-learn • NLP • Machine Learning</b>
</p>

<p align="center">
  If this project helped you learn something, consider giving it a ⭐
</p>
