# SMS Shield — NLP-Based SMS Spam Classification

> **Project Title:** Implementation and Comparative Analysis of NLP-Based SMS Spam Classification Techniques  
> **Course / Mini-Project:** Natural Language Processing (NLP) & Machine Learning  
> **Stack:** Python, NLTK, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, Streamlit, Joblib

---

## 1. Project Objective

The objective of **SMS Shield** is to build, evaluate, and compare two supervised machine learning algorithms—**Multinomial Naive Bayes** and **Logistic Regression**—for binary SMS spam classification (legitimate message `ham` vs. unwanted/fraudulent message `spam`).

The system implements a rigorous NLP preprocessing pipeline, ensures zero data leakage during feature extraction, trains both models, logs actual evaluation metrics and confusion matrices, and presents an interactive **Streamlit** dashboard for real-time inference and analysis.

---

## 2. Dataset Overview

The project uses the **UCI SMS Spam Collection**:
- **Source:** [UCI Machine Learning Repository — SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **File Location:** `data/SMSSpamCollection`
- **Initial Size:** 5,574 raw records (tab-separated: label and message)
- **Deduplication:** 403 duplicate messages identified and dropped
- **Cleaned Dataset:** 5,171 unique messages
  - **Ham (Legitimate):** 4,518 (87.37%)
  - **Spam (Unwanted):** 653 (12.63%)
- **Class Imbalance:** Highly skewed towards ham (~7:1 ratio), demonstrating why **stratified splitting** and **precision/recall/F1-score** are essential metrics rather than accuracy alone.

---

## 3. NLP Preprocessing Pipeline

Every message is transformed via a consistent 5-stage NLP pipeline implemented in [`src/preprocessing.py`](file:///Users/arnavpotale/Code/nlp/src/preprocessing.py):

```
Raw Text ──> [1. Lowercasing] ──> [2. Punctuation Removal] ──> [3. Tokenization] ──> [4. Stopword Removal] ──> [5. Porter Stemming] ──> Cleaned Tokens
```

1. **Lowercase Conversion:** Standardizes capitalization so that words like `"Free"`, `"FREE"`, and `"free"` map to the same token.
2. **Punctuation Removal:** Strips ASCII/Unicode punctuation characters while replacing them with whitespace to avoid accidentally concatenating adjacent words (e.g., `"call:09061701461"` -> `"call 09061701461"`).
3. **Tokenization:** Splits normalized text into individual word tokens using NLTK's `word_tokenize`.
4. **Stopword Removal:** Eliminates high-frequency, non-discriminative English function words (e.g., `"the"`, `"is"`, `"at"`, `"which"`, `"on"`) using NLTK's English stopword corpus.
5. **Porter Stemming:** Employs the algorithmic Porter Stemmer to reduce morphological variants to their root base (e.g., `"winning"` -> `"win"`, `"awarded"` -> `"award"`).

### Data Leakage Prevention
To strictly prevent data leakage:
- The dataset is split into **80% training (4,136 samples)** and **20% testing (1,035 samples)** using stratified sampling (`random_state=42`).
- **TF-IDF Vectorization** (`max_features=5000`) is **fitted exclusively on the training split** (`fit_transform`).
- The held-out test split is transformed using the fitted vocabulary and IDF weights (`transform`), preventing any test set distribution information from leaking into the training phase.

---

## 4. Machine Learning Algorithms

### Model 1: Multinomial Naive Bayes (`MultinomialNB`)
- Applies Bayes' Theorem with the naive assumption of conditional independence between word tokens given the class label:
  $$\hat{y} = \arg\max_{c} P(c) \prod_{i=1}^{n} P(w_i \mid c)$$
- Uses Laplace smoothing ($\alpha = 1.0$) to avoid zero probabilities for unseen terms.
- Highly effective for discrete text counts and TF-IDF features with fast training and low variance.

### Model 2: Logistic Regression (`LogisticRegression`)
- Fits a linear decision boundary using the sigmoid link function:
  $$P(y = 1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- Optimized with L2 regularization (`random_state=42`, `max_iter=1000`) to prevent overfitting across high-dimensional TF-IDF sparse vectors.

---

## 5. Experimental Results & Comparative Analysis

The models were evaluated on the held-out test split of **1,035 messages** (904 Ham, 131 Spam). Results are saved in [`metrics/evaluation_metrics.csv`](file:///Users/arnavpotale/Code/nlp/metrics/evaluation_metrics.csv):

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
|---|---|---|---|---|---|---|---|---|
| **Multinomial Naive Bayes** | **96.71%** | **98.99%** | **74.81%** | **85.22%** | **903** | **1** | 33 | **98** |
| **Logistic Regression** | 96.33% | 96.97% | 73.28% | 83.48% | 901 | 3 | 35 | 96 |

### Confusion Matrix Breakdown

```
Multinomial Naive Bayes (Test Set: 1,035)
               Predicted Ham     Predicted Spam
Actual Ham          903 (TN)             1 (FP)  <-- Only 1 false alarm!
Actual Spam          33 (FN)            98 (TP)

Logistic Regression (Test Set: 1,035)
               Predicted Ham     Predicted Spam
Actual Ham          901 (TN)             3 (FP)
Actual Spam          35 (FN)            96 (TP)
```

### Key Engineering & Operational Takeaways
1. **Precision is Paramount in Spam Detection:**
   - In SMS filtering, a **False Positive (FP)** means a legitimate SMS (such as a bank transaction OTP, flight alert, or doctor appointment confirmation) is incorrectly diverted or blocked.
   - A **False Negative (FN)** simply means an unwanted spam message reaches the inbox.
   - Therefore, a false positive is significantly more damaging to user experience than a false negative. **Multinomial Naive Bayes** demonstrated outstanding precision of **98.99%**, causing only **1 single false positive** across 904 legitimate test messages.
2. **Confidence Score Transparency:**
   - Model confidence values represent raw statistical output probabilities from TF-IDF feature distributions and prior probabilities, **not calibrated real-world risk scores**. The application explicitly labels these as model probabilities to maintain technical clarity.

---

## 6. Project Structure

```
nlp/
├── app.py                      # Main Streamlit interactive dashboard
├── requirements.txt            # Python package dependencies
├── test_system.py              # Automated test suite for verification
├── README.md                   # Complete documentation and user guide
├── src/
│   ├── __init__.py             # Python package marker
│   ├── preprocessing.py        # 5-step NLP preprocessing pipeline
│   ├── train.py                # End-to-end training, evaluation & saving pipeline
│   └── evaluate.py             # Evaluation metrics and plotting functions
├── data/
│   ├── SMSSpamCollection       # Raw UCI dataset (tab-separated)
│   └── readme                  # UCI dataset documentation
├── models/
│   ├── tfidf_vectorizer.joblib # Persisted TF-IDF vectorizer (5,000 features)
│   ├── multinomial_nb.joblib   # Persisted MultinomialNB model
│   └── logistic_regression.joblib # Persisted Logistic Regression model
├── metrics/
│   ├── evaluation_metrics.csv  # Actual recorded evaluation metrics
│   ├── classification_report_nb.csv # Full per-class report for Naive Bayes
│   ├── classification_report_lr.csv # Full per-class report for Logistic Regression
│   ├── confusion_matrix_nb.png # Seaborn confusion matrix heatmap (NB)
│   ├── confusion_matrix_lr.png # Seaborn confusion matrix heatmap (LR)
│   ├── model_comparison_bar.png# Comparative metric bar chart
│   └── class_distribution.png  # Ham vs. Spam distribution plots
└── nltk_data/                  # Local NLTK corpora (punkt, punkt_tab, stopwords)
```

---

## 7. Installation & Setup Instructions

### Prerequisites
- Python **3.8 or higher** (tested with Python 3.11)
- Standard Unix/macOS or Windows terminal

### Step 1: Clone or Navigate to the Project Directory
```bash
cd /path/to/nlp
```

### Step 2: Create and Activate a Virtual Environment
On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
On Windows:
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 8. Execution Instructions

### A. Run Training & Evaluation Pipeline
To retrain the models from the dataset, fit the vectorizer, and regenerate all metrics and plots:
```bash
python src/train.py
```
*Output:*
- Prints dataset statistics and deduplication counts.
- Trains both models with zero data leakage.
- Saves models to `models/` using Joblib.
- Saves evaluation CSV and confusion matrix plots to `metrics/`.

### B. Launch the Streamlit Interactive Dashboard
Start the dashboard application locally:
```bash
streamlit run app.py
```
Open your browser and navigate to:
```
http://localhost:8501
```

### C. Run the Verification Test Suite
Verify that all components (preprocessing, inference, metric persistence) operate reliably:
```bash
python test_system.py
```

---

## 9. Streamlit Dashboard Features

The dashboard includes four comprehensive sections:

1. **📊 Section 1: Dataset Overview & Class Distribution**
   - KPI metrics: Total raw records, duplicate messages dropped, unique records, ham/spam split.
   - Deduplicated class frequency chart and percentage ratio.
   - Character count and message length distribution analysis (Ham vs Spam).
   - Interactive sample message browser with filtering.

2. **🔍 Section 2: SMS Message Prediction (Both Models)**
   - Text area for custom message input with 4 instant preset sample buttons (Lottery Scam, Friendly Chat, Urgent Alert, Casual Chat).
   - Handles empty input, whitespace, and punctuation-only messages gracefully.
   - Step-by-step NLP transformation inspector displaying raw text, lowercasing, punctuation stripping, tokenization, stopword filtering, and Porter stemming.
   - Side-by-side model verdict cards (Ham/Spam badges, visual probability progress bars).
   - Clear consensus / divergence alerts.
   - Explicit disclaimer explaining model probabilities vs calibrated real-world risk.

3. **📈 Section 3: Performance Comparison Table and Bar Chart**
   - Quantitative evaluation table dynamically loaded from `metrics/evaluation_metrics.csv` (Accuracy, Precision, Recall, F1-Score).
   - Grouped bar chart comparing Multinomial Naive Bayes and Logistic Regression across all four metrics.
   - Analytical breakdown explaining operational trade-offs in SMS filtering.

4. **🔲 Section 4: Confusion Matrices and Classification Reports**
   - Side-by-side confusion matrix heatmaps with raw counts and percentages.
   - Per-class classification reports (Ham & Spam Precision, Recall, F1-score, Support).
   - Detailed conceptual explanation of Type I errors (False Positives) and Type II errors (False Negatives).

---

## 10. Constraints & Compliance
- **100% Local Execution:** No cloud services, external LLMs, paid APIs, authentication, or databases are required.
- **Robustness:** Includes local NLTK corpora fallback, defensive input handling, and clear error notifications if dataset or model artifacts are missing.
