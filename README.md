# SMS SHIELD 2.0

> **Intelligent SMS Spam Detection & NLP Analytics Platform**  
> *A Production-Grade Natural Language Processing and Machine Learning System for Academic Evaluation and Real-Time SMS Classification.*

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement & Objectives](#2-problem-statement--objectives)
3. [Technology Stack](#3-technology-stack)
4. [Dataset Details & Preprocessing](#4-dataset-details--preprocessing)
5. [System Architecture](#5-system-architecture)
6. [Machine Learning Algorithms & Methodology](#6-machine-learning-algorithms--methodology)
7. [Experimental Results & Comparative Analysis](#7-experimental-results--comparative-analysis)
8. [Installation & Setup Instructions](#8-installation--setup-instructions)
9. [Execution Guide](#9-execution-guide)
10. [Streamlit Multi-Page Dashboard Tour](#10-streamlit-multi-page-dashboard-tour)
11. [College Presentation & Demonstration Guide](#11-college-presentation--demonstration-guide)
12. [System Limitations & Future Enhancements](#12-system-limitations--future-enhancements)

---

## 1. Project Overview

**SMS Shield 2.0** is an upgraded, locally executable Natural Language Processing (NLP) analytics and classification platform designed to detect spam, phishing, and fraudulent SMS messages.

The system combines:
- A rigorous **5-stage text preprocessing pipeline** (lowercasing, punctuation stripping, tokenization, stopword filtering, Porter stemming).
- A **three-model machine learning architecture** featuring **Multinomial Naive Bayes**, **Logistic Regression**, and **Linear Support Vector Machine (LinearSVC)** with Platt probability calibration.
- **Explainable AI capabilities** providing per-token TF-IDF weight contributions, global coefficient inspection, and heuristic signal extraction (URLs, phone shortcodes, currency symbols, all-caps ratio).
- A **five-page interactive cybersecurity-themed Streamlit dashboard** featuring single-message analysis, step-by-step NLP inspection, diagnostic curves (ROC, Precision-Recall), and bulk CSV processing.

---

## 2. Problem Statement & Objectives

### Problem Statement
Mobile messaging is bombarded by unsolicited commercial advertisements, phishing attacks (smishing), and financial lottery scams. In SMS filtering, the operational cost of classification errors is highly asymmetric:
- **False Positive (Type I Error):** Diverting or dropping a critical legitimate message (e.g., banking OTP, two-factor authentication code, flight change notification, emergency alert).
- **False Negative (Type II Error):** Letting a spam message slip into the user's inbox (minor annoyance or potential phishing risk).

Therefore, an optimal SMS classification system must maximize **Precision** on legitimate messages while achieving competitive **Recall** on spam.

### Objectives of SMS Shield 2.0
1. **Prevent Data Leakage:** Ensure TF-IDF feature extraction is fitted strictly on the training set, eliminating information leakage.
2. **Three-Model Benchmarking:** Benchmark Multinomial Naive Bayes, Logistic Regression, and Linear SVM across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
3. **Multi-Model Consensus:** Implement ensemble consensus voting (Unanimous 3/3 vs. Majority 2/1).
4. **Feature Representation Experiments:** Compare unigram, bigram, and trigram TF-IDF configurations.
5. **Bulk CSV Processing & Export:** Enable enterprise-style batch CSV classification with filtering and downloadable reports.
6. **100% Local Execution:** Operate completely offline without paid APIs, external cloud databases, or unnecessary deep learning overhead.

---

## 3. Technology Stack

- **Programming Language:** Python 3.8+ (tested on Python 3.11)
- **Natural Language Processing:** NLTK 3.8+ (Tokenization, Stopwords, Porter Stemmer)
- **Machine Learning:** scikit-learn 1.3+ (MultinomialNB, LogisticRegression, LinearSVC, CalibratedClassifierCV, TfidfVectorizer, Pipeline, Metrics)
- **Data Manipulation:** pandas & NumPy
- **Data Visualization:** Matplotlib & Seaborn
- **User Interface:** Streamlit (Multi-Page Architecture)
- **Model Serialization:** Joblib

---

## 4. Dataset Details & Preprocessing

The project utilizes the **UCI SMS Spam Collection**:
- **Dataset Source:** [UCI Machine Learning Repository — SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **File Location:** `data/SMSSpamCollection`
- **Raw Records:** 5,574 messages (tab-separated)
- **Deduplication:** 403 exact duplicate messages removed
- **Cleaned Dataset:** 5,171 unique messages
  - **Legitimate (Ham):** 4,518 (87.37%)
  - **Spam:** 653 (12.63%)
- **Class Imbalance:** ~7:1 Ham-to-Spam ratio, necessitating **stratified train/test splitting** (`random_state=42`).

### 5-Step NLP Preprocessing Pipeline

Implemented in [`src/preprocessing.py`](file:///Users/arnavpotale/Code/nlp/src/preprocessing.py):

```
Raw SMS Text
     │
     ▼
[Step 1: Lowercase Conversion]   ── Standardizes casing ("FREE" -> "free")
     │
     ▼
[Step 2: Punctuation Removal]    ── Strips symbols; inserts space to preserve boundaries ("call:123" -> "call 123")
     │
     ▼
[Step 3: Word Tokenization]      ── Uses NLTK word_tokenize to produce discrete word tokens
     │
     ▼
[Step 4: Stopword Filtering]     ── Removes 179+ standard English stopwords ("the", "is", "at", "you")
     │
     ▼
[Step 5: Porter Stemming]        ── Algorithmic suffix stripping ("winning" -> "win", "awarded" -> "award")
     │
     ▼
Clean Stemmed Token String
```

---

## 5. System Architecture

```
nlp/
├── app.py                      # Main Welcome Hub & Architecture Directory
├── requirements.txt            # Environment package requirements
├── README.md                   # Comprehensive documentation & presentation guide
├── test_system.py              # Automated 9-checkpoint validation suite
│
├── pages/
│   ├── 1_Dashboard.py          # Page 1: Global metrics & class distributions
│   ├── 2_Message_Analyzer.py   # Page 2: 3-model live inference & consensus
│   ├── 3_NLP_Analysis.py       # Page 3: Step-by-step pipeline & n-gram distributions
│   ├── 4_Model_Evaluation.py   # Page 4: ROC/PR curves, CV & n-gram experiments
│   └── 5_Bulk_Analysis.py      # Page 5: Bulk CSV upload, processing & report export
│
├── src/
│   ├── __init__.py             # Python package marker
│   ├── preprocessing.py        # 5-step NLP pipeline
│   ├── train.py                # End-to-end training & artifact export
│   ├── evaluate.py             # ROC, PR curves, confusion matrices, evaluation
│   ├── prediction.py           # Unified inference engine for all 3 models
│   ├── explainability.py       # Token contributions & heuristic signal extraction
│   ├── bulk_analysis.py        # Batch CSV processing & sample generator
│   └── ui_components.py        # Shared cybersecurity styling & cached loaders
│
├── data/
│   ├── SMSSpamCollection       # UCI dataset
│   ├── readme                  # UCI dataset readme
│   └── sample_bulk_sms.csv     # 25-message test file for bulk classification
│
├── models/
│   ├── tfidf_vectorizer.joblib # TF-IDF vectorizer (max 5,000 features)
│   ├── multinomial_nb.joblib   # Trained Naive Bayes model
│   ├── logistic_regression.joblib # Trained Logistic Regression model
│   ├── linear_svc.joblib       # Trained Linear SVM model
│   └── calibrated_linear_svc.joblib # Calibrated SVM for probability estimation
│
├── metrics/
│   ├── evaluation_metrics.csv  # Actual empirical test results
│   ├── baseline_metrics.csv    # Historical baseline results preserved
│   ├── tfidf_experiments.csv   # N-gram & feature capacity experiments
│   ├── cross_val_metrics.csv   # 5-Fold Stratified Cross-Validation records
│   ├── classification_report_*.csv # Per-class reports
│   ├── confusion_matrix_*.png  # Confusion matrix heatmaps
│   ├── roc_curves.png          # Diagnostic ROC curves (all 3 models)
│   ├── pr_curves.png           # Diagnostic Precision-Recall curves
│   └── model_comparison_bar.png# Comparative bar chart
│
└── exports/                    # Directory for downloaded bulk classification reports
```

---

## 6. Machine Learning Algorithms & Methodology

### 1. Multinomial Naive Bayes (`MultinomialNB`)
- Uses Bayes' Theorem under conditional term independence:
  $$\hat{y} = \arg\max_{c \in \{0, 1\}} P(c) \prod_{i=1}^{n} P(w_i \mid c)$$
- Uses Laplace smoothing ($\alpha = 1.0$) to prevent zero-probability terms.
- **Strengths:** Outstanding precision on sparse text data; minimal false positive rate.

### 2. Logistic Regression (`LogisticRegression`)
- Fits a linear decision boundary using the sigmoid link function:
  $$P(y = 1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- Uses L2 ridge regularization (`random_state=42`, `max_iter=1000`) to prevent overfitting.
- **Strengths:** Highly calibrated probabilities and transparent feature coefficients.

### 3. Linear Support Vector Machine (`LinearSVC`)
- Maximizes the geometric margin separating the hyperplanes:
  $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^m \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$$
- Uses `CalibratedClassifierCV` (Platt scaling with 5-fold internal CV) to estimate calibrated probabilities while retaining the optimal margin.
- **Strengths:** Superior recall and boundary separation in high-dimensional text space.

---

## 7. Experimental Results & Comparative Analysis

All models were evaluated on the exact held-out test split of **1,035 messages** (904 legitimate Ham, 131 Spam). Results are recorded in [`metrics/evaluation_metrics.csv`](file:///Users/arnavpotale/Code/nlp/metrics/evaluation_metrics.csv):

### Empirical Test Set Results (1,035 Samples)

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-Score | ROC-AUC | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
|---|---|---|---|---|---|---|---|---|
| **Multinomial Naive Bayes** | 96.71% | **98.99%** | 74.81% | 85.22% | 0.9830 | **903** | **1** | 33 | 98 |
| **Logistic Regression** | 96.33% | 96.97% | 73.28% | 83.48% | 0.9900 | 901 | 3 | 35 | 96 |
| **Linear SVM** | **98.16%** | 98.28% | **87.02%** | **92.31%** | **0.9935** | 902 | 2 | **17** | **114** |

### Confusion Matrix Comparisons
- **Naive Bayes:** 903 TN, **1 FP**, 33 FN, 98 TP (Only 1 false alarm across 904 legitimate messages!)
- **Logistic Regression:** 901 TN, 3 FP, 35 FN, 96 TP
- **Linear SVM:** 902 TN, 2 FP, **17 FN**, **114 TP** (Caught 16 more spam messages than NB, cutting missed spam in half!)

### 5-Fold Stratified Cross-Validation

Recorded in [`metrics/cross_val_metrics.csv`](file:///Users/arnavpotale/Code/nlp/metrics/cross_val_metrics.csv):

| Model | Mean CV Accuracy | Mean CV Precision | Mean CV Recall | Mean CV F1-Score |
|---|---|---|---|---|
| **Multinomial Naive Bayes** | 96.64% ± 0.28% | **99.79% ± 0.42%** | 73.51% ± 2.05% | 84.64% ± 1.40% |
| **Logistic Regression** | 96.36% ± 0.38% | 96.99% ± 1.66% | 73.51% ± 2.34% | 83.61% ± 1.78% |
| **Linear SVM** | **98.26% ± 0.50%** | 97.62% ± 1.36% | **88.36% ± 3.21%** | **92.74% ± 2.18%** |

### TF-IDF Feature Representation Experiments

Recorded in [`metrics/tfidf_experiments.csv`](file:///Users/arnavpotale/Code/nlp/metrics/tfidf_experiments.csv):

| Configuration | Features / Range | Naive Bayes F1 | Logistic Regression F1 | Linear SVM F1 | Key Finding |
|---|---|---|---|---|---|
| **Config A** | Unigrams (max=5000) | 85.22% | 83.48% | 92.31% | Clean baseline; robust generalization. |
| **Config B** | Unigram + Bigram (max=5000) | **86.58%** | 82.97% | **92.37%** | Bigrams like `'claim call'`, `'cash prize'` boost precision and recall. |
| **Config C** | Unigram + Bigram + Trigram (max=10000) | 80.91% | 81.74% | 91.63% | **Performance degraded!** Short messages make trigrams sparse, leading to parameter dilution. |

---

## 8. Installation & Setup Instructions

### Prerequisites
- Python **3.8 or higher** (tested with Python 3.11.9)
- Operating System: macOS, Linux, or Windows

### Step 1: Clone or Navigate to the Workspace
```bash
cd /Users/arnavpotale/Code/nlp
```

### Step 2: Activate the Virtual Environment
On macOS / Linux:
```bash
source venv/bin/activate
```
On Windows:
```cmd
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 9. Execution Guide

### 1. Launch the Streamlit Multi-Page Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run the Verification & Test Suite
To verify all 9 system components:
```bash
python test_system.py
```

### 3. Retrain All Models & Regenerate Artifacts
To execute the complete training and experimental pipeline:
```bash
python src/train.py
```

---

## 10. Streamlit Multi-Page Dashboard Tour

Navigate via the sidebar among the five specialized pages:

### 📊 Page 1: Dashboard (`pages/1_Dashboard.py`)
- KPI summary cards (Total messages, spam count, legitimate count, active models, peak F1, peak precision).
- Deduplicated class frequency and percentage ratio charts.
- SMS character length distribution (Ham median ~52 chars vs. Spam median ~149 chars).
- Comparative performance grouped bar chart and quantitative metrics table.

### 🔍 Page 2: Message Analyzer (`pages/2_Message_Analyzer.py`)
- Interactive text area with quick-fill presets (Lottery Scam, Bank Phish, Friendly Lunch, Casual Chat).
- Unified consensus banner (Unanimous 3/3 vs. Majority 2/1).
- Three individual model verdict cards with visual probability progress bars.
- Heuristic indicator extraction (promotional keywords, phone shortcodes, URLs, currencies, ALL-CAPS).
- Token TF-IDF contribution table showing directional feature weights.

### 🔬 Page 3: NLP Analysis (`pages/3_NLP_Analysis.py`)
- Step-by-step 7-stage NLP pipeline trace for any custom input message.
- Pedagogical spotlight on trigger keywords (`FREE`, `WINNER`, `CLAIM`, `URGENT`, `PRIZE`, `CASH`).
- Dataset-wide N-gram tabs comparing top Unigrams, Bigrams, and Trigrams for Ham vs. Spam.
- Global top 10 influential TF-IDF features driving Spam and Ham predictions.

### 📈 Page 4: Model Evaluation (`pages/4_Model_Evaluation.py`)
- Quantitative test set evaluation table across all three algorithms.
- Side-by-side Confusion Matrix heatmaps with sample counts and percentages.
- Receiver Operating Characteristic (ROC) curves with AUC scores.
- Precision-Recall curves focusing on the minority spam class.
- 5-Fold Stratified Cross-Validation summary table.
- Experimental analysis of TF-IDF feature configurations (Configs A, B, C).

### 📁 Page 5: Bulk SMS Analysis (`pages/5_Bulk_Analysis.py`)
- Drag-and-drop CSV batch upload or one-click loading of the pre-packaged 25-message test file.
- Column mapping selector.
- Batch processing with progress bar.
- Summary metrics and agreement distribution charts.
- Interactive filtering (by consensus class, by model disagreement, or keyword search).
- One-click downloadable CSV classification report.

---

## 11. College Presentation & Demonstration Guide

Follow this structured 6-step walkthrough for an undergraduate NLP project demonstration:

### Step 1: Context & Problem Framing (1 minute)
- **What to say:** *"In mobile telecommunications, spam classification has asymmetric error costs. A false positive blocks an OTP or doctor appointment notification, while a false negative is merely an unwanted text. Therefore, our primary technical objective in SMS Shield 2.0 is maximizing legitimate precision while optimizing spam recall."*

### Step 2: Architecture & Zero Data Leakage (1 minute)
- **What to show:** Open [`src/train.py`](file:///Users/arnavpotale/Code/nlp/src/train.py#L182-L195).
- **What to say:** *"We implemented a stratified 80/20 train/test split. Notice that our TF-IDF vectorizer is fitted strictly on the 4,136 training messages. The test split is transformed only, guaranteeing zero data leakage."*

### Step 3: Live Message Analyzer Demonstration (2 minutes)
- **What to show:** Navigate to **Page 2: Message Analyzer**.
- **Action:** Click the preset button **"🎁 Lottery Scam (Spam)"**.
- **What to say:** *"Notice that all three models—Multinomial Naive Bayes, Logistic Regression, and Linear SVM—reach a unanimous 3/3 consensus for SPAM. The system extracts heuristic indicators such as phone shortcodes, currency symbols, and promotional keywords, while showing the exact TF-IDF weight contribution of each token."*
- **Action:** Next, click **"☕ Friendly Lunch (Ham)"** to demonstrate unanimous HAM consensus.

### Step 4: Explainable NLP Pipeline Walkthrough (1.5 minutes)
- **What to show:** Navigate to **Page 3: NLP Analysis**.
- **What to say:** *"Here we illustrate the 7-stage NLP pipeline. We can trace how raw text is lowercased, stripped of punctuation, tokenized, filtered through English stopwords, and stemmed using the Porter Stemmer. In the N-Gram tabs, notice how bigrams like 'claim call' and 'co uk' provide critical multi-word context that unigrams cannot capture."*

### Step 5: Comparative Evaluation & Engineering Insights (2 minutes)
- **What to show:** Navigate to **Page 4: Model Evaluation**.
- **What to say:** *"Here are the experimental results. Naive Bayes achieved 98.99% precision with only 1 false alarm out of 904 legitimate messages. However, Linear SVM achieved an 87.02% recall, capturing 114 out of 131 spam messages, giving it the highest overall F1-score of 92.31%. Furthermore, in our TF-IDF experiments, adding trigrams degraded performance from 86.5% to 80.9%, proving that excessive feature dimensionality on short texts causes term sparsity."*

### Step 6: Bulk CSV Processing & Export (1.5 minutes)
- **What to show:** Navigate to **Page 5: Bulk SMS Analysis**.
- **Action:** Click **"🚀 Load Pre-Packaged Sample Dataset"** and then **"🚀 Run 3-Model Batch Classification"**.
- **What to say:** *"In enterprise deployments, administrators process batches of messages. Our bulk module classifies all rows without silently dropping empty messages, computes model agreement rates, and exports downloadable CSV reports."*

### Frequently Asked Viva / Oral Examination Questions
1. **Q: Why is Laplace smoothing necessary in Multinomial Naive Bayes?**  
   *A:* If a word in the test SMS never appeared in the training spam vocabulary, $P(w_i \mid \text{Spam}) = 0$. Because Naive Bayes multiplies conditional probabilities, a single zero would set the entire spam probability to zero. Laplace smoothing ($\alpha = 1.0$) adds 1 to all counts, ensuring non-zero probabilities.
2. **Q: Why did Linear SVM outperform Logistic Regression in Recall?**  
   *A:* Linear SVM optimizes the maximum-margin hyperplane focusing on support vectors near the decision boundary, whereas Logistic Regression minimizes log-loss across all points. On sparse TF-IDF vectors, the margin-maximizing property of SVM isolates borderline promotional messages more effectively.
3. **Q: Why did adding trigrams decrease F1-score in Config C?**  
   *A:* SMS messages average only 15 words. Specific 3-word combinations are extremely rare, causing feature vectors to become excessively sparse (10,000 features). Without sufficient training frequency, weights undergo over-regularization, diluting discriminative unigram/bigram signals.

---

## 12. System Limitations & Future Enhancements

### Current System Limitations
1. **Adversarial Obfuscation & Leetspeak:** Scammers frequently bypass lexical filters using substituted characters (e.g., `"Fr33"`, `"W!NNER"`, `"C@SH"`), which standard stemming cannot normalize.
2. **Short SMS Acronyms:** Highly colloquial abbreviations (`"idk"`, `"omw"`, `"atm"`) are not part of standard NLTK stemmer dictionaries.
3. **Class Imbalance Sensitivity:** While stratified splitting mitigates imbalance, extreme class skew (~87% Ham vs ~13% Spam) naturally limits raw recall without threshold tuning.

### Future Enhancements
1. **Character-Level N-Gram Embeddings:** Implementing sub-word or character n-grams (FastText) to identify leetspeak and typo variations.
2. **Transformer Fine-Tuning:** Fine-tuning lightweight transformer models (DistilBERT or MobileBERT) for deep contextual sentence embeddings.
3. **Adaptive Threshold Calibration:** Dynamic decision threshold tuning based on user-defined operational risk tolerances (e.g., zero false-positive constraints).
4. **Multilingual Support:** Extending tokenization and stopword filters to Hindi, Hinglish, Spanish, and regional dialects.
