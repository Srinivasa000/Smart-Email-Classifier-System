# 📧 Smart Email Classifier System

A full production-style AI/ML system that automatically classifies incoming emails into **Category**, **Department**, and **Priority**, with a complete NLP pipeline, machine learning models, IMAP inbox integration, and a professional Streamlit UI.

This README is written in **full documentation style**, suitable for GitHub, interviews, academic submission, and corporate project portfolios.

---

# 🚀 Project Overview

Modern organizations process thousands of emails daily. Automatically routing and prioritizing these emails can dramatically reduce manual workload. This project demonstrates how to build an **end-to-end intelligent email classification system**, capable of:

* Detecting email **Category** (Spam / Ham / Promotions / Notification)
* Suggesting **Department Routing** (IT, HR, Finance, Sales, Support)
* Predicting **Priority Levels** (High / Medium / Low)
* Connecting directly to your **inbox (IMAP)**
* Providing an interactive **Streamlit user interface**

This project showcases your understanding of:

* NLP preprocessing
* ML model building
* Modular architecture
* UI engineering
* Email protocol integration

---

# ⭐ Features

## ✔ Multi-Target Classification

The system predicts **three separate outputs** for every incoming email:

1. **Category** → Spam, Ham, Notification, Promotional, etc.
2. **Department** → IT, HR, Finance, Sales, Support
3. **Priority** → High, Medium, Low

Each output includes:

* Predicted label
* Confidence score
* Full probability breakdown
* Visualization via pie chart

---

## ✔ NLP Pipeline

Includes complete text processing flows:

* Lowercasing
* URL and email removal
* Special character cleaning
* Stopword removal
* Tokenization
* Stemming (PorterStemmer)
* TF-IDF vectorization (1–2 grams)
* Meta-feature extraction:

  * subject length
  * body length
  * exclamation marks count

---

## ✔ Machine Learning

| Component               | Algorithm                  |
| ----------------------- | -------------------------- |
| Category classification | Multinomial Naive Bayes    |
| Department routing      | Logistic Regression        |
| Priority prediction     | Logistic Regression        |
| Feature extraction      | TF-IDF 5000 features       |
| Class imbalance         | Random upsampling          |
| Extra features          | Sparse meta-feature vector |

The pipeline is **modular and scalable**, enabling easy upgrades to more advanced models like SVMs or Transformers.

---

## ✔ Streamlit Web Application

The UI is designed with a **clean, responsive layout**, including:

### 🔍 Classifier Page

* Enter Subject + Body
* Get predictions instantly
* View classification cards
* View pie-chart probability distributions

### 📥 Inbox Page (IMAP Integration)

* Enter email credentials (App Password recommended)
* Connect to Gmail, Outlook, Yahoo, Office365
* Fetch last 20 real inbox emails
* Auto-classifies each email
* Shows subject, preview, predictions, and charts

### ℹ About Page

* Project explanation
* Features summary
* Architecture diagram

---

## ✔ IMAP Email Integration

Supports any email provider that supports IMAP:

| Provider  | IMAP Server           |
| --------- | --------------------- |
| Gmail     | imap.gmail.com        |
| Outlook   | imap-mail.outlook.com |
| Office365 | outlook.office365.com |
| Yahoo     | imap.mail.yahoo.com   |

Features:

* Fetch last **20 emails**
* Auto-classify them
* Display clean preview UI
* (Optional) Move emails to folder

---

# 🏛 System Architecture

```
Raw Email (Subject + Body)
            ↓
     Preprocessing
(clean → tokenize → stem)
            ↓
     Feature Engineering
 (TF-IDF + Meta Features)
            ↓
  ----------------------------
  | Category Model (NB)     |
  | Department Model (LR)   |
  | Priority Model (LR)     |
  ----------------------------
            ↓
  Predictions + Confidence
            ↓
   Streamlit Web Interface
            ↓
   (Optional) IMAP Inbox
```

---

# 📁 Project Structure

```
smart-email-classifier/
│── README.md
│── requirements.txt
│── app.py
│── sample_data/
│   └── email_dataset_100.csv
│── models/
│   ├── vectorizer.joblib
│   ├── models.joblib
│   └── encoders.joblib
│── src/
    ├── data_preprocessing.py
    ├── feature_engineering.py
    ├── model.py
    ├── train.py
    ├── predict.py
    └── imap_reader.py
```

---

# ⚙️ Installation

### 1️⃣ Create and activate virtual environment

```
python -m venv .venv
.\.venv\Scripts\activate
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

### 3️⃣ Train the model

```
python src/train.py --data sample_data/email_dataset_100.csv --out_dir models
```

### 4️⃣ Run prediction (CLI)

```
python src/predict.py --subject "Server down" --body "Database not responding"
```

### 5️⃣ Launch the Streamlit App

```
streamlit run app.py
```

---

# 📬 IMAP Setup Guide

### Gmail Example

```
Username: your_email@gmail.com
Password: <App Password>
IMAP Server: imap.gmail.com
Port: 993
```

💡 *Gmail requires an App Password. Normal passwords do not work for IMAP.*

---

# 🎯 Skills Demonstrated

### Machine Learning

* Multi-output classification
* NLP preprocessing
* TF-IDF feature extraction
* Probability-based inference
* Resampling techniques

### Software Engineering

* Modular Python architecture
* Handling external services (IMAP)
* Error management
* Model serialization
* Environment variable usage (.env)

### UI/UX Development

* Streamlit layout design
* Responsive columns/cards
* Altair chart visualizations

### System Design Thinking

* End-to-end ML workflow design
* Real-world email processing
* Scalability considerations

---

# 🔮 Future Enhancements

* Transformer-based models (BERT)
* Dark mode UI
* Detailed analytics dashboard
* Auto-routing emails to folders
* HTML email parsing
* Larger dataset integration

---

# 🏁 Conclusion

This **Smart Email Classifier System** is a complete working AI solution demonstrating:

* Practical NLP engineering
* Multi-target ML modelling
* Real email integration
* Clean and modular software design
* Professional interactive interface


---

**Author:** *Srinivasa Rao Maddikunta*
**Project:** Smart Email Classifier System
**Technology Stack:** Python, Scikit-learn, Streamlit, IMAP, NLP
