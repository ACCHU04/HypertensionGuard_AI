# PulseGuard AI — Intelligent Blood Pressure Prediction System
### SmartBridge Internship Project | AI & Machine Learning Track

---

## 📌 Project Overview
An advanced ML-powered web application that predicts and classifies hypertension stages
(Normal, Stage-1, Stage-2, Hypertensive Crisis) using patient clinical data and lifestyle parameters.

---

## 🗂 Project Structure
```
HYPERTENSION_PREDICTION/
├── static/
│   └── style.css              # Additional CSS styles
├── templates/
│   └── index.html             # Flask HTML frontend
├── app.py                     # Flask backend application
├── train_model.py             # ML training script (Milestones 1-4)
├── logreg_model.pkl           # Trained Logistic Regression model
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## ⚙️ Setup & Installation

### Step 1: Install Anaconda Navigator
- Download from: https://youtu.be/1ra4zH2G4o0

### Step 2: Install Dependencies
Open Anaconda Prompt as Administrator and run:
```bash
pip install flask numpy pandas scikit-learn matplotlib seaborn joblib scipy
```
Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model (Optional — model already included)
```bash
python train_model.py
```

### Step 4: Run the Web Application
```bash
python app.py
```

### Step 5: Open Browser
Navigate to: http://localhost:5000

---

## 🧠 ML Algorithms Compared

| Algorithm           | Accuracy | Status      |
|---------------------|----------|-------------|
| Decision Tree       | 100%     | ❌ Overfitted |
| Random Forest       | 100%     | ❌ Overfitted |
| SVM                 | 100%     | ❌ Overfitted |
| KNN                 | 98.1%    | ⚠ Considered |
| **Logistic Regression** | **95.2%** | **✅ Selected** |
| Ridge Classifier    | 90.0%    | ⚠ Considered |
| Naive Bayes         | 84.4%    | ⚠ Considered |

**Why Logistic Regression?** — Best balance between accuracy and generalization.
100% accuracy models showed classic overfitting patterns inappropriate for clinical use.

---

## 📊 Key Features

### Data Pipeline
- 1,825 patient records from Kaggle
- 477 duplicate records removed
- Label encoding + MinMaxScaler applied
- 13 input features → 4 output classes

### Web Application
- Medical-grade dark UI design
- Real-time form validation
- Color-coded risk assessment results
- Personalized clinical recommendations
- Blood pressure reference guidelines

### Input Features
- Gender, Age Group
- Family history, Medication status
- Symptom severity (Mild/Moderate/Severe)
- Shortness of breath, Visual changes, Nosebleeds
- Systolic & Diastolic blood pressure ranges
- Controlled diet adherence

### Output Classes
- 🟢 NORMAL
- 🟡 HYPERTENSION (Stage-1)
- 🔴 HYPERTENSION (Stage-2)
- 🚨 HYPERTENSIVE CRISIS

---

## 🔮 Future Implementations
- EMR Integration via REST API
- Wearable device data support
- Explainable AI (SHAP values)
- Multi-language support
- Mobile application

---

## ⚠️ Disclaimer
This system is for educational and preliminary screening purposes only.
It does not replace professional medical diagnosis or consultation.

---

**SmartBridge x SkillWallet | Internship Project 2024**
