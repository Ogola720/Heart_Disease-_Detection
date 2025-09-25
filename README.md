
# 🩺 Heart Disease Classification

A machine learning project to predict the presence of heart disease from patient data. This repository contains the code, models, and analysis for a binary classification task, aimed at creating a decision-support tool for clinics, with a specific focus on the Kenyan healthcare context.

-----

## 🚀 Key Results & Performance

This project compares three models on the task of heart disease prediction. The **Random Forest** model achieved the best overall performance on the test set.

| Model | Accuracy | F1-Score | ROC AUC |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 0.803 | 0.833 | 0.869 |
| **Random Forest 🏆** | **0.836** | **0.865** | **0.916** |
| Neural Network (MLP) | 0.803 | 0.838 | **0.916** |

  - **Dataset:** 303 samples, 14 features.
  - **Split:** 80% Training (242), 20% Testing (61), stratified.
  - **Primary Goal:** Maximize **recall** (sensitivity) to minimize missed diagnoses, while maintaining a reasonable precision to avoid unnecessary patient alarm.

-----

## 📋 Table of Contents

1.  [Project Overview](https://www.google.com/search?q=%23-project-overview)
2.  [Tech Stack](https://www.google.com/search?q=%23-tech-stack)
3.  [Repository Structure](https://www.google.com/search?q=%23-repository-structure)
4.  [Getting Started](https://www.google.com/search?q=%23-getting-started)
5.  [Example Usage](https://www.google.com/search?q=%23-example-usage)
6.  [Modeling Pipeline](https://www.google.com/search?q=%23-modeling-pipeline)
7.  [Feature Importance](https://www.google.com/search?q=%23-feature-importance)
8.  [Deployment Ideas](https://www.google.com/search?q=%23-deployment-ideas)
9.  [Limitations & Ethics](https://www.google.com/search?q=%23%EF%B8%8F-limitations--ethical-considerations)
10. [Future Work](https://www.google.com/search?q=%23-future-work)

-----

## 🎯 Project Overview

The primary goal is to develop a reliable binary classification pipeline to predict heart disease. The project evaluates three distinct modeling approaches:

  * **Logistic Regression**: An interpretable baseline model.
  * **Random Forest**: An ensemble model that yielded the best performance.
  * **Neural Network (MLP)**: A deep learning approach using Keras.

The ultimate aim is to produce a **decision-support screening tool** that could assist clinics in identifying high-risk patients, enabling them to prioritize follow-ups and diagnostic resources effectively.

-----

## 🛠️ Tech Stack

This project utilizes standard Python libraries for data science and machine learning:

  * **Data Handling:** `Pandas`, `NumPy`
  * **Visualization:** `Matplotlib`, `Seaborn`
  * **Modeling:** `Scikit-learn`, `TensorFlow (Keras)`
  * **Utilities:** `Joblib` for model persistence

-----

## 📁 Repository Structure

The repository is organized to separate data, notebooks, models, and source code for clarity and reproducibility.

```text
.
├── README.md
├── data/
│   └── heart.csv
├── notebooks/
│   └── heart_disease_analysis.ipynb
├── models/
│   ├── logistic_pipeline.joblib
│   ├── random_forest.joblib
│   ├── scaler.joblib
│   └── neural_network.keras
├── src/
│   └── predict.py
├── requirements.txt
├── .gitignore
└── LICENSE
```

-----

## ⚙️ Getting Started

You can reproduce the results using Google Colab or by setting up a local environment.

### 1\. Prerequisites

  - Python 3.8+
  - Access to a terminal or command prompt.

### 2\. Local Setup

Clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3\. Running the Analysis

Open and run the `notebooks/heart_disease_analysis.ipynb` notebook using Jupyter Lab or Google Colab. Ensure the `data/heart.csv` file is accessible to the notebook.

-----

## ⚡ Example Usage

The trained models are saved in the `models/` directory. The following script demonstrates how to load them and make a prediction on a new sample.

```python
import joblib
import pandas as pd
import tensorflow as tf

# Define paths to saved artifacts
MODEL_DIR = "models/"
LOGISTIC_PATH = f"{MODEL_DIR}logistic_pipeline.joblib"
RF_PATH = f"{MODEL_DIR}random_forest.joblib"
SCALER_PATH = f"{MODEL_DIR}scaler.joblib"
NN_PATH = f"{MODEL_DIR}neural_network.keras"

# Load models
pipe_lr = joblib.load(LOGISTIC_PATH)
rf_model = joblib.load(RF_PATH)
scaler = joblib.load(SCALER_PATH)
mlp_model = tf.keras.models.load_model(NN_PATH)

# Create a new patient sample
sample = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
    "ca": 0, "thal": 1
}
X_new = pd.DataFrame([sample])

# --- Make Predictions ---

# 1. Random Forest (Best Performer)
pred_rf = rf_model.predict(X_new)[0]
prob_rf = rf_model.predict_proba(X_new)[:, 1][0]
print(f"Random Forest Prediction: {pred_rf} (Probability: {prob_rf:.4f})")

# 2. Neural Network (requires manual scaling)
X_scaled = scaler.transform(X_new)
prob_nn = mlp_model.predict(X_scaled, verbose=0).ravel()[0]
pred_nn = int(prob_nn > 0.5)
print(f"Neural Network Prediction: {pred_nn} (Probability: {prob_nn:.4f})")
```

-----

## 🔬 Modeling Pipeline

### Data Preprocessing

  * **Seed:** A global random seed (`42`) was used for all operations to ensure reproducibility.
  * **Train-Test Split:** The data was split into 80% for training and 20% for testing, using stratification to maintain the same class proportion in both sets.
  * **Scaling:** `StandardScaler` was applied to numerical features before training the Logistic Regression and Neural Network models.

### Models

  * **Logistic Regression:** Implemented within a `scikit-learn` Pipeline that includes the scaler. Trained with 5-fold stratified cross-validation.
  * **Random Forest:** Tuned using `GridSearchCV` to find the optimal hyperparameters for `n_estimators`, `max_depth`, and `min_samples_split`.
  * **Neural Network (MLP):** A Keras Sequential model with an architecture of `Dense(32) -> Dropout(0.2) -> Dense(16) -> Dense(1, 'sigmoid')`. `EarlyStopping` was used to prevent overfitting.

-----

## 📊 Feature Importance

The Random Forest model identified the following features as most influential in predicting heart disease. This information can help guide clinical focus.

| Feature | Importance Score |
| :--- | :--- |
| `cp` (Chest Pain Type) | 0.173 |
| `thal` (Thallium Stress Test) | 0.138 |
| `thalach` (Max Heart Rate) | 0.105 |
| `oldpeak` (ST Depression) | 0.104 |
| `ca` (Major Vessels Colored) | 0.096 |

-----

## ☁️ Deployment Ideas

This project can be extended into a practical application through several deployment strategies:

  * **🚀 Simple API**: Use **FastAPI** or **Flask** to wrap the model in a REST API for easy integration.
  * **🖥️ Interactive Dashboard**: Build a **Streamlit** or **Gradio** web app for clinicians to input patient data and get instant risk assessments.
  * **📱 Mobile Application**: Develop a mobile front-end that communicates with the API, designed for use in clinics with limited access to desktop computers.
  * **🧠 Explainable AI (XAI)**: Integrate **SHAP** or **LIME** to provide explanations for each prediction, increasing clinician trust and transparency.

-----

## ⚠️ Limitations & Ethical Considerations

  * **Small Dataset:** With only 303 samples, the models may not generalize well to different demographic or clinical populations without further training on local data.
  * **Clinical Impact:** False positives can cause unnecessary patient anxiety and strain resources, while false negatives can lead to missed diagnoses. The model's prediction threshold must be carefully calibrated with clinical experts.
  * **Not a Diagnostic Tool:** This system is intended for **decision support only** and should never replace the professional judgment of a qualified clinician.

-----

## 📈 Future Work

  * **Data Augmentation:** Collect and integrate more diverse, localized patient data to improve model robustness and fairness.
  * **Model Calibration:** Apply calibration techniques (e.g., Platt Scaling) to ensure prediction probabilities are more reliable.
  * **Explainability Dashboard:** Develop a user interface that presents SHAP-based explanations for individual predictions.
  * **Prospective Validation:** Conduct a study in a real clinical setting to evaluate the model's real-world performance and utility.

-----

## 📞 Contact

Ogola720 -ayienga.peter@gmail.com

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
