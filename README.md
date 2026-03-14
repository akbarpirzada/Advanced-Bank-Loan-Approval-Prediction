# 💵 Loan Approval Prediction (LAP)

> A Machine Learning web app built with **Python**, **Scikit-learn**, and **Streamlit** that predicts whether a loan application will be approved or rejected — with a polished, interactive UI.

---

## 📸 Preview

| Section | Description |
|---|---|
| 📊 Model Metrics | Live accuracy, precision, recall & F1 score cards |
| 🔲 Confusion Matrix | Actual vs predicted results on holdout test set |
| 🧪 Prediction Form | Fill in applicant details and get instant approval verdict |
| ✅ / ❌ Result Banner | Animated approval/rejection card with probability bar |

---

## 🚀 Features

- **One-click training** — load any CSV and retrain the model from the sidebar
- **Logistic Regression pipeline** with automatic preprocessing (imputation, scaling, one-hot encoding)
- **Live predictions** with approval probability score and animated result banner
- **Application summary** metrics (credit score, income, loan amount, existing loans)
- **Custom CSS UI** — gradient metric cards, styled sidebar, grouped input sections
- **Cached model & data** — fast reloads using `@st.cache_resource` and `@st.cache_data`

---

## 🗂️ Project Structure

```
loan-approval-prediction/
│
├── loan_approval_app.py     # Main Streamlit application
├── loan_dataset.csv         # Dataset (place in same directory)
└── README.md
```

---

## 📦 Requirements

```
python >= 3.8
streamlit
pandas
numpy
scikit-learn
```

Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn
```

---

## ▶️ Running the App

```bash
streamlit run loan_approval_app.py
```

Then open your browser at `http://localhost:8501`

---

## 📊 Dataset

The app expects a CSV file with the following columns:

| Column | Type | Description |
|---|---|---|
| `applicant_name` | string | Applicant's full name |
| `gender` | string | `M` or `F` |
| `age` | int | Age of the applicant |
| `city` | string | City of residence |
| `employment_type` | string | e.g. Salaried, Self-Employed |
| `bank` | string | Applicant's bank name |
| `monthly_income_pkr` | float | Monthly income in PKR |
| `credit_score` | int | Credit score (300–900) |
| `loan_amount_pkr` | float | Requested loan amount in PKR |
| `loan_tenure_months` | int | Loan term in months |
| `existing_loans` | int | Number of existing active loans |
| `default_history` | int | `0` = No default, `1` = Has defaulted |
| `has_credit_card` | int | `0` = No, `1` = Yes |
| `approved` | int | **Target** — `0` = Rejected, `1` = Approved |

> Default CSV path is `loan_dataset.csv`. You can change this from the sidebar.

---

## 🧠 Model Details

| Step | Detail |
|---|---|
| Algorithm | Logistic Regression |
| Train/Test Split | 80% / 20% (stratified) |
| Numeric features | Median imputation → Standard scaling |
| Categorical features | Most-frequent imputation → One-hot encoding |
| Pipeline | `sklearn.pipeline.Pipeline` + `ColumnTransformer` |
| Max iterations | 2000 |

---

## 📈 Metrics Explained

| Metric | What it means |
|---|---|
| **Accuracy** | Overall percentage of correct predictions |
| **Precision** | Of all predicted approvals, how many were actually approved |
| **Recall** | Of all actual approvals, how many did we correctly catch |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Table of TP, TN, FP, FN counts |

---

## ⚠️ Disclaimer

This project is built **for learning and practice purposes only**. It is not intended for real-world financial decision-making. Loan approvals in production environments require far more rigorous models, fairness audits, and regulatory compliance.

---

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) — Web UI framework
- [Scikit-learn](https://scikit-learn.org/) — Machine learning pipeline
- [Pandas](https://pandas.pydata.org/) — Data manipulation
- [NumPy](https://numpy.org/) — Numerical computing

---

## 👤 Author

**Akbar Pirzada**
- GitHub: https://github.com/akbarpirzada
- LinkedIn: https://www.linkedin.com/in/akbar-pirzada
- Live Demo:https://advanced-bank-loan-approval-prediction-lap-py.streamlit.app/

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
