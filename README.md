# Heart Disease Prediction & Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A professional, end-to-end machine learning application designed to predict the likelihood of heart disease based on clinical parameters. This dashboard facilitates Exploratory Data Analysis (EDA), interactive data preprocessing, model training, hyperparameter tuning, and comprehensive performance comparison.

## 🚀 Features

### 1. Exploratory Data Analysis (EDA)

- **Data Overview:** View raw data statistics, feature types, and missing value analysis.
- **Interactive Visualizations:**
  - Distribution plots (histograms, box plots) for numerical features.
  - Bar charts for categorical feature analysis.
  - Correlation heatmaps to identify relationships between variables.
- **Automated Reporting:** Generate and download EDA reports in **PDF**, **HTML**, and **Text** formats.

### 2. Advanced Preprocessing Pipeline

Configure your ML pipeline dynamically via the UI:

- **Missing Value Handling:** Options to impute (median/mode) or drop rows.
- **Outlier Detection:** IQR-based clipping to handle extreme values.
- **Feature Scaling:** StandardScaler or MinMaxScaler options.
- **Categorical Encoding:** One-Hot Encoding for categorical variables.
- **Feature Selection:** SelectKBest using mutual information to identify top predictors.

### 3. Machine Learning Modeling

Train and compare a wide range of algorithms:

- **Baselines:** Logistic Regression, SVM, KNN, Decision Tree, Dummy Classifier.
- **Ensemble Methods:** Random Forest, Gradient Boosting, AdaBoost, XGBoost (if available).
- **Hyperparameter Tuning:** Optimize models using GridSearchCV or RandomizedSearchCV directly from the interface.

### 4. Explainability & Interpretation

- **Feature Importance:** Visualize which features contribute most to model predictions.
- **SHAP Analysis:** (Optional) Leverage SHAP (SHapley Additive exPlanations) values to understand individual predictions and global model behavior.

## 📂 Project Structure

The project follows a modular architecture for scalability and maintainability:

```
├── app.py                  # Main entry point for the Streamlit application
├── data/
│   ├── raw/                # Original dataset (heart.csv)
│   └── results/            # Exported model results and metrics
├── src/                    # Source code modules
│   ├── utils.py            # Data loading and basic cleaning utilities
│   ├── visualization.py    # Plotting and figure generation functions
│   ├── modeling.py         # ML pipelines, training, and evaluation logic
│   └── reporting.py        # PDF/HTML report generation logic
├── notebooks/              # Jupyter notebooks for experimentation
├── docs/                   # Project documentation, reports, and presentations
├── assets/                 # Images and static assets for reports
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🛠️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/heart-disease-ml.git
   cd heart-disease-ml
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

To launch the dashboard, run the following command from the project root:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

## 📊 Methodology

The pipeline follows these steps:

1. **Data Ingestion:** Loads the dataset from `data/raw/heart.csv`.
2. **Preprocessing:** Applies user-defined transformations (Imputation -> Outlier Clipping -> Scaling -> Encoding).
3. **Training:** Fits selected models using Stratified K-Fold Cross-Validation.
4. **Evaluation:** Computes metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
5. **Tuning:** Performs hyperparameter search to optimize model performance.
6. **Comparison:** Visualizes performance differences to help select the best model.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
