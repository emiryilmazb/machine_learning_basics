# Heart Disease Analysis & Pre-processing Dashboard

This project is an interactive web application built with Streamlit for Exploratory Data Analysis (EDA) and pre-processing of the Heart Disease dataset. It allows users to visualize data, apply filters, and prepare the dataset for machine learning modeling without writing any code.

## Features

### 1. Exploratory Data Analysis (EDA)

- **Data Overview:** View raw data, summary metrics (total records, features), and data types.
- **Statistical Summary:** Analyze descriptive statistics and visualize the distribution of the target variable (Heart Disease vs. No Disease).
- **Interactive Visualizations:**
  - **Distributions:** Analyze feature distributions with histograms and box plots, segmented by the target variable.
  - **Categorical Analysis:** Explore relationships between categorical features and the target using bar charts.
  - **Correlations:** Visualize feature correlations with an interactive heatmap and a scatter matrix.

### 2. Data Pre-processing Pipeline

A dedicated tab to prepare the data for machine learning models, including:

- **Outlier Handling:** Clip outliers from numerical columns using the IQR (Interquartile Range) method.
- **Categorical Encoding:** Convert categorical variables to a numerical format using One-Hot Encoding.
- **Feature Scaling:** Scale numerical features to a standard range using either `StandardScaler` or `MinMaxScaler`.

### 3. Comparison Analysis

- **Before & After View:** A dedicated tab to visually and statistically compare the dataset before and after pre-processing steps. This helps in understanding the impact of outlier handling and scaling on data distribution.

### 4. Interactive Filtering & Download

- **Dynamic Filtering:** Filter the entire dataset by Age, Sex, and Chest Pain Type via a sidebar menu. All charts and statistics update instantly.
- **Download Processed Data:** Download the cleaned and pre-processed data as a CSV file, ready for model training.

## Machine Learning Context

The primary output of this dashboard is a clean, pre-processed dataset ready for the application of supervised machine learning models. The goal is to build a **binary classification model** that can predict the presence of heart disease (the 'target' variable) based on the provided patient attributes.

### Applicable Models

The processed data is suitable for a variety of classification algorithms, including:

- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **Decision Trees and Random Forests**
- **Gradient Boosting Machines (e.g., XGBoost, LightGBM)**
- **K-Nearest Neighbors (KNN)**

This application serves as the foundational first step, ensuring that the data fed into these models is of the highest possible quality, thereby maximizing the potential for accurate and reliable predictions.

## Setup & Installation

### Prerequisites

- Python 3.8+
- A virtual environment tool (e.g., `venv`)

### 1. Clone the Repository

```bash
git clone <your-repository-link>
cd <your-repository-directory>
```

### 2. Create and Activate a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

All required libraries are listed in `requirements.txt`. Install them using pip:

```bash
pip install -r requirements.txt
```

## Running the Application

To start the Streamlit server, run the following command in your terminal:

```bash
streamlit run app.py
```

The application will open automatically in your web browser at `http://localhost:8501`.

## Project Structure

```
.
.venv/                  # Virtual environment directory
app.py                  # Main Streamlit application file
heart.csv               # The dataset
requirements.txt        # Python dependencies
README.md               # This file
```
