import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, label_binarize
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix, average_precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.dummy import DummyClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


DATA_PATH = "heart.csv"
DATASET_SOURCE = "Kaggle - Heart Disease Dataset"
DATASET_LINK = "https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset"


st.set_page_config(
    page_title="Heart Disease Analysis Dashboard",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main { padding: 2rem; }
    .stPlotlyChart { background-color: #ffffff; border-radius: 6px; box-shadow: 0 4px 10px rgba(0,0,0,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)


class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
            X_df = X
        else:
            self.feature_names_in_ = None
            X_df = pd.DataFrame(X)
        q1 = X_df.quantile(0.25)
        q3 = X_df.quantile(0.75)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_clipped = X_df.clip(self.lower_bounds_, self.upper_bounds_, axis=1)
        return X_clipped.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if self.feature_names_in_ is not None:
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray([f"x{i}" for i in range(len(self.lower_bounds_))], dtype=object)


class HeartDashboard:
    def __init__(self):
        self.file_path = DATA_PATH

    @st.cache_data
    def load_data(_self):
        try:
            return pd.read_csv(_self.file_path)
        except FileNotFoundError:
            st.error(f"File not found: {_self.file_path}")
            return None

    def basic_clean(self, df):
        if df is None:
            return None, 0
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        return df, removed

    def drop_missing_target_rows(self, df, target_col):
        if df is None or target_col not in df.columns:
            return df, 0
        missing_mask = df[target_col].isna()
        dropped = int(missing_mask.sum())
        if dropped:
            df = df.loc[~missing_mask].copy()
        return df, dropped

    def drop_missing_target_xy(self, X, y):
        missing_mask = y.isna()
        dropped = int(missing_mask.sum())
        if dropped:
            X = X.loc[~missing_mask]
            y = y.loc[~missing_mask]
        return X, y, dropped

    def infer_problem_type(self, df, target_col):
        if df is None or target_col not in df.columns:
            return "Unknown"
        target = df[target_col]
        unique_count = target.dropna().nunique()
        if target.dtype.kind in "ifu":
            if unique_count <= 10:
                return "Classification"
            return "Regression"
        return "Classification"

    def build_cleaned_dataset(self, df, target_col):
        if df is None:
            return None
        cleaned = df.copy()
        missing_strategy = st.session_state.get("missing_strategy", "Impute")
        outlier_method = st.session_state.get("outlier_method", "None")

        if missing_strategy == "Drop Rows":
            cleaned = cleaned.dropna(subset=[target_col]).dropna()
        else:
            numeric_cols, categorical_cols, low_card = self.detect_columns(
                cleaned, target_col)
            categorical_cols = sorted(set(categorical_cols + low_card))
            for col in numeric_cols:
                if col == target_col:
                    continue
                median_value = cleaned[col].median()
                if np.isnan(median_value):
                    median_value = 0.0
                cleaned[col] = cleaned[col].fillna(median_value)
            for col in categorical_cols:
                if col == target_col:
                    continue
                if cleaned[col].isna().all():
                    cleaned[col] = cleaned[col].fillna("Unknown")
                else:
                    most_common = cleaned[col].mode(dropna=True)
                    fill_value = most_common.iloc[0] if not most_common.empty else "Unknown"
                    cleaned[col] = cleaned[col].fillna(fill_value)

        if outlier_method == "IQR Clip":
            numeric_cols = cleaned.drop(columns=[target_col]).select_dtypes(
                include=[np.number]).columns.tolist()
            if numeric_cols:
                q1 = cleaned[numeric_cols].quantile(0.25)
                q3 = cleaned[numeric_cols].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                cleaned[numeric_cols] = cleaned[numeric_cols].clip(
                    lower, upper, axis=1)

        cleaned, _ = self.drop_missing_target_rows(cleaned, target_col)
        return cleaned

    def build_eda_summary_text(self, df, target_col, scope_label="current filtered view"):
        if df is None:
            return ""
        lines = []
        lines.append("EDA SUMMARY")
        lines.append(f"Dataset Source: {DATASET_SOURCE}")
        lines.append(f"Dataset Link: {DATASET_LINK}")
        lines.append(f"Report scope: {scope_label}")
        lines.append(f"Rows: {len(df)}")
        lines.append(f"Features: {len(df.columns) - 1}")
        lines.append("")
        lines.append("Missing Values:")
        missing = df.isna().sum()
        lines.append(missing.to_string())
        lines.append("")
        lines.append("Numerical Summary:")
        lines.append(df.describe().T.to_string())
        lines.append("")
        if target_col in df.columns:
            lines.append("Target Distribution:")
            lines.append(df[target_col].value_counts().to_string())
        return "\n".join(lines)

    def _render_matplotlib_png(self, fig):
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        return buffer.getvalue()

    def build_eda_figures(self, df, target_col):
        figures = []
        if df is None:
            return figures

        if target_col in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            target_counts = df[target_col].value_counts().sort_index()
            ax.bar(target_counts.index.astype(str),
                   target_counts.values, color="#2a6f97")
            ax.set_title("Target Distribution")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            figures.append(("Target Distribution", fig))

        missing = df.isna().sum()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(missing.index.astype(str), missing.values, color="#9b2226")
        ax.set_title("Missing Values per Feature")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Missing Count")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        figures.append(("Missing Values", fig))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            corr_df = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(corr_df, cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Correlation Heatmap")
            figures.append(("Correlation Heatmap", fig))

        return figures

    def build_eda_report_html(self, df, target_col, scope_label="current filtered view"):
        if df is None:
            return ""

        problem_type = self.infer_problem_type(df, target_col)
        summary_df = df.describe().T.round(
            4).reset_index().rename(columns={"index": "Feature"})
        missing_df = df.isna().sum().reset_index()
        missing_df.columns = ["Feature", "Missing Count"]
        target_table = ""
        if target_col in df.columns:
            target_counts = df[target_col].value_counts().reset_index()
            target_counts.columns = ["Target", "Count"]
            target_table = target_counts.to_html(
                index=False, border=0, classes="table")

        figures = self.build_eda_figures(df, target_col)
        chart_blocks = []
        for title, fig in figures:
            img_bytes = self._render_matplotlib_png(fig)
            img_b64 = base64.b64encode(img_bytes).decode("ascii")
            chart_blocks.append(
                f"<div class='chart'><h3>{title}</h3>"
                f"<img src='data:image/png;base64,{img_b64}' alt='{title}' /></div>"
            )

        html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>EDA Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
h1, h2, h3 {{ color: #1b263b; }}
.meta {{ margin-bottom: 16px; }}
.meta p {{ margin: 4px 0; }}
.table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px 0; }}
.table th, .table td {{ border: 1px solid #ddd; padding: 6px; text-align: left; font-size: 12px; }}
.chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; padding: 6px; }}
.note {{ font-size: 12px; color: #444; }}
</style>
</head>
<body>
  <h1>EDA Report</h1>
  <div class="meta">
    <p><strong>Dataset Source:</strong> {DATASET_SOURCE}</p>
    <p><strong>Dataset Link:</strong> {DATASET_LINK}</p>
    <p><strong>Target Variable:</strong> {target_col}</p>
    <p><strong>Problem Type:</strong> {problem_type}</p>
    <p><strong>Rows:</strong> {len(df)}</p>
    <p><strong>Features:</strong> {len(df.columns) - 1}</p>
    <p class="note">Report scope: {scope_label}</p>
  </div>

  <h2>Missing Values</h2>
  {missing_df.to_html(index=False, border=0, classes="table")}

  <h2>Summary Statistics</h2>
  {summary_df.to_html(index=False, border=0, classes="table")}

  <h2>Target Distribution</h2>
  {target_table if target_table else "<p>No target column available.</p>"}

  <h2>Charts</h2>
  {"".join(chart_blocks)}
</body>
</html>
"""
        return html

    def build_eda_report_pdf(self, df, target_col, scope_label="current filtered view"):
        if df is None:
            return b""

        problem_type = self.infer_problem_type(df, target_col)
        buffer = io.BytesIO()
        with PdfPages(buffer) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.07, 0.95, "EDA Report", fontsize=18, weight="bold")
            lines = [
                f"Dataset Source: {DATASET_SOURCE}",
                f"Dataset Link: {DATASET_LINK}",
                f"Target Variable: {target_col}",
                f"Problem Type: {problem_type}",
                f"Rows: {len(df)}",
                f"Features: {len(df.columns) - 1}",
                f"Report scope: {scope_label}",
            ]
            y_pos = 0.9
            for line in lines:
                fig.text(0.07, y_pos, line, fontsize=11)
                y_pos -= 0.03
            pdf.savefig(fig)
            plt.close(fig)

            summary_df = df.describe().T.round(
                4).reset_index().rename(columns={"index": "Feature"})
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            ax.set_title("Summary Statistics", pad=20)
            table = ax.table(
                cellText=summary_df.values,
                colLabels=summary_df.columns,
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)
            pdf.savefig(fig)
            plt.close(fig)

            missing_df = df.isna().sum().reset_index()
            missing_df.columns = ["Feature", "Missing Count"]
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            ax.set_title("Missing Values", pad=20)
            table = ax.table(
                cellText=missing_df.values,
                colLabels=missing_df.columns,
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)
            pdf.savefig(fig)
            plt.close(fig)

            figures = self.build_eda_figures(df, target_col)
            for title, fig in figures:
                fig.suptitle(title)
                pdf.savefig(fig)
                plt.close(fig)

        buffer.seek(0)
        return buffer.getvalue()

    def detect_columns(self, df, target_col):
        numeric_cols = df.drop(columns=[target_col]).select_dtypes(
            include=[np.number]).columns.tolist()
        categorical_cols = df.drop(columns=[target_col]).select_dtypes(
            exclude=[np.number]).columns.tolist()

        # Treat low-cardinality numeric columns as categorical if user wants
        low_card = []
        for col in numeric_cols:
            if df[col].nunique() < 10:
                low_card.append(col)
        return numeric_cols, categorical_cols, low_card

    def sidebar_filters(self, df):
        st.sidebar.header("Filters")
        st.sidebar.caption("Filters apply to all tabs and downloads.")
        if df is None:
            return None

        filtered_df = df.copy()
        def _format_option(value, labels):
            try:
                return labels.get(int(value), str(value))
            except Exception:
                return str(value)

        if "age" in df.columns:
            min_age = int(df["age"].min())
            max_age = int(df["age"].max())
            selected_age = st.sidebar.slider(
                "Age range (years)", min_age, max_age, (min_age, max_age))
            filtered_df = filtered_df[filtered_df["age"].between(
                selected_age[0], selected_age[1])]

        if "sex" in df.columns:
            sex_options = sorted(df["sex"].unique())
            sex_labels = {
                0: "0 - Female",
                1: "1 - Male",
            }
            selected_sex = st.sidebar.multiselect(
                "Sex",
                options=sex_options,
                default=sex_options,
                format_func=lambda value: _format_option(value, sex_labels),
            )
            filtered_df = filtered_df[filtered_df["sex"].isin(selected_sex)]

        if "cp" in df.columns:
            cp_options = sorted(df["cp"].unique())
            cp_labels = {
                0: "0 - Typical angina",
                1: "1 - Atypical angina",
                2: "2 - Non-anginal pain",
                3: "3 - Asymptomatic",
            }
            selected_cp = st.sidebar.multiselect(
                "Chest pain type (cp)",
                options=cp_options,
                default=cp_options,
                format_func=lambda value: _format_option(value, cp_labels),
            )
            filtered_df = filtered_df[filtered_df["cp"].isin(selected_cp)]

        st.sidebar.markdown("---")
        st.sidebar.write(f"Total Rows: {len(df)}")
        st.sidebar.write(f"Filtered Rows: {len(filtered_df)}")
        return filtered_df

    def show_data_overview(self, df, original_rows, removed_rows, target_col):
        st.header("Data Overview")
        st.caption("Quick snapshot of the filtered dataset and export options.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Duplicates Removed", removed_rows)
        col3.metric("Features", len(df.columns))

        with st.expander("Preview & Downloads", expanded=True):
            st.caption(
                "Preview shows the filtered data. Downloads use your preprocessing settings "
                "(missing values, outliers, encoding, scaling)."
            )
            st.dataframe(df.head(20), width='stretch')
            cleaned_df = self.build_cleaned_dataset(df, target_col)
            if cleaned_df is not None:
                cleaned_csv = cleaned_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download cleaned dataset (missing/outlier handling)",
                    cleaned_csv,
                    "cleaned_dataset.csv",
                    "text/csv"
                )
            try:
                processed_df = self.build_processed_view(df, target_col)
                processed_csv = processed_df.to_csv(
                    index=False).encode("utf-8")
                st.download_button(
                    "Download processed dataset (encoded/scaled)",
                    processed_csv,
                    "processed_dataset.csv",
                    "text/csv"
                )
            except Exception as exc:
                st.warning(
                    f"Processed dataset could not be built with current settings: {exc}")

        st.subheader("Dataset Structure")
        st.caption("Columns, data types, and non-null counts for the filtered view.")
        buffer = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": df.count(),
            "Dtype": df.dtypes.astype(str)
        }).reset_index(drop=True)
        st.dataframe(buffer, width='stretch')

        st.subheader("Dataset Metadata")
        st.markdown(f"[Open source link]({DATASET_LINK})")
        col_a, col_b = st.columns(2)
        problem_type = self.infer_problem_type(df, target_col)
        with col_a:
            st.text_input("Dataset Source",
                          value=DATASET_SOURCE, disabled=True)
            st.text_input("Dataset Link", value=DATASET_LINK, disabled=True)
            st.text_input("Data Type", value="Structured", disabled=True)
        with col_b:
            st.text_input("Target Variable", value=target_col, disabled=True)
            st.text_input("Problem Type", value=problem_type, disabled=True)
            st.text_input("Number of Samples",
                          value=str(len(df)), disabled=True)
            st.text_input("Number of Features", value=str(
                len(df.columns) - 1), disabled=True)

    def show_statistics(self, df, target_col):
        st.header("Statistical Summary")
        st.caption("EDA reports and summary statistics for the filtered dataset.")
        cleaned_df = self.build_cleaned_dataset(df, target_col)
        if cleaned_df is not None and not cleaned_df.empty:
            eda_df = cleaned_df
            scope_label = "cleaned filtered view (missing values handled, outliers clipped)"
        else:
            eda_df = df
            scope_label = "current filtered view (cleaning unavailable)"

        eda_text = self.build_eda_summary_text(
            eda_df, target_col, scope_label=scope_label)
        if eda_text:
            st.download_button(
                "Download EDA summary (text)",
                eda_text.encode("utf-8"),
                "eda_summary.txt",
                "text/plain"
            )
        eda_html = self.build_eda_report_html(
            eda_df, target_col, scope_label=scope_label)
        if eda_html:
            st.download_button(
                "Download EDA report (HTML)",
                eda_html.encode("utf-8"),
                "eda_report.html",
                "text/html"
            )
        eda_pdf = self.build_eda_report_pdf(
            eda_df, target_col, scope_label=scope_label)
        if eda_pdf:
            st.download_button(
                "Download EDA report (PDF)",
                eda_pdf,
                "eda_report.pdf",
                "application/pdf"
            )
        st.caption(f"EDA scope: {scope_label}")
        st.subheader("Numeric Summary")
        st.dataframe(eda_df.describe().T, width='stretch')

        st.subheader("Missing Values by Column")
        missing = eda_df.isna().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        st.dataframe(missing, width='stretch')

        if target_col in eda_df.columns:
            st.subheader("Target Distribution")
            target_counts = eda_df[target_col].value_counts().reset_index()
            target_counts.columns = ["Target", "Count"]
            fig = px.pie(target_counts, values="Count",
                         names="Target", hole=0.4)
            st.plotly_chart(fig, width='stretch')

    def show_visualizations(self, df, target_col):
        st.header("Interactive Visualizations")
        st.caption("Explore distributions, categorical relationships, and correlations.")
        cleaned_df = self.build_cleaned_dataset(df, target_col)
        if cleaned_df is not None and not cleaned_df.empty:
            eda_df = cleaned_df
            scope_label = "cleaned filtered view (missing values handled, outliers clipped)"
        else:
            eda_df = df
            scope_label = "current filtered view (cleaning unavailable)"
        st.caption(f"EDA scope: {scope_label}")

        numeric_cols = eda_df.select_dtypes(
            include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]
        _, categorical_cols, low_card = self.detect_columns(eda_df, target_col)
        categorical_cols = sorted(set(categorical_cols + low_card))

        tab1, tab2, tab3 = st.tabs(
            ["Distributions", "Categorical Relationships", "Correlations"])

        with tab1:
            if numeric_cols:
                numeric_col = st.selectbox(
                    "Choose a numeric feature to plot", numeric_cols)
                fig = px.histogram(
                    eda_df, x=numeric_col, color=target_col if target_col in eda_df.columns else None,
                    barmode="overlay", opacity=0.7
                )
                st.plotly_chart(fig, width='stretch')

                fig_box = px.box(
                    eda_df, x=target_col if target_col in eda_df.columns else None, y=numeric_col
                )
                st.plotly_chart(fig_box, width='stretch')
            else:
                st.info("No numeric columns found for distribution plots.")

        with tab2:
            if categorical_cols and target_col in eda_df.columns:
                cat_col = st.selectbox(
                    "Choose a categorical feature", categorical_cols)
                cat_data = eda_df.groupby(
                    [cat_col, target_col]).size().reset_index(name="count")
                fig = px.bar(cat_data, x=cat_col, y="count",
                             color=target_col, barmode="group")
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No categorical columns found for categorical analysis.")

        with tab3:
            if numeric_cols:
                corr_matrix = eda_df[numeric_cols + [target_col]].corr()
                fig = px.imshow(
                    corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No numeric columns found for correlation heatmap.")

    def show_preprocessing(self, df, target_col):
        st.header("Data Pre-processing")
        st.markdown(
            "Configure preprocessing steps that will be applied inside the ML pipeline.")
        st.caption(
            "These settings affect the training pipeline and the processed dataset preview, "
            "not the raw data itself."
        )

        numeric_cols, categorical_cols, low_card = self.detect_columns(
            df, target_col)

        with st.expander("1) Missing Values"):
            missing_labels = {
                "Impute": "Impute missing values (recommended)",
                "Drop Rows": "Drop rows with any missing values",
            }
            missing_strategy = st.selectbox(
                "How should missing values be handled?",
                ["Impute", "Drop Rows"],
                key="missing_strategy",
                format_func=lambda key: missing_labels.get(key, key),
            )
            if missing_strategy == "Drop Rows":
                st.info("Rows with missing values will be dropped before training.")
            else:
                st.info(
                    "Missing numeric values are filled with the median; categorical values "
                    "use the most frequent category."
                )

        with st.expander("2) Outlier Handling"):
            outlier_labels = {
                "None": "No outlier handling",
                "IQR Clip": "Clip extreme values using IQR bounds",
            }
            outlier_method = st.selectbox(
                "How should outliers be handled?",
                ["None", "IQR Clip"],
                key="outlier_method",
                format_func=lambda key: outlier_labels.get(key, key),
            )
            if outlier_method == "IQR Clip":
                st.info(
                    "Numeric features are clipped to the IQR bounds inside the pipeline; no rows are removed."
                )
            else:
                st.info("Outlier handling is skipped.")

        with st.expander("3) Encoding Categorical Features"):
            suggested_categorical = sorted(set(categorical_cols + low_card))
            st.caption(
                "Selected columns will be one-hot encoded. Low-cardinality numeric columns "
                "are suggested as categorical by default."
            )
            st.multiselect(
                "Categorical columns", options=df.columns.drop(target_col),
                default=suggested_categorical, key="categorical_cols"
            )

        with st.expander("4) Feature Scaling"):
            scaling_labels = {
                "StandardScaler": "StandardScaler (zero mean, unit variance)",
                "MinMaxScaler": "MinMaxScaler (scale to 0-1)",
                "None": "No scaling",
            }
            scale_method = st.selectbox(
                "How should numeric features be scaled?",
                ["StandardScaler", "MinMaxScaler", "None"],
                key="scale_method",
                format_func=lambda key: scaling_labels.get(key, key),
            )
            st.caption("Scaling applies to numeric features only; one-hot encoded columns are left as-is.")

        with st.expander("5) Feature Selection"):
            st.caption(
                "Optional step to keep only the most informative features. "
                "Uses mutual information (classification)."
            )
            enable_fs = st.checkbox("Enable SelectKBest (mutual information)", key="enable_fs")
            k_features = st.slider(
                "Number of features to keep",
                5,
                50,
                20,
                key="k_features",
                disabled=not enable_fs,
            )
            if enable_fs:
                st.info("SelectKBest uses mutual information for classification.")
                X = df.drop(columns=[target_col])
                y = df[target_col]
                X, y, dropped = self.drop_missing_target_xy(X, y)
                if dropped:
                    st.info(
                        f"Dropped {dropped} rows with missing target before feature selection.")
                if st.session_state.get("missing_strategy") == "Drop Rows":
                    mask = X.notna().all(axis=1) & y.notna()
                    X = X[mask]
                    y = y[mask]
                try:
                    preprocessor = self.build_preprocessor(df, target_col)
                    X_processed = preprocessor.fit_transform(X)
                    feature_names = preprocessor.get_feature_names_out()
                    selector = SelectKBest(mutual_info_classif, k=min(
                        k_features, len(feature_names)))
                    selector.fit(X_processed, y)
                    support = selector.get_support()
                    selected_features = np.array(feature_names)[
                        support].tolist()
                    st.write(
                        f"Features before selection: {len(feature_names)}")
                    st.write(
                        f"Features after selection: {len(selected_features)}")
                    st.write("Selected features:")
                    st.code(", ".join(selected_features))
                    st.caption(
                        "Justification: k is user-defined to balance model simplicity, "
                        "interpretability, and performance."
                    )
                except Exception as exc:
                    st.warning(
                        f"Could not compute selected features with current settings: {exc}")

        st.markdown("---")
        st.subheader("Processed Data Preview")
        st.dataframe(df.head(10), width='stretch')

    def build_preprocessor(self, df, target_col):
        numeric_cols, categorical_cols, low_card = self.detect_columns(
            df, target_col)
        selected_cats = st.session_state.get(
            "categorical_cols", sorted(set(categorical_cols + low_card)))

        num_cols = [
            c for c in df.columns if c in numeric_cols and c not in selected_cats and c != target_col]
        cat_cols = [
            c for c in df.columns if c in selected_cats and c != target_col]

        missing_strategy = st.session_state.get("missing_strategy", "Impute")
        outlier_method = st.session_state.get("outlier_method", "None")
        scale_method = st.session_state.get("scale_method", "StandardScaler")

        num_steps = []
        if missing_strategy == "Impute":
            num_steps.append(("imputer", SimpleImputer(strategy="median")))
        if outlier_method == "IQR Clip":
            num_steps.append(("outlier", IQRClipper()))
        if scale_method == "StandardScaler":
            num_steps.append(("scaler", StandardScaler()))
        elif scale_method == "MinMaxScaler":
            num_steps.append(("scaler", MinMaxScaler()))

        if not num_steps:
            num_steps = [("passthrough", "passthrough")]

        cat_steps = []
        if missing_strategy == "Impute":
            cat_steps.append(
                ("imputer", SimpleImputer(strategy="most_frequent")))
        try:
            onehot = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False)
        except TypeError:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        cat_steps.append(("onehot", onehot))

        numeric_transformer = Pipeline(steps=num_steps)
        categorical_transformer = Pipeline(steps=cat_steps)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ],
            remainder="drop"
        )

        return preprocessor

    def build_pipeline(self, model, df, target_col):
        preprocessor = self.build_preprocessor(df, target_col)
        steps = [("preprocess", preprocessor)]

        if st.session_state.get("enable_fs", False):
            k = st.session_state.get("k_features", 20)
            steps.append(("selector", SelectKBest(mutual_info_classif, k=k)))

        steps.append(("model", model))
        return Pipeline(steps)

    def build_processed_view(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col].copy()
        X, y, _ = self.drop_missing_target_xy(X, y)
        if st.session_state.get("missing_strategy") == "Drop Rows":
            mask = X.notna().all(axis=1) & y.notna()
            X = X[mask]
            y = y[mask]
        preprocessor = self.build_preprocessor(df, target_col)
        X_processed = preprocessor.fit_transform(X)
        feature_names = preprocessor.get_feature_names_out()
        processed_df = pd.DataFrame(X_processed, columns=feature_names)
        processed_df[target_col] = y.reset_index(drop=True)
        return processed_df

    def evaluate_train_test(self, pipeline, X_train, X_test, y_train, y_test):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        n_classes = len(np.unique(y_test))
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            probas = pipeline.predict_proba(X_test)
            if n_classes > 2:
                y_prob = probas
            else:
                y_prob = probas[:, 1]
        else:
            y_prob = None

        avg_method = "macro" if n_classes > 2 else "binary"
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average=avg_method, zero_division=0),
            "Recall": recall_score(y_test, y_pred, average=avg_method, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average=avg_method, zero_division=0),
        }

        if y_prob is not None:
            if n_classes > 2:
                classes = np.unique(y_test)
                y_bin = label_binarize(y_test, classes=classes)
                try:
                    metrics["AUC"] = roc_auc_score(
                        y_bin, y_prob, multi_class="ovr", average="macro"
                    )
                except Exception:
                    metrics["AUC"] = np.nan
                try:
                    metrics["PR AUC"] = average_precision_score(
                        y_bin, y_prob, average="macro"
                    )
                except Exception:
                    metrics["PR AUC"] = np.nan
            else:
                metrics["AUC"] = roc_auc_score(y_test, y_prob)
                metrics["PR AUC"] = average_precision_score(y_test, y_prob)
        else:
            metrics["AUC"] = np.nan
            metrics["PR AUC"] = np.nan

        cm = confusion_matrix(y_test, y_pred)
        return metrics, y_pred, pipeline, cm

    def evaluate_cv(self, pipeline, X, y, cv):
        n_classes = len(np.unique(y))
        scoring = {"accuracy": "accuracy"}
        if n_classes > 2:
            scoring.update(
                {
                    "precision": "precision_macro",
                    "recall": "recall_macro",
                    "f1": "f1_macro",
                    "roc_auc": "roc_auc_ovr",
                }
            )
        else:
            scoring.update(
                {
                    "precision": "precision",
                    "recall": "recall",
                    "f1": "f1",
                    "roc_auc": "roc_auc",
                    "pr_auc": "average_precision",
                }
            )
        scores = cross_validate(pipeline, X, y, cv=cv,
                                scoring=scoring, return_train_score=True)
        metrics = {
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1 Score": scores["test_f1"].mean(),
            "AUC": scores["test_roc_auc"].mean() if "test_roc_auc" in scores else np.nan,
            "PR AUC": scores["test_pr_auc"].mean() if "test_pr_auc" in scores else np.nan,
            "Train Accuracy": scores["train_accuracy"].mean(),
            "Train F1": scores["train_f1"].mean(),
        }
        return metrics

    def display_results(self, results, plot_metrics=None):
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, width='stretch')

        plot_df = results_df.copy()
        if plot_metrics is not None:
            numeric_cols = [c for c in plot_metrics if c in results_df.columns]
        else:
            numeric_cols = results_df.select_dtypes(
                include=[np.number]).columns.tolist()
        if not numeric_cols:
            return

        model_col = "Model"
        if "Model" in plot_df.columns and plot_df["Model"].duplicated().any():
            if "Best Params" in plot_df.columns:
                tuned_mask = plot_df["Best Params"].notna()
                variant = np.where(tuned_mask, "Tuned", "Baseline")
                plot_df["Model Label"] = plot_df["Model"] + " (" + variant + ")"
            else:
                run_count = plot_df.groupby("Model").cumcount() + 1
                plot_df["Model Label"] = plot_df["Model"] + " (Run " + run_count.astype(str) + ")"
            model_col = "Model Label"

        chart_df = plot_df[[model_col] + numeric_cols]
        melted = chart_df.melt(
            id_vars=model_col, var_name="Metric", value_name="Score")
        fig = px.bar(melted, x=model_col, y="Score", color="Metric",
                     barmode="group", text_auto=".3f")
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig, width='stretch')

    def show_before_after(self, df, target_col):
        st.header("Before and After Comparison")
        st.markdown(
            "Generate a processed dataset using current preprocessing selections.")
        st.caption(
            "This preview reflects your current preprocessing choices and does not alter the raw data."
        )

        if st.button("Build processed dataset preview"):
            try:
                processed_df = self.build_processed_view(df, target_col)
                st.session_state["processed_df"] = processed_df
            except Exception as exc:
                st.error(f"Failed to build processed dataset: {exc}")
                return

        processed_df = st.session_state.get("processed_df")
        if processed_df is None:
            st.info("Click 'Build processed dataset preview' to create the processed view.")
            return
        csv = processed_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download processed dataset (CSV)",
            csv,
            "processed_dataset.csv",
            "text/csv",
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Data Summary")
            st.dataframe(df.describe().T, width='stretch')
        with col2:
            st.subheader("Processed Data Summary")
            st.dataframe(processed_df.describe().T, width='stretch')

        st.subheader("Target Distribution (Raw vs Processed)")
        raw_counts = df[target_col].value_counts().reset_index()
        raw_counts.columns = ["Target", "Count"]
        proc_counts = processed_df[target_col].value_counts().reset_index()
        proc_counts.columns = ["Target", "Count"]

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(
                px.pie(raw_counts, values="Count", names="Target", hole=0.4),
                width='stretch',
                key="raw_target_pie"
            )
        with col4:
            st.plotly_chart(
                px.pie(proc_counts, values="Count", names="Target", hole=0.4),
                width='stretch',
                key="processed_target_pie"
            )

    def download_results(self, results, label, filename):
        if not results:
            return
        df = pd.DataFrame(results)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(label, csv, filename, "text/csv")

    def summarize_results(self, results, rank_metric):
        if not results:
            return None
        df = pd.DataFrame(results)
        if rank_metric not in df.columns:
            return None

        best_row = df.loc[df[rank_metric].idxmax()]
        worst_row = df.loc[df[rank_metric].idxmin()]

        notes = []
        score_gap = best_row[rank_metric] - worst_row[rank_metric]
        if score_gap < 0.02:
            notes.append("Models perform similarly; metric gap is small.")
        else:
            notes.append(
                f"Best vs worst gap is {score_gap:.3f} on {rank_metric}.")

        if "Train F1" in best_row and "F1 Score" in best_row:
            gap = best_row["Train F1"] - best_row["F1 Score"]
            if gap > 0.05:
                notes.append(
                    "Best model shows signs of overfitting (train F1 notably higher).")
        if "Train F1" in worst_row and "F1 Score" in worst_row:
            gap = worst_row["Train F1"] - worst_row["F1 Score"]
            if gap < 0.01:
                notes.append(
                    "Worst model may be underfitting (train and test F1 both low).")

        summary = {
            "best": best_row["Model"],
            "worst": worst_row["Model"],
            "notes": notes,
        }
        return summary

    def _coerce_shap_arrays(self, shap_values, X_values, class_index=1):
        if hasattr(shap_values, "values"):
            values = shap_values.values
            data = shap_values.data if getattr(
                shap_values, "data", None) is not None else X_values
            values = np.array(values)
            if values.ndim == 3:
                idx = min(class_index, values.shape[-1] - 1)
                values = values[:, :, idx]
            return values, np.array(data)
        if isinstance(shap_values, list):
            values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            return np.array(values), np.array(X_values)
        values = np.array(shap_values)
        if values.ndim == 3:
            idx = min(class_index, values.shape[-1] - 1)
            values = values[:, :, idx]
        return values, np.array(X_values)

    def build_shap_summary_plotly(self, shap_values, X_values, feature_names, max_display=20, class_index=1):
        values, data = self._coerce_shap_arrays(
            shap_values, X_values, class_index=class_index)
        shap_df = pd.DataFrame(values, columns=feature_names)
        data_df = pd.DataFrame(data, columns=feature_names)

        mean_abs = shap_df.abs().mean().sort_values(ascending=False)
        top_features = mean_abs.head(max_display).index.tolist()

        rng = np.random.default_rng(42)
        fig = go.Figure()
        ordered = list(reversed(top_features))
        for i, feat in enumerate(ordered):
            vals = shap_df[feat].values
            feat_vals = data_df[feat].values
            jitter = (rng.random(len(vals)) - 0.5) * 0.6
            y = np.full(len(vals), i, dtype=float) + jitter
            fig.add_trace(
                go.Scattergl(
                    x=vals,
                    y=y,
                    mode="markers",
                    marker=dict(
                        size=6,
                        opacity=0.7,
                        color=feat_vals,
                        colorscale="RdBu",
                        showscale=(i == 0),
                        colorbar=dict(
                            title="Feature value") if i == 0 else None,
                    ),
                    hovertemplate=(
                        f"{feat}<br>SHAP=%{{x:.4f}}<br>Value=%{{marker.color:.4f}}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

        height = min(800, max(420, 35 * len(top_features) + 140))
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(ordered))),
            ticktext=ordered,
            title=""
        )
        fig.update_xaxes(title="SHAP value (impact on model output)")
        fig.update_layout(height=height, margin=dict(l=140, r=20, t=20, b=40))

        importance = mean_abs.loc[top_features].sort_values(ascending=True)
        bar_fig = go.Figure(
            go.Bar(x=importance.values, y=importance.index, orientation="h")
        )
        bar_fig.update_layout(
            height=min(600, max(360, 28 * len(top_features) + 120)),
            margin=dict(l=140, r=20, t=20, b=40),
            xaxis_title="Mean |SHAP|"
        )
        return fig, bar_fig

    def compute_feature_importance(self, model_pipeline, df, target_col, top_n=20):
        preprocessor = model_pipeline.named_steps["preprocess"]
        feature_names = preprocessor.get_feature_names_out()

        if "selector" in model_pipeline.named_steps:
            support = model_pipeline.named_steps["selector"].get_support()
            feature_names = feature_names[support]

        model = model_pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fi_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importances})
        else:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            if st.session_state.get("missing_strategy") == "Drop Rows":
                mask = X.notna().all(axis=1) & y.notna()
                X = X[mask]
                y = y[mask]
            result = permutation_importance(
                model_pipeline, X, y, n_repeats=10, random_state=42, n_jobs=-1
            )
            fi_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": result.importances_mean}
            )

        return fi_df.sort_values(by="Importance", ascending=False).head(top_n)

    def train_models_section(self, df, target_col):
        st.header("Model Training, Tuning, and Comparison")
        st.caption("Train baselines, tune hyperparameters, and compare models with cross-validation.")

        if target_col not in df.columns:
            st.error("Target column not found in dataset.")
            return

        with st.expander("Model descriptions", expanded=False):
            st.markdown(
                """
**Dummy (Most Frequent)**: Baseline classifier that predicts the most common class; useful to set a minimum bar.

**Logistic Regression**: Linear model with a sigmoid function; fast, interpretable, and a strong baseline for binary tasks.

**SVM (RBF)**: Finds a maximum-margin decision boundary with non-linear kernels; strong for complex boundaries but can be slower.

**KNN**: Classifies based on nearest neighbors; simple and flexible but sensitive to scaling and noise.

**Decision Tree**: Rule-based splits for interpretability; can overfit without constraints.

**Random Forest**: Bagged ensemble of trees to reduce variance; robust and handles non-linearities well.

**Gradient Boosting**: Sequentially builds trees to correct errors; often high accuracy but sensitive to tuning.

**AdaBoost**: Boosts weak learners with reweighting; can perform well on clean data but sensitive to noise.

**XGBoost**: Optimized gradient boosting; strong performance with regularization and efficient training.

"""
            )

        X = df.drop(columns=[target_col])
        y = df[target_col]
        X, y, dropped = self.drop_missing_target_xy(X, y)
        if dropped:
            st.warning(
                f"Dropped {dropped} rows with missing target before modeling.")

        missing_strategy = st.session_state.get("missing_strategy", "Impute")
        if missing_strategy == "Drop Rows":
            mask = X.notna().all(axis=1) & y.notna()
            X = X[mask]
            y = y[mask]

        with st.expander("Modeling settings", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                cv_folds = st.slider(
                    "Cross-validation folds (Stratified K-Fold)", 3, 10, 5, key="cv_folds"
                )
                rank_metric = st.selectbox(
                    "Rank models by",
                    ["F1 Score", "AUC", "PR AUC", "Accuracy"],
                    index=0,
                    key="rank_metric"
                )
            with col_b:
                handle_imbalance = st.checkbox(
                    "Use class_weight='balanced' for imbalanced classes",
                    value=False,
                    key="class_weight"
                )
                run_holdout = st.checkbox(
                    "Run optional 80/20 holdout for confusion matrix",
                    value=False,
                    key="run_holdout"
                )
            st.caption(
                f"CV: Stratified K-Fold, {cv_folds} folds. Ranking metric: {rank_metric}."
            )
            if run_holdout:
                st.caption(
                    "Holdout diagnostics use an 80/20 stratified split for confusion matrix only.")

        class_weight = "balanced" if handle_imbalance else None
        if handle_imbalance and y.nunique() == 2:
            pos = (y == 1).sum()
            neg = (y == 0).sum()
            scale_pos_weight = float(neg / max(pos, 1))
        else:
            scale_pos_weight = 1.0

        non_ensemble_models = {
            "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=class_weight),
            "SVM (RBF)": SVC(probability=True, class_weight=class_weight),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight=class_weight),
        }

        ensemble_models = {
            "Random Forest": RandomForestClassifier(random_state=42, class_weight=class_weight),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "AdaBoost": AdaBoostClassifier(random_state=42),
        }
        if XGBOOST_AVAILABLE:
            ensemble_models["XGBoost"] = xgb.XGBClassifier(
                eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight
            )

        train_tab, tune_tab, compare_tab, explain_tab, export_tab = st.tabs(
            ["1) Train", "2) Tune", "3) Compare", "4) Explain", "5) Export"]
        )

        with train_tab:
            st.subheader("Baselines (Non-ensemble)")
            st.caption("Runs cross-validation with the current preprocessing pipeline.")
            if st.button("Run non-ensemble models (CV)"):
                results = []
                cv = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=42)
                for name, model in non_ensemble_models.items():
                    pipeline = self.build_pipeline(model, df, target_col)
                    metrics = self.evaluate_cv(pipeline, X, y, cv)
                    metrics["Model"] = name
                    results.append(metrics)

                self.display_results(results, plot_metrics=[
                                     "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "PR AUC"])
                st.session_state["non_ensemble_results"] = results

                summary = self.summarize_results(results, rank_metric)
                if summary:
                    st.subheader("Auto Discussion (Non-ensemble)")
                    st.markdown(
                        f"- Best model: **{summary['best']}**\n"
                        f"- Worst model: **{summary['worst']}**\n"
                        + "\n".join([f"- {n}" for n in summary["notes"]])
                    )

                    if run_holdout and summary["best"] in non_ensemble_models:
                        st.subheader("Best Model Confusion Matrix (Holdout)")
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y
                        )
                        pipeline = self.build_pipeline(
                            non_ensemble_models[summary["best"]], df, target_col)
                        _, _, _, cm = self.evaluate_train_test(
                            pipeline, X_train, X_test, y_train, y_test)
                        fig = px.imshow(cm, text_auto=True,
                                        color_continuous_scale="Blues")
                        st.plotly_chart(fig, width='stretch')

            st.markdown("---")
            st.subheader("Baselines (Ensemble)")
            st.caption("Baseline ensemble results before hyperparameter tuning.")
            if st.button("Run ensemble models (CV)"):
                results = []
                cv = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=42)
                for name, model in ensemble_models.items():
                    pipeline = self.build_pipeline(model, df, target_col)
                    metrics = self.evaluate_cv(pipeline, X, y, cv)
                    metrics["Model"] = name
                    results.append(metrics)

                self.display_results(results, plot_metrics=[
                                     "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "PR AUC"])
                st.session_state["ensemble_results"] = results

                summary = self.summarize_results(results, rank_metric)
                if summary:
                    st.subheader("Auto Discussion (Ensemble)")
                    st.markdown(
                        f"- Best model: **{summary['best']}**\n"
                        f"- Worst model: **{summary['worst']}**\n"
                        + "\n".join([f"- {n}" for n in summary["notes"]])
                    )

                    if run_holdout and summary["best"] in ensemble_models:
                        st.subheader("Best Model Confusion Matrix (Holdout)")
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y
                        )
                        pipeline = self.build_pipeline(
                            ensemble_models[summary["best"]], df, target_col)
                        _, _, _, cm = self.evaluate_train_test(
                            pipeline, X_train, X_test, y_train, y_test)
                        fig = px.imshow(cm, text_auto=True,
                                        color_continuous_scale="Blues")
                        st.plotly_chart(fig, width='stretch')

        with tune_tab:
            st.caption(f"Uses Stratified K-Fold CV with {cv_folds} folds.")
            st.subheader("Hyperparameter Tuning (Non-ensemble)")
            tuning_labels = {
                "GridSearchCV": "Grid search (exhaustive)",
                "RandomizedSearchCV": "Random search (faster)",
            }
            metric_labels = {
                "f1": "F1 (default)",
                "roc_auc": "ROC AUC",
                "accuracy": "Accuracy",
            }
            non_tuning_method = st.selectbox(
                "Tuning strategy (non-ensemble)",
                ["GridSearchCV", "RandomizedSearchCV"],
                key="non_tuning_method",
                format_func=lambda key: tuning_labels.get(key, key),
            )
            non_scoring_metric = st.selectbox(
                "Tuning metric (non-ensemble)",
                ["f1", "roc_auc", "accuracy"],
                key="non_scoring_metric",
                format_func=lambda key: metric_labels.get(key, key),
            )

            if st.button("Run non-ensemble tuning"):
                results = []
                tuned_models = {}
                cv = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=42)

                param_grids = {
                    "Logistic Regression": {
                        "model__C": [0.1, 1, 10],
                        "model__solver": ["lbfgs", "liblinear"],
                    },
                    "SVM (RBF)": {
                        "model__C": [0.1, 1, 10],
                        "model__gamma": ["scale", "auto"],
                    },
                    "KNN": {
                        "model__n_neighbors": [3, 5, 7, 11],
                        "model__weights": ["uniform", "distance"],
                    },
                    "Decision Tree": {
                        "model__max_depth": [None, 3, 5, 10],
                        "model__min_samples_split": [2, 5, 10],
                        "model__criterion": ["gini", "entropy"],
                    },
                }

                for name, model in non_ensemble_models.items():
                    if name.startswith("Dummy"):
                        continue
                    pipeline = self.build_pipeline(model, df, target_col)
                    params = param_grids.get(name, {})
                    if non_tuning_method == "GridSearchCV":
                        search = GridSearchCV(
                            pipeline, params, cv=cv, scoring=non_scoring_metric, n_jobs=-1)
                    else:
                        search = RandomizedSearchCV(
                            pipeline, params, cv=cv, scoring=non_scoring_metric, n_jobs=-1, n_iter=10, random_state=42
                        )

                    search.fit(X, y)
                    tuned_models[name] = search.best_estimator_

                    metrics = self.evaluate_cv(search.best_estimator_, X, y, cv)
                    metrics["Model"] = name
                    metrics["Best Params"] = str(search.best_params_)
                    results.append(metrics)

                self.display_results(results, plot_metrics=[
                                     "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "PR AUC"])
                st.session_state["tuned_non_ensemble_results"] = results
                st.session_state["tuned_non_ensemble_models"] = tuned_models
                summary = self.summarize_results(results, rank_metric)
                if summary:
                    st.subheader("Auto Discussion (Tuned Non-ensemble)")
                    st.markdown(
                        f"- Best model: **{summary['best']}**\n"
                        f"- Worst model: **{summary['worst']}**\n"
                        + "\n".join([f"- {n}" for n in summary["notes"]])
                    )

            st.markdown("---")
            st.subheader("Hyperparameter Tuning (Ensemble Models)")
            tuning_method = st.selectbox(
                "Tuning strategy (ensemble)",
                ["GridSearchCV", "RandomizedSearchCV"],
                key="tuning_method",
                format_func=lambda key: tuning_labels.get(key, key),
            )
            scoring_metric = st.selectbox(
                "Tuning metric (ensemble)",
                ["f1", "roc_auc", "accuracy"],
                key="scoring_metric",
                format_func=lambda key: metric_labels.get(key, key),
            )

            if st.button("Run ensemble tuning"):
                results = []
                tuned_models = {}
                cv = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=42)

                param_grids = {
                    "Random Forest": {
                        "model__n_estimators": [100, 200, 400],
                        "model__max_depth": [None, 5, 10],
                        "model__min_samples_split": [2, 5, 10],
                    },
                    "Gradient Boosting": {
                        "model__n_estimators": [100, 200, 300],
                        "model__learning_rate": [0.05, 0.1, 0.2],
                        "model__max_depth": [2, 3, 4],
                    },
                    "AdaBoost": {
                        "model__n_estimators": [50, 100, 200],
                        "model__learning_rate": [0.5, 1.0, 1.5],
                    },
                }
                if XGBOOST_AVAILABLE:
                    param_grids["XGBoost"] = {
                        "model__n_estimators": [200, 400],
                        "model__max_depth": [3, 5],
                        "model__learning_rate": [0.05, 0.1],
                        "model__subsample": [0.8, 1.0],
                        "model__colsample_bytree": [0.8, 1.0],
                    }

                for name, model in ensemble_models.items():
                    pipeline = self.build_pipeline(model, df, target_col)
                    params = param_grids.get(name, {})
                    if tuning_method == "GridSearchCV":
                        search = GridSearchCV(
                            pipeline, params, cv=cv, scoring=scoring_metric, n_jobs=-1)
                    else:
                        search = RandomizedSearchCV(
                            pipeline, params, cv=cv, scoring=scoring_metric, n_jobs=-1, n_iter=10, random_state=42
                        )

                    search.fit(X, y)
                    tuned_models[name] = search.best_estimator_

                    metrics = self.evaluate_cv(search.best_estimator_, X, y, cv)
                    metrics["Model"] = name
                    metrics["Best Params"] = str(search.best_params_)
                    results.append(metrics)

                self.display_results(results, plot_metrics=[
                                     "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "PR AUC"])
                st.session_state["tuned_ensemble_results"] = results
                st.session_state["tuned_models"] = tuned_models

                summary = self.summarize_results(results, rank_metric)
                if summary:
                    st.subheader("Auto Discussion (Tuned Ensemble)")
                    st.markdown(
                        f"- Best model: **{summary['best']}**\n"
                        f"- Worst model: **{summary['worst']}**\n"
                        + "\n".join([f"- {n}" for n in summary["notes"]])
                    )

        with compare_tab:
            st.subheader("Overall Comparison (All Models)")
            if st.button("Build Overall Comparison"):
                combined = []
                combined.extend(st.session_state.get("non_ensemble_results", []))
                combined.extend(st.session_state.get("ensemble_results", []))
                combined.extend(st.session_state.get(
                    "tuned_non_ensemble_results", []))
                combined.extend(st.session_state.get("tuned_ensemble_results", []))

                if combined:
                    self.display_results(combined, plot_metrics=[
                                         "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "PR AUC"])
                    summary = self.summarize_results(combined, rank_metric)
                    if summary:
                        st.subheader("Auto Discussion (Overall)")
                        st.markdown(
                            f"- Best model: **{summary['best']}**\n"
                            f"- Worst model: **{summary['worst']}**\n"
                            + "\n".join([f"- {n}" for n in summary["notes"]])
                        )
                else:
                    st.info(
                        "Run at least one modeling step first to build overall comparison.")

            st.markdown("---")
            st.subheader("Before vs After Tuning Comparison")
            compare_metric = st.selectbox(
                "Compare by metric",
                ["F1 Score", "AUC", "PR AUC", "Accuracy"],
                index=0,
                key="compare_metric"
            )

            def build_comparison(before, after, metric):
                if not before or not after:
                    return None
                before_df = pd.DataFrame(before)[["Model", metric]].rename(
                    columns={metric: "Before"})
                after_df = pd.DataFrame(after)[["Model", metric]].rename(
                    columns={metric: "After"})
                merged = before_df.merge(after_df, on="Model", how="inner")
                merged["Improvement"] = merged["After"] - merged["Before"]
                return merged.sort_values(by="Improvement", ascending=False)

            if st.button("Compare Non-ensemble Before/After"):
                comparison = build_comparison(
                    st.session_state.get("non_ensemble_results", []),
                    st.session_state.get("tuned_non_ensemble_results", []),
                    compare_metric,
                )
                if comparison is not None:
                    st.dataframe(comparison, width='stretch')
                else:
                    st.info("Run non-ensemble models and tuning first.")

            if st.button("Compare Ensemble Before/After"):
                comparison = build_comparison(
                    st.session_state.get("ensemble_results", []),
                    st.session_state.get("tuned_ensemble_results", []),
                    compare_metric,
                )
                if comparison is not None:
                    st.dataframe(comparison, width='stretch')
                else:
                    st.info("Run ensemble models and tuning first.")

            st.markdown("---")
            st.subheader("Before vs After Tuning Tables")
            metric_cols = ["Accuracy", "Precision",
                           "Recall", "F1 Score", "AUC", "PR AUC"]

            def build_before_after_table(before, after):
                if not before or not after:
                    return None
                before_df = pd.DataFrame(before)[["Model"] + metric_cols]
                after_df = pd.DataFrame(after)[["Model"] + metric_cols]
                merged = before_df.merge(
                    after_df,
                    on="Model",
                    how="inner",
                    suffixes=(" (Before)", " (After)")
                )
                return merged

            non_table = build_before_after_table(
                st.session_state.get("non_ensemble_results", []),
                st.session_state.get("tuned_non_ensemble_results", []),
            )
            if non_table is not None:
                st.markdown("**Non-ensemble Models**")
                st.dataframe(non_table, width='stretch')
            else:
                st.info(
                    "Run non-ensemble models and tuning to build the before/after table.")

            ens_table = build_before_after_table(
                st.session_state.get("ensemble_results", []),
                st.session_state.get("tuned_ensemble_results", []),
            )
            if ens_table is not None:
                st.markdown("**Ensemble Models**")
                st.dataframe(ens_table, width='stretch')
            else:
                st.info("Run ensemble models and tuning to build the before/after table.")

        with explain_tab:
            st.subheader("Feature Importance (Ensemble Models)")
            tuned_models = st.session_state.get("tuned_models", {})
            if tuned_models:
                model_names = list(tuned_models.keys())
                default_models = model_names[:3]
                selected_models = st.multiselect(
                    "Select models for feature importance",
                    model_names,
                    default=default_models,
                    key="fi_models"
                )
                top_n = st.slider("Top features to show", 5, 30, 20, key="fi_top_n")

                if not selected_models:
                    st.info("Select at least one model to view feature importance.")
                else:
                    tabs = st.tabs(selected_models)
                    for tab, model_name in zip(tabs, selected_models):
                        with tab:
                            model_pipeline = tuned_models[model_name]
                            try:
                                fi_df = self.compute_feature_importance(
                                    model_pipeline, df, target_col, top_n=top_n
                                )
                                fig = px.bar(fi_df, x="Importance",
                                             y="Feature", orientation="h")
                                st.plotly_chart(fig, width='stretch')
                            except Exception as exc:
                                st.warning(
                                    f"Feature importance could not be computed: {exc}")
            else:
                st.info("Run hyperparameter tuning to see feature importances.")

            st.markdown("---")
            st.subheader("SHAP Analysis (Optional)")
            if not SHAP_AVAILABLE:
                st.info("SHAP is not available in this environment.")
            else:
                if tuned_models:
                    shap_model_name = st.selectbox(
                        "Model for SHAP analysis",
                        list(tuned_models.keys()),
                        key="shap_model",
                    )
                    col_a, col_b = st.columns(2)
                    with col_a:
                        max_sample = min(1000, len(X))
                        min_sample = min(50, max_sample)
                        default_sample = min(200, max_sample)
                        step_size = 10 if max_sample < 100 else 50
                        sample_size = st.slider(
                            "SHAP sample size (rows)",
                            min_sample,
                            max_sample,
                            default_sample,
                            step=step_size,
                        )
                    with col_b:
                        max_display = st.slider(
                            "Max features to display", 5, 30, 20)
                    if st.button("Run SHAP summary"):
                        model_pipeline = tuned_models[shap_model_name]
                        X_sample = X.sample(
                            min(sample_size, len(X)), random_state=42)
                        with st.spinner("Computing SHAP values..."):
                            model_pipeline.fit(X, y)
                            model = model_pipeline.named_steps["model"]
                            preprocessor = model_pipeline.named_steps["preprocess"]
                            X_transformed = preprocessor.transform(X_sample)
                            feature_names = preprocessor.get_feature_names_out()

                            if "selector" in model_pipeline.named_steps:
                                selector = model_pipeline.named_steps["selector"]
                                support = selector.get_support()
                                X_transformed = selector.transform(X_transformed)
                                feature_names = feature_names[support]

                            if hasattr(model, "predict_proba"):
                                if hasattr(X_transformed, "toarray"):
                                    X_array = X_transformed.toarray()
                                else:
                                    X_array = np.asarray(X_transformed)

                                shap_values = None
                                use_kernel = isinstance(model, AdaBoostClassifier)
                                if not use_kernel:
                                    try:
                                        explainer = shap.Explainer(
                                            model, X_array, feature_names=feature_names)
                                        shap_values = explainer(X_array)
                                    except Exception:
                                        try:
                                            explainer = shap.TreeExplainer(model)
                                            shap_values = explainer.shap_values(
                                                X_array)
                                        except Exception:
                                            shap_values = None

                                if shap_values is None:
                                    try:
                                        rng = np.random.default_rng(42)
                                        background_size = min(50, len(X_array))
                                        background_idx = rng.choice(
                                            len(X_array), size=background_size, replace=False)
                                        background = X_array[background_idx]

                                        def predict_fn(data):
                                            proba = model.predict_proba(data)
                                            if proba.ndim == 2 and proba.shape[1] > 1:
                                                return proba[:, 1]
                                            return proba

                                        explainer = shap.KernelExplainer(
                                            predict_fn, background)
                                        shap_values = explainer.shap_values(
                                            X_array, nsamples=100
                                        )
                                    except Exception as exc:
                                        st.warning(f"SHAP failed: {exc}")
                                        shap_values = None
                            else:
                                st.info(
                                    "Selected model does not support SHAP with current setup.")
                                shap_values = None

                        if shap_values is not None:
                            beeswarm_fig, bar_fig = self.build_shap_summary_plotly(
                                shap_values,
                                X_array,
                                feature_names,
                                max_display=max_display,
                                class_index=1
                            )
                            tab_a, tab_b = st.tabs(
                                ["Beeswarm (Interactive)", "Bar (Mean |SHAP|)"])
                            with tab_a:
                                st.plotly_chart(beeswarm_fig, width='stretch')
                            with tab_b:
                                st.plotly_chart(bar_fig, width='stretch')
                else:
                    st.info("Run hyperparameter tuning first to enable SHAP analysis.")
                with st.expander("SHAP Code Snippet"):
                    st.code(
                        """
from shap import Explainer, KernelExplainer
from sklearn.ensemble import AdaBoostClassifier

model_pipeline.fit(X, y)
model = model_pipeline.named_steps["model"]
preprocessor = model_pipeline.named_steps["preprocess"]
X_transformed = preprocessor.transform(X_sample)
feature_names = preprocessor.get_feature_names_out()
if "selector" in model_pipeline.named_steps:
    selector = model_pipeline.named_steps["selector"]
    support = selector.get_support()
    X_transformed = selector.transform(X_transformed)
    feature_names = feature_names[support]

if isinstance(model, AdaBoostClassifier):
    background = X_transformed[:50]
    def predict_fn(data):
        return model.predict_proba(data)[:, 1]
    explainer = KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_transformed, nsamples=100)
else:
    explainer = Explainer(model, X_transformed, feature_names=feature_names)
    shap_values = explainer(X_transformed)
""",
                        language="python"
                    )
                    st.code(
                        """
# Use a custom Plotly-based summary for interactivity
beeswarm_fig, bar_fig = dashboard.build_shap_summary_plotly(
    shap_values, X_transformed, feature_names, max_display=20
)
""",
                        language="python"
                    )

        with export_tab:
            st.subheader("Results Export")
            self.download_results(st.session_state.get(
                "non_ensemble_results"), "Download non-ensemble results (CSV)", "non_ensemble_results.csv")
            self.download_results(st.session_state.get(
                "ensemble_results"), "Download ensemble results (CSV)", "ensemble_results.csv")
            self.download_results(st.session_state.get("tuned_non_ensemble_results"),
                                  "Download tuned non-ensemble results (CSV)", "tuned_non_ensemble_results.csv")
            self.download_results(st.session_state.get("tuned_ensemble_results"),
                                  "Download tuned ensemble results (CSV)", "tuned_ensemble_results.csv")


    def run(self):
        st.title("Heart Disease Analysis Dashboard")
        st.markdown(
            "End-to-end EDA, preprocessing, modeling, and comparison pipeline.")
        with st.expander("Project Information", expanded=False):
            st.markdown(
                """
### Abstract
This project builds a classification pipeline to predict heart disease using clinical features.
It provides end-to-end EDA, preprocessing, model comparison, and tuning to identify strong
predictive baselines and understand feature importance.

### Problem Statement
Predict whether a patient has heart disease based on clinical measurements.
Early risk prediction supports decision-making and can guide follow-up testing.

### Limitations
- The dataset is structured and relatively small; results may not generalize to new populations.
- Labels reflect historical diagnoses and may carry bias or noise.

### Objectives
1. Compare multiple machine learning classifiers for heart disease prediction.
2. Provide transparent preprocessing with encoding, scaling, and feature selection.
3. Evaluate models with cross-validation and hyperparameter tuning.
"""
            )

        raw_df = self.load_data()
        if raw_df is None:
            return

        df, removed_rows = self.basic_clean(raw_df)

        target_col = st.sidebar.selectbox("Target Column", options=df.columns, index=df.columns.get_loc(
            "target") if "target" in df.columns else 0)

        filtered_df = self.sidebar_filters(df)
        if filtered_df is None:
            return

        if "processed_df" not in st.session_state:
            st.session_state["processed_df"] = filtered_df

        tab_labels = [
            "Data Overview",
            "Statistics",
            "Visualizations",
            "Pre-processing",
            "Modeling",
        ]
        if "active_tab" not in st.session_state or st.session_state["active_tab"] not in tab_labels:
            st.session_state["active_tab"] = tab_labels[0]

        # Use a stateful selector to avoid the initial st.tabs reset on first rerun.
        if hasattr(st, "segmented_control"):
            active_tab = st.segmented_control(
                "Navigation",
                tab_labels,
                key="active_tab",
                label_visibility="collapsed",
            )
        else:
            active_tab = st.radio(
                "Navigation",
                tab_labels,
                horizontal=True,
                key="active_tab",
                label_visibility="collapsed",
            )

        if active_tab == "Data Overview":
            self.show_data_overview(filtered_df, len(
                raw_df), removed_rows, target_col)
        elif active_tab == "Statistics":
            self.show_statistics(filtered_df, target_col)
        elif active_tab == "Visualizations":
            self.show_visualizations(filtered_df, target_col)
        elif active_tab == "Pre-processing":
            self.show_preprocessing(filtered_df, target_col)
            st.markdown("---")
            self.show_before_after(filtered_df, target_col)
        elif active_tab == "Modeling":
            self.train_models_section(filtered_df, target_col)


if __name__ == "__main__":
    dashboard = HeartDashboard()
    dashboard.run()
