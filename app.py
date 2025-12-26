import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix, average_precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

    def fit(self, X, y=None):
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

    def detect_columns(self, df, target_col):
        numeric_cols = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.drop(columns=[target_col]).select_dtypes(exclude=[np.number]).columns.tolist()

        # Treat low-cardinality numeric columns as categorical if user wants
        low_card = []
        for col in numeric_cols:
            if df[col].nunique() < 10:
                low_card.append(col)
        return numeric_cols, categorical_cols, low_card

    def sidebar_filters(self, df):
        st.sidebar.header("Filter Options")
        if df is None:
            return None

        filtered_df = df.copy()

        if "age" in df.columns:
            min_age = int(df["age"].min())
            max_age = int(df["age"].max())
            selected_age = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))
            filtered_df = filtered_df[filtered_df["age"].between(selected_age[0], selected_age[1])]

        if "sex" in df.columns:
            sex_options = sorted(df["sex"].unique())
            selected_sex = st.sidebar.multiselect(
                "Sex (0: Female, 1: Male)", options=sex_options, default=sex_options
            )
            filtered_df = filtered_df[filtered_df["sex"].isin(selected_sex)]

        if "cp" in df.columns:
            cp_options = sorted(df["cp"].unique())
            selected_cp = st.sidebar.multiselect(
                "Chest Pain Type (cp)", options=cp_options, default=cp_options
            )
            filtered_df = filtered_df[filtered_df["cp"].isin(selected_cp)]

        st.sidebar.markdown("---")
        st.sidebar.write(f"Total Rows: {len(df)}")
        st.sidebar.write(f"Filtered Rows: {len(filtered_df)}")
        return filtered_df

    def show_data_overview(self, df, original_rows, removed_rows, target_col):
        st.header("Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Duplicates Removed", removed_rows)
        col3.metric("Features", len(df.columns))

        with st.expander("Show Raw Data", expanded=True):
            st.dataframe(df.head(20), width='stretch')

        st.subheader("Dataset Structure")
        buffer = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": df.count(),
            "Dtype": df.dtypes.astype(str)
        }).reset_index(drop=True)
        st.dataframe(buffer, width='stretch')

        st.subheader("Dataset Description")
        st.markdown(f"[Open Dataset Link]({DATASET_LINK})")
        col_a, col_b = st.columns(2)
        with col_a:
            st.text_input("Dataset Source", value=DATASET_SOURCE, disabled=True)
            st.text_input("Dataset Link", value=DATASET_LINK, disabled=True)
            st.text_input("Data Type", value="Structured", disabled=True)
        with col_b:
            st.text_input("Target Variable", value=target_col, disabled=True)
            st.text_input("Number of Samples", value=str(len(df)), disabled=True)
            st.text_input("Number of Features", value=str(len(df.columns) - 1), disabled=True)

    def show_statistics(self, df, target_col):
        st.header("Statistical Summary")
        st.subheader("Numerical Statistics")
        st.dataframe(df.describe().T, width='stretch')

        st.subheader("Missing Values")
        missing = df.isna().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        st.dataframe(missing, width='stretch')

        if target_col in df.columns:
            st.subheader("Target Distribution")
            target_counts = df[target_col].value_counts().reset_index()
            target_counts.columns = ["Target", "Count"]
            fig = px.pie(target_counts, values="Count", names="Target", hole=0.4)
            st.plotly_chart(fig, width='stretch')

    def show_visualizations(self, df, target_col):
        st.header("Interactive Visualizations")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]
        _, categorical_cols, low_card = self.detect_columns(df, target_col)
        categorical_cols = sorted(set(categorical_cols + low_card))

        tab1, tab2, tab3 = st.tabs(["Distributions", "Categorical Analysis", "Correlations"])

        with tab1:
            if numeric_cols:
                numeric_col = st.selectbox("Select Feature for Histogram", numeric_cols)
                fig = px.histogram(
                    df, x=numeric_col, color=target_col if target_col in df.columns else None,
                    barmode="overlay", opacity=0.7
                )
                st.plotly_chart(fig, width='stretch')

                fig_box = px.box(
                    df, x=target_col if target_col in df.columns else None, y=numeric_col
                )
                st.plotly_chart(fig_box, width='stretch')
            else:
                st.info("No numeric columns found for distribution plots.")

        with tab2:
            if categorical_cols and target_col in df.columns:
                cat_col = st.selectbox("Select Categorical Feature", categorical_cols)
                cat_data = df.groupby([cat_col, target_col]).size().reset_index(name="count")
                fig = px.bar(cat_data, x=cat_col, y="count", color=target_col, barmode="group")
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No categorical columns found for categorical analysis.")

        with tab3:
            if numeric_cols:
                corr_matrix = df[numeric_cols + [target_col]].corr()
                fig = px.imshow(
                    corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No numeric columns found for correlation heatmap.")

    def show_preprocessing(self, df, target_col):
        st.header("Data Pre-processing")
        st.markdown("Configure preprocessing steps that will be applied inside the ML pipeline.")

        numeric_cols, categorical_cols, low_card = self.detect_columns(df, target_col)

        with st.expander("1) Missing Values"):
            missing_strategy = st.selectbox(
                "Missing value strategy", ["Impute", "Drop Rows"], key="missing_strategy"
            )
            if missing_strategy == "Drop Rows":
                st.info("Rows with missing values will be dropped before training.")

        with st.expander("2) Outlier Handling"):
            outlier_method = st.selectbox(
                "Outlier method", ["None", "IQR Clip"], key="outlier_method"
            )
            if outlier_method == "IQR Clip":
                st.info("Outliers are clipped using IQR bounds inside the pipeline (no row removal).")

        with st.expander("3) Encoding Categorical Features"):
            suggested_categorical = sorted(set(categorical_cols + low_card))
            st.multiselect(
                "Categorical columns", options=df.columns.drop(target_col),
                default=suggested_categorical, key="categorical_cols"
            )

        with st.expander("4) Feature Scaling"):
            scale_method = st.selectbox(
                "Scaling method", ["StandardScaler", "MinMaxScaler", "None"], key="scale_method"
            )

        with st.expander("5) Feature Selection"):
            enable_fs = st.checkbox("Enable SelectKBest", key="enable_fs")
            k_features = st.slider("Number of features to keep", 5, 50, 20, key="k_features")
            if enable_fs:
                st.info("SelectKBest uses mutual information for classification.")

        st.markdown("---")
        st.subheader("Processed Data Preview")
        st.dataframe(df.head(10), width='stretch')

    def build_preprocessor(self, df, target_col):
        numeric_cols, categorical_cols, low_card = self.detect_columns(df, target_col)
        selected_cats = st.session_state.get("categorical_cols", sorted(set(categorical_cols + low_card)))

        num_cols = [c for c in df.columns if c in numeric_cols and c not in selected_cats and c != target_col]
        cat_cols = [c for c in df.columns if c in selected_cats and c != target_col]

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
            cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        try:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
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

    def evaluate_train_test(self, pipeline, X_train, X_test, y_train, y_test):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = None

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
        }

        if y_prob is not None:
            metrics["AUC"] = roc_auc_score(y_test, y_prob)
            metrics["PR AUC"] = average_precision_score(y_test, y_prob)
        else:
            metrics["AUC"] = np.nan
            metrics["PR AUC"] = np.nan

        cm = confusion_matrix(y_test, y_pred)
        return metrics, y_pred, pipeline, cm

    def evaluate_cv(self, pipeline, X, y, cv):
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
        }
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True)
        metrics = {
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1 Score": scores["test_f1"].mean(),
            "AUC": scores["test_roc_auc"].mean(),
            "PR AUC": scores["test_pr_auc"].mean(),
            "Train Accuracy": scores["train_accuracy"].mean(),
            "Train F1": scores["train_f1"].mean(),
        }
        return metrics

    def display_results(self, results, plot_metrics=None):
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, width='stretch')

        if plot_metrics is not None:
            numeric_cols = [c for c in plot_metrics if c in results_df.columns]
        else:
            numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return

        chart_df = results_df[["Model"] + numeric_cols]
        melted = chart_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group", text_auto=".3f")
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig, width='stretch')

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
            notes.append(f"Best vs worst gap is {score_gap:.3f} on {rank_metric}.")

        if "Train F1" in best_row and "F1 Score" in best_row:
            gap = best_row["Train F1"] - best_row["F1 Score"]
            if gap > 0.05:
                notes.append("Best model shows signs of overfitting (train F1 notably higher).")
        if "Train F1" in worst_row and "F1 Score" in worst_row:
            gap = worst_row["Train F1"] - worst_row["F1 Score"]
            if gap < 0.01:
                notes.append("Worst model may be underfitting (train and test F1 both low).")

        summary = {
            "best": best_row["Model"],
            "worst": worst_row["Model"],
            "notes": notes,
        }
        return summary

    def train_models_section(self, df, target_col):
        st.header("Model Training, Tuning, and Comparison")

        if target_col not in df.columns:
            st.error("Target column not found in dataset.")
            return

        X = df.drop(columns=[target_col])
        y = df[target_col]

        missing_strategy = st.session_state.get("missing_strategy", "Impute")
        if missing_strategy == "Drop Rows":
            mask = X.notna().all(axis=1) & y.notna()
            X = X[mask]
            y = y[mask]

        eval_method = st.radio("Evaluation Method", ["Train-Test Split", "Stratified K-Fold CV"], key="eval_method")
        cv_folds = st.slider("CV Folds", 3, 10, 5, key="cv_folds")
        rank_metric = st.selectbox(
            "Ranking Metric (best/worst)",
            ["F1 Score", "AUC", "PR AUC", "Accuracy"],
            index=0,
            key="rank_metric"
        )
        handle_imbalance = st.checkbox("Handle class imbalance (class_weight=balanced)", value=False, key="class_weight")

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
        }

        ensemble_models = {
            "Random Forest": RandomForestClassifier(random_state=42, class_weight=class_weight),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
        if XGBOOST_AVAILABLE:
            ensemble_models["XGBoost"] = xgb.XGBClassifier(
                eval_metric="logloss", random_state=42, use_label_encoder=False, scale_pos_weight=scale_pos_weight
            )

        st.subheader("Non-ensemble Models")
        if st.button("Run Non-ensemble Models"):
            results = []
            details = {}
            if eval_method == "Train-Test Split":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                for name, model in non_ensemble_models.items():
                    pipeline = self.build_pipeline(model, df, target_col)
                    metrics, _, _, cm = self.evaluate_train_test(pipeline, X_train, X_test, y_train, y_test)
                    metrics["Model"] = name
                    results.append(metrics)
                    details[name] = {"cm": cm, "pipeline": pipeline}
            else:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                for name, model in non_ensemble_models.items():
                    pipeline = self.build_pipeline(model, df, target_col)
                    metrics = self.evaluate_cv(pipeline, X, y, cv)
                    metrics["Model"] = name
                    results.append(metrics)

            self.display_results(results, plot_metrics=["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "PR AUC"])
            st.session_state["non_ensemble_results"] = results
            st.session_state["non_ensemble_details"] = details

            summary = self.summarize_results(results, rank_metric)
            if summary:
                st.subheader("Auto Discussion (Non-ensemble)")
                st.markdown(
                    f"- Best model: **{summary['best']}**\n"
                    f"- Worst model: **{summary['worst']}**\n"
                    + "\n".join([f"- {n}" for n in summary["notes"]])
                )

                if eval_method == "Train-Test Split" and summary["best"] in details:
                    st.subheader("Best Model Confusion Matrix")
                    cm = details[summary["best"]]["cm"]
                    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
                    st.plotly_chart(fig, width='stretch')

        st.subheader("Ensemble Models (Before Tuning)")
        if st.button("Run Ensemble Models"):
            results = []
            details = {}
            if eval_method == "Train-Test Split":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                for name, model in ensemble_models.items():
                    pipeline = self.build_pipeline(model, df, target_col)
                    metrics, _, _, cm = self.evaluate_train_test(pipeline, X_train, X_test, y_train, y_test)
                    metrics["Model"] = name
                    results.append(metrics)
                    details[name] = {"cm": cm, "pipeline": pipeline}
            else:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                for name, model in ensemble_models.items():
                    pipeline = self.build_pipeline(model, df, target_col)
                    metrics = self.evaluate_cv(pipeline, X, y, cv)
                    metrics["Model"] = name
                    results.append(metrics)

            self.display_results(results, plot_metrics=["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "PR AUC"])
            st.session_state["ensemble_results"] = results
            st.session_state["ensemble_details"] = details

            summary = self.summarize_results(results, rank_metric)
            if summary:
                st.subheader("Auto Discussion (Ensemble)")
                st.markdown(
                    f"- Best model: **{summary['best']}**\n"
                    f"- Worst model: **{summary['worst']}**\n"
                    + "\n".join([f"- {n}" for n in summary["notes"]])
                )

                if eval_method == "Train-Test Split" and summary["best"] in details:
                    st.subheader("Best Model Confusion Matrix")
                    cm = details[summary["best"]]["cm"]
                    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
                    st.plotly_chart(fig, width='stretch')

        st.subheader("Hyperparameter Tuning (Non-ensemble)")
        non_tuning_method = st.selectbox("Tuning Strategy (Non-ensemble)", ["GridSearchCV", "RandomizedSearchCV"], key="non_tuning_method")
        non_scoring_metric = st.selectbox("Scoring Metric (Non-ensemble)", ["f1", "roc_auc", "accuracy"], key="non_scoring_metric")

        if st.button("Run Non-ensemble Tuning"):
            results = []
            tuned_models = {}
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

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
            }

            for name, model in non_ensemble_models.items():
                if name.startswith("Dummy"):
                    continue
                pipeline = self.build_pipeline(model, df, target_col)
                params = param_grids.get(name, {})
                if non_tuning_method == "GridSearchCV":
                    search = GridSearchCV(pipeline, params, cv=cv, scoring=non_scoring_metric, n_jobs=-1)
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

            self.display_results(results, plot_metrics=["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "PR AUC"])
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

        st.subheader("Hyperparameter Tuning (Ensemble Models)")
        tuning_method = st.selectbox("Tuning Strategy", ["GridSearchCV", "RandomizedSearchCV"], key="tuning_method")
        scoring_metric = st.selectbox("Scoring Metric", ["f1", "roc_auc", "accuracy"], key="scoring_metric")

        if st.button("Run Hyperparameter Tuning"):
            results = []
            tuned_models = {}
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

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
                    search = GridSearchCV(pipeline, params, cv=cv, scoring=scoring_metric, n_jobs=-1)
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

            self.display_results(results, plot_metrics=["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "PR AUC"])
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

        st.subheader("Overall Comparison (All Models)")
        if st.button("Build Overall Comparison"):
            combined = []
            combined.extend(st.session_state.get("non_ensemble_results", []))
            combined.extend(st.session_state.get("ensemble_results", []))
            combined.extend(st.session_state.get("tuned_non_ensemble_results", []))
            combined.extend(st.session_state.get("tuned_ensemble_results", []))

            if combined:
                self.display_results(combined, plot_metrics=["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "PR AUC"])
                summary = self.summarize_results(combined, rank_metric)
                if summary:
                    st.subheader("Auto Discussion (Overall)")
                    st.markdown(
                        f"- Best model: **{summary['best']}**\n"
                        f"- Worst model: **{summary['worst']}**\n"
                        + "\n".join([f"- {n}" for n in summary["notes"]])
                    )
            else:
                st.info("Run at least one modeling step first to build overall comparison.")

        st.subheader("Feature Importance (Ensemble Models)")
        tuned_models = st.session_state.get("tuned_models", {})
        if tuned_models:
            model_name = st.selectbox("Select Model", list(tuned_models.keys()), key="fi_model")
            model_pipeline = tuned_models[model_name]
            try:
                preprocessor = model_pipeline.named_steps["preprocess"]
                feature_names = preprocessor.get_feature_names_out()

                if "selector" in model_pipeline.named_steps:
                    support = model_pipeline.named_steps["selector"].get_support()
                    feature_names = feature_names[support]

                model = model_pipeline.named_steps["model"]
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                    fi_df = fi_df.sort_values(by="Importance", ascending=False).head(20)
                    fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h")
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("Selected model does not provide feature_importances_.")
            except Exception as exc:
                st.warning(f"Feature importance could not be computed: {exc}")
        else:
            st.info("Run hyperparameter tuning to see feature importances.")

        st.subheader("SHAP Analysis (Optional)")
        if not SHAP_AVAILABLE:
            st.info("SHAP is not available in this environment.")
        else:
            if tuned_models:
                shap_model_name = st.selectbox("Select Model for SHAP", list(tuned_models.keys()), key="shap_model")
                if st.button("Run SHAP Summary"):
                    model_pipeline = tuned_models[shap_model_name]
                    X_sample = X.sample(min(200, len(X)), random_state=42)
                    model_pipeline.fit(X, y)
                    model = model_pipeline.named_steps["model"]
                    preprocessor = model_pipeline.named_steps["preprocess"]
                    X_transformed = preprocessor.transform(X_sample)
                    feature_names = preprocessor.get_feature_names_out()
                    if "selector" in model_pipeline.named_steps:
                        support = model_pipeline.named_steps["selector"].get_support()
                        feature_names = feature_names[support]

                    if hasattr(model, "predict_proba"):
                        try:
                            explainer = shap.Explainer(model, X_transformed, feature_names=feature_names)
                            shap_values = explainer(X_transformed)
                            st.set_option("deprecation.showPyplotGlobalUse", False)
                            shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
                            st.pyplot(bbox_inches="tight")
                        except Exception:
                            try:
                                explainer = shap.TreeExplainer(model)
                                shap_values = explainer.shap_values(X_transformed)
                                st.set_option("deprecation.showPyplotGlobalUse", False)
                                shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
                                st.pyplot(bbox_inches="tight")
                            except Exception as exc:
                                st.warning(f"SHAP failed: {exc}")
                    else:
                        st.info("Selected model does not support SHAP with current setup.")
            else:
                st.info("Run hyperparameter tuning first to enable SHAP analysis.")
            with st.expander("SHAP Code Snippet"):
                st.code(
                    """
from shap import Explainer

model_pipeline.fit(X, y)
model = model_pipeline.named_steps["model"]
preprocessor = model_pipeline.named_steps["preprocess"]
X_transformed = preprocessor.transform(X_sample)
feature_names = preprocessor.get_feature_names_out()

explainer = Explainer(model, X_transformed, feature_names=feature_names)
shap_values = explainer(X_transformed)
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)
""",
                    language="python"
                )

        st.subheader("Results Export")
        self.download_results(st.session_state.get("non_ensemble_results"), "Download Non-ensemble Results", "non_ensemble_results.csv")
        self.download_results(st.session_state.get("ensemble_results"), "Download Ensemble Results", "ensemble_results.csv")
        self.download_results(st.session_state.get("tuned_non_ensemble_results"), "Download Tuned Non-ensemble Results", "tuned_non_ensemble_results.csv")
        self.download_results(st.session_state.get("tuned_ensemble_results"), "Download Tuned Ensemble Results", "tuned_ensemble_results.csv")

    def run(self):
        st.title("Heart Disease Analysis Dashboard")
        st.markdown("End-to-end EDA, preprocessing, modeling, and comparison pipeline.")

        raw_df = self.load_data()
        if raw_df is None:
            return

        df, removed_rows = self.basic_clean(raw_df)

        target_col = st.sidebar.selectbox("Target Column", options=df.columns, index=df.columns.get_loc("target") if "target" in df.columns else 0)

        filtered_df = self.sidebar_filters(df)
        if filtered_df is None:
            return

        if "processed_df" not in st.session_state:
            st.session_state["processed_df"] = filtered_df

        tabs = st.tabs([
            "Data Overview",
            "Statistics",
            "Visualizations",
            "Pre-processing",
            "Modeling",
        ])

        with tabs[0]:
            self.show_data_overview(filtered_df, len(raw_df), removed_rows, target_col)

        with tabs[1]:
            self.show_statistics(filtered_df, target_col)

        with tabs[2]:
            self.show_visualizations(filtered_df, target_col)

        with tabs[3]:
            self.show_preprocessing(filtered_df, target_col)

        with tabs[4]:
            self.train_models_section(filtered_df, target_col)


if __name__ == "__main__":
    dashboard = HeartDashboard()
    dashboard.run()
