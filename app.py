import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Machine Learning & Utils
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.dummy import DummyClassifier

# Local Imports
from src.utils import load_data, basic_clean, drop_missing_target_rows, drop_missing_target_xy, infer_problem_type, detect_columns
from src.visualization import build_eda_figures, build_shap_summary_plotly
from src.modeling import (
    build_preprocessor, build_pipeline, build_processed_view,
    evaluate_cv, evaluate_train_test, compute_feature_importance
)
from src.reporting import build_eda_summary_text, build_eda_report_html, build_eda_report_pdf

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


DATA_PATH = "data/raw/heart.csv"
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


class HeartDashboard:
    def __init__(self):
        self.file_path = DATA_PATH

    @st.cache_data
    def load_data(_self):
        return load_data(_self.file_path)

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

            # Temporary build cleaned dataset logic here for download (could be moved to src)
            # Keeping it simple by reusing modeling logic if needed, but for now simple clean
            # Actually, let's reuse build_processed_view which handles cleaning implicitly via preprocessor
            # But "Cleaned" usually means just missing/outliers, not encoding.
            # For simplicity, we can let user download raw filtered or processed.

            cleaned_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download filtered raw dataset",
                cleaned_csv,
                "filtered_dataset.csv",
                "text/csv"
            )

        st.subheader("Dataset Structure")
        st.caption(
            "Columns, data types, and non-null counts for the filtered view.")
        buffer = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": df.count(),
            "Dtype": df.dtypes.astype(str)
        }).reset_index(drop=True)
        st.dataframe(buffer, width='stretch')

        st.subheader("Dataset Metadata")
        st.markdown(f"[Open source link]({DATASET_LINK})")
        col_a, col_b = st.columns(2)
        problem_type = infer_problem_type(df, target_col)
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

        # In this refactored version, we use the current DF directly for EDA
        eda_df = df
        scope_label = "current filtered view"

        eda_text = build_eda_summary_text(
            eda_df, target_col, DATASET_SOURCE, DATASET_LINK, scope_label=scope_label)
        if eda_text:
            st.download_button(
                "Download EDA summary (text)",
                eda_text.encode("utf-8"),
                "eda_summary.txt",
                "text/plain"
            )
        eda_html = build_eda_report_html(
            eda_df, target_col, DATASET_SOURCE, DATASET_LINK, scope_label=scope_label)
        if eda_html:
            st.download_button(
                "Download EDA report (HTML)",
                eda_html.encode("utf-8"),
                "eda_report.html",
                "text/html"
            )
        eda_pdf = build_eda_report_pdf(
            eda_df, target_col, DATASET_SOURCE, DATASET_LINK, scope_label=scope_label)
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
        st.caption(
            "Explore distributions, categorical relationships, and correlations.")

        eda_df = df
        scope_label = "current filtered view"
        st.caption(f"EDA scope: {scope_label}")

        numeric_cols = eda_df.select_dtypes(
            include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]
        _, categorical_cols, low_card = detect_columns(eda_df, target_col)
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

        _, categorical_cols, low_card = detect_columns(df, target_col)

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
            st.caption(
                "Scaling applies to numeric features only; one-hot encoded columns are left as-is.")

        with st.expander("5) Feature Selection"):
            st.caption(
                "Optional step to keep only the most informative features. "
                "Uses mutual information (classification)."
            )
            enable_fs = st.checkbox(
                "Enable SelectKBest (mutual information)", key="enable_fs")
            st.slider(
                "Number of features to keep",
                5,
                50,
                20,
                key="k_features",
                disabled=not enable_fs,
            )

        st.markdown("---")
        st.subheader("Processed Data Preview")
        st.dataframe(df.head(10), width='stretch')

    def _get_pipeline_params(self, df, target_col):
        numeric_cols, categorical_cols, low_card = detect_columns(
            df, target_col)
        selected_cats = st.session_state.get(
            "categorical_cols", sorted(set(categorical_cols + low_card)))
        missing_strategy = st.session_state.get("missing_strategy", "Impute")
        outlier_method = st.session_state.get("outlier_method", "None")
        scale_method = st.session_state.get("scale_method", "StandardScaler")

        return {
            "df": df,
            "target_col": target_col,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "selected_cats": selected_cats,
            "missing_strategy": missing_strategy,
            "outlier_method": outlier_method,
            "scale_method": scale_method
        }

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
                plot_df["Model Label"] = plot_df["Model"] + \
                    " (" + variant + ")"
            else:
                run_count = plot_df.groupby("Model").cumcount() + 1
                plot_df["Model Label"] = plot_df["Model"] + \
                    " (Run " + run_count.astype(str) + ")"
            model_col = "Model Label"

        chart_df = plot_df[[model_col] + numeric_cols]
        melted = chart_df.melt(
            id_vars=model_col, var_name="Metric", value_name="Score")
        fig = px.bar(melted, x=model_col, y="Score", color="Metric",
                     barmode="group", text_auto=".3f")
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig, width='stretch')

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

    def download_results(self, results, label, filename):
        if not results:
            return
        df = pd.DataFrame(results)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(label, csv, filename, "text/csv")

    def show_before_after(self, df, target_col):
        st.header("Before and After Comparison")
        st.markdown(
            "Generate a processed dataset using current preprocessing selections.")
        st.caption(
            "This preview reflects your current preprocessing choices and does not alter the raw data."
        )

        if st.button("Build processed dataset preview"):
            try:
                params = self._get_pipeline_params(df, target_col)
                preprocessor = build_preprocessor(
                    df, target_col,
                    params["numeric_cols"], params["categorical_cols"],
                    params["missing_strategy"], params["outlier_method"],
                    params["scale_method"], params["selected_cats"]
                )
                processed_df = build_processed_view(
                    df, target_col, preprocessor, params["missing_strategy"])
                st.session_state["processed_df"] = processed_df
            except Exception as exc:
                st.error(f"Failed to build processed dataset: {exc}")
                return

        processed_df = st.session_state.get("processed_df")
        if processed_df is None:
            st.info(
                "Click 'Build processed dataset preview' to create the processed view.")
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

    def train_models_section(self, df, target_col):
        st.header("Model Training, Tuning, and Comparison")
        st.caption(
            "Train baselines, tune hyperparameters, and compare models with cross-validation.")

        if target_col not in df.columns:
            st.error("Target column not found in dataset.")
            return

        # Prepare Pipeline Params
        params = self._get_pipeline_params(df, target_col)

        # Prepare Data for CV
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X, y, dropped = drop_missing_target_xy(X, y)
        if dropped:
            st.warning(
                f"Dropped {dropped} rows with missing target before modeling.")

        if params["missing_strategy"] == "Drop Rows":
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

        # Shared Helper to get pipeline
        def get_model_pipeline(model):
            preprocessor = build_preprocessor(
                df, target_col,
                params["numeric_cols"], params["categorical_cols"],
                params["missing_strategy"], params["outlier_method"],
                params["scale_method"], params["selected_cats"]
            )
            return build_pipeline(
                model, preprocessor,
                enable_fs=st.session_state.get("enable_fs", False),
                k_features=st.session_state.get("k_features", 20)
            )

        with train_tab:
            st.subheader("Baselines (Non-ensemble)")
            if st.button("Run non-ensemble models (CV)"):
                results = []
                cv = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=42)
                for name, model in non_ensemble_models.items():
                    pipeline = get_model_pipeline(model)
                    metrics = evaluate_cv(pipeline, X, y, cv)
                    metrics["Model"] = name
                    results.append(metrics)

                self.display_results(results, plot_metrics=[
                                     "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "PR AUC"])
                st.session_state["non_ensemble_results"] = results
                summary = self.summarize_results(results, rank_metric)
                if summary:
                    st.markdown(
                        f"**Best:** {summary['best']}, **Worst:** {summary['worst']}")

            st.markdown("---")
            st.subheader("Baselines (Ensemble)")
            if st.button("Run ensemble models (CV)"):
                results = []
                cv = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=42)
                for name, model in ensemble_models.items():
                    pipeline = get_model_pipeline(model)
                    metrics = evaluate_cv(pipeline, X, y, cv)
                    metrics["Model"] = name
                    results.append(metrics)

                self.display_results(results, plot_metrics=[
                                     "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "PR AUC"])
                st.session_state["ensemble_results"] = results
                summary = self.summarize_results(results, rank_metric)
                if summary:
                    st.markdown(
                        f"**Best:** {summary['best']}, **Worst:** {summary['worst']}")

        with tune_tab:
            st.subheader("Hyperparameter Tuning")
            tuning_method = st.selectbox(
                "Tuning strategy", ["GridSearchCV", "RandomizedSearchCV"], key="tuning_method")
            scoring_metric = st.selectbox(
                "Tuning metric", ["f1", "roc_auc", "accuracy"], key="scoring_metric")

            if st.button("Run ensemble tuning"):
                results = []
                tuned_models = {}
                cv = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=42)

                # Define param grids (simplified)
                param_grids = {
                    "Random Forest": {"model__n_estimators": [100, 200], "model__max_depth": [None, 5, 10]},
                    "Gradient Boosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]},
                    "AdaBoost": {"model__n_estimators": [50, 100], "model__learning_rate": [0.5, 1.0]},
                }
                if XGBOOST_AVAILABLE:
                    param_grids["XGBoost"] = {"model__n_estimators": [
                        200], "model__max_depth": [3, 5]}

                for name, model in ensemble_models.items():
                    pipeline = get_model_pipeline(model)
                    grid = param_grids.get(name, {})
                    if tuning_method == "GridSearchCV":
                        search = GridSearchCV(
                            pipeline, grid, cv=cv, scoring=scoring_metric, n_jobs=-1)
                    else:
                        search = RandomizedSearchCV(
                            pipeline, grid, cv=cv, scoring=scoring_metric, n_jobs=-1, n_iter=5, random_state=42)

                    search.fit(X, y)
                    tuned_models[name] = search.best_estimator_
                    metrics = evaluate_cv(search.best_estimator_, X, y, cv)
                    metrics["Model"] = name
                    metrics["Best Params"] = str(search.best_params_)
                    results.append(metrics)

                self.display_results(results, plot_metrics=[
                                     "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "PR AUC"])
                st.session_state["tuned_ensemble_results"] = results
                st.session_state["tuned_models"] = tuned_models

        with compare_tab:
            st.subheader("Comparison")
            if st.button("Compare All"):
                combined = []
                combined.extend(st.session_state.get(
                    "non_ensemble_results", []))
                combined.extend(st.session_state.get("ensemble_results", []))
                combined.extend(st.session_state.get(
                    "tuned_ensemble_results", []))
                if combined:
                    self.display_results(combined)
                else:
                    st.info("Run models first.")

        with explain_tab:
            st.subheader("Feature Importance")
            tuned_models = st.session_state.get("tuned_models", {})
            if tuned_models:
                selected_model = st.selectbox(
                    "Select model", list(tuned_models.keys()))
                model_pipeline = tuned_models[selected_model]
                fi_df = compute_feature_importance(
                    model_pipeline, df, target_col, missing_strategy=params["missing_strategy"])
                fig = px.bar(fi_df, x="Importance",
                             y="Feature", orientation="h")
                st.plotly_chart(fig, width='stretch')

                st.markdown("---")
                st.subheader("SHAP Analysis")
                if SHAP_AVAILABLE and st.button("Run SHAP"):
                    X_sample = X.sample(min(100, len(X)), random_state=42)
                    # Simplified SHAP logic for demo
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

                    try:
                        explainer = shap.Explainer(model, X_transformed)
                        shap_values = explainer(X_transformed)
                        fig, bar_fig = build_shap_summary_plotly(
                            shap_values, X_transformed, feature_names)
                        st.plotly_chart(fig, width='stretch')
                    except Exception as e:
                        st.warning(f"SHAP failed: {e}")
            else:
                st.info("Run tuning first.")

        with export_tab:
            st.subheader("Export Results")
            self.download_results(st.session_state.get(
                "non_ensemble_results"), "Non-Ensemble CSV", "non_ensemble.csv")
            self.download_results(st.session_state.get(
                "ensemble_results"), "Ensemble CSV", "ensemble.csv")
            self.download_results(st.session_state.get(
                "tuned_ensemble_results"), "Tuned CSV", "tuned.csv")

    def run(self):
        st.title("Heart Disease Analysis Dashboard")
        st.markdown(
            "End-to-end EDA, preprocessing, modeling, and comparison pipeline.")

        raw_df = self.load_data()
        if raw_df is None:
            return

        df, removed_rows = basic_clean(raw_df)
        target_col = st.sidebar.selectbox("Target Column", options=df.columns, index=df.columns.get_loc(
            "target") if "target" in df.columns else 0)
        filtered_df = self.sidebar_filters(df)

        if filtered_df is None:
            return

        if "processed_df" not in st.session_state:
            st.session_state["processed_df"] = filtered_df

        tab_labels = ["Data Overview", "Statistics",
                      "Visualizations", "Pre-processing", "Modeling"]
        active_tab = st.radio("Navigation", tab_labels, horizontal=True)

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
