import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix, average_precision_score
from sklearn.inspection import permutation_importance
from src.utils import detect_columns, drop_missing_target_xy, IQRClipper


def build_preprocessor(df, target_col, numeric_cols, categorical_cols,
                       missing_strategy="Impute", outlier_method="None",
                       scale_method="StandardScaler", selected_cats=None):

    if selected_cats is None:
        selected_cats = categorical_cols

    num_cols = [
        c for c in df.columns if c in numeric_cols and c not in selected_cats and c != target_col]
    cat_cols = [c for c in df.columns if c in selected_cats and c != target_col]

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


def build_pipeline(model, preprocessor, enable_fs=False, k_features=20):
    steps = [("preprocess", preprocessor)]

    if enable_fs:
        steps.append(("selector", SelectKBest(
            mutual_info_classif, k=k_features)))

    steps.append(("model", model))
    return Pipeline(steps)


def build_processed_view(df, target_col, preprocessor, missing_strategy="Impute"):
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    X, y, _ = drop_missing_target_xy(X, y)

    if missing_strategy == "Drop Rows":
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(X_processed, columns=feature_names)
    processed_df[target_col] = y.reset_index(drop=True)
    return processed_df


def evaluate_train_test(pipeline, X_train, X_test, y_train, y_test):
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


def evaluate_cv(pipeline, X, y, cv):
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


def compute_feature_importance(model_pipeline, df, target_col, top_n=20, missing_strategy="Impute"):
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
        if missing_strategy == "Drop Rows":
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
