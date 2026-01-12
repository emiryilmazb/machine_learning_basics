import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import io


def render_matplotlib_png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def build_eda_figures(df, target_col):
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


def _coerce_shap_arrays(shap_values, X_values, class_index=1):
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


def build_shap_summary_plotly(shap_values, X_values, feature_names, max_display=20, class_index=1):
    values, data = _coerce_shap_arrays(
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
