import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.utils import infer_problem_type
from src.visualization import build_eda_figures, render_matplotlib_png


def build_eda_summary_text(df, target_col, dataset_source, dataset_link, scope_label="current filtered view"):
    if df is None:
        return ""
    lines = []
    lines.append("EDA SUMMARY")
    lines.append(f"Dataset Source: {dataset_source}")
    lines.append(f"Dataset Link: {dataset_link}")
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


def build_eda_report_html(df, target_col, dataset_source, dataset_link, scope_label="current filtered view"):
    if df is None:
        return ""

    problem_type = infer_problem_type(df, target_col)
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

    figures = build_eda_figures(df, target_col)
    chart_blocks = []
    for title, fig in figures:
        img_bytes = render_matplotlib_png(fig)
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
    <p><strong>Dataset Source:</strong> {dataset_source}</p>
    <p><strong>Dataset Link:</strong> {dataset_link}</p>
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


def build_eda_report_pdf(df, target_col, dataset_source, dataset_link, scope_label="current filtered view"):
    if df is None:
        return b""

    problem_type = infer_problem_type(df, target_col)
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, "EDA Report", fontsize=18, weight="bold")
        lines = [
            f"Dataset Source: {dataset_source}",
            f"Dataset Link: {dataset_link}",
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

        figures = build_eda_figures(df, target_col)
        for title, fig in figures:
            fig.suptitle(title)
            pdf.savefig(fig)
            plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()
