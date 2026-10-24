from __future__ import annotations

import hashlib
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from analysis_engine import (
    ask_groq_for_plan,
    data_quality_profile,
    execute_plan,
    infer_columns,
    list_excel_sheets,
    load_dataframe,
    summarize_result,
)

load_dotenv()
try:
    streamlit_secrets = st.secrets
except StreamlitSecretNotFoundError:
    streamlit_secrets = {}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip() or str(
    streamlit_secrets.get("GROQ_API_KEY", "")
).strip()
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

APP_TITLE = "Universal AI Data Analyst"
MODEL_NAME = (
    os.getenv("GROQ_MODEL", "").strip()
    or str(streamlit_secrets.get("GROQ_MODEL", "openai/gpt-oss-20b")).strip()
)
MAX_RESULT_ROWS = 100

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
st.title("📊 Universal AI Data Analyst")
st.caption("Upload CSV or Excel data, explore it automatically, and ask questions in natural language.")

with st.sidebar:
    st.header("Upload dataset")
    uploaded = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"], help="No sample data is loaded automatically.")
    st.divider()
    st.caption(f"AI model: {MODEL_NAME}")
    st.caption("Your uploaded table is analyzed locally with Pandas. Groq receives the question and schema metadata, then receives the summarized calculation result for explanation.")

if uploaded is None:
    st.info("Upload your own CSV or Excel file using the sidebar to get started.")
    st.markdown("""
    ### What this system does
    - Detects numeric, date/time and categorical columns
    - Reports rows, columns, missing values, duplicates and data types
    - Builds charts using columns found in your file
    - Interprets questions with Groq and performs supported calculations locally
    - Displays an answer, a chart when appropriate, and the result table

    **Supported file types:** `.csv`, `.xlsx`, `.xls`
    """)
    st.stop()

file_bytes = uploaded.getvalue()
file_hash = hashlib.sha256(file_bytes).hexdigest()
suffix = uploaded.name.rsplit(".", 1)[-1].lower()
sheet_name = None
if suffix in {"xlsx", "xls"}:
    try:
        sheets = list_excel_sheets(file_bytes)
    except Exception as exc:
        st.error(f"Could not inspect this Excel workbook: {exc}")
        st.stop()
    if not sheets:
        st.error("This workbook contains no worksheets.")
        st.stop()
    if len(sheets) > 1:
        sheet_name = st.sidebar.selectbox("Worksheet", sheets)
    else:
        sheet_name = sheets[0]

try:
    df = load_dataframe(file_bytes, uploaded.name, sheet_name)
except Exception as exc:
    st.error(f"Could not load the uploaded file: {exc}")
    st.stop()

numeric_cols, categorical_cols, date_cols = infer_columns(df)
profile = data_quality_profile(df)

# Prevent chat history from carrying across unrelated uploaded files.
active_dataset_key = f"{file_hash}:{sheet_name or ''}"
if st.session_state.get("active_dataset_key") != active_dataset_key:
    st.session_state["active_dataset_key"] = active_dataset_key
    st.session_state["chat_history"] = []

# Main metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", f"{len(df.columns):,}")
m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
m4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

overview_tab, explore_tab, ask_tab = st.tabs(["📋 Overview", "📈 Explore", "💬 Ask your data"])

with overview_tab:
    st.subheader("Dataset profile")
    a, b = st.columns([1, 1.3])
    with a:
        st.write(f"**File:** `{uploaded.name}`")
        if sheet_name:
            st.write(f"**Worksheet:** `{sheet_name}`")
        st.write(f"**Numeric ({len(numeric_cols)}):** " + (", ".join(numeric_cols) if numeric_cols else "None detected"))
        st.write(f"**Categorical/text ({len(categorical_cols)}):** " + (", ".join(categorical_cols[:40]) if categorical_cols else "None detected"))
        st.write(f"**Date/time ({len(date_cols)}):** " + (", ".join(date_cols) if date_cols else "None detected"))
    with b:
        st.subheader("Column quality")
        st.dataframe(profile, use_container_width=True, hide_index=True)

    st.subheader("Data preview")
    preview_rows = st.slider("Rows to display", min_value=5, max_value=min(100, max(5, len(df))), value=min(20, max(5, len(df))))
    st.dataframe(df.head(preview_rows), use_container_width=True)
    if numeric_cols:
        with st.expander("Numeric descriptive statistics"):
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

with explore_tab:
    st.subheader("Dynamic chart builder")
    chart_type = st.selectbox("Chart type", ["Bar chart", "Line chart", "Histogram", "Scatter plot", "Box plot", "Correlation matrix"])

    if chart_type == "Histogram":
        if not numeric_cols:
            st.warning("No numeric columns were detected in this file.")
        else:
            column = st.selectbox("Numeric column", numeric_cols, key="hist_column")
            fig = px.histogram(df, x=column, title=f"Distribution of {column}")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter plot":
        if len(numeric_cols) < 2:
            st.warning("A scatter plot needs at least two numeric columns.")
        else:
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_options = [c for c in numeric_cols if c != x_col] or numeric_cols
            y_col = c2.selectbox("Y-axis", y_options, key="scatter_y")
            color_by = c3.selectbox("Color by", ["None"] + categorical_cols, key="scatter_color")
            fig = px.scatter(df, x=x_col, y=y_col, color=None if color_by == "None" else color_by, title=f"{y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box plot":
        if not numeric_cols:
            st.warning("No numeric columns were detected.")
        else:
            c1, c2 = st.columns(2)
            value_col = c1.selectbox("Numeric measure", numeric_cols, key="box_value")
            group_col = c2.selectbox("Group by", ["None"] + categorical_cols, key="box_group")
            fig = px.box(df, x=None if group_col == "None" else group_col, y=value_col, title=f"Distribution of {value_col}")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Correlation matrix":
        if len(numeric_cols) < 2:
            st.warning("A correlation matrix needs at least two numeric columns.")
        else:
            corr = df[numeric_cols].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation matrix", zmin=-1, zmax=1, color_continuous_scale="RdBu")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(corr, use_container_width=True)

    else:
        x_options = date_cols + categorical_cols
        if not x_options or not numeric_cols:
            st.warning("Bar/line charts require at least one categorical/date column and one numeric measure.")
        else:
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("Category or date", x_options, key="group_x")
            y_col = c2.selectbox("Numeric measure", numeric_cols, key="group_y")
            aggregation = c3.selectbox("Aggregation", ["sum", "mean", "median", "count", "min", "max"], key="group_agg")
            chart_df = df[[x_col, y_col]].copy()
            if x_col in date_cols:
                chart_df[x_col] = pd.to_datetime(chart_df[x_col], errors="coerce")
            chart_df = chart_df.dropna(subset=[x_col])
            if chart_type == "Line chart" and x_col in date_cols:
                chart_df["_period"] = chart_df[x_col].dt.to_period("M").dt.start_time
                group_key = "_period"
            else:
                group_key = x_col
            if aggregation == "count":
                out = chart_df.groupby(group_key, dropna=False)[y_col].count().reset_index(name=f"{y_col} count")
                y_key = f"{y_col} count"
            else:
                out = chart_df.groupby(group_key, dropna=False)[y_col].agg(aggregation).reset_index(name=f"{y_col} {aggregation}")
                y_key = f"{y_col} {aggregation}"
            if len(out) > 40:
                out = out.sort_values(y_key, ascending=False).head(40) if chart_type == "Bar chart" else out.sort_values(group_key)
            if chart_type == "Line chart":
                out = out.sort_values(group_key)
                fig = px.line(out, x=group_key, y=y_key, markers=True, title=f"{aggregation.title()} of {y_col} by {x_col}")
            else:
                fig = px.bar(out, x=group_key, y=y_key, title=f"{aggregation.title()} of {y_col} by {x_col}")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(out, use_container_width=True, hide_index=True)

with ask_tab:
    st.subheader("Ask questions in natural language")
    st.caption("Examples: “Which region has the highest total revenue?”, “Compare online and offline sales”, “Show monthly profit trend”, “What are the missing values?”")
    if not GROQ_API_KEY:
        st.warning("Groq API key is missing. Set GROQ_API_KEY in your local `.env` file or Streamlit Cloud app secrets.")
    st.info("Safety design: the model selects from predefined analysis operations. Pandas performs calculations locally. The full dataset is not sent to Groq; the question and column metadata are sent for planning, and calculated results may be sent back for a written explanation.")

    for item in st.session_state["chat_history"]:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("table"):
                st.dataframe(pd.DataFrame(item["table"]), use_container_width=True, hide_index=True)
            if item.get("chart_json"):
                try:
                    import plotly.io as pio
                    st.plotly_chart(pio.from_json(item["chart_json"]), use_container_width=True)
                except Exception:
                    pass

    question = st.chat_input("Ask a question about the uploaded dataset...") if GROQ_API_KEY else None
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.spinner("Interpreting your question and analyzing the data..."):
                plan = ask_groq_for_plan(question, df, numeric_cols, categorical_cols, date_cols, MODEL_NAME)
                answer, result_df, plot_spec = execute_plan(df, plan, numeric_cols, categorical_cols, date_cols)
                narrative = answer
                if result_df is not None and not result_df.empty:
                    short_result = result_df.head(30).to_csv(index=False)
                    try:
                        explanation = summarize_result(question, short_result, MODEL_NAME)
                        narrative = f"{answer}\n\n{explanation}"
                    except Exception:
                        # The deterministic calculation/table is still useful if narrative generation fails.
                        pass

            chart_json = None
            fig = None
            if plot_spec:
                ptype = plot_spec.get("type")
                title = plot_spec.get("title", "Analysis result")
                if ptype == "bar" and result_df is not None and plot_spec.get("x") in result_df.columns and plot_spec.get("y") in result_df.columns:
                    fig = px.bar(result_df, x=plot_spec["x"], y=plot_spec["y"], title=title)
                elif ptype == "line" and result_df is not None and plot_spec.get("x") in result_df.columns and plot_spec.get("y") in result_df.columns:
                    fig = px.line(result_df, x=plot_spec["x"], y=plot_spec["y"], markers=True, title=title)
                elif ptype == "histogram" and plot_spec.get("x") in df.columns:
                    fig = px.histogram(df, x=plot_spec["x"], title=title)
                elif ptype == "scatter" and plot_spec.get("x") in df.columns and plot_spec.get("y") in df.columns:
                    fig = px.scatter(df, x=plot_spec["x"], y=plot_spec["y"], title=title)
                elif ptype == "heatmap" and plot_spec.get("columns"):
                    fig = px.imshow(df[plot_spec["columns"]].corr(numeric_only=True), text_auto=".2f", aspect="auto", title=title, zmin=-1, zmax=1, color_continuous_scale="RdBu")
                if fig is not None:
                    import plotly.io as pio
                    chart_json = pio.to_json(fig)

            with st.chat_message("assistant"):
                st.markdown(narrative)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                if result_df is not None:
                    st.dataframe(result_df.head(MAX_RESULT_ROWS), use_container_width=True, hide_index=True)
                    if len(result_df) > MAX_RESULT_ROWS:
                        st.caption(f"Showing first {MAX_RESULT_ROWS} of {len(result_df):,} rows.")
            history_item = {"role": "assistant", "content": narrative}
            if result_df is not None:
                history_item["table"] = result_df.head(MAX_RESULT_ROWS).astype(str).to_dict(orient="records")
            if chart_json:
                history_item["chart_json"] = chart_json
            st.session_state["chat_history"].append(history_item)
        except Exception as exc:
            error_text = f"I couldn't complete that analysis: {exc}"
            with st.chat_message("assistant"):
                st.error(error_text)
            st.session_state["chat_history"].append({"role": "assistant", "content": error_text})
