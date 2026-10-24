"""Safe analysis engine for Universal AI Data Analyst.

The LLM chooses from a small list of allowed analytical operations. All calculations
are performed locally with pandas; the model does not generate executable Python.
"""
from __future__ import annotations

import json
import re
from io import BytesIO
from typing import Any

import pandas as pd

MAX_SCHEMA_COLUMNS = 120
MAX_RESULT_ROWS = 100
ALLOWED_INTENTS = {
    "aggregate", "top_n", "trend", "correlation", "distribution",
    "missing", "describe", "preview", "unsupported",
}
ALLOWED_AGGREGATIONS = {"sum", "mean", "median", "min", "max", "count", "nunique"}
ALLOWED_CHARTS = {"bar", "line", "scatter", "histogram", "box", "heatmap", "none"}
ALLOWED_GRANULARITIES = {"day", "week", "month", "quarter", "year"}


def load_dataframe(file_bytes: bytes, filename: str, sheet_name: str | None = None) -> pd.DataFrame:
    """Load a CSV or Excel worksheet and normalize blank/duplicate column names."""
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix == "csv":
        try:
            df = pd.read_csv(BytesIO(file_bytes), sep=None, engine="python")
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(file_bytes), sep=None, engine="python", encoding="latin-1")
    elif suffix in {"xlsx", "xls"}:
        df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name if sheet_name is not None else 0)
    else:
        raise ValueError("Unsupported file format. Upload a .csv, .xlsx, or .xls file.")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The selected worksheet could not be loaded as a table.")
    if df.empty:
        raise ValueError("The selected file or worksheet has no data rows.")
    if len(df.columns) == 0:
        raise ValueError("No columns were found in this file.")

    names: list[str] = []
    used: dict[str, int] = {}
    for idx, raw_name in enumerate(df.columns, start=1):
        name = str(raw_name).strip() or f"Column_{idx}"
        used[name] = used.get(name, 0) + 1
        if used[name] > 1:
            name = f"{name}_{used[name]}"
        names.append(name)
    df.columns = names
    return df


def list_excel_sheets(file_bytes: bytes) -> list[str]:
    return pd.ExcelFile(BytesIO(file_bytes)).sheet_names


def infer_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Detect numeric/date/categorical columns without forcing arbitrary text into dates."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols: list[str] = []

    for col in df.columns:
        if col in numeric_cols:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            continue

        # Use both the column label and actual values to avoid parsing ordinary text as dates.
        looks_like_date_name = bool(re.search(r"date|time|month|year|created|updated|timestamp", str(col), re.I))
        sample = df[col].dropna().astype(str).head(500)
        if sample.empty:
            continue
        if looks_like_date_name:
            parsed_sample = pd.to_datetime(sample, errors="coerce")
            threshold = 0.70
        else:
            # Only infer date values for strings that visibly resemble common date formats.
            pattern_share = sample.str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{4}-\d{2}", regex=True).mean()
            if pattern_share < 0.70:
                continue
            parsed_sample = pd.to_datetime(sample, errors="coerce")
            threshold = 0.80
        if parsed_sample.notna().mean() >= threshold:
            parsed_full = pd.to_datetime(df[col], errors="coerce")
            if parsed_full.notna().mean() >= threshold:
                df[col] = parsed_full
                date_cols.append(col)

    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in date_cols]
    return numeric_cols, categorical_cols, date_cols


def data_quality_profile(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Column": [str(c) for c in df.columns],
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Missing %": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })


def build_schema(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], date_cols: list[str]) -> dict[str, Any]:
    cols = []
    for c in list(df.columns)[:MAX_SCHEMA_COLUMNS]:
        kind = "numeric" if c in numeric_cols else "date/time" if c in date_cols else "categorical/text"
        cols.append({
            "name": str(c),
            "type": str(df[c].dtype),
            "kind": kind,
            "unique_values": int(df[c].nunique(dropna=True)),
            "missing_values": int(df[c].isna().sum()),
        })
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": cols,
        "column_list_truncated": len(df.columns) > MAX_SCHEMA_COLUMNS,
    }


def parse_json_response(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    text = str(content).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The AI did not return a valid analysis plan. Please rephrase the question.")
    result = json.loads(text[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("The AI returned an invalid analysis plan.")
    return result


def validate_plan(plan: dict[str, Any], df: pd.DataFrame, numeric_cols: list[str],
                  categorical_cols: list[str], date_cols: list[str]) -> dict[str, Any]:
    """Validate model output against a fixed operation set and real dataset columns."""
    intent = str(plan.get("intent", "unsupported")).lower()
    if intent not in ALLOWED_INTENTS:
        intent = "unsupported"

    aggregation = str(plan.get("aggregation", "sum")).lower()
    if aggregation not in ALLOWED_AGGREGATIONS:
        aggregation = "sum"
    chart = str(plan.get("chart", "none")).lower()
    if chart not in ALLOWED_CHARTS:
        chart = "none"
    granularity = str(plan.get("date_granularity", "month")).lower()
    if granularity not in ALLOWED_GRANULARITIES:
        granularity = "month"

    valid_columns = set(df.columns)
    def col(value: Any) -> str | None:
        return value if isinstance(value, str) and value in valid_columns else None

    raw_groups = plan.get("group_by", [])
    if not isinstance(raw_groups, list):
        raw_groups = []
    groups: list[str] = []
    for item in raw_groups:
        valid = col(item)
        if valid is not None and valid not in groups:
            groups.append(valid)
    groups = groups[:2]

    metric = col(plan.get("metric"))
    x_col = col(plan.get("x_column"))
    y_col = col(plan.get("y_column"))
    target_col = col(plan.get("target_column"))

    if metric and metric not in numeric_cols and aggregation not in {"count", "nunique"}:
        metric = None
    if intent == "trend" and x_col not in date_cols:
        x_col = None
    if intent == "correlation":
        if x_col not in numeric_cols:
            x_col = None
        if y_col not in numeric_cols:
            y_col = None
    if intent == "distribution" and target_col is None:
        target_col = groups[0] if groups else (categorical_cols + numeric_cols + date_cols or [None])[0]

    try:
        limit = max(1, min(int(plan.get("limit", 10)), 100))
    except (TypeError, ValueError):
        limit = 10

    return {
        "intent": intent,
        "group_by": groups,
        "metric": metric,
        "aggregation": aggregation,
        "x_column": x_col,
        "y_column": y_col,
        "target_column": target_col,
        "limit": limit,
        "date_granularity": granularity,
        "chart": chart,
        "title": str(plan.get("title") or "Analysis result")[:120],
        "explanation": str(plan.get("explanation") or "")[:400],
    }


SYSTEM_PROMPT = """You are the query planner for a universal tabular-data analysis application.
Return ONLY a valid JSON object (no markdown, no Python, no SQL). Never invent column names.
Use only exact names from the supplied schema.

Required schema:
{
 "intent":"aggregate|top_n|trend|correlation|distribution|missing|describe|preview|unsupported",
 "group_by":[], "metric":null,
 "aggregation":"sum|mean|median|min|max|count|nunique",
 "x_column":null, "y_column":null, "target_column":null,
 "limit":10,
 "date_granularity":"day|week|month|quarter|year",
 "chart":"bar|line|scatter|histogram|box|heatmap|none",
 "title":"short title", "explanation":"one brief sentence"
}

Planning rules:
- aggregate: compute the selected aggregation over a numeric column, optionally grouped by one or two columns.
- top_n: rank a category by a numeric measure; group_by must include category. For questions like "which region has the most records", use row counts.
- trend: set x_column to a date/time column and metric to a numeric measure; choose aggregation and granularity.
- correlation: select x_column and y_column from numeric columns if the question refers to two measures; otherwise choose correlation matrix with both null.
- distribution: set target_column to the requested column; use a histogram for numeric values, value counts for category/text.
- missing: user asks about blanks/null/missing values.
- describe: descriptive statistics for numeric values.
- preview: show/display/sample rows.
- unsupported: request cannot reasonably be answered from tabular data.
- For "best", "highest", "top selling", "most profitable", pick the closest appropriate numeric column based on names and schema.
- For grouped count, metric may be null and aggregation="count".
- Use maximum limit 100. Return only JSON.
"""


def ask_groq_for_plan(question: str, df: pd.DataFrame, numeric_cols: list[str],
                      categorical_cols: list[str], date_cols: list[str], model_name: str) -> dict[str, Any]:
    from langchain_groq import ChatGroq

    schema = build_schema(df, numeric_cols, categorical_cols, date_cols)
    user_prompt = (
        f"Question: {question}\n\n"
        f"Dataset schema (metadata only; not the entire table):\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Numeric columns: {json.dumps(numeric_cols)}\n"
        f"Categorical/text columns: {json.dumps(categorical_cols)}\n"
        f"Date/time columns: {json.dumps(date_cols)}"
    )
    llm = ChatGroq(model=model_name, temperature=0)
    response = llm.invoke([("system", SYSTEM_PROMPT), ("human", user_prompt)])
    plan = parse_json_response(response.content)
    return validate_plan(plan, df, numeric_cols, categorical_cols, date_cols)


def aggregate_value(series: pd.Series, aggregation: str):
    if aggregation == "sum":
        return series.sum(min_count=1)
    if aggregation == "mean":
        return series.mean()
    if aggregation == "median":
        return series.median()
    if aggregation == "min":
        return series.min()
    if aggregation == "max":
        return series.max()
    if aggregation == "nunique":
        return series.nunique(dropna=True)
    return series.count()


def grouped_aggregate(df: pd.DataFrame, groups: list[str], metric: str | None,
                      aggregation: str) -> pd.DataFrame:
    if not groups:
        if metric is None:
            return pd.DataFrame({"Rows": [len(df)]})
        value = aggregate_value(df[metric], aggregation)
        return pd.DataFrame({"Measure": [metric], "Aggregation": [aggregation], "Value": [value]})

    if metric is None:
        return df.groupby(groups, dropna=False).size().reset_index(name="Row count")
    if aggregation == "count":
        return df.groupby(groups, dropna=False)[metric].count().reset_index(name=f"{metric} count")
    result_name = f"{metric} {aggregation}"
    return df.groupby(groups, dropna=False)[metric].agg(lambda s: aggregate_value(s, aggregation)).reset_index(name=result_name)


def execute_plan(df: pd.DataFrame, plan: dict[str, Any], numeric_cols: list[str],
                 categorical_cols: list[str], date_cols: list[str]):
    """Execute only predefined pandas operations; return text, result table, optional plot type."""
    intent = plan["intent"]
    groups = plan["group_by"]
    metric = plan["metric"]
    agg = plan["aggregation"]
    limit = plan["limit"]
    message = "Analysis completed using the uploaded dataset."
    result: pd.DataFrame | None = None
    plot: dict[str, Any] | None = None

    if intent == "unsupported":
        return "I couldn't map that question to a supported analysis. Try asking for a total, average, top values, date trend, distribution, correlation, missing-value summary, descriptive statistics, or a row preview.", None, None

    if intent == "missing":
        result = pd.DataFrame({
            "Column": list(df.columns),
            "Missing values": [int(df[c].isna().sum()) for c in df.columns],
            "Missing %": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
        }).sort_values("Missing values", ascending=False)
        return "Missing-value counts and percentages by column:", result, None

    if intent == "describe":
        if not numeric_cols:
            return "No numeric columns were detected for descriptive statistics.", None, None
        result = df[numeric_cols].describe().T.reset_index(names="Column")
        return "Descriptive statistics for numeric columns:", result, None

    if intent == "preview":
        return f"Showing the first {min(limit, len(df))} rows:", df.head(limit).copy(), None

    if intent == "distribution":
        target = plan["target_column"]
        if target not in df.columns:
            return "No suitable column was selected for distribution analysis. Try naming a column explicitly.", None, None
        if target in numeric_cols:
            values = pd.to_numeric(df[target], errors="coerce").dropna()
            if values.empty:
                return f"Column {target} has no valid numeric values.", None, None
            result = values.describe().rename_axis("Statistic").reset_index(name="Value")
            plot = {"type": "histogram", "x": target, "title": f"Distribution of {target}"}
        elif target in date_cols:
            values = pd.to_datetime(df[target], errors="coerce").dropna()
            result = values.dt.to_period("M").value_counts().sort_index().rename_axis("Month").reset_index(name="Count")
            result["Month"] = result["Month"].astype(str)
            plot = {"type": "bar", "x": "Month", "y": "Count", "title": f"Records by month: {target}"}
        else:
            result = df[target].fillna("(Missing)").astype(str).value_counts().head(limit).rename_axis(target).reset_index(name="Count")
            plot = {"type": "bar", "x": target, "y": "Count", "title": f"Most common values: {target}"}
        return f"Distribution analysis for {target}:", result, plot

    if intent == "correlation":
        x, y = plan["x_column"], plan["y_column"]
        if x and y:
            paired = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(paired) < 2:
                return "At least two complete numeric rows are required to calculate correlation.", None, None
            coefficient = float(paired[x].corr(paired[y]))
            result = pd.DataFrame({"Column X": [x], "Column Y": [y], "Pearson correlation": [coefficient], "Rows used": [len(paired)]})
            plot = {"type": "scatter", "x": x, "y": y, "title": f"{y} vs {x}"}
            return f"Pearson correlation between {x} and {y}: {coefficient:.4f}. Correlation measures association and does not prove causation.", result, plot
        if len(numeric_cols) < 2:
            return "Correlation requires at least two numeric columns.", None, None
        corr = df[numeric_cols].corr(numeric_only=True)
        result = corr.reset_index(names="Column")
        plot = {"type": "heatmap", "columns": numeric_cols, "title": "Numeric correlation matrix"}
        return "Correlation matrix for numeric columns. Correlation does not prove causation:", result, plot

    if intent == "trend":
        date_col = plan["x_column"]
        if date_col not in date_cols or metric not in numeric_cols:
            return "A trend needs one detected date/time column and one numeric measure. Try naming both in your question.", None, None
        work = df[[date_col, metric]].copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.dropna(subset=[date_col])
        if work.empty:
            return "The selected date/time column has no valid dates.", None, None
        gran = plan["date_granularity"]
        freq = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}[gran]
        work["Period"] = work[date_col].dt.to_period(freq).dt.start_time
        val_name = f"{metric} {agg}"
        if agg == "count":
            result = work.groupby("Period")[metric].count().reset_index(name=val_name)
        else:
            result = work.groupby("Period")[metric].agg(lambda s: aggregate_value(s, agg)).reset_index(name=val_name)
        result = result.sort_values("Period")
        plot = {"type": "line", "x": "Period", "y": val_name, "title": plan["title"] or f"{gran.title()} trend of {metric}"}
        return f"{gran.title()} trend using {agg} of {metric}:", result, plot

    if intent in {"aggregate", "top_n"}:
        if intent == "top_n" and not groups:
            # A validated fallback to a categorical column, if one exists.
            groups = categorical_cols[:1]
        if intent == "aggregate" and not groups and metric is None:
            return "Please specify a numeric measure or a category for this aggregation.", None, None
        result = grouped_aggregate(df, groups, metric, agg)
        if result.empty:
            return "No results were produced. Check the selected columns and try again.", result, None
        value_columns = [c for c in result.columns if c not in groups]
        if intent == "top_n" and value_columns:
            sort_column = value_columns[0]
            result = result.sort_values(sort_column, ascending=False).head(limit)
        plot = None
        if groups and value_columns:
            plot = {"type": "bar", "x": groups[0], "y": value_columns[0], "title": plan["title"]}
        return message, result, plot

    return "Unable to complete this analysis. Try a clearer question with column names.", None, None


def summarize_result(question: str, result_text: str, model_name: str) -> str:
    """Generate a compact narrative from already-calculated results; no raw source rows sent."""
    from langchain_groq import ChatGroq
    llm = ChatGroq(model=model_name, temperature=0)
    prompt = (
        "You are a careful data analyst. Answer the user's question using only the calculated "
        "result supplied below. Do not invent facts. Keep the answer concise, mention the actual "
        "measure/group where possible, and state uncertainty if the result is insufficient.\n\n"
        f"Question: {question}\n\nCalculated result:\n{result_text[:12000]}"
    )
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    return str(content).strip()
