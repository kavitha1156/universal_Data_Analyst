import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_engine import infer_columns, validate_plan, execute_plan, load_dataframe


def test_load_csv_and_duplicate_headers():
    raw = b"Region,Region,Revenue\nNorth,West,20\nSouth,East,10\n"
    df = load_dataframe(raw, "sales.csv")
    assert list(df.columns) == ["Region", "Region.1", "Revenue"]
    assert len(df) == 2


def test_infer_columns_and_aggregate():
    df = pd.DataFrame({"Region": ["North", "South", "North"], "Revenue": [10, 20, 30]})
    num, cat, dates = infer_columns(df)
    plan = validate_plan({
        "intent": "aggregate", "group_by": ["Region"], "metric": "Revenue",
        "aggregation": "sum", "chart": "bar"
    }, df, num, cat, dates)
    _, result, plot = execute_plan(df, plan, num, cat, dates)
    assert result.set_index("Region")["Revenue sum"].to_dict() == {"North": 40, "South": 20}
    assert plot["type"] == "bar"


def test_missing_values_summary():
    df = pd.DataFrame({"A": [1, None], "B": ["x", "y"]})
    num, cat, dates = infer_columns(df)
    plan = validate_plan({"intent": "missing"}, df, num, cat, dates)
    _, result, _ = execute_plan(df, plan, num, cat, dates)
    assert int(result.loc[result["Column"] == "A", "Missing values"].iloc[0]) == 1


def test_monthly_trend():
    df = pd.DataFrame({
        "Order Date": ["2025-01-01", "2025-01-17", "2025-02-02"],
        "Total Revenue": [100, 50, 300],
    })
    num, cat, dates = infer_columns(df)
    assert "Order Date" in dates
    plan = validate_plan({
        "intent": "trend", "x_column": "Order Date", "metric": "Total Revenue",
        "aggregation": "sum", "date_granularity": "month", "chart": "line"
    }, df, num, cat, dates)
    _, result, plot = execute_plan(df, plan, num, cat, dates)
    assert len(result) == 2
    assert result["Total Revenue sum"].tolist() == [150, 300]
    assert plot["type"] == "line"
