# Universal AI Data Analyst

A general-purpose data analysis dashboard for CSV, XLSX, and XLS files. It does not hardcode sales-specific columns and does not generate/load sample data automatically.

## Features

- File upload with support for CSV, XLSX, and XLS
- Excel worksheet selection for workbooks with multiple sheets
- Automatic data profiling: dimensions, types, missing values, unique values, duplicates, numeric summary
- Dynamic chart builder using columns detected in the uploaded file
- Natural-language Q&A powered by Groq + LangChain
- The LLM chooses from predefined analysis operations; Pandas performs the calculations locally
- Results are displayed as a narrative, chart when appropriate, and supporting table

## Setup (Windows / VS Code)

1. Extract this folder and open it in VS Code.
2. Open Terminal > New Terminal.
3. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

4. Install the packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Copy `.env.example` to `.env` and insert your Groq key:

```env
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Create a key at https://console.groq.com/keys. Check the current model list at https://console.groq.com/docs/models.

6. Run the app:

```powershell
streamlit run app.py
```

Streamlit will print a local URL, normally http://localhost:8501.

## Example questions

- What is the total revenue by product?
- Which region has the highest total profit?
- Compare online and offline sales.
- Show monthly revenue trend.
- What are the top 10 countries by total sales?
- Show missing values in every column.
- What are the strongest correlations between numeric columns?
- Show the first 20 records.

Questions work best when their measures and groups match existing column names. If a file has generic or ambiguous columns, mention their exact names in your question.

## Important notes

- No sample CSV is automatically created or loaded. Upload your actual dataset from the sidebar.
- The model receives the question and column metadata for planning, not the entire raw dataset. A small calculated result table may be sent to Groq for a written explanation. Do not use sensitive data unless you have permission to send its metadata and aggregate results to the API provider.
- Calculations use predefined pandas operations. The app does not run model-generated Python code.
- Date detection is heuristic. Check detected date columns and use the exact date column name in trend questions when needed.
- This is a project starter, not a fully hardened multi-user production deployment.
