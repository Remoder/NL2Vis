"""Centralized DVCR protocol and prompt builders.

编辑建议：
1. 常改规则：优先修改 `DVCR_CORE_PHILOSOPHY` / `DVCR_SYNTAX_SPEC`。
2. 常改示例：修改 `DVCR_SHOTS`。
3. 业务场景差异：修改对应的 build_* 函数尾部任务说明。
"""

DVCR_PROTOCOL_SYSTEM_PROMPT = (
    "You are an expert about Data Visualization and Pandas. "
    "Always strictly follow the DVCR_PROTOCOL."
)


DVCR_CORE_PHILOSOPHY = """### CORE PHILOSOPHY:
- Modular Reasoning: Separate "data retrieval" ([DATA_FLOW]) from "visual transformation" ([VIS_TRANSFORM]).
- Dimensionality: If the question implies a comparison (e.g., "by gender", "stacked"), identify the classification column and assign it to the "color" key in [VIS_CONFIG].
- Casing: Use the EXACT casing for table and column names provided in the schema.
- Preserving Zeroes: When the question asks for "each" item or compares two independent groups (e.g., "faculties vs students per activity"), you MUST use **LEFT JOIN** or **FULL JOIN** to ensure items with zero counts are not filtered out. Mismatched data lengths (e.g., 12 instead of 14) will cause total failure.
"""


DVCR_SYNTAX_SPEC = """### DVCR 2.1 Syntax Specification (Multi-Dimensional Support)
All DVCR outputs must strictly adhere to this four-section structure:

1. [DATA_FLOW] (The Retrieval Layer):
- Use `source(table1, table2, ...)` to identify tables.
- Use `.where(condition)` for row-level filtering.
- Use `.join()` for associations. Do NOT aggregate here.

2. [VIS_TRANSFORM] (The Shaping Layer):
- Use `.bin_by(`Col`, 'YEAR'|'MONTH'|'WEEKDAY')` for time resampling.
- Use `.groupby([cols...])` for defining dimensions (X-axis and optional Color-axis).
- Use `.aggregate(func(`Col`) as `alias`)` for calculations. Assign aliases to ALL aggregations.
- Use `.orderby(col, asc|desc)` and `.limit(n)` for sorting.

3. [VIS_CONFIG] (The Mapping Layer):
- JSON keys: "chart" (bar, line, scatter, pie), "x_name", "y_name", "color" (optional).
- PROHIBITION: Refer only to columns or aliases defined in [VIS_TRANSFORM].

4. [EXECUTE]:
- Fixed format: `visualize(res, config=VIS_CONFIG)`
"""


DVCR_SHOTS = """question = "Show the number of faculty members for each rank in a bar chart."
schema = ['Faculty.Rank', 'Faculty.FacID']
gold_sql = "SELECT Rank, COUNT(FacID) FROM Faculty GROUP BY Rank"
gold_vis = {"chart": "bar", "x_name": "Faculty.Rank", "y_name": "count(Faculty.FacID)"}

```DVCR
[DATA_FLOW]
# Stage 1: Retrieve raw records from source tables
df1 = source(`Faculty`)

[VIS_TRANSFORM]
# Stage 2: Transform raw data into visual aggregates
df2 = df1.groupby(`Faculty`.`Rank`).aggregate(count(`Faculty`.`FacID`) as `member_count`)
res = df2.select(`Faculty`.`Rank`, `member_count`)

[VIS_CONFIG]
{
  "chart": "bar",
  "x_name": "Faculty.Rank",
  "y_name": "member_count",
  "intent": "Comparison"
}

[EXECUTE]
visualize(res, config=VIS_CONFIG)
```

question = "Use a scatter plot to show the relationship between age and GPA."
schema = ['Student.Age', 'Student.GPA']
gold_sql = "SELECT Age, GPA FROM Student"
gold_vis = {"chart": "scatter", "x_name": "Student.Age", "y_name": "Student.GPA"}

```DVCR
[DATA_FLOW]
df1 = source(`Student`)

[VIS_TRANSFORM]
res = df1.select(`Student`.`Age`, `Student`.`GPA`)

[VIS_CONFIG]
{
  "chart": "scatter",
  "x_name": "Student.Age",
  "y_name": "Student.GPA",
  "intent": "Correlation"
}

[EXECUTE]
visualize(res, config=VIS_CONFIG)
```

question = "What is the highest eligible free rate for K-12 students in Alameda County?"
schema = ['schools.County', 'frpm.Free Meal Count (K-12)', 'frpm.Enrollment (K-12)', 'schools.School Name']
gold_sql = "SELECT (Free Meal Count (K-12) / Enrollment (K-12)) FROM frpm JOIN schools ON ... WHERE County = 'Alameda' ORDER BY ... LIMIT 1"
gold_vis = {"chart": "bar", "x_name": "School Name", "y_name": "Rate"}

```DVCR
[DATA_FLOW]
# Identify and filter the raw data needed
df1 = source(`schools`, `frpm`).where(`schools`.`County` == 'Alameda')

[VIS_TRANSFORM]
# Perform calculation and ranking for visualization
df1[`free_rate`] = `frpm`.`Free Meal Count (K-12)` / `frpm`.`Enrollment (K-12)`
df2 = df1.orderby(`free_rate`, desc).limit(1)
res = df2.select(`schools`.`School Name`, `free_rate`)

[VIS_CONFIG]
{
  "chart": "bar",
  "x_name": "School Name",
  "y_name": "free_rate",
  "intent": "Rank"
}

[EXECUTE]
visualize(res, config=VIS_CONFIG)
```

question = "How many invoices were issued each year? Show as a line chart."
schema = ['Invoice.InvoiceDate', 'Invoice.InvoiceId']
gold_sql = "SELECT InvoiceDate, COUNT(InvoiceId) FROM Invoice GROUP BY InvoiceDate"
binning = "BIN InvoiceDate BY YEAR"

```DVCR
[DATA_FLOW]
df1 = source(`Invoice`)

[VIS_TRANSFORM]
# Use bin_by to handle visualization resampling (e.g., YEAR, MONTH, WEEKDAY)
df2 = df1.bin_by(`Invoice`.`InvoiceDate`, 'YEAR').aggregate(count(`Invoice`.`InvoiceId`) as `invoice_count`)
res = df2.select(`Invoice`.`InvoiceDate`, `invoice_count`)

[VIS_CONFIG]
{
  "chart": "line",
  "x_name": "Invoice.InvoiceDate",
  "y_name": "invoice_count",
  "intent": "Trend"
}

[EXECUTE]
visualize(res, config=VIS_CONFIG)

question = "Show the number of faculty members for each rank and gender using a stacked bar chart."
gold_sql = "SELECT Rank, Sex, COUNT(FacID) FROM Faculty GROUP BY Rank, Sex"

```DVCR
[DATA_FLOW]
df1 = source(`Faculty`)

[VIS_TRANSFORM]
df2 = df1.groupby([`Faculty`.`Rank`, `Faculty`.`Sex`]).aggregate(count(`Faculty`.`FacID`) as `member_count`)
res = df2.select([`Faculty`.`Rank`, `Faculty`.`Sex`, `member_count`]).orderby(`member_count`, asc)

[VIS_CONFIG]
{
  "chart": "bar",
  "x_name": "Faculty.Rank",
  "y_name": "member_count",
  "color": "Faculty.Sex",
  "intent": "Comparison"
}

[EXECUTE]
visualize(res, config=VIS_CONFIG)
```
"""


def build_sql2dvcr_prompt(question, sql, binning, vis, schema, shots=DVCR_SHOTS):
    return """You are a Senior Data Scientist specializing in NL2Vis. Your task is to transform a natural language question, its corresponding Gold SQL, and Binning constraints into a structured DVCR 2.1 plan.

{core}

You MUST strictly follow the "DVCR 2.1 Syntax Specification" below.

{syntax}

### CRITICAL
1. If the question asks for "each [Entity]" or "relationship between [A] and [B]", you MUST use LEFT JOIN from the main entity table to preserve zero counts. Results with missing rows (due to INNER JOIN) will fail validation.

### 📚 Examples:
{shots}

---
### INPUT DATA:
- **Question**: "{question}"
- **Gold SQL**: "{sql}"
- **Visualization Constraint (Binning)**: "{binning}"
- **Gold Vis Ground Truth**: {vis}
- **Enriched Schema**: {schema}

### TASK:
Generate the DVCR 2.1 representation. Align the [DATA_FLOW] with the raw data source and the [VIS_TRANSFORM] with the analytical intent and binning requirements. And ensure multi-dimensional intents are captured via the 'color' key and correct 'groupby' logic.

```DVCR
""".format(
        core=DVCR_CORE_PHILOSOPHY,
        syntax=DVCR_SYNTAX_SPEC,
        shots=shots,
        question=question,
        sql=sql,
        binning=binning,
        vis=vis,
        schema=schema,
    )


def build_sr2sql_check_prompt(question, vis_config, schema, data_flow, vis_transform):
    return """You are a SQL Translation Expert. Your task is to synthesize a single, executable SQLite/DuckDB SQL query from two logic sections of a DVCR 2.1 plan.

{core}

You MUST strictly follow the "DVCR 2.1 Syntax Specification" below.

{syntax}

### 🛠 TRANSLATION RULES:
1. Retrieval: Map [DATA_FLOW] to FROM, JOIN, and WHERE clauses.
2. Reshaping: Map [VIS_TRANSFORM] to SELECT, GROUP BY, and ORDER BY.
3. Multi-Dimension: If "color" exists in [VIS_CONFIG], it MUST be in both SELECT and GROUP BY.
4. Binning: Map `bin_by(Col, 'YEAR')` to `strftime('%Y', Col)`, 'MONTH' to `strftime('%m', Col)`, 'WEEKDAY' to `strftime('%w', Col)`.
5. Strictness: All non-aggregated columns in SELECT must be in GROUP BY. No CTEs.
6. Identifiers: Use full `Table`.`Column` names.

### TASK:
Synthesize the provided DVCR 2.0 sections into one valid SQL query.

### CONTEXT:
- **Question**: "{question}"
- **Target Visualization**: {vis_config}
- **Database Schema**: {schema}

### INPUT DVCR SECTIONS:
**[DATA_FLOW]**: 
{data_flow}

**[VIS_TRANSFORM]**: 
{vis_transform}

### INSTRUCTIONS:
1. Use the "Target Visualization" and "VIS_TRANSFORM" to determine the final SELECT projections.
2. Maintain the aliases defined in `aggregate(... as alias)`.
3. Ensure there is a SPACE after the SELECT keyword.
4. Reference columns exactly as defined in the DVCR (using backticks if provided).

```sqlite
""".format(
        core=DVCR_CORE_PHILOSOPHY,
        syntax=DVCR_SYNTAX_SPEC,
        question=question,
        vis_config=vis_config,
        schema=schema,
        data_flow=data_flow,
        vis_transform=vis_transform,
    )
