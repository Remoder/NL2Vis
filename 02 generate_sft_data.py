import os
import json
import pandas as pd
import glob, argparse
from openai import OpenAI
import time
from tqdm import tqdm

# --- Configuration ---
DEFAULT_KEY = "sk-EWwCihmo7aEgCAKZeVV82P3vdQcy6jBg02JBozZ3Ix7Q2ESu"
DEFAULT_BASE = "https://api.wenwen-ai.com/v1"

API_KEY = os.getenv("OPENAI_API_KEY", DEFAULT_KEY)
BASE_URL = os.getenv("OPENAI_API_BASE", DEFAULT_BASE)

if not BASE_URL.endswith("/v1"):
    BASE_URL += "/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def call_llm(user_prompt, system_prompt="You are an expert about text-to-SQL and pandas code.", max_retries=3, model="gpt-4o"):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ LLM Error (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(3) # 如果报错，停顿 3 秒再试，防止服务器过载
            else:
                print("❌ LLM API permanently failed for this request.")
                return ""

# --- Prompts ---
# DVCR: Data-Vis Co-Representation
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
```
"""

SQL2DVCR_PROMPT = """You are a Senior Data Scientist specializing in NL2Vis. Your task is to transform a natural language question, its corresponding Gold SQL, and Binning constraints into a structured DVCR 2.0 (Data-Vis Co-Representation).

### 🧠 CORE PHILOSOPHY (DVCR 2.0):
1. **[DATA_FLOW] (The Retrieval Layer)**: Focus ONLY on identifying raw data. Use `source`, `join`, and `where`. Do NOT perform visual aggregations or binning here.
2. **[VIS_TRANSFORM] (The Shaping Layer)**: Transform the raw data for visualization. This is where `groupby`, `aggregate`, `bin_by`, `orderby`, and `limit` belong.
3. **[VIS_CONFIG] (The Mapping Layer)**: Map the processed columns to chart axes.

You MUST strictly follow the "DVCR 2.0 Syntax Specification" below.

### 📜 DVCR 2.0 Syntax Specification:
1. **Structural Integrity**: Output must contain exactly four sections: [DATA_FLOW], [VIS_TRANSFORM], [VIS_CONFIG], and [EXECUTE]. Headers must be on their own lines.
2. **Identifier Formatting**: ALL table and column names must be enclosed in backticks using the fully qualified `Table`.`Column` format (e.g., `` `Physician`.`Name` ``).
3. **Layer Constraints**:
   - **[DATA_FLOW] Operators**: `source(t1, t2)`, `where(condition)`. Keep data at the atomic record level.
   - **[VIS_TRANSFORM] Operators**: 
     - `bin_by(`Col`, 'YEAR'|'MONTH'|'WEEKDAY')`: Use this for all time-based resampling.
     - `aggregate(func(`Col`) as `alias`)`: MANDATORY aliasing for all counts/sums/avgs.
     - `groupby(`Col`)`, `select(cols...)`, `orderby(col, asc|desc)`, `limit(n)`.
4. **[VIS_CONFIG]**:
   - Must be valid JSON.
   - Keys: `"chart"`, `"x_name"`, `"y_name"`. 
   - Refer only to identifiers or aliases defined in [VIS_TRANSFORM].
5. **[EXECUTE]**: Fixed format: `visualize(res, config=VIS_CONFIG)`

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
Generate the DVCR 2.0 representation. Align the [DATA_FLOW] with the raw data source and the [VIS_TRANSFORM] with the analytical intent and binning requirements.

```DVCR
"""

# --- Schema Analysis ---
class SchemaAnalyzer:
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.cache = {}

    def get_simple_schema(self, db_id):
        if db_id in self.cache:
            return self.cache[db_id]
            
        db_path = os.path.join(self.dataset_root, "databases", db_id)
        if not os.path.exists(db_path):
            return []
        
        csv_files = glob.glob(os.path.join(db_path, "*.csv"))
        simple_schema_list = []
        
        for path in csv_files:
            try:
                df = pd.read_csv(path, nrows=0) 
                table_name = os.path.basename(path).replace('.csv', '')
                df.columns = [c.strip() for c in df.columns]
                for col in df.columns:
                    safe_col = f"`{col}`" if " " in col else col
                    full_name = f"{table_name}.{safe_col}"
                    simple_schema_list.append(full_name)
            except Exception as e:
                print(f"Error reading {path}: {e}")
        
        self.cache[db_id] = simple_schema_list
        return simple_schema_list

    def get_full_schema(self, db_id):
        """ 将之前生成的 schema 关系同样传入 """
        simple_schema_list = self.get_simple_schema(db_id)
        
        # 1. 构建 schema JSON 的预期路径
        json_path = os.path.join(os.path.dirname(self.dataset_root), "temp_results", "schema", f"{db_id}.json")
        
        schema_context = None
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    schema_context = json.load(f)
            except Exception as e:
                print(f"Error reading schema JSON {json_path}: {e}")

        # 2. 如果存在上下文，将其与 simple_schema_list 合并返回
        if schema_context:
            # 假设 schema_context 是一个字典或列表，根据需求合并
            # 这里演示返回一个包含两者的字典，更清晰
            return {
                "simple_schema": simple_schema_list,
                "extended_context": schema_context
            }
            
        return {"simple_schema": simple_schema_list, "extended_context": None}


def main():
    # ---------------------------------------------------------
    # 1. 增加命令行参数解析
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Generate DVCR SFT Data")
    parser.add_argument("--db_id", type=str, default=None, help="指定要处理的数据库ID (例如: hospital_1)")
    args = parser.parse_args()

    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    vis_eval_path = os.path.join(project_root, "dataset/visEval.json")
    dataset_root = os.path.join(project_root, "dataset")
    output_file = os.path.join(project_root, "temp_results", "sft_data", "sft_origin_data.jsonl")
    
    # ---------------------------------------------------------
    # 2. 预处理：确保目录存在并清空旧的目标文件
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Cleaning and preparing output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        pass # 以 'w' 模式打开即清空文件内容

    print(f"Reading dataset from: {vis_eval_path}")
    if not os.path.exists(vis_eval_path):
        print(f"Error: {vis_eval_path} not found.")
        return

    with open(vis_eval_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    analyzer = SchemaAnalyzer(dataset_root)
    
    items = list(data.items())
    
    # ---------------------------------------------------------
    # 3. 筛选逻辑
    # ---------------------------------------------------------
    if args.db_id:
        print(f"Filtering items for Database: {args.db_id}")
        items = [(q_id, entry) for q_id, entry in items if entry.get('db_id') == args.db_id]

    if not items:
        print(f"No items found to process{' for ' + args.db_id if args.db_id else ''}.")
        return

    print(f"Found {len(items)} items. Processing...")
    
    results = []
    
    # Process batch
    for q_id, entry in tqdm(items): 
        db_id = entry.get('db_id')
        nl_queries = entry.get('nl_queries', [])
        
        try:
            data_part = entry['vis_query']['data_part']
            sql = data_part.get('sql_part', '')
            binning = data_part.get('binning', '')
        except KeyError:
            continue
            
        # Extract Visualization Ground Truth
        vis_obj = entry.get('vis_obj', {})
        gold_vis = {
            "chart": vis_obj.get('chart'),
            "x_name": vis_obj.get('x_name'),
            "y_name": vis_obj.get('y_name')
        }

        if not db_id or not sql or not nl_queries:
            continue

        schema = analyzer.get_full_schema(db_id)
        if not schema:
            continue

        # Use the first NL query for Teacher Forcing
        question = nl_queries[0]
        
        # 建议：这里将 schema 转为 json 字符串通常比直接 str(schema) 对 LLM 更友好
        schema_json = json.dumps(schema, ensure_ascii=False)

        user_prompt = SQL2DVCR_PROMPT.format(
            shots=DVCR_SHOTS,
            question=question,
            schema=schema_json,
            sql=sql,
            binning=binning,
            vis=json.dumps(gold_vis)
        )
        
        # 建议：注入你之前定义的 DVCR_PROTOCOL 强化约束
        system_prompt = "You are an expert about Data Visualization and Pandas. Always strictly follow the DVCR_PROTOCOL."
        
        dvcr_response = call_llm(system_prompt, user_prompt)
        
        # Clean response
        dvcr_content = dvcr_response.replace("```DVCR", "").replace("```", "").strip()
        
        if not dvcr_content:
            continue
            
        # Pair with all NL variations
        for nl_q in nl_queries:
            example = {
                "id": f"{q_id}",
                "db_id": db_id,
                "instruction": f"Generate DVCR for the following query based on schema: {schema_json}",
                "input": nl_q,
                "output": dvcr_content,
                "gold_sql": sql,
                "gold_vis": gold_vis
            }
            results.append(example)
            
            # 使用 'a' 模式追加。由于 main 开始时已清空，这里追加是安全的。
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(example) + "\n")
                
    print(f"Done! Generated {len(results)} DVCR examples in {output_file}")

if __name__ == "__main__":
    main()
