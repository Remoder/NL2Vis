import os
import sys
import json
import pandas as pd
import re
from openai import OpenAI
import traceback
import argparse
import glob
import warnings

# --- Configuration ---
# 配置 API Key 和 Base URL，默认使用环境变量，否则使用硬编码的默认值
DEFAULT_KEY = "sk-EWwCihmo7aEgCAKZeVV82P3vdQcy6jBg02JBozZ3Ix7Q2ESu"
DEFAULT_BASE = "https://api.wenwen-ai.com/v1"

API_KEY = os.getenv("OPENAI_API_KEY", DEFAULT_KEY)
BASE_URL = os.getenv("OPENAI_API_BASE", DEFAULT_BASE)

if not BASE_URL.endswith("/v1"):
    BASE_URL += "/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def call_llm(system_prompt, user_prompt, model="gpt-4o"):
    """
    调用 LLM 的通用函数。
    system_prompt: 系统提示词，设定角色。
    user_prompt: 用户提示词，包含具体任务。
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0 # 设置为 0 以保证输出的确定性
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        return ""

# ==========================================
# Prompt Templates (DVCR -> Python)
# ==========================================

# 核心 Prompt：将 DVCR (Data-Vis Co-Representation) 中间表示转换为 Python 代码
# 这里包含了详细的思维链 (CoT) 和约束条件
DVCR2PYTHON_PROMPT = """You are an expert of Python Data Visualization. 
Your task is to convert a "DVCR 2.0" logic plan into executable Python code using `pandas` and `matplotlib`.

## Input Information:
1. Schema: {schema}
2. Column Description:
{column_description}
3. Question: "{question}"
4. Relationship: "{relationship}"
5. DVCR 2.0 (Logic Plan):
```DVCR
{dvcr}
```

## PANDAS IMPLEMENTATION RULES (MANDATORY):

1. Naming & Bracket Notation:
ALL DataFrames are pre-processed. Column names are exactly Table.Column (e.g., 'Physician.Name').
MANDATORY: You MUST use bracket notation: df['Table.Column']. Never use dot notation like df.Table.Column.

2. [DATA_FLOW] Implementation (Data Retrieval):
Joins: Use safe_merge(df1, df2, left_on='...', right_on='...', how='inner').
Filtering: Use boolean indexing: df = df[df['Table.Col'] == val].
Cross-Table Logic (CRITICAL): NEVER compare columns from two different DataFrames directly (e.g., df1[df1['A'] == df2['B']]). This causes a "identically-labeled Series" error.
MANDATORY: You MUST use safe_merge first to join tables, then perform the comparison on the columns of the same merged DataFrame.

3. [VIS_TRANSFORM] Implementation (Data Shaping):
bin_by(Col, 'YEAR'|'MONTH'|'WEEKDAY'):
Convert to datetime: df['Col'] = pd.to_datetime(df['Col']).
Extract:
'YEAR' -> df['Col'] = df['Col'].dt.year.
'MONTH' -> df['Col'] = df['Col'].dt.month.
'WEEKDAY' -> df['Col'] = df['Col'].dt.day_name().str[:3].
aggregate(...): Use .groupby(...).agg(alias=('Col', 'func')).reset_index().
orderby(...): Use df = df.sort_values(by='...', ascending=True/False). Ensure this is done BEFORE plotting.
MANDATORY: ALWAYS call .reset_index() after any groupby to keep columns accessible.

4. [VIS_CONFIG] Implementation (Final Mapping):
The final DataFrame MUST contain columns that match the exact strings in VIS_CONFIG['x_name'] and VIS_CONFIG['y_name'].
Mapping: If you created an alias or renamed a column during transformation, assign it back: df[VIS_CONFIG['x_name']] = df['your_computed_column'].

## ABSOLUTE MANDATORY RULES (FORBIDDEN ACTIONS):
1. **DO NOT REDEFINE FUNCTIONS**: 
   - NEVER write `def safe_merge` or `def safe_pie_plot` in your code. They are ALREADY DEFINED in the environment. Redefining them will break the execution.
2. **USE safe_merge ONLY**: 
   - NEVER use the pandas method `df.merge()`. 
   - ALWAYS use the global function `safe_merge(df1, df2, left_on='...', right_on='...', how='inner')`.
3. **TYPE SAFETY**: 
   - Our `safe_merge` handles string casting for you. Do not attempt to manually cast IDs to strings.
4. **NO .values ASSIGNMENT**: 
   - Never assign columns using `.values`. If you need to join data, use `safe_merge`.

## Execution Requirements:
No pd.read_csv: DataFrames matching Schema table names are already in the namespace.
Helper Functions: Use safe_merge, safe_bar_plot(x, y, ...), and safe_pie_plot(x, y, ...).
Safety: Wrap plotting in if not df.empty:.
Output: Return ONLY the Python code block inside python ....
"""

# ==========================================
# DVCR Validator & Constraint Checker (Innovation Module)
# ==========================================
class DVCRValidator:
    """
    DVCR 2.0 验证器：支持 [VIS_TRANSFORM] 段落及别名追踪逻辑。
    """
    def __init__(self, column_types=None):
        self.column_types = column_types or {} # {"Table.Col": "dtype"}
        self.chart_requirements = {
            "bar": ["x_name", "y_name"],
            "line": ["x_name", "y_name"],
            "scatter": ["x_name", "y_name"],
            "pie": ["x_name", "y_name"], 
            "box": ["x_name", "y_name"], 
            "histogram": ["x_name"]
        }

    def _extract_section(self, text, section_name):
        """支持 DVCR 2.0 标签解析"""
        pattern = rf"\[\s*{section_name}\s*\](.*?)(?=\[\s*\w+\s*\]|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _fix_json(self, json_str):
        try:
            clean_json = re.sub(r'```json|```', '', json_str).strip()
            return json.loads(clean_json)
        except:
            fixed = clean_json.replace("'", '"')
            fixed = re.sub(r',\s*}', '}', fixed)
            try: return json.loads(fixed)
            except: return None

    def validate(self, dvcr_text):
        """
        升级版静态检查 (适配 DVCR 2.0)
        """
        messages = []
        is_valid = True

        # 1. 结构完整性检查 (增加 VIS_TRANSFORM)
        data_flow = self._extract_section(dvcr_text, "DATA_FLOW")
        vis_transform = self._extract_section(dvcr_text, "VIS_TRANSFORM")
        vis_config_str = self._extract_section(dvcr_text, "VIS_CONFIG")

        if not data_flow:
            return False, ["❌ Missing [DATA_FLOW] section."]
        if not vis_transform:
            # 2.0 允许 transform 为空（如果不做任何变换），但标签应该存在
            messages.append("⚠️ Note: [VIS_TRANSFORM] section is missing or empty.")
        if not vis_config_str:
            return False, ["❌ Missing [VIS_CONFIG] section."]

        # 2. JSON 有效性检查
        vis_config = self._fix_json(vis_config_str)
        if not vis_config:
            return False, ["❌ [VIS_CONFIG] is not valid JSON."]

        # 3. 语义约束检查 (Chart Type)
        chart_type = vis_config.get("chart") or vis_config.get("chart_type")
        if not chart_type:
            messages.append("❌ Missing 'chart' key in VIS_CONFIG.")
            is_valid = False
        else:
            req_fields = self.chart_requirements.get(chart_type.lower(), [])
            for f in req_fields:
                # 同时兼容 x_axis/x_name
                if f not in vis_config and f.replace("name", "axis") not in vis_config:
                    messages.append(f"❌ Chart '{chart_type}' requires '{f}'.")
                    is_valid = False

        # 4. 数据流与变换层字段存在性检查
        # 提取两个段落中所有 `Table`.`Column` 格式的列
        all_logic_text = (data_flow or "") + " " + (vis_transform or "")
        referenced_cols_raw = re.findall(r'`([^`\s]+)`\.`([^`\s]+)`', all_logic_text)
        referenced_cols = [f"{t}.{c}" for t, c in referenced_cols_raw]
        
        if self.column_types:
            schema_keys = {k.replace('`',''): v for k,v in self.column_types.items()}
            for ref in referenced_cols:
                if ref not in schema_keys:
                    messages.append(f"⚠️ Warning: Referenced column '{ref}' not found in schema.")

        # 5. 别名追踪 (Alias Tracking)
        # 在 VIS_TRANSFORM 中寻找 `as `alias`` 模式
        aliases = re.findall(r'as\s+`([^`\s]+)`', vis_transform or "", re.IGNORECASE)
        
        # 6. 可视化配置字段合规性检查
        used_cols_in_vis = []
        for key in ["x_name", "x_axis", "y_name", "y_axis", "color"]:
            if key in vis_config: used_cols_in_vis.append(str(vis_config[key]))

        for col in used_cols_in_vis:
            clean_col = col.replace("`", "").strip()
            # 只要列在原始 Schema 中，或者在 VIS_TRANSFORM 中被定义为别名，就放行
            in_schema = any(clean_col == k.split('.')[-1] or clean_col == k for k in self.column_types.keys())
            is_alias = clean_col in aliases
            
            if not in_schema and not is_alias:
                 # 排除 count(*) 等特殊情况
                 if "(" not in clean_col and clean_col != "*":
                    messages.append(f"⚠️ Warning: Column/Alias '{clean_col}' in VIS_CONFIG may not be defined in logic.")

        # 7. 聚合规范检查
        if "groupby" in (vis_transform or "") and "reset_index" not in (vis_transform or ""):
             messages.append("💡 Suggestion: Ensure '.reset_index()' is used after groupby in Python implementation.")

        if not messages:
            messages.append("✅ DVCR 2.0 Validated Successfully.")

        return is_valid, messages
                
# ==========================================
# Schema Analyzer
# ==========================================
class SchemaAnalyzer:
    """
    Schema 分析器：读取 CSV 文件，提取表名、列名、数据类型和样本值。
    """
    def analyze(self, csv_paths):
        schema_info = []
        column_descriptions = [] 
        simple_schema_list = []
        column_types = {}
        
        for path in csv_paths:
            try:
                df = pd.read_csv(path)
                table_name = os.path.basename(path).replace('.csv', '')
                df.columns = [c.strip() for c in df.columns]
                
                cols = list(df.columns)
                for col in cols:
                    safe_col = f"`{col}`" if " " in col else col
                    full_name = f"{table_name}.{safe_col}"
                    simple_schema_list.append(full_name)
                    column_types[full_name] = str(df[col].dtype)
                    
                    # 提取样本值用于 Prompt
                    examples = df[col].dropna().unique()
                    example_str_list = []
                    for ex in examples[:3]:
                        if isinstance(ex, str):
                            example_str_list.append(f"'{ex}'")
                        else:
                            example_str_list.append(str(ex))
                    ex_str = ", ".join(example_str_list)
                    
                    desc_text = f"The column '{col}' in table '{table_name}'."
                    desc_line = f"# {full_name}: {desc_text} Type: {df[col].dtype}. Example values: [{ex_str}]."
                    column_descriptions.append(desc_line)
                
            except Exception as e:
                print(f"Error reading {path}: {e}")

        col_desc_str = "\n".join(column_descriptions)
        return {
            "simple_schema_list": simple_schema_list,
            "column_descriptions": col_desc_str,
            "column_types": column_types
        }

# ==========================================
# Coder
# ==========================================
class Coder:
    """
    代码生成器：负责调用 LLM 生成代码，并注入必要的 Header 和 Helper Functions。
    """
    def gen_code(self, dvcr_text, query, csv_files, db_path, schema_data):
        simple_schema = schema_data['simple_schema_list']
        col_desc = schema_data['column_descriptions']
        relationship = schema_data["relationship"]

        user_prompt = DVCR2PYTHON_PROMPT.format(
            schema=simple_schema,
            column_description=col_desc,
            question=query,
            relationship=relationship,
            dvcr=dvcr_text
        )

        system_prompt = "You are a Python Visualization Expert. You MUST follow the instructions about JOIN types strictly."
        code_resp = call_llm(system_prompt, user_prompt)
        
        match = re.search(r'```python(.*?)```', code_resp, re.DOTALL)
        if match:
            llm_code = match.group(1).strip()
        else:
            llm_code = code_resp

        # --- 注入代码头部 (Header Injection) ---
        injection_lines = []
        injection_lines.append("# --- Auto-Injected Header ---")
        injection_lines.append("import matplotlib.pyplot as plt")
        injection_lines.append("import pandas as pd")
        injection_lines.append("import os")
        injection_lines.append("import numpy as np")
        injection_lines.append("import warnings")
        injection_lines.append("warnings.filterwarnings('ignore')")
        injection_lines.append("")
        
        injection_lines.append(f"data_dir = '{db_path}'")
        injection_lines.append("")
        
        all_tables = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

        # --- 自动加载数据 (Data Loading Injection) ---
        # 自动生成读取 CSV 的代码，并进行基础的数据清洗（去空格、类型转换）
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            table_name = os.path.splitext(filename)[0]
            
            injection_lines.append(f"if '{table_name}' not in locals():")
            injection_lines.append(f"    try:")
            injection_lines.append(f"        {table_name} = pd.read_csv(os.path.join(data_dir, '{filename}'))")
            injection_lines.append(f"    except FileNotFoundError:")
            injection_lines.append(f"        {table_name} = pd.read_csv('{filename}')")
            
            # --- 【关键修改】：清洗列名，并强制加上 "表名." 前缀 ---
            injection_lines.append(f"if not any(str(c).startswith('{table_name}.') for c in {table_name}.columns):")
            injection_lines.append(f"    {table_name}.columns = [c.strip() for c in {table_name}.columns]")
            
            # 尝试修复 ID 类列的类型（转为字符串），注意：此时还没有加前缀
            injection_lines.append(f"    for col in {table_name}.columns:")
            injection_lines.append(f"        if col in {all_tables} or col.endswith('ID') or col.endswith('Code') or col.endswith('SSN') or col == 'Treatment':")
            injection_lines.append(f"            if {table_name}[col].dtype != 'object':")
            injection_lines.append(f"                {table_name}[col] = {table_name}[col].astype(str)")
            
            # --- 强制重命名为 Table.Column 格式 ---
            injection_lines.append(f"    {table_name}.columns = [f'{table_name}.{{c}}' for c in {table_name}.columns]")
            injection_lines.append("")
            
        # --- 注入辅助函数 (Helper Functions Injection) ---
        injection_lines.append("# --- Helper Functions ---")
        # safe_merge: 自动处理连接键类型不一致的问题
        injection_lines.append("def safe_merge(df1, df2, left_on=None, right_on=None, on=None, how='inner'):")
        injection_lines.append("    if on:")
        injection_lines.append("        left_on = [on] if isinstance(on, str) else on")
        injection_lines.append("        right_on = [on] if isinstance(on, str) else on")
        injection_lines.append("    else:")
        injection_lines.append("        left_on = [left_on] if isinstance(left_on, str) else left_on")
        injection_lines.append("        right_on = [right_on] if isinstance(right_on, str) else right_on")
        injection_lines.append("    ")
        injection_lines.append("    # Force cast to string to prevent int64 vs str merge errors")
        injection_lines.append("    if left_on:")
        injection_lines.append("        for k in left_on:")
        injection_lines.append("            if k in df1.columns: df1[k] = df1[k].astype(str)")
        injection_lines.append("    if right_on:")
        injection_lines.append("        for k in right_on:")
        injection_lines.append("            if k in df2.columns: df2[k] = df2[k].astype(str)")
        injection_lines.append("                ")
        # 【关键修复】：只传 left_on 和 right_on，绝对不要再传 on=on 进去！
        injection_lines.append("    return pd.merge(df1, df2, left_on=left_on, right_on=right_on, how=how)")
        injection_lines.append("")
        
        # safe_pie_plot / safe_bar_plot: 封装绘图逻辑，增加判空保护
        injection_lines.append("def safe_pie_plot(x, y, title=''):")
        injection_lines.append("    if len(y) > 0 and len(x) == len(y):")
        injection_lines.append("        plt.figure(figsize=(8, 8))")
        injection_lines.append("        plt.pie(y, labels=x, autopct='%1.1f%%')")
        injection_lines.append("        plt.title(title)")
        injection_lines.append("        plt.savefig('output.png')")
        injection_lines.append("    else:")
        injection_lines.append("        print('Warning: Data empty or mismatch for pie chart')")
        injection_lines.append("")
        injection_lines.append("def safe_bar_plot(x, y, title='', xlabel='', ylabel=''):")
        injection_lines.append("    if len(y) > 0 and len(x) == len(y):")
        injection_lines.append("        plt.figure(figsize=(10, 6))")
        injection_lines.append("        plt.bar(x, y)")
        injection_lines.append("        plt.title(title)")
        injection_lines.append("        plt.xlabel(xlabel)")
        injection_lines.append("        plt.ylabel(ylabel)")
        injection_lines.append("        plt.xticks(rotation=45)")
        injection_lines.append("        plt.savefig('output.png')")
        injection_lines.append("    else:")
        injection_lines.append("        print('Warning: Data empty or mismatch for bar chart')")
        injection_lines.append("")
            
        final_code = "\n".join(injection_lines) + "\n" + llm_code
        return final_code

# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Generate Python Code from DVCR (SFT Data)")
    # parser.add_argument("--sft_file", type=str, default="", help="Path to SFT data file")
    parser.add_argument("--db_id", type=str, required=True, help="Database ID (e.g. activity_1) to filter and process")
    # parser.add_argument("--output_dir", type=str, default="generated_codes_dvcr", help="Output directory for python codes")
    args = parser.parse_args()

    # sft_file = args.sft_file
    target_db_id = args.db_id
    
    # 路径设置
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir) 
    dataset_root = os.path.join(project_root, "dataset")
    sft_file = os.path.join(project_root, "temp_results", "sft_data", "sft_advised_data.jsonl")
    
    # if not os.path.exists(dataset_root):
    #     dataset_root = "/root/nl2vis/VisEval-main/visEval_dataset"

    db_path = os.path.join(dataset_root, "databases", target_db_id)
    if not os.path.exists(db_path):
        print(f"Error: Database path {db_path} does not exist.")
        return

    csv_files = glob.glob(os.path.join(db_path, "*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return

    # 1. Analyze Schema (分析数据库结构)
    print(f"Analyzing schema for {target_db_id}...")
    analyzer = SchemaAnalyzer()
    schema_data = analyzer.analyze(csv_files)
    schema_json_path = os.path.join(project_root, "temp_results/schema", f"{target_db_id}.json")
    with open(schema_json_path, 'r') as f:
        schema_enhanced = json.load(f)
        relationship = schema_enhanced.get('relationship', [])
    schema_data["relationship"] = relationship

    # 2. Load SFT Data (加载 DVCR 训练数据)
    print(f"Loading SFT data from {sft_file}...")
    data_to_process = []
    with open(sft_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get('db_id') == target_db_id:
                data_to_process.append(item)

    print(f"Found {len(data_to_process)} samples for DB: {target_db_id}")

    output_path = os.path.join(project_root, "temp_results", "generated_code", target_db_id)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    coder = Coder()
    validator = DVCRValidator(schema_data.get('column_types'))

    # 3. Generate Code (循环处理每条数据)
    id_counts = {}

    for i, item in enumerate(data_to_process):
        q_id = item['id']
        question = item['input']
        dvcr_content = item['output'] # This is the DVCR plan
        
        if q_id not in id_counts:
            id_counts[q_id] = 0
        else:
            id_counts[q_id] += 1
            
        current_idx = id_counts[q_id]
        
        print(f"[{i+1}/{len(data_to_process)}] Processing Query ID: {q_id} (Index: {current_idx})")
        
        # --- Innovation: DVCR Validation (静态检查) ---
        is_valid, report_msgs = validator.validate(dvcr_content)
        if not is_valid:
            print(f"    ❌ DVCR Validation Failed (Proceeding with warnings):")
            for msg in report_msgs:
                print(f"        {msg}")
        else:
            if any("Warning" in m for m in report_msgs):
                 print(f"    ⚠️ DVCR Warnings:")
                 for msg in report_msgs:
                     if "Warning" in msg: print(f"        {msg}")
            else:
                 print(f"    ✅ DVCR Validated.")
        # -----------------------------------
        
        try:
            # 生成代码
            code = coder.gen_code(dvcr_content, question, csv_files, db_path, schema_data)
            
            # Save with correct index suffix
            safe_name = f"{q_id}_{current_idx}.txt" 
            output_file = os.path.join(output_path, safe_name)
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"    ✅ Saved to {os.path.basename(output_file)}")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
