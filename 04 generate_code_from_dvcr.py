import os
import sys
import json
import pandas as pd
import re, keyword
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
DVCR2PYTHON_PROMPT = """You are an expert Python Data Scientist. 
Your task is to convert a "DVCR 2.1" logic plan into executable Python code using `pandas`, `matplotlib`.

## Input Information:
1. **Schema**: {schema}
2. **Column Description**: {column_description}
3. **Database Relationships**: {relationship}
4. **Natural Language Question**: "{question}"
5. **DVCR 2.1 (Logic Plan)**:
```DVCR
{dvcr}
```

## PANDAS IMPLEMENTATION RULES (MANDATORY):

1 .Naming & Bracket Notation:
ALL DataFrames are pre-processed. Column names are exactly Table.Column.
MANDATORY: You MUST use bracket notation: df['Table.Column']. Never use dot notation like df.Table.Column.

2. [DATA_FLOW] Implementation (Data Retrieval):
Joins: Use the provided Relationships to identify keys. Use safe_merge(df1, df2, left_on='...', right_on='...', how='inner').
Cross-Table Logic: NEVER compare columns from two different DataFrames directly. You MUST use safe_merge first to join tables, then perform comparison on the resulting single DataFrame.

3. [VIS_TRANSFORM] Implementation (Data Shaping):
bin_by(Col, 'YEAR'|'MONTH'|'WEEKDAY'):
Convert to datetime: df['Col'] = pd.to_datetime(df['Col']).
Extract: 'YEAR' -> .dt.year, 'MONTH' -> .dt.month, 'WEEKDAY' -> .dt.day_name().str[:3].
Multi-Dimensional Grouping: Use VIS_CONFIG["classify"] as the semantic grouping key. For aggregated grouped/stacked/multi-series tasks, ensure .groupby([...]) includes both the X-axis column and the classification key.
Aggregation: Use .groupby(...).agg(alias=('Col', 'func')).reset_index().
MANDATORY: ALWAYS call .reset_index() after aggregation to keep columns accessible for plotting.

4. [VIS_CONFIG] & Visualization (Final Mapping):
Classification (3D): Prefer VIS_CONFIG["classify"] for series semantics. "color" is optional rendering channel and can inherit from "classify".
MANDATORY Call Formats:
Scatter Chart: safe_scatter_plot(df, x_name=VIS_CONFIG['x_name'], y_name=VIS_CONFIG['y_name'], classify_name=(VIS_CONFIG.get('classify') or VIS_CONFIG.get('color')), mode=VIS_CONFIG.get('scatter_mode', 'plain'), title="...")
Bar Chart: safe_bar_plot(df, x_name=VIS_CONFIG['x_name'], y_name=VIS_CONFIG['y_name'], color_name=(VIS_CONFIG.get('color') or VIS_CONFIG.get('classify')), bar_mode=VIS_CONFIG.get('bar_mode', 'plain'), sort_info=VIS_CONFIG.get('sort_info'), title="...")
Pie Chart: safe_pie_plot(df, x_name=VIS_CONFIG['x_name'], y_name=VIS_CONFIG['y_name'], title="...")

Grouped Scatter Strict Rule:
If chart is scatter and mode is grouped (or classify exists), you MUST draw one scatter trace per classify group and show legend.
FORBIDDEN in grouped scatter: using `c=...`, `cmap=...`, and `plt.colorbar(...)` to encode classify.

Bar Sorting Strict Rule:
If VIS_CONFIG includes sort_info for bar charts, you MUST pass sort_info into safe_bar_plot and preserve that order in the plotted result.

## ABSOLUTE MANDATORY RULES (FORBIDDEN ACTIONS):
1. DO NOT REDEFINE FUNCTIONS: NEVER write def safe_merge, def safe_scatter_plot, def safe_bar_plot, or def safe_pie_plot. They are already defined in the environment.
2. USE safe_merge ONLY: NEVER use df.merge(). Use the global safe_merge.
3. NO .values ASSIGNMENT: Never assign columns using .values. Use proper pandas joining or mapping.
4. NO CUSTOM PLOTTING HELPERS: NEVER define extra plotting helpers in the generated code (e.g., def my_bar_plot / def plot_grouped). Always call the provided safe_* helpers directly.
5. BAR MODE CONSISTENCY: If VIS_CONFIG['bar_mode'] is provided, plotting behavior MUST match it exactly. Do not emulate grouped/stacked with ad-hoc loops.

## Execution Requirements:
Tables matching Schema names are already loaded as variables.
Wrap all plotting calls in if not df.empty:.
Return ONLY the Python code block inside python ....
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

        # 2.5 兼容归一化：classify <-> color
        if not vis_config.get("classify") and vis_config.get("color"):
            vis_config["classify"] = vis_config.get("color")
            messages.append("ℹ️ Compatibility: mapped classify <- color.")
        if not vis_config.get("color") and vis_config.get("classify"):
            vis_config["color"] = vis_config.get("classify")

        # 3. 语义约束检查 (Chart Type)
        chart_type = vis_config.get("chart") or vis_config.get("chart_type")
        if not chart_type:
            messages.append("❌ Missing 'chart' key in VIS_CONFIG.")
            is_valid = False
        else:
            chart_lower = chart_type.lower()
            req_fields = self.chart_requirements.get(chart_lower, [])
            for f in req_fields:
                # 同时兼容 x_axis/x_name
                if f not in vis_config and f.replace("name", "axis") not in vis_config:
                    messages.append(f"❌ Chart '{chart_type}' requires '{f}'.")
                    is_valid = False

            def _contains_col_in_transform(col_name):
                if not col_name:
                    return False
                transform_text = (vis_transform or "").replace("`", "").lower()
                full_name = str(col_name).replace("`", "").strip().lower()
                short_name = full_name.split(".")[-1]
                if full_name in transform_text:
                    return True
                return re.search(rf"\b{re.escape(short_name)}\b", transform_text) is not None

            classify_col = vis_config.get("classify")
            transform_lower = (vis_transform or "").lower()

            def _has_sort_intent(text):
                t = (text or "").lower()
                return (
                    "orderby(" in t
                    or "order by" in t
                    or ".sort_values(" in t
                    or "sort_values(" in t
                )

            def _validate_sort_info(sort_info):
                if sort_info is None:
                    return False, "missing 'sort_info'."
                if not isinstance(sort_info, dict):
                    return False, "'sort_info' must be an object."
                by = str(sort_info.get("by", "")).lower()
                order = str(sort_info.get("order", "")).lower()
                if by not in {"x", "y"}:
                    return False, "'sort_info.by' must be 'x' or 'y'."
                if order not in {"asc", "desc", "ascending", "descending"}:
                    return False, "'sort_info.order' must be asc|desc."
                return True, ""

            if chart_lower == "scatter":
                scatter_mode = (vis_config.get("scatter_mode") or ("grouped" if classify_col else "plain")).lower()
                vis_config["scatter_mode"] = scatter_mode
                if scatter_mode not in {"plain", "grouped"}:
                    messages.append("❌ Invalid scatter_mode. Allowed: plain, grouped.")
                    is_valid = False
                if scatter_mode == "grouped":
                    if not classify_col:
                        messages.append("❌ scatter_mode='grouped' requires 'classify'.")
                        is_valid = False
                    elif not _contains_col_in_transform(classify_col):
                        messages.append("❌ grouped scatter requires classify to appear in [VIS_TRANSFORM].")
                        is_valid = False

            elif chart_lower == "bar":
                bar_mode = (vis_config.get("bar_mode") or ("grouped" if classify_col else "plain")).lower()
                vis_config["bar_mode"] = bar_mode
                if bar_mode not in {"plain", "grouped", "stacked"}:
                    messages.append("❌ Invalid bar_mode. Allowed: plain, grouped, stacked.")
                    is_valid = False
                if bar_mode in {"grouped", "stacked"}:
                    if not classify_col:
                        messages.append(f"❌ bar_mode='{bar_mode}' requires 'classify'.")
                        is_valid = False
                    else:
                        if "groupby" not in transform_lower:
                            messages.append(f"❌ bar_mode='{bar_mode}' requires groupby in [VIS_TRANSFORM].")
                            is_valid = False
                        if not _contains_col_in_transform(classify_col):
                            messages.append(f"❌ bar_mode='{bar_mode}' requires classify to appear in [VIS_TRANSFORM].")
                            is_valid = False
                sort_info = vis_config.get("sort_info")
                sort_intent_detected = _has_sort_intent(vis_transform)
                if sort_intent_detected:
                    ok, reason = _validate_sort_info(sort_info)
                    if not ok:
                        messages.append(f"❌ bar sort intent detected in [VIS_TRANSFORM], but {reason}")
                        is_valid = False
                elif sort_info is not None:
                    ok, reason = _validate_sort_info(sort_info)
                    if not ok:
                        messages.append(f"❌ Invalid sort_info: {reason}")
                        is_valid = False

            elif chart_lower == "line":
                line_mode = (vis_config.get("line_mode") or ("multi_series" if classify_col else "single")).lower()
                vis_config["line_mode"] = line_mode
                if line_mode not in {"single", "multi_series"}:
                    messages.append("❌ Invalid line_mode. Allowed: single, multi_series.")
                    is_valid = False
                if line_mode == "multi_series":
                    if not classify_col:
                        messages.append("❌ line_mode='multi_series' requires 'classify'.")
                        is_valid = False
                    elif not _contains_col_in_transform(classify_col):
                        messages.append("❌ multi_series line requires classify to appear in [VIS_TRANSFORM].")
                        is_valid = False
                    if "orderby" not in transform_lower:
                        messages.append("💡 Suggestion: multi_series line usually needs orderby on x axis.")

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
        for key in ["x_name", "x_axis", "y_name", "y_axis", "classify", "color"]:
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

        user_prompt = DVCR2PYTHON_PROMPT.format(
            schema=simple_schema,
            column_description=col_desc,
            question=query,
            relationship=json.dumps(schema_data['relationship'], indent=2),
            dvcr=dvcr_text
        )

        system_prompt = "You are a Python Visualization Expert. Use full Table.Column prefixes."
        code_resp = call_llm(system_prompt, user_prompt)
        
        match = re.search(r'```python(.*?)```', code_resp, re.DOTALL)
        llm_code = match.group(1).strip() if match else code_resp

        # --- 注入代码头部 (Header Injection) ---
        injection_lines = []
        injection_lines.append("# --- Auto-Injected Header ---")
        injection_lines.append("import matplotlib.pyplot as plt")
        injection_lines.append("import pandas as pd")
        injection_lines.append("import os")
        injection_lines.append("import numpy as np")
        injection_lines.append("import warnings")
        injection_lines.append("warnings.filterwarnings('ignore')")
        # 【防御】：废掉模型的 plt.show，防止它清空画布导致 05 抓不到图
        injection_lines.append("plt.show = lambda: None") 
        injection_lines.append("")
        
        injection_lines.append(f"data_dir = '{db_path}'")
        injection_lines.append("")
        
        all_tables = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

# --- 自动加载数据 ---
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            table_name = os.path.splitext(filename)[0]
            
            # 1. 数据加载逻辑 (保持原样)
            injection_lines.append(f"if '{table_name}' not in locals() and '{table_name}' not in globals():")
            injection_lines.append(f"    try: {table_name} = pd.read_csv(os.path.join(data_dir, '{filename}'))")
            injection_lines.append(f"    except: {table_name} = pd.read_csv('{filename}')")
            
            # 2. 列名清洗与重命名 (保持原样)
            injection_lines.append(f"if not any('.' in str(c) for c in {table_name}.columns):")
            injection_lines.append(f"    {table_name}.columns = [c.strip() for c in {table_name}.columns]")
            injection_lines.append(f"    for col in {table_name}.columns:")
            injection_lines.append(f"        if col in {all_tables} or col.endswith('ID') or col.endswith('Code') or col.endswith('SSN') or col == 'Treatment':")
            injection_lines.append(f"            if {table_name}[col].dtype != 'object': {table_name}[col] = {table_name}[col].astype(str)")
            injection_lines.append(f"    {table_name}.columns = [f'{table_name}.{{c}}' for c in {table_name}.columns]")
            
            # 3. 【合并后的别名映射逻辑】：安全、唯一
            if table_name[0].isupper():
                lower_name = table_name.lower()
                # 检查是否为 Python 关键字
                if keyword.iskeyword(lower_name):
                    final_alias = f"{lower_name}_"
                else:
                    final_alias = lower_name
                
                # 只写一行映射，彻底杜绝 SyntaxError
                injection_lines.append(f"{final_alias} = {table_name}")
            
            injection_lines.append("")

        # --- 注入辅助函数 ---
        injection_lines.append("# --- Helper Functions ---")
        
        # 1. safe_merge
        injection_lines.append("def safe_merge(df1, df2, left_on=None, right_on=None, on=None, how='inner'):")
        injection_lines.append("    l_on = [on] if on else ([left_on] if isinstance(left_on, str) else left_on)")
        injection_lines.append("    r_on = [on] if on else ([right_on] if isinstance(right_on, str) else right_on)")
        injection_lines.append("    if l_on: ")
        injection_lines.append("        for k in l_on: ")
        injection_lines.append("            if k in df1.columns: df1[k] = df1[k].astype(str)")
        injection_lines.append("    if r_on: ")
        injection_lines.append("        for k in r_on: ")
        injection_lines.append("            if k in df2.columns: df2[k] = df2[k].astype(str)")
        injection_lines.append("    return pd.merge(df1, df2, left_on=l_on, right_on=r_on, how=how, suffixes=('_src', '_tgt'))")
        injection_lines.append("")
        
        # 2. safe_scatter_plot：严格分组散点（避免连续色条导致通道丢失）
        injection_lines.append("def safe_scatter_plot(df, x_name, y_name, classify_name=None, mode='plain', title=''):")
        injection_lines.append("    fig, ax = plt.subplots(figsize=(10, 6))")
        injection_lines.append("    if df is None or df.empty:")
        injection_lines.append("        ax.text(0.5, 0.5, 'Empty Data', ha='center'); plt.savefig('output.png'); return")
        injection_lines.append("    def fix_c(c):")
        injection_lines.append("        if not c or c in df.columns: return c")
        injection_lines.append("        matched = [col for col in df.columns if col.endswith('.' + c)]")
        injection_lines.append("        return matched[0] if matched else c")
        injection_lines.append("    x, y, k = fix_c(x_name), fix_c(y_name), fix_c(classify_name)")
        injection_lines.append("    scatter_mode = (mode or ('grouped' if k else 'plain')).lower()")
        injection_lines.append("    if scatter_mode == 'grouped' and k and k in df.columns:")
        injection_lines.append("        for label, grp in df.groupby(k, observed=False):")
        injection_lines.append("            ax.scatter(grp[x], grp[y], label=str(label), alpha=0.8)")
        injection_lines.append("        if df[k].nunique(dropna=False) > 1:")
        injection_lines.append("            ax.legend(title=str(k).split('.')[-1])")
        injection_lines.append("    else:")
        injection_lines.append("        ax.scatter(df[x], df[y], alpha=0.8)")
        injection_lines.append("    ax.set_title(title)")
        injection_lines.append("    ax.set_xlabel(str(x).split('.')[-1]); ax.set_ylabel(str(y).split('.')[-1])")
        injection_lines.append("    plt.tight_layout(); plt.savefig('output.png')")
        injection_lines.append("")

        # 3. 升级版 safe_bar_plot：内建强制排序与透视
        injection_lines.append("def safe_bar_plot(df, x_name, y_name, color_name=None, bar_mode='plain', title='', sort_info=None):")
        injection_lines.append("    fig, ax = plt.subplots(figsize=(10, 6))")
        injection_lines.append("    if df is None or df.empty:")
        injection_lines.append("        ax.text(0.5, 0.5, 'Empty Data', ha='center'); plt.savefig('output.png'); return")
        
        # A. 智能列名修复
        injection_lines.append("    def fix_c(c):")
        injection_lines.append("        if not c or c in df.columns: return c")
        injection_lines.append("        matched = [col for col in df.columns if col.endswith('.' + c)]")
        injection_lines.append("        return matched[0] if matched else c")
        injection_lines.append("    x, y, c = fix_c(x_name), fix_c(y_name), fix_col(color_name) if 'fix_col' in locals() else fix_c(color_name)")
        injection_lines.append("    mode = (bar_mode or ('grouped' if c else 'plain')).lower()")
        injection_lines.append("    if mode not in {'plain', 'grouped', 'stacked'}:")
        injection_lines.append("        mode = 'plain'")
        
        # B. DVCR 2.2 视觉意图排序
        injection_lines.append("    totals = df.groupby(x, observed=False)[y].sum()")
        injection_lines.append("    if sort_info and sort_info.get('by') == 'y':")
        injection_lines.append("        is_asc = sort_info.get('order', 'asc').lower() == 'asc'")
        injection_lines.append("        order = totals.sort_values(ascending=is_asc).index.tolist()")
        injection_lines.append("    elif sort_info and sort_info.get('by') == 'x':")
        injection_lines.append("        is_asc = sort_info.get('order', 'asc').lower() == 'asc'")
        injection_lines.append("        order = totals.sort_index(ascending=is_asc).index.tolist()")
        injection_lines.append("    else:")
        injection_lines.append("        order = df[x].drop_duplicates().tolist()")
        
        # 将 X 轴锁定为 Categorical，绝对保证后续作图的顺序
        injection_lines.append("    df[x] = pd.Categorical(df[x], categories=order, ordered=True)")
        
        # C. 绘图执行
        injection_lines.append("    if c and c in df.columns and mode in {'grouped', 'stacked'}:")
        injection_lines.append("        pivot_df = df.pivot_table(index=x, columns=c, values=y, aggfunc='sum', observed=False)")
        injection_lines.append("        pivot_df = pivot_df.reindex(order).fillna(0)")
        injection_lines.append("        pivot_df.plot(kind='bar', stacked=(mode == 'stacked'), ax=ax)")
        injection_lines.append("    else:")
        injection_lines.append("        plot_df = df.groupby(x, observed=False)[y].sum().reindex(order)")
        injection_lines.append("        plot_df.plot(kind='bar', ax=ax)")
        
        # D. 收尾
        injection_lines.append("    plt.title(title)")
        injection_lines.append("    plt.xlabel(str(x).split('.')[-1]); plt.ylabel(str(y).split('.')[-1])")
        injection_lines.append("    plt.xticks(rotation=45); plt.tight_layout()")
        injection_lines.append("    plt.savefig('output.png')")
        injection_lines.append("")

        # 4. 升级版 safe_pie_plot
        injection_lines.append("def safe_pie_plot(df, x_name, y_name, title=''):")
        injection_lines.append("    fig, ax = plt.subplots(figsize=(8, 8))")
        injection_lines.append("    if df is None or df.empty:")
        injection_lines.append("        ax.text(0.5, 0.5, 'Empty Data', ha='center'); plt.savefig('output.png'); return")
        injection_lines.append("    def fix_c(c):")
        injection_lines.append("        if not c or c in df.columns: return c")
        injection_lines.append("        matched = [col for col in df.columns if col.endswith('.' + c)]")
        injection_lines.append("        return matched[0] if matched else c")
        injection_lines.append("    x, y = fix_c(x_name), fix_c(y_name)")
        injection_lines.append("    plot_df = df.groupby(x, observed=False)[y].sum()")
        injection_lines.append("    ax.pie(plot_df, labels=plot_df.index, autopct='%1.1f%%')")
        injection_lines.append("    plt.title(title); plt.savefig('output.png')")
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
