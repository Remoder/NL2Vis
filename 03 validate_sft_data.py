import os
import json
import sqlite3
import pandas as pd
import multiprocessing as mp
import re, time
import glob, duckdb, sqlglot
from collections import Counter
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
from openai import OpenAI

# --- Configuration ---
DEFAULT_KEY = "sk-EWwCihmo7aEgCAKZeVV82P3vdQcy6jBg02JBozZ3Ix7Q2ESu"
DEFAULT_BASE = "https://api.wenwen-ai.com/v1"

API_KEY = os.getenv("OPENAI_API_KEY", DEFAULT_KEY)
BASE_URL = os.getenv("OPENAI_API_BASE", DEFAULT_BASE)

if not BASE_URL.endswith("/v1"):
    BASE_URL += "/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def call_llm(user_prompt, system_prompt="You are an expert about text-to-SQL and pandas code.", max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
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
SR2SQL_CHECK_PROMPT = """You are a SQL Translation Expert. Your task is to synthesize a single, executable SQLite/DuckDB SQL query from two logic sections of a DVCR 2.1 plan.

### CORE PHILOSOPHY:
- Modular Reasoning: Separate "data retrieval" ([DATA_FLOW]) from "visual transformation" ([VIS_TRANSFORM]).
- Dimensionality: If the question implies a comparison (e.g., "by gender", "stacked"), identify the classification column and assign it to the "color" key in [VIS_CONFIG].
- Casing: Use the EXACT casing for table and column names provided in the schema.
- Preserving Zeroes: When the question asks for "each" item or compares two independent groups (e.g., "faculties vs students per activity"), you MUST use **LEFT JOIN** or **FULL JOIN** to ensure items with zero counts are not filtered out. Mismatched data lengths (e.g., 12 instead of 14) will cause total failure.

You MUST strictly follow the "DVCR 2.1 Syntax Specification" below.

### DVCR 2.1 Syntax Specification (Multi-Dimensional Support)
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
"""

# --- Parsing Utils ---
def parse_dvcr(dvcr_text):
    """适配 DVCR 2.1 的四段式解析器"""
    parts = {}
    def get_section(name, text):
        pattern = rf"\[\s*{name}\s*\](.*?)(?=\[\s*\w+\s*\]|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    parts['data_flow'] = get_section("DATA_FLOW", dvcr_text)
    parts['vis_transform'] = get_section("VIS_TRANSFORM", dvcr_text) # 新增
    
    vis_config_str = get_section("VIS_CONFIG", dvcr_text)
    if vis_config_str:
        try:
            clean_json = re.sub(r'```json|```', '', vis_config_str).strip()
            parts['vis_config'] = json.loads(clean_json)
        except:
            parts['vis_config'] = None
    else:
        parts['vis_config'] = None
    return parts

def extract_sql(llm_response):
    match = re.search(r'```sqlite(.*?)```', llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```sql(.*?)```', llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return llm_response.replace("```sqlite", "").replace("```", "").strip()

# ================= 替换开始 =================

# --- 全局数据库连接缓存（提升 10 倍以上速度） ---
DB_CACHE = {}

def get_duckdb_connection(db_path):
    """缓存 DuckDB 内存连接，避免每验证一条数据就重新读取一遍 CSV 磁盘文件"""
    if db_path in DB_CACHE:
        return DB_CACHE[db_path]
        
    con = duckdb.connect(database=':memory:')
    csv_files = glob.glob(os.path.join(db_path, "*.csv"))
    for file in csv_files:
        table_name = os.path.basename(file).replace('.csv', '')
        # 添加 header 和类型推断，防止某些奇怪的 CSV 格式报错
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file}', ignore_errors=true)")
    
    DB_CACHE[db_path] = con
    return con

def execute_query_compatible(con, sql_str, max_retries=3):
    """
    V7 终极安全版执行器：
    1. 严格切分 SQL 的各个子句（SELECT, FROM/JOIN, WHERE, GROUP BY, HAVING, ORDER BY）。
    2. 只在允许使用聚合函数的子句（SELECT, HAVING, ORDER BY）中尝试修复 ANY_VALUE。
    3. 避免破坏 WHERE, ON 和 GROUP BY 导致新的语法错误。
    """
    current_sql = sql_str
    
    for attempt in range(max_retries):
        # 1. 基础清理，保证关键字可以被正则安全捕获
        current_sql = re.sub(r'(?i)\b(SELECT|FROM|WHERE|GROUP BY|HAVING|ORDER BY|LIMIT)\b', r' \1 ', current_sql)
        # 压缩多余空格
        current_sql = re.sub(r'\s+', ' ', current_sql).strip()
        
        try:
            # 2. 转译并执行
            try:
                duckdb_sql = sqlglot.transpile(current_sql, read="sqlite", write="duckdb")[0]
            except Exception:
                duckdb_sql = current_sql
                
            df = con.execute(duckdb_sql).df()
            return df, None, duckdb_sql
            
        except duckdb.BinderException as e:
            error_msg = str(e)
            
            # 核心修复：捕获需要修复的列
            match_groupby = re.search(r'column "(.*?)" must appear in the GROUP BY clause', error_msg)
            if match_groupby:
                offending_col = match_groupby.group(1)
                col_base_name = offending_col.split('.')[-1].strip('`"')
                
                # --- V7 核心：安全的分区替换逻辑 ---
                
                # 定义一个安全的局部替换函数，处理逗号分隔的表达式
                def safe_wrap_clause(clause_text, target_col_base):
                    if not clause_text: return clause_text
                    
                    exprs = clause_text.split(',')
                    new_exprs = []
                    modified = False
                    
                    for expr in exprs:
                        # 如果包含该列，且没有被聚合函数包裹
                        if target_col_base in expr and not re.search(r'(?i)\b(ANY_VALUE|COUNT|SUM|AVG|MAX|MIN)\s*\(', expr):
                            # 正则：匹配独立的列名（可能带表前缀和反引号）
                            pattern = rf'(?<![a-zA-Z0-9_])([`"\w]+\.)?([`"]?{re.escape(target_col_base)}[`"]?)(?![a-zA-Z0-9_])'
                            
                            # 执行包裹
                            new_expr, subs = re.subn(pattern, r'ANY_VALUE(\g<0>)', expr, count=1, flags=re.IGNORECASE)
                            if subs > 0:
                                new_exprs.append(new_expr)
                                modified = True
                            else:
                                new_exprs.append(expr)
                        else:
                            new_exprs.append(expr)
                            
                    return ",".join(new_exprs) if modified else clause_text

                # 尝试用正则粗略切分 SQL 结构
                # 这种切分不完美（如果子查询里有这些关键字会切错），但在当前 NL2Vis 场景下足够了
                pattern_clauses = r'(?i)(SELECT\s+)(.*?)(\s+FROM\s+.*?)(?:\s+(WHERE|GROUP BY|HAVING|ORDER BY|LIMIT)\s+|$)'
                
                # 为了安全，我们只处理主查询的结构
                # 我们通过寻找最后的 ORDER BY 或 HAVING 来尝试修复
                
                fixed_sql = current_sql
                
                # 1. 尝试修复 SELECT 部分
                match_select = re.search(r'(?i)(SELECT\s+)(.*?)(\s+FROM\s+)', fixed_sql, re.DOTALL)
                if match_select:
                    select_body = match_select.group(2)
                    new_select_body = safe_wrap_clause(select_body, col_base_name)
                    if new_select_body != select_body:
                        fixed_sql = fixed_sql[:match_select.start(2)] + new_select_body + fixed_sql[match_select.end(2):]

                # 2. 尝试修复 ORDER BY 部分
                match_orderby = re.search(r'(?i)(ORDER\s+BY\s+)(.*?)(\s+LIMIT\s+|$)', fixed_sql, re.DOTALL)
                if match_orderby:
                    ob_body = match_orderby.group(2)
                    # 注意：ORDER BY 可能带有 DESC/ASC
                    # safe_wrap_clause 依然适用，因为它只会把列包裹，如 ANY_VALUE(Name) DESC
                    new_ob_body = safe_wrap_clause(ob_body, col_base_name)
                    if new_ob_body != ob_body:
                        fixed_sql = fixed_sql[:match_orderby.start(2)] + new_ob_body + fixed_sql[match_orderby.end(2):]
                
                # 3. 尝试修复 HAVING 部分 (虽然 DVCR 很少生成)
                match_having = re.search(r'(?i)(HAVING\s+)(.*?)(\s+ORDER\s+BY|\s+LIMIT\s+|$)', fixed_sql, re.DOTALL)
                if match_having:
                    hav_body = match_having.group(2)
                    new_hav_body = safe_wrap_clause(hav_body, col_base_name)
                    if new_hav_body != hav_body:
                        fixed_sql = fixed_sql[:match_having.start(2)] + new_hav_body + fixed_sql[match_having.end(2):]

                # 如果有任何修改，进入下一次重试
                if fixed_sql != current_sql:
                    current_sql = fixed_sql
                    continue
            
            # 其他 Binder 错误直接退出
            return None, error_msg, current_sql
            
        except Exception as e:
            return None, str(e), current_sql
            
    return None, "Max auto-fix retries exceeded", current_sql

def fix_sql_quotes(sql_str):
    """
    将 Gold SQL 中误用的双引号替换为单引号，
    防止 DuckDB 将其误认为列名。
    """
    # 简单的启发式逻辑：如果双引号内是值（如 "Lisa", "food"），转为单引号
    # 这里我们只对常见的比较操作符后的双引号进行替换
    return re.sub(r'=\s*"([^"]+)"', r"= '\1'", sql_str)

def execute_with_timeout(gen_sql, gold_sql, db_path):
    """独立执行对比，修复之前的幽灵报错 bug"""
    
    # 1. 获取（或创建）缓存的数据库连接
    con = get_duckdb_connection(db_path)
    
    # 1.5. 修改 Gold_sql 中的双引号问题
    gold_sql = fix_sql_quotes(gold_sql)

    # 2. 隔离执行 Gen SQL 和 Gold SQL
    df_gen, gen_err, final_gen_sql = execute_query_compatible(con, gen_sql)
    df_gold, gold_err, final_gold_sql = execute_query_compatible(con, gold_sql)
    
    # 将结果转换为字符串用于日志输出
    gen_str = str(gen_err) if gen_err else "DataFrame Executed Successfully."
    gold_str = str(gold_err) if gold_err else "DataFrame Executed Successfully."
    
    # 3. 如果有任何一个执行失败，直接返回 0 (False)
    if gen_err or gold_err:
        return 0,[gen_str, gold_str]
        
    # 4. 如果都成功了，比对 DataFrame 结果
    try:
        # 定义一个内部函数，将 DataFrame 转换为“内容无关”的规范化格式
        def get_canonical_representation(df):
            # a. 填充空值并统一转为字符串（解决 1.0 vs 1 的问题）
            df_str = df.fillna('').astype(str)
            
            # b. 处理“列顺序无关”：
            # 将每一行转为列表并进行内部排序。
            # 这样 [100, 'food', '1'] 和 [100, '1', 'food'] 都会变成 ['1', '100', 'food']
            rows = [sorted(list(row)) for row in df_str.values.tolist()]
            
            # c. 处理“行顺序无关”：
            # 对整个行列表进行排序
            rows.sort()
            return rows

        # 获取两个结果的规范化表达
        gen_canonical = get_canonical_representation(df_gen)
        gold_canonical = get_canonical_representation(df_gold)
        
        # 为了日志输出，我们依然保留一个易读的字符串版本
        # 排序原 DataFrame 用于展示（仅为辅助）
        gen_out = df_gen.fillna('').astype(str).to_string(index=False)
        gold_out = df_gold.fillna('').astype(str).to_string(index=False)
        
        # 核心比对：比较规范化后的列表
        if gen_canonical == gold_canonical:
            return 1, [gen_out, gold_out]
        else:
            # 如果不匹配，在 log 中打印出前几行规范化数据，方便排查
            debug_info = f"Mismatch! \nGen(canonical sample): {gen_canonical[:2]}\nGold(canonical sample): {gold_canonical[:2]}"
            # print(debug_info) # 调试时可开启
            return 0, [gen_out, gold_out]
            
    except Exception as e:
        return 0, [f"Compare Error: {str(e)}", f"Compare Error: {str(e)}"]

# ================= 替换结束 =================

# --- Validation Logic ---
def normalize(s):
    """更强大的归一化，处理全称、别名和聚合"""
    if not s: return ""
    # 转小写，去反引号，去空格
    s = str(s).lower().replace("`", "").replace('"', '').replace("'", "").strip()
    
    # --- 新增：如果包含 BIN(...) 逻辑，尝试提取核心列名 ---
    # 比如 BIN(weather.date, WEEKDAY) -> date
    bin_match = re.search(r'bin\((.*?)[,\)]', s)
    if bin_match:
        s = bin_match.group(1).split('.')[-1].strip()
        return s

    # 如果是全称 Table.Column，只保留 Column
    if "." in s:
        s = s.split(".")[-1]
    
    # 处理常见的聚合别名映射 (Alias -> Aggregation Keyword)
    # 如果别名里包含 count, num, amount 等，归一化为 'count'
    if any(kw in s for kw in ['count', 'num', 'amount', 'total_number', 'n_']):
        return "count"
    if 'sum' in s or 'total' in s:
        return "sum"
    if 'avg' in s or 'average' in s or 'mean' in s:
        return "avg"
    if 'max' in s or 'highest' in s:
        return "max"
    if 'min' in s or 'lowest' in s:
        return "min"
        
    # 处理 SQL 原始聚合字符串: count(name) -> count
    s = re.sub(r'(count|sum|avg|max|min)\(.*\)', r'\1', s)
    if s == "*": return "count" # 处理 count(*) 剩下的 *
    
    return s

def validate_vis_config(pred_config, gold_vis):
    """基于语义和 Protocol v1.0 的可视化匹配"""
    if not pred_config or not gold_vis:
        return False
        
    # 1. 校验图表类型 (chart vs chart_type)
    pred_chart = (pred_config.get('chart') or pred_config.get('chart_type', '')).lower()
    gold_chart = gold_vis.get('chart', '').lower()
    if pred_chart != gold_chart:
        return False
        
    # 2. 校验轴字段 (允许全称匹配和别名语义匹配)
    pred_x = pred_config.get('x_name') or pred_config.get('x_axis') or pred_config.get('labels')
    pred_y = pred_config.get('y_name') or pred_config.get('y_axis') or pred_config.get('values')
    
    gold_x = gold_vis.get('x_name')
    gold_y = gold_vis.get('y_name')

    norm_pred_x = normalize(pred_x)
    norm_gold_x = normalize(gold_x)
    norm_pred_y = normalize(pred_y)
    norm_gold_y = normalize(gold_y)
    
    # X 轴匹配逻辑：只要 Got 包含 Expected，或者归一化后一致
    x_match = (norm_gold_x in norm_pred_x) or (norm_pred_x == norm_gold_x)
    
    # Y 轴匹配逻辑：对于聚合操作，归一化后的关键字一致即可
    y_match = (norm_gold_y == norm_pred_y) or (norm_gold_y in norm_pred_y)
    
    # 特殊情况：如果 Gold 的 Y 轴是列名（非聚合），比如 'Cost'
    if norm_gold_y and norm_gold_y not in ['count', 'sum', 'avg', 'max', 'min']:
        y_match = (norm_gold_y in norm_pred_y)

    return x_match and y_match

def validate_against_vis_obj(df_gen, vis_obj):
    """支持多维度 (X, Y, Color) 的语义级比对"""
    from collections import Counter
    if df_gen is None or df_gen.empty or vis_obj is None:
        return False, "Data empty"

    def normalize_val(v):
        if v is None: return ""
        s = str(v).strip()
        try: # Date
            return pd.to_datetime(s).strftime('%Y-%m-%d')
        except: pass
        try: # Number
            f = float(s)
            if f.is_integer(): return str(int(f))
            return "{:.2f}".format(f)
        except: pass
        return s.lower()

    try:
        # 1. 提取金标 (支持 y_data 列表嵌套)
        g_x_all = vis_obj.get('x_data', [[]])[0]
        g_y_lists = vis_obj.get('y_data', []) 
        g_c_list = vis_obj.get('classify', [])
        
        gold_points = []
        for i, y_list in enumerate(g_y_lists):
            c_label = g_c_list[i] if i < len(g_c_list) else None
            for x_val, y_val in zip(g_x_all, y_list):
                point = [normalize_val(x_val), normalize_val(y_val)]
                if c_label is not None: point.append(normalize_val(c_label))
                gold_points.append(tuple(sorted(point)))
        gold_set = Counter(gold_points)

        # 2. 提取模型数据
        gen_points = []
        for row in df_gen.values:
            point = [normalize_val(v) for v in row]
            gen_points.append(tuple(sorted(point)))
        gen_set = Counter(gen_points)

        # 3. 核心比对与零值容错
        if gen_set == gold_set:
            return True, "Perfect Match"
        else:
            missing = gold_set - gen_set
            # 允许模型漏掉数值为 0 的行（Inner Join 常见现象）
            if all("0" in p for p in missing.keys()) and len(gen_set) > 0:
                return True, "Match (Zeroes ignored)"
            return False, f"Missing non-zero points: {list(missing.items())[:2]}"
    except Exception as e:
        return False, f"Comparison Error: {e}"

def validate_and_filter(sft_file_path, dataset_root, output_file_path):
    from tqdm import tqdm
    
    vis_eval_path = os.path.join(dataset_root, "visEval.json")
    tqdm.write(f"Loading ground truth data from {vis_eval_path}...")
    with open(vis_eval_path, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    
    valid_data = []
    total = 0
    passed_vis = 0
    passed_sql = 0
    
    with open(sft_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    tqdm.write(f"🚀 Start validating {len(lines)} samples...")
    
    pbar = tqdm(lines, desc="Validating")
    for line in pbar:
        item = json.loads(line)
        total += 1
        
        q_id = item['id']
        db_id = item['db_id']
        question = item['input']
        dvcr_output = item['output']
        gold_vis = item['gold_vis']

        if q_id not in ground_truth_data: continue
        target_vis_obj = ground_truth_data[q_id].get('vis_obj')
        
        # 准备标准答案的展示样本
        gold_x_sample = target_vis_obj.get('x_data', [[]])[0]
        gold_y_sample = target_vis_obj.get('y_data', [[]])[0]
        gold_data_preview = list(zip(gold_x_sample, gold_y_sample))[:3]

        # 初始化本次循环的日志信息
        report = []
        report.append(f"\n{'='*30} 🔍 EXAMINING ID: {q_id} {'='*30}")
        report.append(f"DB: {db_id} | Q: {question}")
        
        # --- 解析 DVCR ---
        is_parse_ok = True
        try:
            parsed = parse_dvcr(dvcr_output)
            data_flow = parsed['data_flow']
            vis_config = parsed['vis_config']
            vis_transform=parsed['vis_transform']
        except Exception as e:
            report.append(f"❌ [PARSE ERROR]: {e}")
            is_parse_ok = False

        # --- 1. 校验 VIS_CONFIG ---
        vis_ok = False
        if is_parse_ok:
            vis_ok = validate_vis_config(vis_config, gold_vis)
            if not vis_ok:
                report.append(f"❌ [VIS MISMATCH]:")
                report.append(f"   - Expected: {gold_vis}")
                report.append(f"   - Got:      {vis_config}")
            else:
                passed_vis += 1

        # --- 2. 执行与比对 (仅在 VIS 校验通过后进行，或者你想看全量报错也可以去掉 if) ---
        sql_ok = False
        if vis_ok:
            try:
                schema_str = item['instruction'].split("based on schema: ")[1]
                db_path = os.path.join(dataset_root, "databases", db_id)
                con = get_duckdb_connection(db_path)
                
                # 转换并执行
                prompt = SR2SQL_CHECK_PROMPT.format(
                    schema=schema_str, question=question, data_flow=data_flow, vis_transform=vis_transform, vis_config=json.dumps(vis_config)
                )
                gen_sql_resp = call_llm(prompt)
                gen_sql = extract_sql(gen_sql_resp)
                
                df_gen, gen_err, _ = execute_query_compatible(con, gen_sql)
                
                if gen_err:
                    report.append(f"❌ [SQL EXEC ERROR]: {gen_err}")
                    report.append(f"   - GEN SQL: {gen_sql.replace(os.linesep, ' ')}")
                else:
                    # 数据比对
                    is_match, msg = validate_against_vis_obj(df_gen, target_vis_obj)
                    if is_match:
                        sql_ok = True
                        passed_sql += 1
                    else:
                        report.append(f"❌ [DATA MISMATCH]: {msg}")
                        report.append(f"   - GEN SQL: {gen_sql.replace(os.linesep, ' ')}")
                        # 展示标准答案 vs 模型答案
                        report.append(f"   - GOLD DATA (Sample): {gold_data_preview}")
                        gen_preview = df_gen.head(3).values.tolist()
                        report.append(f"   - GEN DATA (Sample):  {gen_preview}")
            except Exception as e:
                report.append(f"❌ [SYSTEM ERROR]: {e}")

        # --- 判定最终结果 ---
        if sql_ok:
            # tqdm.write(f"✅ ID {q_id}: Passed") # 成功时可以不打印，保持日志清爽
            valid_data.append(item)
        else:
            # 只有失败时才打印整份报告
            tqdm.write("\n".join(report))
            tqdm.write(f"{'='*80}")
            
    # 最终汇总
    tqdm.write(f"\n✨ FINAL STATS ✨")
    tqdm.write(f"Final Yield: {len(valid_data)}/{total} ({len(valid_data)/total:.1%})")
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")
            
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    dataset_root = os.path.join(project_root, "dataset")
    input_file = os.path.join(project_root, "temp_results", "sft_data", "sft_origin_data.jsonl")
    output_file = os.path.join(project_root, "temp_results", "sft_data", "sft_advised_data.jsonl")
    
    print(f"Cleaning output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        pass # 打开并关闭 w 模式，即清空文件内容
    # if not os.path.exists(dataset_root):
    #     dataset_root = "/root/nl2vis/VisEval-main/visEval_dataset"
    
    if os.path.exists(input_file):
        validate_and_filter(input_file, dataset_root, output_file)
    else:
        print(f"Error: {input_file} not found. Run generation script first.")
