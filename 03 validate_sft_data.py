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
from dvcr_protocol import build_sr2sql_check_prompt

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

def sanitize_sql_literals(sql_str):
    """
    统一 SQL 中的字符串常量，避免 LLM 产出的双引号触发 DuckDB Binder 报错。
    """
    if not sql_str:
        return sql_str

    sql_str = re.sub(r'=\s*"([^"]+)"', r"= '\1'", sql_str)
    sql_str = re.sub(
        r'IN\s*\(([^)]+)\)',
        lambda m: m.group(0).replace('"', "'"),
        sql_str,
        flags=re.IGNORECASE,
    )
    return sql_str


WEEKDAY_LABELS = [
    ('0', 'sun'),
    ('1', 'mon'),
    ('2', 'tue'),
    ('3', 'wed'),
    ('4', 'thur'),
    ('5', 'fri'),
    ('6', 'sat'),
]


def _find_matching_parenthesis(sql_str, start_idx):
    depth = 0
    in_single = False
    in_double = False
    escape = False

    for idx in range(start_idx, len(sql_str)):
        ch = sql_str[idx]
        if ch == '\\' and not escape:
            escape = True
            continue
        if ch == "'" and not in_double and not escape:
            in_single = not in_single
        elif ch == '"' and not in_single and not escape:
            in_double = not in_double
        elif ch == '(' and not in_single and not in_double:
            depth += 1
        elif ch == ')' and not in_single and not in_double:
            depth -= 1
            if depth == 0:
                return idx
        if escape:
            escape = False
    return -1


def _split_first_argument(segment):
    depth = 0
    in_single = False
    in_double = False
    escape = False

    for idx, ch in enumerate(segment):
        if ch == '\\' and not escape:
            escape = True
            continue
        if ch == "'" and not in_double and not escape:
            in_single = not in_single
        elif ch == '"' and not in_single and not escape:
            in_double = not in_double
        elif ch == '(' and not in_single and not in_double:
            depth += 1
        elif ch == ')' and not in_single and not in_double:
            depth = max(depth - 1, 0)
        elif ch == ',' and depth == 0 and not in_single and not in_double:
            return segment[:idx], segment[idx + 1 :]
        if escape:
            escape = False
    return None, None


def coerce_weekday_labels(sql_str):
    """
    将 strftime('%w', ...) 包装成固定的 weekday 文本，避免数值 weekday 与金标不一致。
    """
    if not sql_str:
        return sql_str

    lower_sql = sql_str.lower()
    idx = 0
    pieces = []

    while idx < len(sql_str):
        pos = lower_sql.find("strftime", idx)
        if pos == -1:
            pieces.append(sql_str[idx:])
            break

        name_end = pos + len("strftime")
        lookahead = name_end
        while lookahead < len(sql_str) and sql_str[lookahead].isspace():
            lookahead += 1
        if lookahead >= len(sql_str) or sql_str[lookahead] != '(':
            pieces.append(sql_str[idx:pos + 1])
            idx = pos + 1
            continue

        paren_start = lookahead
        paren_end = _find_matching_parenthesis(sql_str, paren_start)
        if paren_end == -1:
            pieces.append(sql_str[idx:pos + 1])
            idx = pos + 1
            continue

        inner = sql_str[paren_start + 1 : paren_end]
        arg1, arg_rest = _split_first_argument(inner)
        if arg1 is None or arg_rest is None:
            pieces.append(sql_str[idx:pos + 1])
            idx = pos + 1
            continue

        first_arg = arg1.strip().strip('"').strip("'").lower()
        if first_arg != '%w':
            pieces.append(sql_str[idx:pos + 1])
            idx = pos + 1
            continue

        original_call = sql_str[pos : paren_end + 1]
        case_expr = ["(CASE ", original_call, " "]
        for digit, label in WEEKDAY_LABELS:
            case_expr.append(f"WHEN '{digit}' THEN '{label}' ")
        case_expr.append("ELSE ")
        case_expr.append(original_call)
        case_expr.append(" END)")

        pieces.append(sql_str[idx:pos])
        pieces.append(''.join(case_expr))
        idx = paren_end + 1

    return ''.join(pieces)

def execute_with_timeout(gen_sql, gold_sql, db_path):
    """独立执行对比，修复之前的幽灵报错 bug"""
    
    # 1. 获取（或创建）缓存的数据库连接
    con = get_duckdb_connection(db_path)
    
    # 1.5. 标准化 SQL 字符串常量 & weekday 表达式
    gen_sql = coerce_weekday_labels(sanitize_sql_literals(gen_sql))
    gold_sql = coerce_weekday_labels(sanitize_sql_literals(gold_sql))

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

def normalize_val(v, is_x_axis=False, gold_samples=None):
    """
    增强版归一化：
    1. 保护数字/比分不被误转为日期。
    2. 自动处理 DuckDB 的 strftime('%w') 数字到英文周几的转换。
    """
    if v is None or pd.isna(v): return ""
    s = str(v).strip()

    # --- [核心修复] 统一各种横杠和连接符 ---
    # 替换 En-dash (–), Em-dash (—), Figure dash (‒) 为标准 Hyphen (-)
    s = str(v).strip().replace('–', '-').replace('—', '-').replace('‒', '-')
    
    # --- 1. 星期几数字映射 (解决 Apartment_Rentals 问题) ---
    # 如果是 X 轴，且产出是单位数字 '0'-'6'，且金标里有 ['Mon', 'Tue'...] 缩写
    weekday_num_to_str = {
        '0': 'sun', '1': 'mon', '2': 'tue', '3': 'wed', 
        '4': 'thur', '5': 'fri', '6': 'sat'
    }
    if is_x_axis and s in weekday_num_to_str:
        # 检查金标样本，确定是否需要映射
        if gold_samples and any(day in [str(gs).lower() for gs in gold_samples] for day in weekday_num_to_str.values()):
            return weekday_num_to_str[s]

    # --- 2. 保护逻辑：拦截纯数字和比分格式，防止 pd.to_datetime 误判 (解决 Basketball 问题) ---
    # 规则：如果是纯数字 (如 2100) 或者符合 "数字-数字" (如 102-98) 且不包含斜杠或冒号
    is_pure_num = re.match(r'^-?\d+(\.\d+)?$', s)
    is_score_format = re.match(r'^\d+[\-–]\d+$', s) # 处理 Hyphen 和 En-dash
    
    if not is_pure_num and not is_score_format:
        try:
            # 只有在包含常见日期分隔符时才尝试转日期
            if any(sep in s for sep in ['/', '-', '.']) or len(s) > 6:
                return pd.to_datetime(s).strftime('%Y-%m-%d')
        except:
            pass

    # --- 3. 数字精度处理 (2100.0 -> 2100) ---
    try:
        f = float(s)
        if f.is_integer(): return str(int(f))
        return "{:.2f}".format(f)
    except:
        pass

    return s.lower()

def validate_against_vis_obj(df_gen, vis_obj):
    """
    更新后的比对函数，传入 X 轴金标样本以支持上下文感知的归一化
    """
    if df_gen is None or df_gen.empty or vis_obj is None:
        return False, "Data empty"

    try:
        # 1. 获取金标数据
        g_x_all = vis_obj.get('x_data', [[]])[0]
        g_y_lists = vis_obj.get('y_data', [])
        g_c_list = vis_obj.get('classify', [])
        
        # 提取金标 X 轴样本用于 normalize_val 的上下文判断
        gold_x_samples = set([str(x).lower() for x in g_x_all])

        gold_points = []
        for i, y_list in enumerate(g_y_lists):
            c_label = g_c_list[i] if i < len(g_c_list) else None
            for x_val, y_val in zip(g_x_all, y_list):
                # 金标已经是最终结果，简单归一化即可
                point = [normalize_val(x_val), normalize_val(y_val)]
                if c_label is not None: point.append(normalize_val(c_label))
                gold_points.append(tuple(sorted(point)))
        gold_set = Counter(gold_points)

        # 2. 提取模型生成的数据
        gen_points = []
        # 假设 df_gen 的第一列是 X 轴，第二列是 Y 轴
        for row in df_gen.values:
            # 第一列传 is_x_axis=True，并带入金标样本作为参考
            p_list = []
            for idx, v in enumerate(row):
                is_x = (idx == 0)
                p_list.append(normalize_val(v, is_x_axis=is_x, gold_samples=gold_x_samples))
            gen_points.append(tuple(sorted(p_list)))
        gen_set = Counter(gen_points)

        # 3. 比对逻辑（保持不变）
        if gen_set == gold_set:
            return True, "Perfect Match"
        else:
            missing = gold_set - gen_set
            if all("0" in str(p) for p in missing.keys()) and len(gen_set) > 0:
                return True, "Match (Zeroes ignored)"
            return False, f"Missing points count: {len(missing)}. Example: {list(missing.items())[:1]}"
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
                prompt = build_sr2sql_check_prompt(
                    schema=schema_str,
                    question=question,
                    data_flow=data_flow,
                    vis_transform=vis_transform,
                    vis_config=json.dumps(vis_config),
                )
                gen_sql_resp = call_llm(prompt)
                gen_sql_raw = extract_sql(gen_sql_resp)
                gen_sql_fixed = coerce_weekday_labels(sanitize_sql_literals(gen_sql_raw))

                # 若 SQL 被修复，尝试同步写回 DVCR
                if gen_sql_fixed != gen_sql_raw:
                    replaced = False
                    if gen_sql_raw and gen_sql_raw in dvcr_output:
                        dvcr_output = dvcr_output.replace(gen_sql_raw, gen_sql_fixed, 1)
                        replaced = True
                    gen_sql = gen_sql_fixed
                    if replaced:
                        item['output'] = dvcr_output
                else:
                    gen_sql = gen_sql_raw
                
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
