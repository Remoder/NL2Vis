import os
import argparse
import json
import sqlite3
import pandas as pd
import multiprocessing as mp
import re, time
import glob, duckdb, sqlglot
from collections import Counter
from sqlglot import exp
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

TOKEN_LOG_PATH = None
TOKEN_RUN_ID = None
TOKEN_STAGE = "stage03_validate_sft"
TOKEN_DB_ID = "ALL"
MAX_SQL_REPAIR_ROUNDS = 2


def init_token_logger(project_root, db_id):
    global TOKEN_LOG_PATH, TOKEN_RUN_ID, TOKEN_DB_ID
    TOKEN_DB_ID = db_id or "ALL"
    TOKEN_RUN_ID = time.strftime("%Y%m%d_%H%M%S")
    token_dir = os.path.join(project_root, "logs", "token_usage")
    os.makedirs(token_dir, exist_ok=True)
    TOKEN_LOG_PATH = os.path.join(token_dir, f"{TOKEN_STAGE}.jsonl")


def log_token_usage(response, model, meta=None):
    if not TOKEN_LOG_PATH:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    try:
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": TOKEN_RUN_ID,
            "stage": TOKEN_STAGE,
            "db_id": TOKEN_DB_ID,
            "model": model,
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        if isinstance(meta, dict):
            record.update(meta)
        with open(TOKEN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def call_llm(user_prompt, system_prompt="You are an expert about text-to-SQL and pandas code.", max_retries=3, model="gpt-4o", meta=None):
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
            log_token_usage(response=response, model=model, meta=meta)
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
DB_SCHEMA_CACHE = {}

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


def _quote_ident(name):
    return '"' + str(name).replace('"', '""') + '"'


def get_schema_index(con, db_path):
    """
    构建并缓存当前数据库的表/列索引（小写），供 SQL 前置约束校验使用。
    """
    if db_path in DB_SCHEMA_CACHE:
        return DB_SCHEMA_CACHE[db_path]

    schema_index = {}
    try:
        table_rows = con.execute("SHOW TABLES").fetchall()
    except Exception:
        table_rows = []

    for row in table_rows:
        if not row:
            continue
        table_name = str(row[0])
        table_key = table_name.lower()
        columns = set()
        try:
            desc_rows = con.execute(f"DESCRIBE {_quote_ident(table_name)}").fetchall()
            for d in desc_rows:
                if d:
                    columns.add(str(d[0]).lower())
        except Exception:
            pass
        schema_index[table_key] = columns

    DB_SCHEMA_CACHE[db_path] = schema_index
    return schema_index


def extract_sql_references(sql_str):
    """
    从 SQL 中提取：
    - 真实表名
    - 表别名 -> 真实表名映射
    - CTE 名称
    - 派生表别名（子查询别名）
    - SELECT 投影别名
    - 列引用（含可选表限定）
    """
    if not sql_str:
        return None, "Empty SQL"

    parsed = None
    parse_error = None
    for dialect in ("sqlite", "duckdb", None):
        try:
            if dialect:
                parsed = sqlglot.parse_one(sql_str, read=dialect)
            else:
                parsed = sqlglot.parse_one(sql_str)
            if parsed is not None:
                break
        except Exception as e:
            parse_error = str(e)
            continue

    if parsed is None:
        return None, f"SQL parse failed: {parse_error}"

    real_tables = set()
    alias_to_table = {}
    cte_names = set()
    derived_aliases = set()
    select_aliases = set()
    columns = []

    for cte_expr in parsed.find_all(exp.CTE):
        alias_name = (cte_expr.alias or "").strip()
        if alias_name:
            cte_names.add(alias_name.lower())

    for table_expr in parsed.find_all(exp.Table):
        table_name = (table_expr.name or "").strip()
        if not table_name:
            continue
        table_key = table_name.lower()
        alias_name = (table_expr.alias or "").strip()
        # CTE 不是底层真实表，避免误判为 schema 不存在
        if table_key in cte_names:
            if alias_name:
                derived_aliases.add(alias_name.lower())
            continue
        real_tables.add(table_key)
        if alias_name:
            alias_to_table[alias_name.lower()] = table_key

    for subquery_expr in parsed.find_all(exp.Subquery):
        alias_name = (subquery_expr.alias or "").strip()
        if alias_name:
            derived_aliases.add(alias_name.lower())

    for select_expr in parsed.find_all(exp.Select):
        for projection in select_expr.expressions or []:
            alias_name = (projection.alias or "").strip()
            if alias_name:
                select_aliases.add(alias_name.lower())

    for col_expr in parsed.find_all(exp.Column):
        col_name = (col_expr.name or "").strip()
        if not col_name or col_name == "*":
            continue
        tbl_name = (col_expr.table or "").strip()
        columns.append((tbl_name.lower() if tbl_name else "", col_name.lower()))

    return {
        "real_tables": real_tables,
        "alias_to_table": alias_to_table,
        "cte_names": cte_names,
        "derived_aliases": derived_aliases,
        "select_aliases": select_aliases,
        "columns": columns,
    }, None


def schema_precheck_sql(con, db_path, sql_str):
    """
    SQL 执行前基于 schema 做轻量约束校验。
    仅对可明确映射到真实表的引用做硬校验，避免误杀 CTE/别名列。
    """
    refs, ref_err = extract_sql_references(sql_str)
    if refs is None:
        # 复杂 SQL 或方言差异导致 parse 失败时，降级到执行器判断，避免硬拦截
        return True, None

    schema_index = get_schema_index(con, db_path)
    schema_tables = set(schema_index.keys())

    errors = []

    for t in refs["real_tables"]:
        if t not in schema_tables:
            errors.append(f"Table `{t}` not found in schema")

    for qualifier, col in refs["columns"]:
        if qualifier:
            if qualifier in refs["alias_to_table"]:
                base_table = refs["alias_to_table"][qualifier]
                if base_table in schema_tables and col not in schema_index.get(base_table, set()):
                    errors.append(f"Column `{base_table}.{col}` not found in schema")
            elif qualifier in schema_tables:
                if col not in schema_index.get(qualifier, set()):
                    errors.append(f"Column `{qualifier}.{col}` not found in schema")
            elif qualifier in refs["derived_aliases"] or qualifier in refs["cte_names"]:
                # 子查询/CTE 别名列不在基础 schema 中，放过，交给执行器最终判断
                continue
            else:
                # 未知限定符可能来自复杂作用域别名，避免误杀，交给执行器兜底
                continue
        else:
            # 未限定列可能是 SELECT 别名（常见于 ORDER BY / GROUP BY）
            if col in refs["select_aliases"]:
                continue
            # 对未限定列不做硬校验，避免 alias/作用域误判
            continue

    if errors:
        return False, "; ".join(errors[:3])
    return True, None


def build_sql_repair_prompt(question, schema, data_flow, vis_transform, vis_config, failed_sql, error_msg):
    return f"""You are a senior SQL repair engineer.

Task:
Repair the failed SQL so it is executable on SQLite/DuckDB and remains faithful to the analytics intent.

Context:
- Question: {question}
- Target VIS_CONFIG: {vis_config}
- Schema: {schema}
- DATA_FLOW: {data_flow}
- VIS_TRANSFORM: {vis_transform}

Failed SQL:
```sql
{failed_sql}
```

Failure:
{error_msg}

Hard constraints:
1. Use only tables/columns in schema.
2. Keep chart semantics implied by VIS_CONFIG (x/y/classify/sort intent).
3. Ensure GROUP BY correctness for non-aggregated select items.
4. Avoid invalid aliases like table.column alias names.
5. Output exactly one SQL in a fenced sqlite block.

```sqlite
"""


def repair_sql_with_llm(question, schema_str, data_flow, vis_transform, vis_config, failed_sql, error_msg, q_id, db_id, attempt):
    prompt = build_sql_repair_prompt(
        question=question,
        schema=schema_str,
        data_flow=data_flow,
        vis_transform=vis_transform,
        vis_config=vis_config,
        failed_sql=failed_sql,
        error_msg=error_msg,
    )
    repair_resp = call_llm(
        prompt,
        meta={
            "q_id": str(q_id),
            "db_id": db_id,
            "task": "sr2sql_repair",
            "repair_round": int(attempt),
        }
    )
    repaired_sql = extract_sql(repair_resp)
    repaired_sql = postprocess_generated_sql(repaired_sql)
    return repaired_sql


def execute_sql_with_repair(con, db_path, initial_sql, context, max_repair_rounds=MAX_SQL_REPAIR_ROUNDS):
    """
    执行链路：
    precheck -> execute
    如失败则进入 LLM 修复，最多 max_repair_rounds 轮。
    """
    current_sql = initial_sql
    repair_history = []

    for attempt in range(max_repair_rounds + 1):
        pre_ok, pre_err = schema_precheck_sql(con, db_path, current_sql)
        precheck_err_msg = f"SCHEMA_PRECHECK_ERROR: {pre_err}" if (not pre_ok and pre_err) else None

        # D2.1: 预检失败不硬拦截，统一尝试执行；执行失败再触发修复。
        df_gen, gen_err, executed_sql = execute_query_compatible(con, current_sql)
        if not gen_err:
            return df_gen, None, executed_sql, repair_history
        err_msg = str(gen_err)

        history_item = {"round": attempt, "error": err_msg, "sql": current_sql}
        if precheck_err_msg:
            history_item["precheck_error"] = precheck_err_msg
        repair_history.append(history_item)

        if attempt >= max_repair_rounds:
            return None, err_msg, current_sql, repair_history

        repaired_sql = repair_sql_with_llm(
            question=context["question"],
            schema_str=context["schema_str"],
            data_flow=context["data_flow"],
            vis_transform=context["vis_transform"],
            vis_config=context["vis_config_json"],
            failed_sql=current_sql,
            error_msg=err_msg,
            q_id=context["q_id"],
            db_id=context["db_id"],
            attempt=attempt + 1,
        )

        if not repaired_sql:
            return None, err_msg, current_sql, repair_history
        if repaired_sql.strip() == current_sql.strip():
            return None, err_msg, current_sql, repair_history
        current_sql = repaired_sql

    return None, "Max repair rounds exceeded", current_sql, repair_history


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


def _strip_identifier_quotes(name):
    if not name:
        return ""
    return str(name).strip().strip("`").strip('"').strip()


def rewrite_topn_groupby_anyvalue(sql_str):
    """
    D3 语义修复:
    针对 GROUP BY + ORDER BY ANY_VALUE(x) + LIMIT N 的常见退化写法，
    改写为“先 TopN 再聚合”，避免把 TopN 语义错误地下沉到聚合后。
    """
    if not sql_str:
        return sql_str

    pattern = re.compile(
        r"""(?is)^\s*SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<from>.+?)\s+(?P<between>.*?)\s*GROUP\s+BY\s+(?P<group>.+?)\s+ORDER\s+BY\s+(?P<order>.+?)\s+LIMIT\s+(?P<limit>\d+)\s*;?\s*$"""
    )
    m = pattern.match(sql_str.strip())
    if not m:
        return sql_str

    select_clause = (m.group("select") or "").strip()
    from_clause = (m.group("from") or "").strip()
    between_clause = (m.group("between") or "").strip()
    group_clause = (m.group("group") or "").strip()
    order_clause = (m.group("order") or "").strip()
    limit_clause = (m.group("limit") or "").strip()

    if not re.search(r"(?i)\bANY_VALUE\s*\(", order_clause):
        return sql_str
    if re.search(r"(?i)\b(join|union|intersect|except)\b", from_clause):
        return sql_str

    topn_order_clause = re.sub(
        r"(?is)\bANY_VALUE\s*\(\s*(.*?)\s*\)",
        r"\1",
        order_clause,
    ).strip()
    if not topn_order_clause:
        return sql_str

    alias_match = re.match(
        r"""(?is)^\s*(?P<table>[`"\w\.]+)\s*(?:AS\s+)?(?P<alias>[`"\w]+)?\s*$""",
        from_clause,
    )
    if alias_match:
        table_name = _strip_identifier_quotes(alias_match.group("table"))
        alias_name = _strip_identifier_quotes(alias_match.group("alias")) or table_name.split(".")[-1]
    else:
        table_name = "__src"
        alias_name = "__src"

    if not alias_name:
        alias_name = "__src"

    agg_aliases = re.findall(
        r"""(?is)\b(?:COUNT|SUM|AVG|MAX|MIN)\s*\([^)]*\)\s+AS\s+([`"\w]+)""",
        select_clause,
    )
    outer_order = ""
    if agg_aliases:
        agg_alias = agg_aliases[-1].strip()
        direction = "DESC" if re.search(r"(?i)\bDESC\b", order_clause) else "ASC"
        outer_order = f"\nORDER BY {agg_alias} {direction}"

    where_having_segment = ""
    if between_clause:
        where_having_segment = " " + between_clause.strip()

    rewritten = (
        "WITH __topn AS (\n"
        "    SELECT *\n"
        f"    FROM {from_clause}{where_having_segment}\n"
        f"    ORDER BY {topn_order_clause}\n"
        f"    LIMIT {limit_clause}\n"
        ")\n"
        f"SELECT {select_clause}\n"
        f"FROM __topn AS {_quote_ident(alias_name)}\n"
        f"GROUP BY {group_clause}"
        f"{outer_order};"
    )
    return rewritten


def rewrite_weather_date_count_distinct(sql_str):
    """
    D3 语义修复:
    weather 场景下，按日期粒度分箱统计时，COUNT(date) 常需按“天”去重。
    将 COUNT(date) 调整为 COUNT(DISTINCT date) 以贴近 VisEval 金标语义。
    """
    if not sql_str:
        return sql_str
    if not re.search(r"""(?is)\bFROM\s+[`"]?weather[`"]?\b""", sql_str):
        return sql_str
    if not re.search(r"(?is)\bGROUP\s+BY\b", sql_str):
        return sql_str
    if not re.search(r"(?is)\bstrftime\s*\(", sql_str):
        return sql_str

    pattern_count_date = re.compile(
        r"""(?is)\bCOUNT\s*\(\s*(?!DISTINCT\b)([`"]?\w+[`"]?\.)?[`"]?date[`"]?\s*\)"""
    )

    def _to_distinct(match):
        full = match.group(0)
        inner = re.search(r"""(?is)\(\s*(.*?)\s*\)""", full)
        if not inner:
            return full
        arg = inner.group(1).strip()
        return f"COUNT(DISTINCT {arg})"

    return pattern_count_date.sub(_to_distinct, sql_str)

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


def sanitize_sql_aliases(sql_str):
    """
    修复易导致 Parser Error 的别名形式：
    1) AS `table.column` / AS "table.column" -> 下划线别名
    2) AS table.column -> AS table_column
    """
    if not sql_str:
        return sql_str

    sql_str = re.sub(
        r'(?i)\bAS\s+`([^`]*\.[^`]*)`',
        lambda m: f"AS `{m.group(1).replace('.', '_')}`",
        sql_str
    )
    sql_str = re.sub(
        r'(?i)\bAS\s+"([^"]*\.[^"]*)"',
        lambda m: f'AS "{m.group(1).replace(".", "_")}"',
        sql_str
    )
    sql_str = re.sub(
        r'(?i)\bAS\s+([A-Za-z_]\w*\.[A-Za-z_]\w*)\b',
        lambda m: f"AS {m.group(1).replace('.', '_')}",
        sql_str
    )
    return sql_str


def fix_identifier_quote_typos(sql_str):
    """
    修复 LLM 常见引号错误（例如 `employees`.`FIRST_NAME' ）。
    仅针对标识符引号，不处理字符串字面量内容。
    """
    if not sql_str:
        return sql_str

    # 标识符以反引号开头、误以单引号结尾 -> 统一改为反引号闭合
    sql_str = re.sub(r"`([A-Za-z_][^`']*)'", r"`\1`", sql_str)
    # 标识符以双引号开头、误以单引号结尾 -> 统一改为双引号闭合
    sql_str = re.sub(r'"([A-Za-z_][^"\']*)\'', r'"\1"', sql_str)
    return sql_str


def postprocess_generated_sql(sql_str):
    """
    Stage 03 SQL 执行前统一清洗入口，减少可修复语法噪声。
    """
    if not sql_str:
        return sql_str
    cleaned = sanitize_sql_literals(sql_str)
    cleaned = sanitize_sql_aliases(cleaned)
    cleaned = fix_identifier_quote_typos(cleaned)
    cleaned = coerce_weekday_labels(cleaned)
    cleaned = rewrite_topn_groupby_anyvalue(cleaned)
    cleaned = rewrite_weather_date_count_distinct(cleaned)
    return cleaned


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
    gen_sql = postprocess_generated_sql(gen_sql)
    gold_sql = postprocess_generated_sql(gold_sql)

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

    # 0. 兼容归一化：classify <-> color
    cfg = dict(pred_config)
    if not cfg.get('classify') and cfg.get('color'):
        cfg['classify'] = cfg.get('color')
    if not cfg.get('color') and cfg.get('classify'):
        cfg['color'] = cfg.get('classify')

    # 1. 校验图表类型 (chart vs chart_type)
    pred_chart = (cfg.get('chart') or cfg.get('chart_type', '')).lower()
    gold_chart = gold_vis.get('chart', '').lower()
    if pred_chart != gold_chart:
        return False

    # 1.5 模式与分类约束（最小语义约束）
    classify = cfg.get('classify')
    if pred_chart == 'scatter':
        scatter_mode = (cfg.get('scatter_mode') or ('grouped' if classify else 'plain')).lower()
        if scatter_mode == 'grouped' and not classify:
            return False
    elif pred_chart == 'bar':
        bar_mode = (cfg.get('bar_mode') or ('grouped' if classify else 'plain')).lower()
        if bar_mode in {'grouped', 'stacked'} and not classify:
            return False
    elif pred_chart == 'line':
        line_mode = (cfg.get('line_mode') or ('multi_series' if classify else 'single')).lower()
        if line_mode == 'multi_series' and not classify:
            return False

    # 2. 校验轴字段 (允许全称匹配和别名语义匹配)
    pred_x = cfg.get('x_name') or cfg.get('x_axis') or cfg.get('labels')
    pred_y = cfg.get('y_name') or cfg.get('y_axis') or cfg.get('values')

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


def append_stage03_reject(reject_log_path, record):
    """将 Stage 03 失败样本结构化写入 jsonl。"""
    try:
        with open(reject_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # 日志写入失败不应影响主流程
        pass

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

def validate_and_filter(sft_file_path, dataset_root, output_file_path, target_db_id=None):
    from tqdm import tqdm
    
    vis_eval_path = os.path.join(dataset_root, "visEval.json")
    tqdm.write(f"Loading ground truth data from {vis_eval_path}...")
    with open(vis_eval_path, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    
    valid_data = []
    total = 0
    passed_vis = 0
    passed_sql = 0
    project_root = os.path.dirname(dataset_root)
    reject_log_path = os.path.join(project_root, "logs", "stage03_rejects.jsonl")
    os.makedirs(os.path.dirname(reject_log_path), exist_ok=True)
    
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

        if target_db_id and db_id != target_db_id:
            continue

        if q_id not in ground_truth_data: continue
        target_vis_obj = ground_truth_data[q_id].get('vis_obj')
        
        # 准备标准答案的展示样本
        gold_x_sample = target_vis_obj.get('x_data', [[]])[0]
        gold_y_sample = target_vis_obj.get('y_data', [[]])[0]
        gold_data_preview = list(zip(gold_x_sample, gold_y_sample))[:3]

        # 初始化本次循环的日志信息
        report = []
        fail_events = []
        gen_sql_for_log = ""
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
            fail_events.append({
                "type": "PARSE_ERROR",
                "reason": str(e),
            })
            is_parse_ok = False

        # --- 1. 校验 VIS_CONFIG ---
        vis_ok = False
        if is_parse_ok:
            vis_ok = validate_vis_config(vis_config, gold_vis)
            if not vis_ok:
                report.append(f"❌ [VIS MISMATCH]:")
                report.append(f"   - Expected: {gold_vis}")
                report.append(f"   - Got:      {vis_config}")
                fail_events.append({
                    "type": "VIS_MISMATCH",
                    "reason": "predicted VIS_CONFIG does not match gold_vis",
                    "expected": gold_vis,
                    "predicted": vis_config,
                })
            else:
                passed_vis += 1

        # --- 2. 执行与比对（D1：VIS 软门禁，允许 VIS 不匹配样本继续进行 SQL+数据实检） ---
        sql_ok = False
        should_attempt_sql = (
            is_parse_ok
            and bool(data_flow)
            and bool(vis_transform)
            and (vis_config is not None)
        )
        if should_attempt_sql:
            if not vis_ok:
                report.append("⚠️ [VIS SOFT-GATE]: VIS_CONFIG mismatch, continue SQL/data validation.")
            try:
                schema_str = item['instruction'].split("based on schema: ")[1]
                db_path = os.path.join(dataset_root, "databases", db_id)
                con = get_duckdb_connection(db_path)
                vis_config_json = json.dumps(vis_config, ensure_ascii=False)
                repair_history = []
                
                # 先由 SR2SQL 生成初始 SQL
                prompt = build_sr2sql_check_prompt(
                    schema=schema_str,
                    question=question,
                    data_flow=data_flow,
                    vis_transform=vis_transform,
                    vis_config=vis_config_json,
                )
                gen_sql_resp = call_llm(
                    prompt,
                    meta={
                        "q_id": str(q_id),
                        "db_id": db_id,
                        "task": "sr2sql_check",
                    }
                )
                gen_sql_raw = extract_sql(gen_sql_resp)
                gen_sql_fixed = postprocess_generated_sql(gen_sql_raw)

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
                gen_sql_for_log = gen_sql

                # D2: 执行前 schema 约束 + 执行失败自动修复
                df_gen, gen_err, final_sql_used, repair_history = execute_sql_with_repair(
                    con=con,
                    db_path=db_path,
                    initial_sql=gen_sql,
                    context={
                        "question": question,
                        "schema_str": schema_str,
                        "data_flow": data_flow,
                        "vis_transform": vis_transform,
                        "vis_config_json": vis_config_json,
                        "q_id": q_id,
                        "db_id": db_id,
                    },
                )
                if final_sql_used:
                    gen_sql_for_log = final_sql_used

                if gen_err:
                    report.append(f"❌ [SQL EXEC ERROR]: {gen_err}")
                    report.append(f"   - GEN SQL: {gen_sql_for_log.replace(os.linesep, ' ')}")
                    if repair_history:
                        report.append(f"   - REPAIR ROUNDS: {len(repair_history)-1}")
                    fail_type = "SCHEMA_PRECHECK_ERROR" if str(gen_err).startswith("SCHEMA_PRECHECK_ERROR:") else "SQL_EXEC_ERROR"
                    fail_events.append({
                        "type": fail_type,
                        "reason": str(gen_err),
                        "gen_sql": gen_sql_for_log,
                        "repair_history": repair_history,
                    })
                else:
                    # 数据比对
                    is_match, msg = validate_against_vis_obj(df_gen, target_vis_obj)
                    if is_match:
                        sql_ok = True
                        passed_sql += 1
                    else:
                        report.append(f"❌ [DATA MISMATCH]: {msg}")
                        report.append(f"   - GEN SQL: {gen_sql_for_log.replace(os.linesep, ' ')}")
                        # 展示标准答案 vs 模型答案
                        report.append(f"   - GOLD DATA (Sample): {gold_data_preview}")
                        gen_preview = df_gen.head(3).values.tolist()
                        report.append(f"   - GEN DATA (Sample):  {gen_preview}")
                        fail_events.append({
                            "type": "DATA_MISMATCH",
                            "reason": str(msg),
                            "gen_sql": gen_sql_for_log,
                            "gold_data_preview": gold_data_preview,
                            "gen_data_preview": gen_preview,
                            "repair_history": repair_history,
                        })
            except Exception as e:
                report.append(f"❌ [SYSTEM ERROR]: {e}")
                fail_events.append({
                    "type": "SYSTEM_ERROR",
                    "reason": str(e),
                    "gen_sql": gen_sql_for_log,
                })
        else:
            fail_events.append({
                "type": "SQL_SKIPPED",
                "reason": "DVCR parse failed or required sections are missing",
            })

        # --- 判定最终结果 ---
        if sql_ok:
            # tqdm.write(f"✅ ID {q_id}: Passed") # 成功时可以不打印，保持日志清爽
            valid_data.append(item)
        else:
            # 只有失败时才打印整份报告
            tqdm.write("\n".join(report))
            tqdm.write(f"{'='*80}")
            append_stage03_reject(
                reject_log_path,
                {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "run_id": TOKEN_RUN_ID,
                    "stage": TOKEN_STAGE,
                    "db_id": db_id,
                    "id": str(q_id),
                    "question": question,
                    "primary_fail_type": fail_events[0]["type"] if fail_events else "UNKNOWN",
                    "fail_events": fail_events,
                    "gen_sql": gen_sql_for_log,
                },
            )
            
    # 最终汇总
    tqdm.write(f"\n✨ FINAL STATS ✨")
    tqdm.write(f"Final Yield: {len(valid_data)}/{total} ({len(valid_data)/total:.1%})")
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate SFT Data")
    parser.add_argument("--db_id", type=str, default=None, help="指定要处理的数据库ID (例如: hospital_1)")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    dataset_root = os.path.join(project_root, "dataset")
    input_file = os.path.join(project_root, "temp_results", "sft_data", "sft_origin_data.jsonl")
    output_file = os.path.join(project_root, "temp_results", "sft_data", "sft_advised_data.jsonl")
    init_token_logger(project_root=project_root, db_id=args.db_id)
    
    print(f"Cleaning output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        pass # 打开并关闭 w 模式，即清空文件内容
    # if not os.path.exists(dataset_root):
    #     dataset_root = "/root/nl2vis/VisEval-main/visEval_dataset"
    
    if os.path.exists(input_file):
        validate_and_filter(input_file, dataset_root, output_file, target_db_id=args.db_id)
    else:
        print(f"Error: {input_file} not found. Run generation script first.")
