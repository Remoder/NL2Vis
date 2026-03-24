import os
import json
import pandas as pd
import glob, argparse
from openai import OpenAI
import time
from tqdm import tqdm
from dvcr_protocol import (
    DVCR_PROTOCOL_SYSTEM_PROMPT,
    build_sql2dvcr_prompt,
)

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
TOKEN_STAGE = "stage02_generate_sft"
TOKEN_DB_ID = "ALL"


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
    init_token_logger(project_root=project_root, db_id=args.db_id)
    
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

        user_prompt = build_sql2dvcr_prompt(
            question=question,
            sql=sql,
            binning=binning,
            vis=json.dumps(gold_vis),
            schema=schema_json,
        )
        system_prompt = DVCR_PROTOCOL_SYSTEM_PROMPT

        dvcr_response = call_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            meta={
                "q_id": str(q_id),
                "db_id": db_id,
                "task": "sql2dvcr",
            }
        )
        
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
