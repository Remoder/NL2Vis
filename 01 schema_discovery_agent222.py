import os
import json
import pandas as pd
import re
import glob
import argparse
from typing import List, Dict, Any
from custom_agent import CustomLLMAdapter
from langchain_core.messages import HumanMessage

# ==========================================
# 1. 数据层 (保持不变)
# ==========================================
class DataRepository:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.dfs = {}
        self._load_all_csvs()

    def _load_all_csvs(self):
        csv_files = glob.glob(os.path.join(self.folder_path, "*.csv"))
        for file_path in csv_files:
            try:
                table_name = os.path.splitext(os.path.basename(file_path))[0]
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                df.columns = [c.strip() for c in df.columns]
                self.dfs[table_name] = df
            except:
                pass

    def get_table(self, table_name: str) -> pd.DataFrame:
        return self.dfs.get(table_name)

# ==========================================
# 2. 工具层 (增强统计输出)
# ==========================================
class ExplorerTools:
    def __init__(self, repo: DataRepository):
        self.repo = repo

    def get_db_profile(self) -> str:
        profile = "Database Overview:\n"
        for name, df in self.repo.dfs.items():
            profile += f"- Table '{name}' ({len(df)} rows): {', '.join(df.columns.tolist())}\n"
        return profile

    def get_column_stats(self, table_name: str, column_name: str) -> str:
        """🔥 修正了 NumPy 类型无法 JSON 序列化的问题"""
        df = self.repo.get_table(table_name)
        if df is None or column_name not in df.columns:
            return "Error: Column not found."
        
        series = df[column_name]
        valid_series = series.dropna()
        
        # 使用 .tolist() 会自动把 numpy.int64 转换为 python int
        unique_vals = valid_series.unique()
        n_unique = len(unique_vals)
        
        stats = {
            "dtype": str(series.dtype),
            "distinct_count": int(n_unique), # 强制转为原生 int
            "sample_values": pd.Series(unique_vals[:5]).tolist() # 核心：使用 tolist() 转换类型
        }
        
        # 增加 default=str 作为一个最终保险，防止其他不可预见的类型报错
        return json.dumps(stats, default=str)

    def verify_relationship(self, table_a: str, column_a: str, table_b: str, column_b: str) -> str:
        df1 = self.repo.get_table(table_a)
        df2 = self.repo.get_table(table_b)
        if df1 is None or df2 is None or column_a not in df1.columns or column_b not in df2.columns:
            return "Overlap: 0"
        vals1 = set(df1[column_a].dropna().astype(str))
        vals2 = set(df2[column_b].dropna().astype(str))
        intersection = vals1.intersection(vals2)
        return f"Overlap: {len(intersection)}"

# ==========================================
# 3. 🕵️‍♂️ Agent A: Topology Agent (增强语义角色)
# ==========================================
class TopologyAgent:
    def __init__(self, llm, tools: ExplorerTools):
        self.llm = llm
        self.tools = tools

    def run(self) -> List[Dict]:
        print("\n🕵️‍♂️ [Topology Agent] Identifying relationship roles...")
        db_profile = self.tools.get_db_profile()
        
        prompt = f"""
Analyze the table schema below and identify ALL potential Foreign Key relationships.
For each link, you MUST determine its semantic role:
1. relationship_role: 
   - "Ownership/Primary": The main entity association (e.g., Patient's PCP, Department's Head). Essential for identifying the primary join path.
   - "Action/Event": A transactional association (e.g., Surgeon in an Operation, Student in a specific Class session).
2. priority: "High" for core links, "Low" for secondary links.

Schema:
{db_profile}

Response strictly in JSON format:
```json
{{
  "candidates": [
    {{
      "table_a": "...", "column_a": "...", 
      "table_b": "...", "column_b": "...",
      "relationship_role": "Ownership/Primary OR Action/Event",
      "priority": "High/Medium/Low",
      "reason": "..."
    }}
  ]
}}

"""
        response = self.llm.invoke([HumanMessage(content=prompt)]).content
        candidates = self._extract_json(response).get("candidates", [])
        verified_relationships = []

        for cand in candidates:
            res = self.tools.verify_relationship(cand['table_a'], cand['column_a'], cand['table_b'], cand['column_b'])
            if "Overlap: 0" not in res:
                verified_relationships.append({
                    "source": f"{cand['table_a']}.{cand['column_a']}",
                    "target": f"{cand['table_b']}.{cand['column_b']}",
                    "role": cand.get('relationship_role', 'Action/Event'),
                    "priority": cand.get('priority', 'Medium'),
                    "reason": cand.get('reason', '')
                })

        return verified_relationships

    def _extract_json(self, text):
        try:
            # 强化正则：贪婪匹配 JSON 块
            match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
            if match: return json.loads(match.group(1))
            s, e = text.find('{'), text.rfind('}')
            if s != -1 and e != -1: return json.loads(text[s:e+1])
            return {}
        except: return {}

# ==========================================
# 4. 📊 Agent B: Profiling Agent (注入拓扑上下文)
# ==========================================
class ProfilingAgent:
    def __init__(self, llm, tools: ExplorerTools):
        self.llm = llm
        self.tools = tools

    def analyze_table(self, table_name: str, verified_relationships: List[Dict]) -> Dict:
        print(f"📊 [Profiling Agent] Deep analysis: {table_name}...")
        df = self.tools.repo.get_table(table_name)
        
        # 1. 注入拓扑上下文
        relevant_rels = [r for r in verified_relationships if table_name in r['source'] or table_name in r['target']]
        rel_context = "RELATED RELATIONSHIPS & ROLES:\n" + "\n".join(
            [f"- {r['source']} -> {r['target']} | Role: {r['role']} | Priority: {r['priority']}" for r in relevant_rels]
        )

        # 2. 准备列统计
        col_stats_summary = ""
        for col in df.columns:
            stats = self.tools.get_column_stats(table_name, col)
            col_stats_summary += f"Column '{col}': {stats}\n"

        sample_rows = df.head(2).to_json(orient='records')

        prompt = f"""
You are a Data Visualization & SQL Expert. Analyze table '{table_name}'.

RELATIONSHIP CONTEXT:
{rel_context}

COLUMN STATISTICS:
{col_stats_summary}

VISUALIZATION RULES:

If Categorical and distinct_count < 10: Suggest "Pie Chart".

If Categorical and distinct_count >= 10: Suggest "Bar Chart".

If Quantitative: Suggest "Bar Chart" (for comparison) or "Histogram" (for distribution).

If Time: Suggest "Line Chart" (for trends).

CRITICAL INSTRUCTIONS:

If a column has an 'Ownership/Primary' role in the context, add 'PRIMARY LINK' to its description.

If it's an 'Action/Event' role, describe it as a 'Transactional Link'.

CRITICAL RULES FOR SEMANTIC_TYPE:
1. IDENTIFIERS ARE NOT QUANTITATIVE: 
   - Any column that is a Primary Key, Foreign Key, ID, SSN, Phone Number, or Zip Code MUST be 'Categorical' or 'Text', NEVER 'Quantitative'. 
   - Even if the data consists of numbers, if you don't perform math (SUM/AVG) on it, it is NOT Quantitative.
   
2. VISUALIZATION REFINEMENT:
   - For 'Categorical' columns with very high cardinality (like Name, EmployeeID, SSN): 
     Suggest "Table" or "None" instead of "Bar Chart". 
   - Only suggest "Bar Chart" for Categorical columns with repeated values (like Position, Department).

3. PRIMARY LINK MARKING:
   - {rel_context}
   - As you've done, mark columns involved in 'Ownership/Primary' roles as 'PRIMARY LINK'.

Output JSON:
{{
"{table_name}": {{
"ColumnName": {{
"semantic_type": "Categorical/Quantitative/Time/Text",
"description": "...",
"visualization_hints": {{ "preferred_chart": "..." }}
}},
"_sample_rows": {sample_rows}
}}
}}
    """
        response = self.llm.invoke([HumanMessage(content=prompt)]).content
        result = self._extract_json(response)

        # 鲁棒性修正：确保返回结果包含表名层级
        if table_name not in result and len(result) > 0:
            return {table_name: result}
        return result

    def _extract_json(self, text):
        try:
            match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
            if match: return json.loads(match.group(1))
            s, e = text.find('{'), text.rfind('}')
            if s != -1 and e != -1: return json.loads(text[s:e+1])
            return {}
        except: return {}

# ==========================================
# 5. 🧠 Master Pipeline (闭环流控)
# ==========================================

class NL2VISPipeline:
    def __init__(self, llm, folder_path):
        self.repo = DataRepository(folder_path)
        self.tools = ExplorerTools(self.repo)
        self.topo_agent = TopologyAgent(llm, self.tools)
        self.prof_agent = ProfilingAgent(llm, self.tools)

    def run(self):
        # Phase 1: 拓扑发现 (先跑，生成全局关系网)
        relationships = self.topo_agent.run()
        
        # Phase 2: 逐表画像 (注入 Phase 1 的结果)
        semantic_analysis = {}
        table_names = list(self.repo.dfs.keys())
        
        for t_name in table_names:
            table_info = self.prof_agent.analyze_table(t_name, relationships)
            if t_name in table_info:
                semantic_analysis[t_name] = table_info[t_name]
            else:
                semantic_analysis[t_name] = table_info

        return {
            "relationships": relationships,
            "semantic_analysis": semantic_analysis
        }
        
# ==========================================
# 启动函数与入口
# ==========================================

def discover_database_schema(llm, dataset_folder: str) -> Dict[str, Any]:
    pipeline = NL2VISPipeline(llm, dataset_folder)
    return pipeline.run()

def process_database(db_name: str, databases_root: str, output_dir: str, llm) -> bool:
    dataset_folder = os.path.join(databases_root, db_name)
    if not os.path.isdir(dataset_folder): return False

    print(f"\n🔄 Processing DB: {db_name}")
    result_json = discover_database_schema(llm, dataset_folder)

    output_path = os.path.join(output_dir, f"{db_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False, default=str)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_id", type=str, default=None)
    parser.add_argument("--engine", type=str, default="gpt-4o")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    databases_root = os.path.join(project_root, "dataset", "databases")
    output_dir = os.path.join(project_root, "temp_results", "schema")
    os.makedirs(output_dir, exist_ok=True)

    llm = CustomLLMAdapter(engine=args.engine, temp=0.0)

    if args.db_id:
        process_database(args.db_id, databases_root, output_dir, llm)
    else:
        for db in sorted(os.listdir(databases_root)):
            if os.path.isdir(os.path.join(databases_root, db)) and not db.startswith('.'):
                try:
                    process_database(db, databases_root, output_dir, llm)
                except Exception as e:
                    print(f"Error {db}: {e}")