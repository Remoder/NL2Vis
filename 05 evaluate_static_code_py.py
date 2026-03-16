# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from pathlib import Path
import sys
import os
import glob
import re
import warnings
from contextlib import redirect_stdout
import io
import dotenv

# [Mod] 强制优先使用当前目录下的 viseval 源码
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from viseval import Evaluator
from viseval.dataset import Dataset as VisevalDataset
from viseval.agent import ChartExecutionResult
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def show_svg(plt_obj, log_name=None):
    """Generate SVG content from matplotlib plot."""
    f = io.StringIO()
    try:
        plt_obj.savefig(f, format="svg")
        if log_name:
            plt_obj.savefig(f"{log_name}")
        svg_content = f.getvalue()
    except Exception as e:
        print(f"Error generating SVG: {e}")
        return None
    finally:
        plt_obj.close()
    return svg_content

dotenv.load_dotenv()

class StaticCodeAgent:
    """
    一个“静态代码”Agent，它不调用 LLM 生成代码，
    而是从文件中读取预先生成的 Python 代码并返回。
    """
    def __init__(self, code_map):
        self.code_map = code_map

    def generate(self, nl_query: str, tables: list[str], config: dict):
        """
        返回预置的代码。如果找不到，则返回特定标记。
        """
        # 1. 精确匹配
        if isinstance(self.code_map, dict) and nl_query in self.code_map:
            return self.code_map[nl_query], {"tables": tables}
        
        # 2. 去空格模糊匹配
        if isinstance(self.code_map, dict):
            for k, v in self.code_map.items():
                if k.strip() == nl_query.strip():
                    return v, {"tables": tables}

        # 3. 如果找不到代码，返回标记，说明该实例在之前的 SQL 校验阶段失败了
        return "# SQL_VALIDATION_FAILED", {"tables": tables}

    def execute(self, code: str, context: dict, log_name: str = None) -> ChartExecutionResult:
        """
        执行代码并生成可视化。
        """
        # 识别 SQL 校验失败标记
        if not code or code.strip() == "# SQL_VALIDATION_FAILED":
            return ChartExecutionResult(status=False, error_msg="SQL Validation Failed (No code generated in Stage 03)")

        tables = context.get("tables", [])
        
        # Load tables
        dfs = {}
        for table_path in tables:
            try:
                table_name = os.path.basename(table_path).split('.')[0]
                try:
                    df = pd.read_csv(table_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(table_path, encoding='latin1')
                dfs[table_name] = df
            except Exception as e:
                print(f"Error loading {table_path}: {e}")
                return ChartExecutionResult(status=False, error_msg=f"Failed to load data: {e}")

        # Prepare execution environment
        def safe_exit(code=0):
            raise RuntimeError(f"Code called exit({code})")

        exec_globals = {
            "pd": pd,
            "plt": plt,
            "show_svg": show_svg,
            "exit": safe_exit,
            "quit": safe_exit,
        }
        # Inject dataframes
        for name, df in dfs.items():
            exec_globals[name] = df.copy()

        plt.clf()

        try:
            # Execute code
            exec(code, exec_globals)
            
            fig = plt.gcf()
            if not fig.axes:
                 return ChartExecutionResult(status=False, error_msg="No plot generated (empty figure).")
            
            svg_content = show_svg(plt, log_name)
            if svg_content:
                return ChartExecutionResult(status=True, svg_string=svg_content)
            else:
                return ChartExecutionResult(status=False, error_msg="Failed to generate SVG output.")

        except Exception as e:
            return ChartExecutionResult(status=False, error_msg=f"Execution Error: {str(e)}")

def load_code_files(code_folder: str, dataset):
    """
    加载文件夹下的 .py/.txt 代码文件。
    """
    code_map = {}
    relevant_ids = set()
    files = []
    files.extend(glob.glob(os.path.join(code_folder, "**", "*.py"), recursive=True))
    files.extend(glob.glob(os.path.join(code_folder, "**", "*.txt"), recursive=True))
    
    files = sorted(files)
    print(f"Found {len(files)} code files in {code_folder}")
    
    dataset_queries = {}
    for item in dataset:
        item_id = str(item['id'])
        queries = item['nl_queries']
        dataset_queries[item_id] = queries

    for f_path in files:
        filename = os.path.basename(f_path)
        stem = os.path.splitext(filename)[0]
        
        if '_' in stem:
            parts = stem.rsplit('_', 1)
            if parts[1].isdigit():
                query_id_part = parts[0]
                paraphrase_idx = int(parts[1])
                
                target_query_id = None
                for ds_query_id in dataset_queries.keys():
                    safe_query_id = str(ds_query_id).replace("@", "_").replace("/", "_")
                    if safe_query_id == query_id_part:
                        target_query_id = ds_query_id
                        break

                if not target_query_id and query_id_part in dataset_queries:
                    target_query_id = query_id_part

                if target_query_id and target_query_id in dataset_queries:
                    relevant_ids.add(target_query_id)
                    queries = dataset_queries[target_query_id]
                    if 0 <= paraphrase_idx < len(queries):
                        target_query_text = queries[paraphrase_idx]
                        with open(f_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            content = re.sub(r'```python', '', content)
                            content = re.sub(r'```', '', content)
                            code_map[target_query_text] = content

    print(f"Matched code for {len(relevant_ids)} original IDs.")
    return code_map, relevant_ids

class DatasetWrapper:
    def __init__(self, items):
        self.benchmark = items
        self.dict = {str(item['id']): item for item in items}

def _main():
    parser = argparse.ArgumentParser(description="Evaluate static code files against VisEval benchmark")
    parser.add_argument("--code_folder", type=str, required=True, help="Path to folder containing code files")
    parser.add_argument("--benchmark", type=Path, required=True, help="Path to VisEval dataset")
    parser.add_argument("--type", type=str, choices=["all", "single", "multiple"], default="all")
    parser.add_argument("--irrelevant_tables", type=bool, default=False)
    parser.add_argument("--library", type=str, default="matplotlib", choices=["matplotlib", "seaborn"])
    parser.add_argument("--webdriver", type=Path, default=None)
    parser.add_argument("--logs", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--db_id", type=str, default=None)
    parser.add_argument("--limit_id", type=str, default=None)

    args = parser.parse_args()

    if args.logs is None:
        db_id = args.db_id if args.db_id else "unknown"
        args.logs = Path(f"../logs/logs_static_eval_{db_id}")

    # 1. Config Dataset
    dataset_kwargs = {}
    if args.limit is not None:
        dataset_kwargs["limit"] = args.limit
    if args.db_id is not None:
        dataset_kwargs["db_id"] = args.db_id
    
    raw_dataset = VisevalDataset(args.benchmark, args.type, args.irrelevant_tables, **dataset_kwargs)
    dataset_items = list(raw_dataset.benchmark)
    
    if args.limit_id:
        dataset_items = [item for item in dataset_items if str(item['id']) == args.limit_id]

    print(f"Loaded dataset with {len(dataset_items)} examples.")

    # 2. Load Code
    code_map, relevant_ids = load_code_files(args.code_folder, dataset_items)
    
    # 【核心修改点 1】：不再过滤 relevant_ids，直接评测全量数据集
    filtered_items = dataset_items 
    print(f"Starting FULL evaluation for all {len(filtered_items)} instances.")

    # 3. Config Static Agent
    agent = StaticCodeAgent(code_map)

    # 4. Config Evaluator
    evaluator = Evaluator(webdriver_path=args.webdriver, vision_model=None)

    # 5. Run Evaluation
    config = {"library": args.library, "logs": args.logs, "force_rerun": True}
    dataset_wrapper = DatasetWrapper(filtered_items)
    result = evaluator.evaluate(agent, dataset_wrapper, config)
    score = result.score()
    
    print("\n" + "="*50)
    print(f"🏆 全量端到端通过率 (End-to-End Pass Rate): {score['pass_rate']:.2%}")
    print(f"   (基于 {len(filtered_items)} 个实例的完整统计)")
    print("="*50 + "\n")

    # 6. Generate Report
    if args.logs:
        try:
            print("\n" + "=" * 80)
            print("📊 正在生成错误分析报告...")
            print("=" * 80 + "\n")
            
            import importlib.util
            analyze_results_path = Path(current_dir) / "codes/analyze_results.py"
            
            if not analyze_results_path.exists():
                print(f"❌ 错误：找不到分析脚本 {analyze_results_path}")
                return 

            spec = importlib.util.spec_from_file_location("analyze_results", analyze_results_path)
            analyze_results = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(analyze_results)
            
            load_results = analyze_results.load_results
            analyze_instance = analyze_results.analyze_instance
            print_analysis_summary = analyze_results.print_analysis_summary
            print_instance_details = analyze_results.print_instance_details
            
            logs_dir = Path(args.logs)
            if logs_dir.exists():
                results = load_results(logs_dir)
                
                if not results:
                    print(f"❌ 警告：未找到结果。")
                else:
                    all_analyses = []
                    # 统计 SQL 校验失败的数量
                    sql_fail_count = 0

                    for instance_id, data in sorted(results.items()):
                        analysis = analyze_instance(instance_id, data['evaluations'], data['codes'])
                        
                        # 检查错误详情，如果是我们标记的“SQL 校验失败”，则在分析中体现
                        for evals in data['evaluations']:
                            for e in evals:
                                if "SQL Validation Failed" in str(e.get('rationale', '')):
                                    sql_fail_count += 1
                        
                        all_analyses.append(analysis)
                    
                    print_analysis_summary(all_analyses)
                    print(f"\n📢 其中由于 Stage 03 导致的代码缺失 (SQL 校验失败): {sql_fail_count} 次")
                    
                    report_file = logs_dir / "error_analysis_report.txt"
                    with open(report_file, 'w', encoding='utf-8') as f:
                        output_buffer = io.StringIO()
                        with redirect_stdout(output_buffer):
                            print_analysis_summary(all_analyses)
                            print(f"\n📢 统计说明：")
                            print(f" - 总实例数: {len(filtered_items)}")
                            print(f" - SQL 校验失败 (无代码生成): {sql_fail_count} 次")
                            
                            failed_instances = [a for a in all_analyses if a['failed_queries'] > 0]
                            if failed_instances:
                                print(f"\n{'='*80}")
                                print(f"📋 失败实例详情 ({len(failed_instances)} 个)")
                                for analysis in failed_instances:
                                    print_instance_details(analysis, verbose=True)
                        f.write(output_buffer.getvalue())
                    print(f"\n✓ 错误分析报告已保存到: {report_file}")
        except Exception as e:
            import traceback
            print(f"⚠️  分析报告生成失败: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    _main()