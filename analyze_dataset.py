import json
import os
from collections import defaultdict, Counter

def analyze_dataset(json_path):
    if not os.path.exists(json_path):
        print(f"❌ 错误：找不到文件 {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 存储统计结果: db_id -> stats
    # stats 包含: total_count, hardness_distribution, id_details
    db_stats = defaultdict(lambda: {
        "total": 0,
        "nl_query_total": 0,
        "hardness_counts": Counter(),
        "ids": []
    })

    for q_id, entry in data.items():
        db_id = entry.get("db_id", "unknown")
        hardness = entry.get("hardness", "Unknown")
        nl_queries = entry.get("nl_queries", [])
        
        stats = db_stats[db_id]
        stats["total"] += 1
        stats["nl_query_total"] += len(nl_queries)
        stats["hardness_counts"][hardness] += 1
        stats["ids"].append((q_id, hardness))

    # 打印汇总报告
    print("=" * 80)
    print(f"{'Database ID':<25} | {'Total Tasks':<12} | {'NL Queries':<12} | {'Hardness Distribution'}")
    print("-" * 80)
    
    # 按任务总数降序排列
    sorted_dbs = sorted(db_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    
    for db_id, stats in sorted_dbs:
        h_dist = ", ".join([f"{h}: {c}" for h, c in stats["hardness_counts"].items()])
        print(f"{db_id:<25} | {stats['total']:<12} | {stats['nl_query_total']:<12} | {h_dist}")

    # 打印详细清单（可选）
    print("\n" + "=" * 80)
    print("📂 详细编号与难度映射 (前 3 个库示例)")
    print("=" * 80)
    for db_id, stats in sorted_dbs[:3]:
        print(f"\n🔹 数据库: {db_id}")
        # 每行显示 5 个 ID
        id_list = stats["ids"]
        for i in range(0, len(id_list), 4):
            chunk = id_list[i:i+4]
            formatted_ids = "  ".join([f"[{id_}: {h}]" for id_, h in chunk])
            print(f"  {formatted_ids}")

    # 总计统计
    all_total_tasks = sum(s['total'] for s in db_stats.values())
    all_total_queries = sum(s['nl_query_total'] for s in db_stats.values())
    print("\n" + "=" * 80)
    print(f"📊 全局统计:")
    print(f"  - 总数据库数: {len(db_stats)}")
    print(f"  - 总任务实例 (Instance): {all_total_tasks}")
    print(f"  - 总自然语言提问 (NL Queries): {all_total_queries}")
    print("=" * 80)

if __name__ == "__main__":
    # 请确保路径指向你的 visEval.json
    path = "/home/wys/data/expriments/NL2Vis/dataset/visEval.json"
    analyze_dataset(path)