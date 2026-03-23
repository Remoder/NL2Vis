#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析评测结果，找出未通过的实例和失败原因
"""
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# 定义检查类型分类
VALIDITY_ASPECTS = ["code execution", "surface-form check"]
LEGALITY_ASPECTS = ["deconstruction", "chart type check", "data check", "order check", "layout check", "scale and ticks check"]
READABILITY_ASPECT = "readability check"

def load_results(logs_dir: Path) -> Dict[str, Dict]:
    """加载所有评测结果"""
    results = {}
    instance_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    
    for instance_dir in sorted(instance_dirs):
        result_file = instance_dir / "result.json"
        if result_file.exists():
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results[instance_dir.name] = {
                        'codes': data.get('codes', []),
                        'evaluations': data.get('evaluations', [])
                    }
            except Exception as e:
                print(f"⚠️  读取 {result_file} 失败: {e}")
    
    return results

def analyze_instance(instance_id: str, evaluations: List[List[Dict]], codes: List[str]) -> Dict:
    """分析单个实例的评测结果"""
    analysis = {
        'instance_id': instance_id,
        'total_queries': len(evaluations),
        'passed_queries': 0,
        'failed_queries': 0,
        'failed_by_validity': 0,
        'failed_by_legality': 0,
        'query_details': [],
        'failure_reasons': Counter()
    }
    
    for query_idx, query_results in enumerate(evaluations):
        # 将结果转为字典方便查找
        results_dict = {r['aspect']: r for r in query_results}
        
        # 检查有效性（validity）
        validity_passed = all(
            results_dict.get(aspect, {}).get('answer', False)
            for aspect in VALIDITY_ASPECTS
        )
        
        # 检查合法性（legality）
        # 注意：某些检查可能因为前置检查失败而未执行
        # 只有实际执行了的检查才计入合法性判断
        executed_legality_aspects = [
            aspect for aspect in LEGALITY_ASPECTS
            if aspect in results_dict and results_dict[aspect].get('rationale') != 'N/A'
        ]
        legality_passed = (
            len(executed_legality_aspects) > 0 and
            all(results_dict[aspect].get('answer', False) for aspect in executed_legality_aspects)
        )
        
        # 总体是否通过
        passed = validity_passed and legality_passed
        
        query_detail = {
            'query_idx': query_idx,
            'passed': passed,
            'validity_passed': validity_passed,
            'legality_passed': legality_passed,
            'code': codes[query_idx] if query_idx < len(codes) else None,
            'failures': []
        }
        
        # 收集失败原因
        if not validity_passed:
            analysis['failed_by_validity'] += 1
            for aspect in VALIDITY_ASPECTS:
                result = results_dict.get(aspect, {})
                if not result.get('answer', False):
                    query_detail['failures'].append({
                        'type': 'validity',
                        'aspect': aspect,
                        'reason': result.get('rationale', 'N/A')
                    })
                    analysis['failure_reasons'][f'validity: {aspect}'] += 1
        
        if not legality_passed:
            analysis['failed_by_legality'] += 1
            for aspect in LEGALITY_ASPECTS:
                result = results_dict.get(aspect, {})
                # 只记录实际执行的检查（rationale 不为 'N/A'）
                if result and result.get('rationale') != 'N/A':
                    if not result.get('answer', False):
                        query_detail['failures'].append({
                            'type': 'legality',
                            'aspect': aspect,
                            'reason': result.get('rationale', 'N/A')
                        })
                        analysis['failure_reasons'][f'legality: {aspect}'] += 1
        
        if passed:
            analysis['passed_queries'] += 1
        else:
            analysis['failed_queries'] += 1
        
        query_detail['all_results'] = {r['aspect']: {
            'passed': r['answer'],
            'reason': r.get('rationale', 'N/A')
        } for r in query_results}
        
        analysis['query_details'].append(query_detail)
    
    return analysis

def print_analysis_summary(all_analyses: List[Dict]):
    """打印分析摘要，包含实例级和查询级双重准确率"""
    # --- 1. 基础统计 ---
    total_instances = len(all_analyses)
    # 实例级通过：该实例下的 passed_queries 等于 total_queries
    full_passed_instances = sum(1 for a in all_analyses if a['passed_queries'] == a['total_queries'] and a['total_queries'] > 0)
    
    total_queries = sum(a['total_queries'] for a in all_analyses)
    total_passed = sum(a['passed_queries'] for a in all_analyses)
    total_failed = sum(a['failed_queries'] for a in all_analyses)

    # --- 2. 计算准确率 ---
    # 算法 A: 实例级端到端准确率 (Hard Pass)
    instance_e2e_acc = (full_passed_instances / total_instances * 100) if total_instances > 0 else 0
    # 算法 B: 查询级准确率 (Soft Pass)
    query_acc = (total_passed / total_queries * 100) if total_queries > 0 else 0
    
    # --- 3. 统计失败原因（含 SQL 校验失败） ---
    all_failure_reasons = Counter()
    stage03_failed_queries = 0
    for analysis in all_analyses:
        all_failure_reasons.update(analysis['failure_reasons'])
        for q_detail in analysis['query_details']:
            # 这里的判定逻辑需与 05 脚本中返回的标记一致
            has_stage03_fail = False
            for fail in q_detail['failures']:
                if "SQL Validation Failed" in str(fail.get('reason', '')):
                    has_stage03_fail = True
            if has_stage03_fail:
                stage03_failed_queries += 1

    # 通过 Stage 03 检验并进入 DVCR 图像生成阶段后的查询准确率
    dvcr_generated_queries = total_queries - stage03_failed_queries
    dvcr_generated_acc = (
        total_passed / dvcr_generated_queries * 100
        if dvcr_generated_queries > 0 else 0
    )
                    
    data_check_failures = []
    for analysis in all_analyses:
        for query_detail in analysis['query_details']:
            if not query_detail['passed']:
                for failure in query_detail['failures']:
                    if failure['aspect'] == 'data check':
                        data_check_failures.append({
                            'instance': analysis['instance_id'],
                            'query': query_detail['query_idx'],
                            'reason': failure['reason']
                        })

    # --- 4. 打印输出 ---
    print("=" * 80)
    print("📊 评测结果分析摘要 (End-to-End Analysis)")
    print("=" * 80)
    
    print(f"\n🚀 核心指标 (Core Metrics):")
    print(f"  ⭐ 实例级端到端准确率 (Instance E2E): {instance_e2e_acc:.2f}%")
    print(f"     (完全通过实例数/总实例数: {full_passed_instances} / {total_instances})")
    
    print(f"\n  🔹 查询级准确率 (Query-level): {query_acc:.2f}%")
    print(f"     (总通过查询数/总查询数: {total_passed} / {total_queries})")
    print(f"\n  🧭 通过 Stage 03 检验后的 DVCR 生成图像正确率: {dvcr_generated_acc:.2f}%")
    print(f"     (总通过查询数/(总查询数-Stage 03 失败查询数): {total_passed} / {dvcr_generated_queries})")
    
    print(f"\n📈 统计详情:")
    print(f"  • 总实例数: {total_instances}")
    print(f"  • 总查询数: {total_queries}")
    print(f"  • 失败查询数: {total_failed}")
    if stage03_failed_queries > 0:
        print(f"    - 其中由于 Stage 03 校验失败导致的无代码生成: {stage03_failed_queries}")
    
    print(f"\n🔍 失败原因分布 (Top 10):")
    if not all_failure_reasons:
        print(f"  🎉 暂无失败记录")
    else:
        for reason, count in all_failure_reasons.most_common(10):
            print(f"  • {reason}: {count} 次")
    
    # 找出完全失败的实例
    failed_instances = [a for a in all_analyses if a['passed_queries'] == 0]
    if failed_instances:
        print(f"\n⚠️  完全失败的实例 ({len(failed_instances)} 个):")
        for analysis in failed_instances:
            print(f"  • 实例 {analysis['instance_id']}: {analysis['failed_queries']}/{analysis['total_queries']} 失败")
    
    
    # 分析 data check 失败
    if data_check_failures:
        print(f"\n📋 数据检查失败分析 ({len(data_check_failures)} 个):")
        print("  失败原因通常是图表中的数据点与期望值不匹配。")
        print("  可能的原因:")
        print("    1. 数据聚合方式不正确（如计数、求和等）")
        print("    2. 数据筛选条件不准确")
        print("    3. 图表中缺少某些预期的数据点")
        print("    4. 数据值计算错误")
        
        # 提取失败的数据点
        missing_data_points = []
        for failure in data_check_failures:
            reason = failure['reason']
            if 'not found' in reason:
                # 尝试提取数据点信息
                import re
                match = re.search(r'\{[^}]+\}', reason)
                if match:
                    missing_data_points.append({
                        'instance': failure['instance'],
                        'query': failure['query'],
                        'data_point': match.group(0)
                    })
        
        if missing_data_points:
            print(f"\n  缺失的数据点 ({len(missing_data_points)} 个):")
            for item in missing_data_points[:5]:  # 只显示前5个
                print(f"    • 实例 {item['instance']} 查询 {item['query']}: {item['data_point']}")
            if len(missing_data_points) > 5:
                print(f"    ... 还有 {len(missing_data_points) - 5} 个")

def print_instance_details(analysis: Dict, verbose: bool = False):
    """打印实例详细信息"""
    instance_id = analysis['instance_id']
    
    print(f"\n{'='*80}")
    print(f"实例 {instance_id} (通过: {analysis['passed_queries']}/{analysis['total_queries']})")
    print(f"{'='*80}")
    
    if analysis['failed_by_validity'] > 0:
        print(f"  ❌ 有效性失败: {analysis['failed_by_validity']} 个查询")
    if analysis['failed_by_legality'] > 0:
        print(f"  ❌ 合法性失败: {analysis['failed_by_legality']} 个查询")
    
    # 打印每个查询的结果
    for query_detail in analysis['query_details']:
        if not query_detail['passed'] or verbose:
            status = "✅" if query_detail['passed'] else "❌"
            print(f"\n  {status} 查询 {query_detail['query_idx']}:")
            
            if query_detail['failures']:
                for failure in query_detail['failures']:
                    print(f"    • {failure['type']} - {failure['aspect']}:")
                    print(f"      原因: {failure['reason'][:150]}...")
            
            if verbose and query_detail['code']:
                code_preview = query_detail['code'][:200].replace('\n', ' ')
                print(f"    代码预览: {code_preview}...")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='分析评测结果')
    parser.add_argument('--logs', type=str, default='./logs_custom_10',
                        help='日志目录路径 (默认: ./logs_custom_10)')
    parser.add_argument('--instance', type=str, default=None,
                        help='只显示指定实例的详情 (例如: 4)')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细信息，包括通过的查询')
    parser.add_argument('--failed-only', action='store_true',
                        help='只显示失败的实例')
    
    args = parser.parse_args()
    
    logs_dir = Path(args.logs)
    if not logs_dir.exists():
        print(f"❌ 错误: 日志目录不存在: {logs_dir}")
        sys.exit(1)
    
    # 加载结果
    print(f"📂 正在加载结果 from {logs_dir}...")
    results = load_results(logs_dir)
    
    if not results:
        print("❌ 未找到任何评测结果")
        sys.exit(1)
    
    print(f"✓ 加载了 {len(results)} 个实例的结果\n")
    
    # 分析每个实例
    all_analyses = []
    for instance_id, data in sorted(results.items()):
        analysis = analyze_instance(instance_id, data['evaluations'], data['codes'])
        all_analyses.append(analysis)
    
    # 打印摘要
    print_analysis_summary(all_analyses)
    
    # 打印详细信息
    if args.instance:
        # 只显示指定实例
        analysis = next((a for a in all_analyses if a['instance_id'] == args.instance), None)
        if analysis:
            print_instance_details(analysis, verbose=True)
        else:
            print(f"❌ 未找到实例 {args.instance}")
    else:
        # 显示所有实例或只显示失败的
        instances_to_show = all_analyses
        if args.failed_only:
            instances_to_show = [a for a in all_analyses if a['failed_queries'] > 0]
        
        print(f"\n{'='*80}")
        print(f"📋 详细结果 ({len(instances_to_show)} 个实例)")
        print(f"{'='*80}")
        
        for analysis in instances_to_show:
            print_instance_details(analysis, verbose=args.verbose)

if __name__ == "__main__":
    main()
