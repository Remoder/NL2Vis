import argparse
import json
import os
from collections import defaultdict


STAGE_FILES = [
    "stage02_generate_sft.jsonl",
    "stage03_validate_sft.jsonl",
    "stage04_generate_code.jsonl",
]


def _safe_int(v):
    try:
        return int(v)
    except Exception:
        return 0


def _load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _fmt_int(n):
    return f"{n:,}"


def _fmt_float(v):
    return f"{v:.2f}"


def _markdown_table(headers, rows):
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [line1, line2]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def infer_total_queries_from_viseval(vis_eval_path, db_id=None):
    if not vis_eval_path or not os.path.exists(vis_eval_path):
        return None
    try:
        with open(vis_eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    total = 0
    for item in data.values():
        cur_db = item.get("db_id")
        if db_id and str(cur_db).lower() != db_id.lower():
            continue
        nl_queries = item.get("nl_queries", [])
        if isinstance(nl_queries, list):
            total += len(nl_queries)
    return total


def summarize(records, db_id=None, target_queries=None, buffer_ratio=0.3):
    if db_id:
        records = [r for r in records if str(r.get("db_id", "")).lower() == db_id.lower()]

    if not records:
        return {
            "records": [],
            "stage_rows": [],
            "db_rows": [],
            "overall": {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "avg_tokens_per_call": 0.0,
                "stage2_unique_qids": 0,
                "avg_tokens_per_query_base_stage2": 0.0,
            },
            "forecast": None,
        }

    by_stage = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "total": 0})
    by_db = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "total": 0})
    overall = {"calls": 0, "prompt": 0, "completion": 0, "total": 0}

    stage2_qids_by_db = defaultdict(set)
    stage2_qids_global = set()

    for r in records:
        stage = str(r.get("stage", "unknown"))
        db = str(r.get("db_id", "UNKNOWN"))
        q_id = r.get("q_id")
        prompt_t = _safe_int(r.get("prompt_tokens", 0))
        completion_t = _safe_int(r.get("completion_tokens", 0))
        total_t = _safe_int(r.get("total_tokens", prompt_t + completion_t))

        by_stage[stage]["calls"] += 1
        by_stage[stage]["prompt"] += prompt_t
        by_stage[stage]["completion"] += completion_t
        by_stage[stage]["total"] += total_t

        by_db[db]["calls"] += 1
        by_db[db]["prompt"] += prompt_t
        by_db[db]["completion"] += completion_t
        by_db[db]["total"] += total_t

        overall["calls"] += 1
        overall["prompt"] += prompt_t
        overall["completion"] += completion_t
        overall["total"] += total_t

        if stage == "stage02_generate_sft" and q_id is not None:
            stage2_qids_global.add(str(q_id))
            stage2_qids_by_db[db].add(str(q_id))

    stage_rows = []
    for stage in sorted(by_stage.keys()):
        stat = by_stage[stage]
        calls = stat["calls"]
        avg = stat["total"] / calls if calls else 0.0
        stage_rows.append(
            {
                "stage": stage,
                "calls": calls,
                "prompt_tokens": stat["prompt"],
                "completion_tokens": stat["completion"],
                "total_tokens": stat["total"],
                "avg_tokens_per_call": avg,
            }
        )

    db_rows = []
    for db in sorted(by_db.keys()):
        stat = by_db[db]
        calls = stat["calls"]
        avg = stat["total"] / calls if calls else 0.0
        stage2_qids = len(stage2_qids_by_db[db])
        avg_per_query = (stat["total"] / stage2_qids) if stage2_qids else 0.0
        db_rows.append(
            {
                "db_id": db,
                "calls": calls,
                "prompt_tokens": stat["prompt"],
                "completion_tokens": stat["completion"],
                "total_tokens": stat["total"],
                "avg_tokens_per_call": avg,
                "stage2_unique_qids": stage2_qids,
                "avg_tokens_per_query_base_stage2": avg_per_query,
            }
        )

    overall_calls = overall["calls"]
    overall_avg = overall["total"] / overall_calls if overall_calls else 0.0
    stage2_unique_qids = len(stage2_qids_global)
    overall_avg_per_query = overall["total"] / stage2_unique_qids if stage2_unique_qids else 0.0

    forecast = None
    if target_queries is not None and stage2_unique_qids > 0:
        stage_forecast_rows = []
        for row in stage_rows:
            per_query = row["total_tokens"] / stage2_unique_qids
            base_total = per_query * target_queries
            buffered_total = base_total * (1.0 + buffer_ratio)
            stage_forecast_rows.append(
                {
                    "stage": row["stage"],
                    "avg_tokens_per_query": per_query,
                    "forecast_total_tokens": base_total,
                    "forecast_total_tokens_with_buffer": buffered_total,
                }
            )
        overall_base = overall_avg_per_query * target_queries
        overall_buffered = overall_base * (1.0 + buffer_ratio)
        forecast = {
            "target_queries": int(target_queries),
            "buffer_ratio": float(buffer_ratio),
            "overall_avg_tokens_per_query": overall_avg_per_query,
            "forecast_total_tokens": overall_base,
            "forecast_total_tokens_with_buffer": overall_buffered,
            "stage_rows": stage_forecast_rows,
        }

    return {
        "records": records,
        "stage_rows": stage_rows,
        "db_rows": db_rows,
        "overall": {
            "calls": overall_calls,
            "prompt_tokens": overall["prompt"],
            "completion_tokens": overall["completion"],
            "total_tokens": overall["total"],
            "avg_tokens_per_call": overall_avg,
            "stage2_unique_qids": stage2_unique_qids,
            "avg_tokens_per_query_base_stage2": overall_avg_per_query,
        },
        "forecast": forecast,
    }


def build_report_md(summary, db_id=None):
    title = "Token Usage Summary"
    scope = db_id if db_id else "ALL"
    overall = summary["overall"]

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Scope: `{scope}`")
    lines.append(f"- Records: `{_fmt_int(len(summary['records']))}`")
    lines.append("")

    overall_headers = [
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "avg_tokens_per_call",
        "stage2_unique_qids",
        "avg_tokens_per_query(base_stage2)",
    ]
    overall_rows = [[
        _fmt_int(overall["calls"]),
        _fmt_int(overall["prompt_tokens"]),
        _fmt_int(overall["completion_tokens"]),
        _fmt_int(overall["total_tokens"]),
        _fmt_float(overall["avg_tokens_per_call"]),
        _fmt_int(overall["stage2_unique_qids"]),
        _fmt_float(overall["avg_tokens_per_query_base_stage2"]),
    ]]
    lines.append("## Overall")
    lines.append(_markdown_table(overall_headers, overall_rows))
    lines.append("")

    stage_headers = [
        "stage",
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "avg_tokens_per_call",
    ]
    stage_rows = []
    for row in summary["stage_rows"]:
        stage_rows.append([
            row["stage"],
            _fmt_int(row["calls"]),
            _fmt_int(row["prompt_tokens"]),
            _fmt_int(row["completion_tokens"]),
            _fmt_int(row["total_tokens"]),
            _fmt_float(row["avg_tokens_per_call"]),
        ])
    lines.append("## By Stage")
    lines.append(_markdown_table(stage_headers, stage_rows if stage_rows else [["(empty)", "0", "0", "0", "0", "0.00"]]))
    lines.append("")

    db_headers = [
        "db_id",
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "avg_tokens_per_call",
        "stage2_unique_qids",
        "avg_tokens_per_query(base_stage2)",
    ]
    db_rows = []
    for row in summary["db_rows"]:
        db_rows.append([
            row["db_id"],
            _fmt_int(row["calls"]),
            _fmt_int(row["prompt_tokens"]),
            _fmt_int(row["completion_tokens"]),
            _fmt_int(row["total_tokens"]),
            _fmt_float(row["avg_tokens_per_call"]),
            _fmt_int(row["stage2_unique_qids"]),
            _fmt_float(row["avg_tokens_per_query_base_stage2"]),
        ])
    lines.append("## By DB")
    lines.append(_markdown_table(db_headers, db_rows if db_rows else [["(empty)", "0", "0", "0", "0", "0.00", "0", "0.00"]]))
    lines.append("")

    forecast = summary.get("forecast")
    if forecast:
        lines.append("## Forecast")
        forecast_headers = [
            "target_queries",
            "buffer_ratio",
            "overall_avg_tokens_per_query",
            "forecast_total_tokens",
            "forecast_total_tokens_with_buffer",
        ]
        forecast_rows = [[
            _fmt_int(forecast["target_queries"]),
            _fmt_float(forecast["buffer_ratio"]),
            _fmt_float(forecast["overall_avg_tokens_per_query"]),
            _fmt_int(int(forecast["forecast_total_tokens"])),
            _fmt_int(int(forecast["forecast_total_tokens_with_buffer"])),
        ]]
        lines.append(_markdown_table(forecast_headers, forecast_rows))
        lines.append("")

        stage_forecast_headers = [
            "stage",
            "avg_tokens_per_query",
            "forecast_total_tokens",
            "forecast_total_tokens_with_buffer",
        ]
        stage_forecast_rows = []
        for row in forecast.get("stage_rows", []):
            stage_forecast_rows.append([
                row["stage"],
                _fmt_float(row["avg_tokens_per_query"]),
                _fmt_int(int(row["forecast_total_tokens"])),
                _fmt_int(int(row["forecast_total_tokens_with_buffer"])),
            ])
        lines.append("### Forecast By Stage")
        lines.append(_markdown_table(stage_forecast_headers, stage_forecast_rows if stage_forecast_rows else [["(empty)", "0.00", "0", "0"]]))
        lines.append("")

    lines.append("## Notes")
    lines.append("- `avg_tokens_per_query(base_stage2)` 的查询基数来自 Stage02 的唯一 `q_id`。")
    lines.append("- 若只跑了 Stage03/Stage04 而没有对应 Stage02，按 query 均值会偏高或不可用。")
    if forecast:
        lines.append("- Forecast 使用 `avg_tokens_per_query(base_stage2) * target_queries`，并提供 `buffer_ratio` 后的上限。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize token usage for NL2Vis pipeline")
    parser.add_argument("--db_id", type=str, default=None, help="仅统计指定数据库")
    parser.add_argument("--token_dir", type=str, default=None, help="token 日志目录，默认 project_root/logs/token_usage")
    parser.add_argument("--target_queries", type=int, default=None, help="预算推断目标查询数（不传则尝试从 visEval 自动推断）")
    parser.add_argument("--buffer_ratio", type=float, default=0.3, help="预算缓冲比例，默认 0.3（即 +30%）")
    parser.add_argument("--vis_eval_path", type=str, default=None, help="visEval.json 路径，用于自动推断 target_queries")
    parser.add_argument("--no_auto_target", action="store_true", help="关闭 visEval 自动推断 target_queries")
    parser.add_argument("--save_md", action="store_true", help="将结果写入 summary_token_usage.md")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    token_dir = args.token_dir or os.path.join(project_root, "logs", "token_usage")
    vis_eval_path = args.vis_eval_path or os.path.join(project_root, "dataset", "visEval.json")

    records = []
    for fn in STAGE_FILES:
        fp = os.path.join(token_dir, fn)
        records.extend(_load_jsonl(fp))

    target_queries = args.target_queries
    if target_queries is None and not args.no_auto_target:
        target_queries = infer_total_queries_from_viseval(vis_eval_path, db_id=args.db_id)

    summary = summarize(
        records,
        db_id=args.db_id,
        target_queries=target_queries,
        buffer_ratio=args.buffer_ratio,
    )
    report_md = build_report_md(summary, db_id=args.db_id)
    print(report_md)

    if args.save_md:
        out_path = os.path.join(token_dir, "summary_token_usage.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
