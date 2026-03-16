## 结构说明
代码结构如下：
```text
.
|-- codes/                   # 核心流水线脚本 (01-05)
|-- dataset/                 # 原始数据集
|   |-- databases/           # 146 个数据库的 CSV 文件
|   |-- visEval.json         # 原始任务定义
|-- logs/                    # 运行日志与报告 (按 db_id 隔离)
|   |-- logs_static_eval_xxx/
|   |   |-- result.json      # 单题评测详情
|   |   |-- all_log.txt      # 完整流水线执行日志 (run_pipeline.py产物)
|   |   |-- error_analysis_report.txt  # 汇总统计与错误分析报告
|-- temp_results/            # 中间产物
    |-- schema/              # Stage 01: 增强版 JSON Schema (可复用)
    |-- sft_data/            # Stage 02/03: 经逻辑校验的 SFT 训练对 (.jsonl)
    |-- generated_code/      # Stage 04: 最终可执行的 Python 绘图代码
```

## 运行说明
运行完整流水线（涵盖 01 探索至 05 最终评测）：
```bash
python run_pipeline.py --db_id [database_id]
```
注：脚本会自动检测 temp_results/schema 下是否有现成的模式文件，若有则自动跳过 Stage 01 以节省 Token。
