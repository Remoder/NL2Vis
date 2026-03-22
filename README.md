## 任务背景
- NL2Vis 流程需要将自然语言可视化需求（VisEval 数据集中的 `nl_queries`）转化为可靠的 DVCR 规划、可执行 SQL，以及最终的 Python 可视化代码。
- 项目以 `/dataset` 下的 146 个 CSV 数据库与 `visEval.json` 为基准，产出过程中的所有中间物存放在 `/temp_results`，方便追踪与复用。
- 通过 `run_pipeline.py --db_id <database>` 可自动串联 5 个阶段脚本，生成日志、错误报告并复盘端到端效果。

## 整体思路
1. **Schema 认知**：`01 schema_discovery_agent222.py` 调用 Topology/Profiling Agent，在 LLM+数据探查的混合策略下生成结构化关系与语义标签，为后续模板提供上下文。
2. **Teacher-Forcing 数据生成**：`02 generate_sft_data.py` 逐题读取 VisEval 样本，结合 SQL Ground Truth 与增强 Schema，使用 `dvcr_protocol.py` 构造提示，批量生成 DVCR SFT 对。
3. **双重校验过滤**：`03 validate_sft_data.py` 解析 DVCR，先校验 `VIS_CONFIG` 与金标是否一致，再借助 DuckDB + SQL 重写器执行、对齐数值，生成 `sft_advised_data.jsonl`。
4. **Python 代码落地**：`04 generate_code_from_dvcr.py` 读取校验后的 DVCR，注入 Schema、关系与 CSV 文件路径，先做 DVCR 静态检查，再调用 Coder 生成 matplotlib 代码。
5. **静态评测与报告**：`05 evaluate_static_code_py.py` 用 Viseval 官方 Evaluator 对所有 NL 查询进行静态执行，导出 SVG、打分，并通过 `analyze_results.py` 生成错误分析。

## 分阶段方案
| 阶段 | 入口脚本 | 输入 / 输出 | 关键要点 |
| --- | --- | --- | --- |
| Stage 01 – Schema Discovery | `codes/01 schema_discovery_agent222.py` | 输入：`dataset/databases/<db_id>`；输出：`temp_results/schema/<db_id>.json` | `TopologyAgent` 识别主外键及角色，`ProfilingAgent` 注入列统计与可视化建议，可复用结果以跳过重复探查。 |
| Stage 02 – SFT Data Generation | `codes/02 generate_sft_data.py` | 输入：VisEval 样本 + Stage 01 Schema；输出：`temp_results/sft_data/sft_origin_data.jsonl` | `SchemaAnalyzer` 合并轻量/增强 Schema，`build_sql2dvcr_prompt` 固化 DVCR 协议，确保每条 `nl_query` 对应 Teacher-Forcing 示例。 |
| Stage 03 – SFT Validation | `codes/03 validate_sft_data.py` | 输入：`sft_origin_data.jsonl`；输出：`sft_advised_data.jsonl` | 解析 DVCR 四段式文本，`validate_vis_config` 匹配图表语义，DuckDB 执行自愈 SQL（ANY_VALUE 包裹、引号修正），并对比金标 `vis_obj`。 |
| Stage 04 – Code Generation | `codes/04 generate_code_from_dvcr.py` | 输入：校验后的 DVCR + CSV；输出：`temp_results/generated_code/<db_id>/*.txt` | `DVCRValidator` 先做静态规则检查，`Coder` 基于 DVCR + Schema 拼接数据载入 + matplotlib 绘图代码，并按 Query/paraphrase 编号落盘。 |
| Stage 05 – Static Evaluation | `codes/05 evaluate_static_code_py.py` | 输入：Stage 04 代码 + VisEval 数据；输出：`logs/logs_static_eval_<db_id>`（含 `result.json`、`error_analysis_report.txt`） | `StaticCodeAgent` 将现有代码映射到 NL 查询，Evaluator 统一执行、生成 SVG，附带错误统计与 SQL 校验失败次数。 |
| Pipeline Orchestration | `codes/run_pipeline.py` | 参数：`--db_id`、可选 `--project_root` | 自动串联各阶段，带时间戳日志，检测 Schema 是否已存在从而跳过 Stage 01，并将输出写入 `logs/.../all_log.txt`。 |

## 核心亮点
- **可复用 Schema 缓存**：`run_pipeline.py` 会检测 `temp_results/schema/<db_id>.json`，存在时直接跳过 Stage 01，显著节省 Token 与时间。
- **两层安全校验**：Stage 03 在 `validate_vis_config` 通过后，才使用 `build_sr2sql_check_prompt` 生成 SQL，并在 DuckDB 中做 ANY_VALUE 自动修补 + 结果对齐，减少伪阳性样本。
- **DVCR → 代码的抗噪设计**：Stage 04 的 `DVCRValidator` 会在代码生成前指出格式 & 语义异常，即使告警也允许继续但会在日志中点亮问题，方便回溯。
- **端到端可解释评测**：Stage 05 不仅给出 pass rate，还写入 `error_analysis_report.txt`，包含 SQL 校验失败计数、逐题失败原因，配合 `logs/.../all_log.txt` 形成闭环。

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
