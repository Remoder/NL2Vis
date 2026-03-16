# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import base64
import json
import logging
import os
from typing import Union

import pandas as pd
from attr import dataclass

# 延迟导入 cairosvg，只在需要时导入
try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except OSError:
    # Cairo 系统库缺失时，延迟导入
    CAIROSVG_AVAILABLE = False
    cairosvg = None

from .check import (
    chart_check,
    data_check,
    deconstruct,
    layout_check,
    order_check,
    readability_check,
    surface_form_check as check_surface_form_check,
    scale_and_ticks_check,
)
from .dataset import Dataset


@dataclass
class CheckResult:
    answer: Union[bool, int]
    aspect: str
    rationale: str

    def get_json(self):
        return {
            "answer": self.answer,
            "aspect": self.aspect,
            "rationale": self.rationale,
        }


@dataclass
class EvaluationDetail:
    id: str
    results: list[list[CheckResult]]


VALID_ASPECTS = ["code execution", "surface-form check"]
LEGAL_ASPECTS = ["deconstruction", "chart type check", "data check", "order check"]
READABILITY_ASPECT = ["layout check", "scale and ticks check", "readability check"]

FAIL_ASPECTS = VALID_ASPECTS + LEGAL_ASPECTS + ["layout check", "scale and ticks check"]


class EvaluationResult:
    dataset: Dataset
    details: list[EvaluationDetail]

    def __init__(self, dataset: Dataset, details: list[EvaluationDetail]):
        self.dataset = dataset
        self.details = details

    def detail_records(self) -> pd.DataFrame:
        records = []
        for detail in self.details:
            id = detail.id
            instance_results = detail.results
            count = len(instance_results)
            record = {
                "id": id,
                "chart": self.dataset.dict[id]["chart"],
                "hardness": self.dataset.dict[id]["hardness"],
            }

            # fail rate
            for aspect in FAIL_ASPECTS:
                evaluate_result = [
                    (
                        all(
                            [
                                item.answer
                                for item in query_results
                                if item.aspect == aspect
                            ]
                        )
                    )
                    for query_results in instance_results
                ]
                fail_result = [item for item in evaluate_result if not item]
                record[f"{aspect}_fail_rate"] = len(fail_result) / count

            high_level_dimensions = [
                ["invalid_rate", VALID_ASPECTS],
                ["illegal rate", LEGAL_ASPECTS],
            ]
            pass_count = count
            for dimension in high_level_dimensions:
                evaluate_result = [
                    (
                        all(
                            [
                                item.answer
                                for item in query_results
                                if (item.aspect in dimension[1])
                            ]
                        )
                    )
                    for query_results in instance_results
                ]
                false_count = len([item for item in evaluate_result if not item])
                record[dimension[0]] = false_count / count
                pass_count -= false_count

            # pass rate
            record["pass_rate"] = pass_count / count

            # readability score
            evaluate_result = [
                (
                    sum(
                        [
                            item.answer
                            for item in query_results
                            if item.aspect == "readability check"
                        ]
                    )
                )
                for query_results in instance_results
            ]
            # 检查是否有 readability check 的结果
            readability_count = sum([1 for r in evaluate_result if r > 0])
            if readability_count > 0:
                record["readability_score"] = sum(evaluate_result) / readability_count
            else:
                record["readability_score"] = None  # 或者使用 0，如果没有 vision model

            record["quality_score"] = sum(evaluate_result) / count if count > 0 else 0

            # 只追加一次 record，确保所有字段都已设置
            records.append(record)

        return pd.DataFrame(records)

    def score(self):
        records = self.detail_records()
        metrics = [
            "invalid_rate",
            "illegal rate",
            "pass_rate",
            "readability_score",
            "quality_score",
        ]
        score = {}
        for metric in metrics:
            if metric in records.columns:
                # 对于 readability_score，可能需要过滤 None 值
                if metric == "readability_score":
                    non_null_values = records[metric].dropna()
                    score[metric] = non_null_values.mean() if len(non_null_values) > 0 else None
                else:
                    score[metric] = records[metric].mean()
            else:
                score[metric] = None

        for key in records.keys():
            if (
                key not in metrics
                and key != "id"
                and key != "chart"
                and key != "hardness"
            ):
                score[key] = records[key].mean()

        return score


def convert_svg_to_base64(svg_string):
    if not CAIROSVG_AVAILABLE or cairosvg is None:
        # 如果 cairosvg 不可用，返回 SVG 的 base64 编码
        base64_encoded = base64.b64encode(svg_string.encode("utf-8")).decode("utf-8")
        return f"data:image/svg+xml;base64,{base64_encoded}"
    png_string = cairosvg.svg2png(bytestring=svg_string)
    base64_encoded = base64.b64encode(png_string).decode("utf-8")
    return f"data:image/png;base64,{base64_encoded}"


class Evaluator:
    def __init__(self, webdriver_path=None, vision_model=None):
        self.webdriver_path = webdriver_path
        self.vision_model = vision_model

    def evaluate(self, agent, dataset, config):
        use_logs = False
        evaluation_details = []
        if "logs" in config:
            log_folder = config["logs"]
            isExists = os.path.exists(log_folder)
            try:
                if not isExists:
                    os.makedirs(log_folder)
                # 配置 logging，但不影响 stdout 的 print 输出
                import sys
                file_handler = logging.FileHandler(log_folder / "evaluation.log", mode="a")
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
                
                # 只添加文件 handler，不影响 stdout
                root_logger = logging.getLogger()
                root_logger.setLevel(logging.INFO)
                # 清除现有的 handlers，避免重复
                root_logger.handlers = []
                root_logger.addHandler(file_handler)
                
                use_logs = True
            except Exception as e:
                print(e)

        instance_count = 0
        for instance in dataset.benchmark:
            instance_count += 1
            codes = []
            instance_results = []
            nl_queries = instance["nl_queries"]
            tables = instance["tables"]
            
            # 打印进度信息
            print(f"\n[{instance_count}] {instance['id']} ({len(nl_queries)} queries)")

            if use_logs:
                instanceFolder = log_folder / instance["id"]
                path = instanceFolder / "result.json"
                # 检查是否有强制重新运行标志
                force_rerun = config.get("force_rerun", False)
                
                print(f"  Cache: {'HIT' if os.path.exists(path) and not force_rerun else 'MISS'}")
                
                if os.path.exists(path) and not force_rerun:
                    with open(path, "r") as f:
                        data = json.load(f)
                        if "codes" in data and "evaluations" in data:
                            print(f"\n{'='*60}")
                            print(f"⚡ [CACHE HIT] Skipping execution for instance {instance['id']}")
                            print(f"{'='*60}")
                            print(f"📁 Cache file: {path}")
                            print(f"📊 Cached queries: {len(data.get('evaluations', []))}")
                            print(f"💡 To see actual LLM calls and execution, use: --force")
                            print(f"{'='*60}\n")
                            instance_results = []
                            for query_result in data["evaluations"]:
                                results = [
                                    CheckResult(
                                        answer=result["answer"],
                                        aspect=result["aspect"],
                                        rationale=result["rationale"],
                                    )
                                    for result in query_result
                                ]
                                instance_results.append(results)
                            evaluation_details.append(
                                EvaluationDetail(instance["id"], instance_results)
                            )
                            continue
                elif os.path.exists(path) and force_rerun:
                    print(f"  Force rerun: regenerating...")
                    # 确保 codes 列表是空的，强制重新生成
                    codes = []
                else:
                    logging.info(f"Instance ({instance['id']}) evaluation began.")
                    isExists = os.path.exists(instanceFolder)
                    if not isExists:
                        os.makedirs(instanceFolder)

            for index in range(len(nl_queries)):
                nl_query = nl_queries[index]
                print(f"  [{instance_count}.{index+1}] {nl_query[:50]}{'...' if len(nl_query) > 50 else ''}")
                if index < len(codes):
                    print(f"    Using cached code")
                    code = codes[index]
                    context = {"tables": tables}
                else:
                    code, context = agent.generate(nl_query, tables, config)
                    if code:
                        print(f"    Generated code ({len(code)} chars)")
                        codes.append(code)
                    else:
                        print(f"    Code generation failed")
                        codes.append(None)
                if code is None:
                    results = [
                        CheckResult(
                            answer=False,
                            aspect="generation",
                            rationale="Code generation failed.",
                        )
                    ]
                else:
                    context["library"] = config["library"]
                    if use_logs:
                        results = self.validity_check(
                            code, context, agent, instanceFolder / f"{index}.svg"
                        )
                    else:
                        results = self.validity_check(code, context, agent)

                    pass_validity = all([result.answer for result in results])
                    if pass_validity:
                        ground_truth = {
                            "chart": instance["chart"],
                            "vis_obj": instance["vis_obj"],
                            "meta_info": instance["query_meta"][index],
                        }
                        results += self.legality_check(context, ground_truth)

                    pass_legality = all([result.answer for result in results])
                    if pass_legality:
                        results += self.readability_evaluate(context, nl_query)

                instance_results.append(results)

            evaluation_details.append(
                EvaluationDetail(instance["id"], instance_results)
            )
            if use_logs:
                logging.info(f"Instance ({instance['id']}) evaluation finished.")
                # convert CheckResult to json
                instance_results = [
                    [result.get_json() for result in results]
                    for results in instance_results
                ]
                with open(log_folder / (instance["id"] + "/result.json"), "w") as f:
                    f.write(
                        json.dumps({"codes": codes, "evaluations": instance_results})
                    )
                    f.close()
        return EvaluationResult(dataset, evaluation_details)

    def execute(self, code, context, agent, log_name=None) -> CheckResult:
        result = agent.execute(code, context, log_name)
        if result.status is False:
            return CheckResult(
                answer=False, aspect="code execution", rationale=result.error_msg
            )

        context["svg_string"] = result.svg_string
        return CheckResult(
            answer=True,
            aspect="code execution",
            rationale="Code executed successfully.",
        )

    def surface_form_check(self, context) -> CheckResult:
        svg_string = context["svg_string"]
        answer, rationale = check_surface_form_check(svg_string)
        return CheckResult(
            answer=answer,
            aspect="surface-form check",
            rationale=rationale,
        )

    def validity_check(self, code, context, agent, log_name=None) -> list[CheckResult]:
        results = []
        result = self.execute(code, context, agent, log_name)
        results.append(result)
        if result.answer:
            result = self.surface_form_check(context)
            results.append(result)

        return results

    def deconstruction(self, context) -> CheckResult:
        svg_string = context["svg_string"]
        library = context["library"]
        if library == "seaborn":
            library = "matplotlib"
        try:
            chart_info, msg = deconstruct(svg_string, library)
            if chart_info is None:
                return CheckResult(
                    answer=False,
                    aspect="deconstruction",
                    rationale=msg,
                )
            context.update(chart_info)
            return CheckResult(
                answer=True,
                aspect="deconstruction",
                rationale="Deconstructed the chart successfully.",
            )
        except:
            return CheckResult(
                answer=False,
                aspect="deconstruction",
                rationale="Cannot parse the visualization.",
            )

    def chart_type_check(self, context, ground_truth) -> CheckResult:
        answer, rationale = chart_check(
            context,
            ground_truth["chart"],
            (
                ground_truth["meta_info"]["stacked_bar"]
                if "stacked_bar" in ground_truth["meta_info"]
                else None
            ),
        )
        return CheckResult(
            answer=answer,
            aspect="chart type check",
            rationale=rationale,
        )

    def data_check(self, context, ground_truth) -> CheckResult:
        answer, rationale = data_check(
            context,
            ground_truth["vis_obj"],
            ground_truth["meta_info"]["channel_specified"],
        )
        return CheckResult(
            answer=answer,
            aspect="data check",
            rationale=rationale,
        )

    def order_check(self, context, ground_truth) -> CheckResult:
        answer, rationale = order_check(
            context,
            ground_truth["vis_obj"],
            (
                ground_truth["meta_info"]["sort_by"]
                if "sort_by" in ground_truth["meta_info"]
                else None
            ),
        )
        return CheckResult(
            answer=answer,
            aspect="order check",
            rationale=rationale,
        )

    def legality_check(self, context, ground_truth) -> list[CheckResult]:
        results = []
        result = self.deconstruction(context)
        results.append(result)
        if result.answer:
            chart_type_check_result = self.chart_type_check(context, ground_truth)
            data_check_result = self.data_check(context, ground_truth)
            results.append(chart_type_check_result)
            results.append(data_check_result)
            if data_check_result.answer and ground_truth["vis_obj"]["sort"] is not None:
                self.order_check(context, ground_truth)
                results.append(self.order_check(context, ground_truth))

        return results

    def layout_check(self, context) -> CheckResult:
        assert "svg_string" in context
        assert self.webdriver_path is not None

        answer, rationale = layout_check(context, self.webdriver_path)
        return CheckResult(
            answer=answer,
            aspect="layout check",
            rationale=rationale,
        )

    def scale_and_ticks_check(self, context, query) -> CheckResult:
        assert "base64" in context and "encoding" in context and "chart" in context
        assert self.vision_model is not None

        answer, rationale = scale_and_ticks_check(context, query, self.vision_model)
        return CheckResult(
            answer=answer,
            aspect="scale and ticks check",
            rationale=rationale,
        )

    def readability_evaluate(self, context, query: str) -> list[CheckResult]:
        results = []
        if self.webdriver_path:
            layout_result = self.layout_check(context)
            if layout_result.answer is not None:
                results.append(layout_result)

        if self.vision_model:
            context["base64"] = convert_svg_to_base64(context["svg_string"])
            scale_and_ticks_result = self.scale_and_ticks_check(context, query)
            if scale_and_ticks_result.answer is not None:
                results.append(scale_and_ticks_result)

            aspect_format = {
                "layout check": "Overflow/Overlap",
                "scale and ticks check": "Scale/Ticks",
            }
            reviews = [
                {
                    "aspect": aspect_format[result.aspect],
                    "content": result.rationale,
                }
                for result in results
            ]
            context["reviews"] = reviews

            answer, rationale = readability_check(context, query, self.vision_model)
            if answer is not None:
                readability_result = CheckResult(
                    answer=answer,
                    aspect="readability check",
                    rationale=rationale,
                )
                results.append(readability_result)

        return results
