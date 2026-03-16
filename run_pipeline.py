import subprocess
import argparse
import time
import os
import sys

def run_command(command, description, log_file):
    header = f"\n{'='*25} 开始执行: {description} {'='*25}\n"
    print(header, end="")
    
    # 将开始执行的时间戳写入日志文件
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(header)
        f.write(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"💻 命令: {' '.join(command)}\n\n")

    start_time = time.time()
    
    # 使用 Popen 实现实时回显并写入文件
    try:
        # bufsize=1 表示行缓冲，text=True 表示使用文本模式
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy()
        )

        with open(log_file, "a", encoding="utf-8") as f:
            for line in process.stdout:
                print(line, end="") # 实时显示在终端
                f.write(line)       # 实时写入文件
                f.flush()           # 确保内容立即写入磁盘

        process.wait()
        end_time = time.time()

        if process.returncode != 0:
            error_msg = f"\n❌ 错误: {description} 执行失败，退出代码: {process.returncode}\n"
            print(error_msg)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(error_msg)
            return False

        success_msg = f"\n✅ 完成: {description} | 耗时: {end_time - start_time:.2f}s\n"
        print(success_msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(success_msg)
        return True

    except Exception as e:
        err_msg = f"\n❌ 运行时异常: {str(e)}\n"
        print(err_msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(err_msg)
        return False

def main():
    parser = argparse.ArgumentParser(description="NL2Vis 全流程自动执行 Pipeline (带日志记录版)")
    parser.add_argument("--db_id", type=str, required=True, help="数据库 ID (例如: hospital_1)")
    parser.add_argument("--project_root", type=str, default="/home/wys/data/expriments/NL2Vis", help="项目根目录")
    args = parser.parse_args()

    db_id = args.db_id
    root = args.project_root
    
    # 定义各项路径
    scripts_dir = os.path.join(root, "codes")
    code_folder = os.path.join(root, "temp_results/generated_code/")
    benchmark_path = os.path.join(root, "dataset/")
    
    # 日志目录与文件
    log_dir = os.path.join(root, "logs", f"logs_static_eval_{db_id}")
    os.makedirs(log_dir, exist_ok=True)
    all_log_file = os.path.join(log_dir, "all_log.txt")
    
    # 预检：Stage 01 的输出文件
    schema_file_path = os.path.join(root, "temp_results", "schema", f"{db_id}.json")

    python_bin = sys.executable

    # 初始化日志文件 (如果是新运行，可以选 'w' 清空，这里建议用 'a' 并在头部加分割)
    with open(all_log_file, "a", encoding="utf-8") as f:
        f.write(f"\n\n{'#'*80}\n")
        f.write(f"🚀 NEW PIPELINE RUN | DB_ID: {db_id} | TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'#'*80}\n")

    print(f"\n🚀 启动流水线 | Target DB: {db_id}")
    print(f"📄 日志文件: {all_log_file}")
    
    pipeline_start = time.time()
    pipeline_steps = []

    # ---------------------------------------------------------
    # 逻辑 1: 检查 Schema 是否存在
    # ---------------------------------------------------------
    if os.path.exists(schema_file_path):
        skip_msg = f"⏭️  [跳过] Stage 1: 检测到 Schema 增强文件已存在 ({schema_file_path})\n"
        print(skip_msg)
        with open(all_log_file, "a", encoding="utf-8") as f:
            f.write(skip_msg)
    else:
        pipeline_steps.append({
            "desc": "Stage 1: Schema Discovery",
            "command": [python_bin, os.path.join(scripts_dir, "01 schema_discovery_agent222.py"), "--db_id", db_id]
        })

    # ---------------------------------------------------------
    # 添加后续阶段
    # ---------------------------------------------------------
    pipeline_steps.extend([
        {
            "desc": "Stage 2: SFT Data Generation",
            "command": [python_bin, os.path.join(scripts_dir, "02 generate_sft_data.py"), "--db_id", db_id]
        },
        {
            "desc": "Stage 3: SFT Data Validation",
            "command": [python_bin, os.path.join(scripts_dir, "03 validate_sft_data.py"), "--db_id", db_id]
        },
        {
            "desc": "Stage 4: Python Code Generation",
            "command": [python_bin, os.path.join(scripts_dir, "04 generate_code_from_dvcr.py"), "--db_id", db_id]
        },
        {
            "desc": "Stage 5: Static Code Evaluation",
            "command": [
                python_bin, 
                os.path.join(scripts_dir, "05 evaluate_static_code_py.py"), 
                "--code_folder", code_folder,
                "--benchmark", benchmark_path,
                "--db_id", db_id
            ]
        }
    ])

    # ---------------------------------------------------------
    # 执行
    # ---------------------------------------------------------
    for step in pipeline_steps:
        success = run_command(step["command"], step["desc"], all_log_file)
        if not success:
            fail_msg = f"\n🛑 流水线在 [{step['desc']}] 异常中断。\n"
            print(fail_msg)
            with open(all_log_file, "a", encoding="utf-8") as f:
                f.write(fail_msg)
            sys.exit(1)

    pipeline_total_time = (time.time() - pipeline_start) / 60
    final_msg = f"\n🎉 所有阶段执行成功！\n🎯 Target DB: {db_id}\n⏱️ 总运行时间: {pipeline_total_time:.2f} 分钟\n"
    print(final_msg)
    with open(all_log_file, "a", encoding="utf-8") as f:
        f.write(final_msg)

if __name__ == "__main__":
    main()