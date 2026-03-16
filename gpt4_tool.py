"""
GPT-4 API调用工具模块

这个模块提供了调用OpenAI GPT-4 API的封装工具，包含重试机制和超时处理。
主要用于NL2Dashboard项目中的AI模型调用，支持生成VEGA-DASH配置和分析故事。

主要功能：
1. 自动重试机制 - 处理网络异常和API限制
2. 超时控制 - 防止长时间等待
3. 错误处理 - 优雅处理各种异常情况

作者: NL2Dashboard 研究团队
用途: 支持AI模型调用和响应处理
"""

import time
from functools import wraps
import threading
from openai import OpenAI


def retry(exception_to_check, tries=3, delay=5, backoff=1):
    """
    自动重试装饰器
    
    当函数执行失败时，自动进行重试，支持指数退避策略。
    主要用于处理网络请求、API调用等可能临时失败的操作。
    
    Args:
        exception_to_check: 需要捕获的异常类型
        tries (int): 最大重试次数，默认3次
        delay (int): 初始等待时间（秒），默认5秒
        backoff (int): 退避倍数，每次重试后等待时间会乘以这个值，默认1倍
    
    Returns:
        decorator: 装饰器函数
        
    Example:
        @retry(Exception, tries=3, delay=2, backoff=2)
        def api_call():
            # 这个函数失败时会自动重试3次
            # 等待时间：2秒 -> 4秒 -> 8秒
            pass
    """

    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exception_to_check as e:
                    print(f"{str(e)}, Retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # 返回真正的装饰器

    return deco_retry


def timeout_decorator(timeout):
    """
    超时控制装饰器
    
    为函数执行设置超时时间，如果函数在指定时间内未完成，会抛出超时异常。
    使用多线程实现，不会阻塞主线程。
    
    Args:
        timeout (int): 超时时间（秒）
    
    Returns:
        decorator: 装饰器函数
        
    Example:
        @timeout_decorator(30)  # 30秒超时
        def long_running_function():
            # 如果这个函数执行超过30秒，会抛出超时异常
            pass
    """
    
    class TimeoutException(Exception):
        """自定义超时异常类"""
        pass

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 使用列表存储结果，因为内部函数需要修改这个值
            result = [
                TimeoutException("Function call timed out")
            ]

            def target():
                """在子线程中执行目标函数"""
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            # 创建并启动子线程
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout)  # 等待指定时间
            
            # 如果线程仍在运行，说明超时了
            if thread.is_alive():
                print(f"Function {func.__name__} timed out, retrying...")
                return wrapper(*args, **kwargs)  # 递归重试
            
            # 如果返回的是异常，则抛出
            if isinstance(result[0], Exception):
                raise result[0]
            
            return result[0]

        return wrapper

    return decorator


@timeout_decorator(180)  # 3分钟超时
@retry(Exception, tries=3, delay=5, backoff=1)  # 失败重试3次
def send_chat_request_azure(
        message_text,
        engine="gpt-4o-2024-08-06",  # 默认使用GPT-4o模型
        temp=0.2,
        logit_bias: dict = {},
        max_new_token=4096,
        sample_n=1,
):
    """
    发送聊天请求到OpenAI API (Azure版本)
    
    这是NL2Dashboard项目的核心API调用函数，用于生成VEGA-DASH配置和分析故事。
    集成了超时控制和自动重试机制，确保API调用的稳定性。
    
    Args:
        message_text (list): 聊天消息列表，格式为 [{"role": "user", "content": "..."}]
        engine (str): 使用的模型名称，默认"gpt-4o-2024-08-06"
        temp (float): 温度参数，控制输出的随机性，0.0-1.0，默认0.2
        logit_bias (dict): 对数偏差，用于调整特定token的概率，默认空字典
        max_new_token (int): 最大生成token数，默认4096
        sample_n (int): 生成样本数量，默认1
    
    Returns:#文心千帆代理
        tuple: (第一个响应内容, 所有响应内容列表)
        
    Raises:
        Exception: API调用失败或超时
        
    Example:
        messages = [{"role": "user", "content": "生成一个销售仪表板"}]
        response, all_responses = send_chat_request_azure(messages)
    
    Environment Variables:
        OPENAI_API_KEY: API密钥（必需）
        OPENAI_API_BASE: API基础URL，默认 "https://api.wenwen-ai.com/v1"
    """
    import os
    
    data_res_list = []
    
    # 从环境变量读取配置，如果没有则使用默认值
    api_key = os.getenv("OPENAI_API_KEY", "sk-EWwCihmo7aEgCAKZeVV82P3vdQcy6jBg02JBozZ3Ix7Q2ESu")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.wenwen-ai.com/v1")
    
    # 确保 base_url 以 /v1 结尾
    if not api_base.endswith("/v1"):
        if api_base.endswith("/"):
            api_base = api_base + "v1"
        else:
            api_base = api_base + "/v1"
    
    # 配置OpenAI客户端
    client = OpenAI(api_key=api_key, base_url=api_base)
 

    # 调用OpenAI Chat Completions API
    response = client.chat.completions.create(
        model=engine,                    # 使用的模型
        messages=message_text,           # 输入消息
        temperature=temp,                # 温度参数
        max_tokens=max_new_token,        # 最大token数
        top_p=0.95,                     # 核采样参数
        frequency_penalty=0,            # 频率惩罚
        presence_penalty=0,             # 存在惩罚
        stop=None,                      # 停止词
        n=sample_n,                     # 生成样本数
    )

    # 提取所有响应内容
    for index in range(sample_n):
        data_res = response.choices[index].message.content
        data_res_list.append(data_res)

    # 返回第一个响应和所有响应
    return data_res_list[0], data_res_list
