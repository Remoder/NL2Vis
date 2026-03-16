# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import warnings
import traceback
import re
import json
import glob
import os
import pandas as pd
import matplotlib
# force 'Agg' backend BEFORE importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from langchain_core.language_models.chat_models import SimpleChatModel
except ImportError:
    try:
        from langchain.chat_models.base import SimpleChatModel
    except ImportError:
        # 如果都不行，尝试从其他地方导入
        from langchain_core.language_models import SimpleChatModel

try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
except ImportError:
    from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing import List, Optional, Dict
try:
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
except ImportError:
    from langchain.callbacks.manager import CallbackManagerForLLMRun

from viseval.agent import Agent, ChartExecutionResult

# 支持相对导入和绝对导入
try:
    from .utils import show_svg
except ImportError:
    from utils import show_svg

# 导入自定义的 GPT-4 工具
# 添加 Data_ana 目录到路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_data_ana_dir = os.path.join(os.path.dirname(os.path.dirname(_current_dir)), 'Data_ana')
if _data_ana_dir not in sys.path:
    sys.path.insert(0, _data_ana_dir)

try:
    from gpt4_tool import send_chat_request_azure
except ImportError as e:
    warnings.warn(f"Failed to import gpt4_tool: {e}. Please ensure gpt4_tool.py is in Data_ana directory.")
    raise


# ==========================================
# LLM 适配器：将自定义的 GPT-4 调用封装成 LangChain SimpleChatModel
# ==========================================
class CustomLLMAdapter(SimpleChatModel):
    """适配器类，将 send_chat_request_azure 封装成 LangChain SimpleChatModel"""
    
    engine: str = "gpt-4o-2024-08-06"
    temp: float = 0.1
    
    def __init__(self, engine="gpt-4o-2024-08-06", temp=0.1, **kwargs):
        super().__init__(engine=engine, temp=temp, **kwargs)
    
    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        # 将 LangChain 消息格式转换为 API 格式
        api_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                api_messages.append({"role": "assistant", "content": msg.content})
        
        import time
        api_start = time.time()
        # 调用自定义的 API
        response_text, _ = send_chat_request_azure(api_messages, engine=self.engine, temp=self.temp)
        api_elapsed = time.time() - api_start
        
        return response_text
    
    @property
    def _llm_type(self) -> str:
        return "custom_azure_gpt4"


# ==========================================
# Schema Enhancer (保持原有逻辑)
# ==========================================
class SchemaEnhancer:
    def __init__(self, dfs: dict):
        self.dfs = dfs

    def get_semantic_schema(self):
        schema_info = []
        join_keys = {}  # 记录可能的连接键
        
        for name, df in self.dfs.items():
            table_info = f"Table Variable: {name}\n"
            table_info += f"  Rows: {len(df)}\n"
            table_info += f"Columns:\n"
            
            # 识别可能的连接键
            potential_join_keys = []
            for col in df.columns:
                dtype = df[col].dtype
                semantic_type = str(dtype)
                is_id_key = False
                
                # 更准确地识别 ID/Key 字段
                col_lower = col.lower()
                if ('id' in col_lower or 'key' in col_lower or 'code' in col_lower) or \
                   (pd.api.types.is_integer_dtype(dtype) and df[col].nunique() > 0.8 * len(df) and len(df) > 0):
                    semantic_type = "ID/Key (Potential Join Key)"
                    is_id_key = True
                    potential_join_keys.append(col)
                elif 'date' in col_lower or 'year' in col_lower or 'time' in col_lower:
                    semantic_type = "Time/Date"
                elif pd.api.types.is_numeric_dtype(dtype):
                    semantic_type = f"Numeric ({dtype})"
                elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                    semantic_type = "Text/String"
                
                # 提供更详细的值信息
                if df[col].nunique() <= 20 and len(df) > 0:
                    # 如果唯一值不多，显示所有值
                    unique_vals = df[col].dropna().unique()[:20]
                    if len(unique_vals) <= 10:
                        val_str = f"All Values: {unique_vals.tolist()}"
                    else:
                        val_str = f"Sample Values: {unique_vals[:10].tolist()} ... (共 {df[col].nunique()} 个唯一值)"
                else:
                    # 显示样本值和统计信息
                    sample_vals = df[col].dropna().astype(str).head(5).tolist()
                    unique_count = df[col].nunique()
                    val_str = f"Sample: {sample_vals} (共 {unique_count} 个唯一值)"
                
                # 对于 ID/Key 字段，显示更多样本
                if is_id_key:
                    id_samples = df[col].dropna().head(10).tolist()
                    val_str += f"\n    ID Samples: {id_samples[:10]}"
                
                table_info += f"  - {col} (Type: {semantic_type}): {val_str}\n"
            
            if potential_join_keys:
                table_info += f"  Potential Join Keys: {potential_join_keys}\n"
            
            join_keys[name] = potential_join_keys
            schema_info.append(table_info)
        
        # 分析表之间的潜在连接关系
        if len(self.dfs) > 1:
            schema_info.append("\n=== TABLE JOIN ANALYSIS ===\n")
            table_names = list(self.dfs.keys())
            for i, table1 in enumerate(table_names):
                for table2 in table_names[i+1:]:
                    keys1 = join_keys.get(table1, [])
                    keys2 = join_keys.get(table2, [])
                    
                    # 查找可能的连接键（名称相似或值匹配）
                    for key1 in keys1:
                        for key2 in keys2:
                            # 检查键名相似度
                            if key1.lower() == key2.lower() or \
                               (key1.lower() in key2.lower() or key2.lower() in key1.lower()):
                                schema_info.append(
                                    f"  Possible Join: {table1}.{key1} <-> {table2}.{key2}\n"
                                )
                                # 显示值的交集以验证
                                val1_set = set(self.dfs[table1][key1].dropna().astype(str).head(50))
                                val2_set = set(self.dfs[table2][key2].dropna().astype(str).head(50))
                                overlap = val1_set.intersection(val2_set)
                                if overlap:
                                    schema_info.append(
                                        f"    Overlapping Values (sample): {list(overlap)[:5]}\n"
                                    )
                    
                    # 也检查非ID字段的连接可能性（如名称字段）
                    for col1 in self.dfs[table1].columns:
                        for col2 in self.dfs[table2].columns:
                            if col1.lower() == col2.lower() and \
                               col1.lower() not in ['id', 'key', 'code'] and \
                               self.dfs[table1][col1].dtype == self.dfs[table2][col2].dtype:
                                val1_set = set(self.dfs[table1][col1].dropna().astype(str).head(50))
                                val2_set = set(self.dfs[table2][col2].dropna().astype(str).head(50))
                                overlap = val1_set.intersection(val2_set)
                                if len(overlap) > 0:
                                    schema_info.append(
                                        f"  Possible Join (by name): {table1}.{col1} <-> {table2}.{col2}\n"
                                    )
        
        return "\n".join(schema_info)


# ==========================================
# Custom Agent (继承 Agent 基类)
# ==========================================
class CustomAgent(Agent):
    def __init__(self, llm, config: dict = None):
        self.llm = llm
        self.config = config or {}
        self.max_retries = self.config.get("max_retries", 3)

    def load_database_from_tables(self, tables: list[str], verbose: bool = False):
        """从表格文件路径列表加载数据"""
        dfs = {}
        for table_path in tables:
            if not os.path.exists(table_path):
                if verbose:
                    print(f"❌ Error: Path not found: {table_path}")
                continue
            
            try:
                file_name = os.path.basename(table_path)
                table_name = os.path.splitext(file_name)[0]
                
                # 如果已经加载过，直接使用缓存（避免重复加载和打印）
                if hasattr(self, '_table_cache') and table_name in self._table_cache:
                    dfs[table_name] = self._table_cache[table_name]
                    continue
                
                # 尝试不同的编码
                try:
                    df = pd.read_csv(table_path, sep=None, engine='python', encoding='utf-8')
                except:
                    df = pd.read_csv(table_path, sep=None, engine='python', encoding='latin1')
                
                if len(df) > 0:
                    dfs[table_name] = df
                    # 缓存加载的表
                    if not hasattr(self, '_table_cache'):
                        self._table_cache = {}
                    self._table_cache[table_name] = df
                    if verbose:
                        print(f"   ✅ Loaded table: '{table_name}' ({df.shape[0]} rows)")
            except Exception as e:
                if verbose:
                    print(f"   ❌ Failed to load {table_path}: {e}")
        
        return dfs

    def construct_system_prompt(self, schema_str):
        return f"""
You are an expert Data Visualization Agent specialized in generating accurate and complete data visualizations.

**AVAILABLE DATA (Pre-loaded Variables):**
{schema_str}

**CRITICAL REQUIREMENTS:**

1. **Multi-table Join (CRITICAL):**
   - Carefully examine the JOIN ANALYSIS section above to identify correct join keys
   - Use exact column names as shown in the schema (case-sensitive, exact spelling)
   - When joining, verify that the join keys actually match (check overlapping values)
   - After joining, verify that all expected data points are present
   - Example: If joining Student and Faculty tables on Advisor and FacID, ensure these columns exist and contain matching values
   - Always use: `pd.merge(table1, table2, left_on='col1', right_on='col2')` with exact column names
   - NEVER create sample data - always use the pre-loaded DataFrames

2. **Data Aggregation (CRITICAL):**
   - Use the CORRECT aggregation method based on the query:
     * COUNT: Use `.size()` or `.count()` or `.value_counts()`
     * SUM: Use `.sum()`
     * AVERAGE/MEAN: Use `.mean()`
   - When counting after a join, ensure you're counting the correct entity
   - Example: To count students by faculty rank after joining:
     * `merged_df.groupby('Rank').size()` counts rows (students)
     * NOT `merged_df.groupby('Rank')['StudentID'].count()` (might miss NULLs)

3. **Data Completeness (CRITICAL):**
   - Your visualization MUST include ALL data points present in the source data
   - Do NOT filter out any categories unless explicitly asked
   - After aggregation, verify that all categories are present in result_df
   - Check for missing categories by comparing unique values before and after aggregation
   - Ensure result_df is NOT empty before plotting
   - When using groupby, use `.size()` or `.agg()` to preserve all groups, even if they have zero values
   - Example: `df.groupby('category').size()` will include all categories, even those with 0 counts

4. **Result Variable (MANDATORY):**
   - ALWAYS assign the final aggregated/prepared data to `result_df`
   - Ensure `result_df` has the correct column names matching the visualization
   - For pie charts: result_df should have categories and counts/values
   - For bar charts: result_df should have x-axis labels and y-axis values
   - Verify result_df is not empty: `assert not result_df.empty, "result_df is empty"`

5. **Field Name Accuracy (CRITICAL - ENHANCED):**
   - Use EXACT column names as shown in the schema (case-sensitive)
   - Pay attention to exact spelling and capitalization
   - When merging, use the exact column names from each table
   - **IMPORTANT: After merging tables, column names may change:**
     - If both tables have the same column name, pandas may add suffixes: `col_x`, `col_y`
     - Example: `pd.merge(df1, df2, left_on='col', right_on='col')` may result in `col_x` and `col_y`
     - Use `suffixes=('', '_y')` parameter to control this, or access columns explicitly
     - After merge, verify column names: `print(merged_df.columns)` or check schema
   - If a column name contains spaces or special characters, use it exactly as shown
   - NEVER guess column names - always refer to the schema above
   - **If you get KeyError, check for column name variants: `col`, `col_x`, `col_y`**

6. **Chart Type Selection (ENHANCED):**
   - "pie chart" or "pie" → Use `plt.pie()`
   - "bar chart" or "bar graph" → Use `plt.bar()` or `plt.barh()`
   - **"stacked bar chart" or "stacked bar" → Use `plt.bar()` with `bottom` parameter**
     Example for stacked bars:
     ```python
     # For stacked bar chart with multiple series
     x = result_df['category']
     y1 = result_df['series1']
     y2 = result_df['series2']
     ax.bar(x, y1, label='Series1')
     ax.bar(x, y2, bottom=y1, label='Series2')
     plt.legend()
     ```
   - "line chart" or "line graph" → Use `plt.plot()`
   - "scatter" → Use `plt.scatter()`
   - Match the chart type exactly as requested in the query
   - **If query mentions "stacked", "grouped", or multiple series, use appropriate stacking/grouping technique**

7. **Visualization (MANDATORY - MUST GENERATE A PLOT):**
   - You MUST create a visualization - this is mandatory
   - Use matplotlib to create the visualization
   - Do NOT call `plt.show()`
   - For pie charts: Use `plt.pie(result_df['values'], labels=result_df['categories'], autopct='%1.1f%%')`
   - For bar charts: Use `plt.bar(result_df['x'], result_df['y'])` or `plt.barh(result_df['x'], result_df['y'])`
   - **For multiple series (CRITICAL):**
     - When plotting multiple series in a loop, ensure each series has a unique label
     - Use: `ax.bar(x, y1, label='Series1')` then `ax.bar(x, y2, label='Series2')`
     - Call `plt.legend()` ONCE after all series are plotted
     - **CRITICAL: Ensure the number of labels matches the number of series**
     - Example:
       ```python
       fig, ax = plt.subplots()
       for key, grp in result_df.groupby('category'):
           ax.bar(grp['x'], grp['y'], label=str(key))
       plt.legend()  # Call once after all bars
       ```
   - Always add labels: `plt.xlabel()`, `plt.ylabel()`, `plt.title()`
   - After plotting, verify a figure was created: `assert len(plt.gcf().axes) > 0, "No plot generated"`

8. **Code Format (ENHANCED):**
   - Return ONLY valid Python code in JSON format
   - No markdown code blocks, no explanations outside JSON
   - The code string should contain actual newlines (\\n), not escaped newlines
   - **CRITICAL: Ensure ALL statements are complete and syntactically correct**
   - **Do NOT truncate code - ensure all strings, parentheses, brackets are properly closed**
   - **Do NOT end statements with backslashes or incomplete expressions**
   - Do NOT include sample/hardcoded data - use the pre-loaded DataFrames
   - **Example of BAD code (incomplete):**
     ```python
     assert not result_df.empty, \\  # ← WRONG: incomplete statement
     ```
   - **Example of GOOD code (complete):**
     ```python
     assert not result_df.empty, "result_df is empty"  # ← CORRECT: complete statement
     ```

**COMMON MISTAKES TO AVOID:**
- Creating sample data instead of using pre-loaded DataFrames (NEVER use pd.DataFrame({...}))
- Using wrong join keys (always verify column names match)
- Missing data points after aggregation (check all categories are included)
- Using incorrect aggregation method (count vs sum vs mean)
- Typos in column names (use exact names from schema, case-sensitive)
- Filtering data unnecessarily (include all relevant data)
- Not generating a plot (MUST call plt.pie(), plt.bar(), etc.)
- Empty result_df (always verify data exists before plotting)
- Incomplete code (ensure all statements are complete, no truncated code)
- Missing categories in groupby results (use .size() to preserve all groups)

**DATA COMPLETENESS CHECKLIST:**
Before plotting, verify:
1. result_df contains all expected categories/values
2. No data points were accidentally filtered out
3. Aggregation preserved all groups (even with 0 values)
4. Column names match exactly with schema
5. result_df is not empty

**OUTPUT FORMAT (JSON):**
{{
    "plan": "Brief plan of your approach (join strategy, aggregation method, chart type, etc.)",
    "code": "Complete Python code starting from imports. Use actual newlines, not escaped newlines."
}}
"""

    def parse_response(self, response_text):
        """解析 LLM 响应，提取 JSON 格式的代码"""
        def unescape_code(code_str):
            """处理 JSON 转义字符"""
            if not isinstance(code_str, str):
                return str(code_str)
            
            # JSON.loads 应该已经处理了标准转义（\n -> 换行符）
            # 但如果代码中包含字面量的 \\n（双重转义），需要处理
            # 检查是否有明显的双重转义模式
            import re
            
            # 如果代码开头就有问题（如 "import pandas\\n"），可能是双重转义
            # 使用正则表达式查找并替换明显的转义序列
            # 但要小心不要破坏字符串字面量中的内容
            
            # 尝试处理双重转义的换行符（\\n -> \n）
            # 只在明显是转义的情况下处理（不在字符串引号内）
            # 这是一个简化的处理，主要针对代码结构中的换行符
            
            # 如果代码看起来像是被错误转义了（很多 \\n 但没有真正的换行）
            if '\\n' in code_str:
                # 检查是否是双重转义：如果有很多 \\n 但很少真正的换行符
                literal_newlines = code_str.count('\\n')
                actual_newlines = code_str.count('\n')
                
                # 如果字面量换行符远多于实际换行符，可能是双重转义
                if literal_newlines > actual_newlines * 2:
                    # 尝试修复：将 \\n 替换为 \n（但要小心字符串字面量）
                    # 简单方法：直接替换（可能不够准确，但通常有效）
                    code_str = code_str.replace('\\n', '\n')
                    code_str = code_str.replace('\\t', '\t')
                    code_str = code_str.replace('\\r', '\r')
            
            return code_str
        
        try:
            # 方法1: 尝试提取完整的 JSON 对象（支持多行）
            # 改进：使用更精确的正则表达式，确保匹配完整的JSON
            json_match = re.search(r'\{[^{}]*"code"[^{}]*\}', response_text, re.DOTALL)
            if not json_match:
                # 尝试匹配嵌套的JSON（包含大括号）
                json_match = re.search(r'\{.*?"code".*?\}', response_text, re.DOTALL)
            
            if json_match:
                clean_text = json_match.group(0)
                # 尝试清理可能的问题
                clean_text = re.sub(r'```json|```python|```', '', clean_text).strip()
                
                # 尝试修复不完整的JSON（如果以"code"结尾但没有闭合）
                if clean_text.count('{') > clean_text.count('}'):
                    # 缺少闭合括号，尝试添加
                    clean_text += '}'
                
                parsed = json.loads(clean_text)
                code = parsed.get("code", "")
                plan = parsed.get("plan", "")
                
                # 处理转义字符
                code = unescape_code(code)
                
                # 检查并修复代码截断问题
                code = self._fix_incomplete_code(code)
                
                # 确保代码是字符串类型
                if isinstance(code, str) and code.strip():
                    return plan, code
                else:
                    return plan, str(code) if code else ""
        except (json.JSONDecodeError, Exception) as e:
            pass
        
        try:
            # 方法2: 移除 markdown 代码块标记后再解析
            clean_text = re.sub(r'```json|```python|```', '', response_text).strip()
            # 尝试找到 JSON 对象
            json_match = re.search(r'\{.*?"code".*?\}', clean_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                code = parsed.get("code", "")
                plan = parsed.get("plan", "")
                code = unescape_code(code)
                code = self._fix_incomplete_code(code)
                return plan, str(code) if code else ""
        except Exception as e:
            pass
        
        try:
            # 方法3: 如果解析失败，尝试直接提取代码块
            code_match = re.search(r'```python\n(.*?)\n```', response_text, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                code = unescape_code(code)
                code = self._fix_incomplete_code(code)
                return "Parse Error", code
            
            # 方法4: 尝试提取代码部分（可能在 JSON 中但格式不标准）
            code_match = re.search(r'"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', response_text, re.DOTALL)
            if code_match:
                code = code_match.group(1)
                code = unescape_code(code)
                code = self._fix_incomplete_code(code)
                return "Parse Error", code
        except Exception as e:
            pass
        
        # 最后尝试：直接返回整个响应作为代码（去除明显的非代码部分）
        clean_text = response_text.strip()
        # 移除常见的 JSON/代码块标记
        clean_text = re.sub(r'^```(?:json|python)?\n?', '', clean_text, flags=re.MULTILINE)
        clean_text = re.sub(r'\n?```$', '', clean_text, flags=re.MULTILINE)
        clean_text = unescape_code(clean_text)
        clean_text = self._fix_incomplete_code(clean_text)
        return "Parse Error", clean_text
    
    def _fix_incomplete_code(self, code: str) -> str:
        """检测并修复不完整的代码（如截断的语句）"""
        if not code or not isinstance(code, str):
            return code
        
        lines = code.split('\n')
        if not lines:
            return code
        
        # 检查最后一行是否不完整
        last_line = lines[-1].strip()
        
        # 如果最后一行以不完整的字符结尾，尝试修复
        incomplete_indicators = [':', '\\', '(', '[', '{']
        if last_line and any(last_line.endswith(ind) for ind in incomplete_indicators):
            # 检查是否是未闭合的字符串
            if last_line.endswith('\\'):
                # 可能是转义字符，检查是否在字符串中
                # 简单修复：移除末尾的反斜杠（如果它导致语法错误）
                # 但要注意：这可能不是最佳修复，只是尝试
                if last_line.count("'") % 2 == 0 and last_line.count('"') % 2 == 0:
                    # 不在字符串中，可能是转义问题
                    # 尝试修复：如果是 assert 语句，补全
                    if 'assert' in last_line and 'empty' in last_line:
                        lines[-1] = last_line.rstrip('\\') + ', "result_df is empty"'
                    else:
                        # 其他情况，移除反斜杠
                        lines[-1] = last_line.rstrip('\\')
                else:
                    # 在字符串中，可能是正常的转义
                    pass
        
        # 检查括号匹配
        code_str = '\n'.join(lines)
        open_parens = code_str.count('(')
        close_parens = code_str.count(')')
        if open_parens > close_parens:
            # 缺少闭合括号，尝试添加
            lines.append(')' * (open_parens - close_parens))
        
        # 检查方括号匹配
        open_brackets = code_str.count('[')
        close_brackets = code_str.count(']')
        if open_brackets > close_brackets:
            lines.append(']' * (open_brackets - close_brackets))
        
        # 检查大括号匹配
        open_braces = code_str.count('{')
        close_braces = code_str.count('}')
        if open_braces > close_braces:
            lines.append('}' * (open_braces - close_braces))
        
        return '\n'.join(lines)

    def generate(self, nl_query: str, tables: list[str], config: dict):
        """生成可视化代码"""
        library = config.get("library", "matplotlib")
        
        # 加载数据（不打印详细信息，减少输出）
        dfs = self.load_database_from_tables(tables, verbose=False)
        if not dfs:
            return None, None
        
        # 只在第一次调用时打印加载信息（通过检查是否有实例ID来判断）
        # 这里简化处理，减少重复输出
        
        # 生成 schema 描述
        enhancer = SchemaEnhancer(dfs)
        schema_context = enhancer.get_semantic_schema()
        
        # 构建 prompt
        system_prompt = self.construct_system_prompt(schema_context)
        user_prompt = f"Query: {nl_query}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                if attempt == 0:
                    print(f"\n--- Generation Attempt {attempt + 1} ---")
                
                # 调用 LLM
                response = self.llm.invoke(messages)
                # SimpleChatModel.invoke 返回 AIMessage，但有 content 属性
                if isinstance(response, AIMessage):
                    response_text = response.content
                elif hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # 解析响应
                plan, code = self.parse_response(response_text)
                if attempt == 0:  # 只在第一次尝试时打印简要信息
                    print(f"    ✓ 代码生成成功 ({len(code)} 字符)")
                else:
                    print(f"    ↻ 重试 {attempt + 1}/{self.max_retries}")
                
                if code:
                    # 验证代码中使用的表名和列名是否存在
                    validation_warnings = self._validate_code(code, dfs)
                    if validation_warnings:
                        # 如果有验证警告，添加到提示中重试
                        warning_msg = "Code validation warnings:\n" + "\n".join(validation_warnings)
                        if attempt == 0:
                            print(f"    ⚠️  {warning_msg[:150]}...")
                        if attempt < self.max_retries - 1:
                            messages.append(AIMessage(content=response_text))
                            
                            # 根据警告类型提供更具体的指导
                            fix_instructions = "Please fix these issues:\n" + warning_msg + "\n\nCRITICAL FIXES REQUIRED:\n"
                            
                            if any("sample data" in w.lower() or "DataFrame" in w for w in validation_warnings):
                                fix_instructions += "1. DO NOT create sample data with pd.DataFrame({}). Use the pre-loaded DataFrames directly.\n"
                            
                            if any("plotting function" in w.lower() for w in validation_warnings):
                                fix_instructions += "2. MUST include a plotting function: plt.pie(), plt.bar(), plt.barh(), or plt.plot()\n"
                            
                            if any("result_df" in w.lower() for w in validation_warnings):
                                fix_instructions += "3. MUST create 'result_df' variable with the aggregated data before plotting\n"
                            
                            if any("Syntax error" in w or "EOF" in w for w in validation_warnings):
                                fix_instructions += "4. Code is incomplete or has syntax errors. Ensure complete, valid Python code.\n"
                            
                            if any("Column" in w and "not found" in w for w in validation_warnings):
                                fix_instructions += "5. Verify all column names match exactly with the schema (case-sensitive)\n"
                            
                            fix_instructions += "\nEnsure:\n- All column names and table names are correct (use exact names from schema)\n- Code uses pre-loaded DataFrames, not sample data\n- Code is complete and syntactically correct\n- Code includes a plotting function and creates result_df"
                            
                            messages.append(HumanMessage(content=fix_instructions))
                            continue
                    
                    # 准备执行环境
                    pre_code = self._generate_pre_code(dfs, library)
                    full_code = pre_code + "\n" + code
                    
                    context = {
                        "tables": tables,
                        "library": library,
                        # 注意：dfs 不能直接序列化，需要在 execute 时重新加载
                    }
                    
                    return full_code, context
                else:
                    # 如果代码为空，添加到对话历史并重试
                    messages.append(AIMessage(content=response_text))
                    messages.append(HumanMessage(content="Please provide valid Python code in JSON format."))
            
            except Exception as e:
                print(f"❌ Error in generation: {e}")
                if attempt < self.max_retries - 1:
                    messages.append(HumanMessage(content=f"An error occurred: {str(e)}. Please retry."))
                else:
                    warnings.warn(str(traceback.format_exc()))
        
        return None, None

    def _validate_code(self, code: str, dfs: dict) -> List[str]:
        """验证代码中使用的表名和列名是否存在"""
        warnings = []
        
        # 检查代码完整性（增强版）
        completeness_warnings = self._validate_code_completeness(code)
        warnings.extend(completeness_warnings)
        
        # 检查代码语法完整性（避免EOF错误）
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            error_msg = str(e)
            if 'EOF' in error_msg or 'unexpected EOF' in error_msg:
                warnings.append(f"SyntaxError: Code appears to be incomplete or truncated. {error_msg[:100]}")
            else:
                warnings.append(f"Syntax error detected: {error_msg[:100]}. Code may be incomplete or truncated.")
        
        # 检查表名
        table_names = set(dfs.keys())
        
        # 查找可能的表名引用（简单启发式方法）
        for table_name in table_names:
            # 检查表名是否在代码中被使用（作为变量）
            # 注意：这只是简单检查，可能不够准确
            if table_name in code or table_name.capitalize() in code:
                # 验证表是否存在
                if table_name not in dfs:
                    warnings.append(f"Table '{table_name}' referenced but not loaded")
        
        # 检查是否包含绘图语句
        plot_functions = ['plt.pie', 'plt.bar', 'plt.barh', 'plt.plot', 'plt.scatter', 'plt.hist']
        has_plot = any(func in code for func in plot_functions)
        if not has_plot:
            warnings.append("Code does not contain any plotting function (plt.pie, plt.bar, etc.)")
        
        # 检查是否创建了 result_df
        if 'result_df' not in code:
            warnings.append("Code does not create 'result_df' variable")
        
        # 检查是否使用了样本数据（不应该）- 更严格的检查
        sample_patterns = [
            'pd.DataFrame({', 
            'Sample data', 
            'sample data', 
            'hardcoded',
            '= pd.DataFrame({',  # 更精确的模式
            'DataFrame({',  # 检查DataFrame创建
        ]
        # 检查是否在创建DataFrame（可能是样本数据）
        import re
        df_creation_pattern = r'(\w+)\s*=\s*pd\.DataFrame\s*\('
        matches = re.findall(df_creation_pattern, code)
        if matches:
            # 检查创建的DataFrame是否使用了预加载的表名
            for match in matches:
                if match not in table_names and match.lower() not in [t.lower() for t in table_names]:
                    warnings.append(f"Code creates DataFrame '{match}' which may be sample data. Use pre-loaded DataFrames instead.")
        
        for pattern in sample_patterns:
            if pattern.lower() in code.lower():
                warnings.append(f"Code appears to create sample data instead of using pre-loaded DataFrames")
                break
        
        # 检查列名（更严格的验证）
        # 提取所有可能的列名引用
        # 排除常见的方法名，避免误报
        pandas_methods = {'groupby', 'merge', 'join', 'agg', 'aggregate', 'apply', 'map', 
                         'fillna', 'dropna', 'sort_values', 'sort_index', 'reset_index',
                         'head', 'tail', 'iloc', 'loc', 'at', 'iat', 'values', 'columns',
                         'index', 'dtypes', 'shape', 'size', 'count', 'sum', 'mean', 'std',
                         'min', 'max', 'describe', 'info', 'copy', 'astype', 'drop',
                         'rename', 'set_index', 'reset_index', 'transpose', 'T'}
        
        for table_name, df in dfs.items():
            if table_name in code:
                # 查找列名引用模式：df['col'] 或 df.col
                # 模式1: df['col'] 或 df["col"]
                bracket_pattern = rf"{re.escape(table_name)}\[['\"]([^'\"]+)['\"]\]"
                bracket_matches = re.findall(bracket_pattern, code)
                for col_name in bracket_matches:
                    if col_name and col_name not in df.columns and col_name.lower() not in pandas_methods:
                        # 检查是否是大小写问题
                        col_lower = col_name.lower()
                        matching_cols = [c for c in df.columns if c.lower() == col_lower]
                        if not matching_cols:
                            # 检查是否是merge后的列名变体（col_x, col_y）
                            col_variants = [f"{col_name}_x", f"{col_name}_y", f"{col_name}_left", f"{col_name}_right"]
                            variant_found = any(v in df.columns for v in col_variants)
                            if not variant_found:
                                # 提供列名建议
                                suggestion = self._suggest_column_name(col_name, list(df.columns))
                                warnings.append(f"Column '{col_name}' not found in table '{table_name}'. {suggestion}")
                
                # 模式2: df.col (但要排除方法调用)
                dot_pattern = rf"{re.escape(table_name)}\.([a-zA-Z_][a-zA-Z0-9_]*)"
                dot_matches = re.findall(dot_pattern, code)
                for col_name in dot_matches:
                    # 检查是否是方法调用（后面有括号）
                    col_pattern = rf"{re.escape(table_name)}\.{re.escape(col_name)}\s*\("
                    is_method_call = re.search(col_pattern, code)
                    if not is_method_call and col_name not in df.columns and col_name.lower() not in pandas_methods:
                        # 检查是否是大小写问题
                        col_lower = col_name.lower()
                        matching_cols = [c for c in df.columns if c.lower() == col_lower]
                        if not matching_cols:
                            # 检查是否是merge后的列名变体
                            col_variants = [f"{col_name}_x", f"{col_name}_y", f"{col_name}_left", f"{col_name}_right"]
                            variant_found = any(v in df.columns for v in col_variants)
                            if not variant_found:
                                suggestion = self._suggest_column_name(col_name, list(df.columns))
                                warnings.append(f"Column '{col_name}' not found in table '{table_name}'. {suggestion}")
        
        return warnings
    
    def _validate_code_completeness(self, code: str) -> List[str]:
        """检查代码是否完整（括号匹配、字符串闭合等）"""
        warnings = []
        
        # 检查括号匹配
        if code.count('(') != code.count(')'):
            warnings.append("Unmatched parentheses detected - code may be incomplete")
        if code.count('[') != code.count(']'):
            warnings.append("Unmatched brackets detected - code may be incomplete")
        if code.count('{') != code.count('}'):
            warnings.append("Unmatched braces detected - code may be incomplete")
        
        # 检查字符串引号匹配（简化检查）
        # 计算单引号（排除转义的）
        single_quotes = len([m for m in re.finditer(r"(?<!\\)'", code)])
        double_quotes = len([m for m in re.finditer(r'(?<!\\)"', code)])
        if single_quotes % 2 != 0:
            warnings.append("Unmatched single quotes detected - string may be incomplete")
        if double_quotes % 2 != 0:
            warnings.append("Unmatched double quotes detected - string may be incomplete")
        
        # 检查最后一行是否不完整
        lines = code.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            # 如果最后一行以不完整的字符结尾
            incomplete_chars = [':', '\\', '(', '[', '{']
            if last_line and any(last_line.endswith(c) for c in incomplete_chars):
                # 检查是否在字符串中（简单检查）
                if not (last_line.count("'") % 2 == 1 or last_line.count('"') % 2 == 1):
                    warnings.append("Code appears to be incomplete (last line ends with continuation character)")
        
        return warnings
    
    def _suggest_column_name(self, wrong_col: str, available_cols: List[str]) -> str:
        """为错误的列名提供建议"""
        from difflib import get_close_matches
        
        # 使用模糊匹配找到最相似的列名
        suggestions = get_close_matches(wrong_col, available_cols, n=3, cutoff=0.6)
        
        if suggestions:
            return f"Did you mean: {', '.join(suggestions)}?"
        else:
            # 检查大小写变体
            col_lower = wrong_col.lower()
            matching = [c for c in available_cols if c.lower() == col_lower]
            if matching:
                return f"Column name case mismatch. Available: {', '.join(matching)}"
            
            # 检查是否是merge后的列名
            col_variants = [f"{wrong_col}_x", f"{wrong_col}_y", f"{wrong_col}_left", f"{wrong_col}_right"]
            found_variants = [v for v in col_variants if v in available_cols]
            if found_variants:
                return f"After merge, column may be: {', '.join(found_variants)}"
        
        # 显示前10个可用列名
        return f"Available columns: {', '.join(available_cols[:10])}"
    
    def _generate_pre_code(self, dfs: dict, library: str):
        """生成预执行代码说明（实际数据在 execute 中加载）"""
        pre_code_parts = [
            "import pandas as pd",
            "import matplotlib",
            "matplotlib.use('Agg')",
            "import matplotlib.pyplot as plt",
        ]
        
        if library == "seaborn":
            pre_code_parts.append("import seaborn as sns")
        
        pre_code_parts.append("\n# DataFrames are pre-loaded as variables with their table names")
        
        return "\n".join(pre_code_parts)

    def execute(self, code: str, context: dict, log_name: str = None):
        """执行代码并生成可视化"""
        tables = context.get("tables", [])
        library = context.get("library", "matplotlib")
        
        # 重新加载数据（因为 context 中的 dfs 无法序列化，不打印详细信息）
        dfs = self.load_database_from_tables(tables, verbose=False)
        if not dfs:
            return ChartExecutionResult(
                status=False,
                error_msg="RuntimeError: Failed to load data tables."
            )
        
        # 准备执行环境
        global_env = {}
        global_env.update({name: df.copy() for name, df in dfs.items()})
        global_env.update({
            "pd": pd,
            "plt": plt,
            "svg_string": None,
            "show_svg": show_svg,
            "svg_name": log_name,
        })
        
        if library == "seaborn":
            import seaborn as sns
            global_env["sns"] = sns
        
        # 清除之前的图
        plt.clf()
        
        # 先执行代码（不调用 show_svg），检查是否生成了图
        try:
            exec(code, {}, global_env)
            
            # 检查是否有结果
            if 'result_df' in global_env:
                res_df = global_env['result_df']
                if res_df.empty:
                    return ChartExecutionResult(
                        status=False,
                        error_msg="RuntimeError: 'result_df' is empty. Please ensure data aggregation produces non-empty results."
                    )
            else:
                # 警告：没有创建 result_df，但继续检查是否有图
                pass
            
            # 检查是否生成了图（在调用 show_svg 之前检查）
            fig = plt.gcf()
            if not fig.axes or len(fig.axes) == 0:
                # 提供更详细的错误信息
                error_details = []
                if 'result_df' not in global_env:
                    error_details.append("'result_df' variable not created")
                else:
                    res_df = global_env['result_df']
                    if res_df.empty:
                        error_details.append("'result_df' is empty")
                    else:
                        error_details.append(f"'result_df' has {len(res_df)} rows but no plot was generated")
                
                plot_functions = ['plt.pie', 'plt.bar', 'plt.barh', 'plt.plot', 'plt.scatter']
                has_plot_call = any(func in code for func in plot_functions)
                if not has_plot_call:
                    error_details.append("Code does not contain any plotting function call")
                
                error_msg = f"RuntimeError: No plot generated. " + "; ".join(error_details) + "."
                return ChartExecutionResult(status=False, error_msg=error_msg)
            
            # 如果检查通过，调用 show_svg 生成 SVG
            svg_string = show_svg(plt, log_name)
            if svg_string:
                return ChartExecutionResult(status=True, svg_string=svg_string)
            else:
                return ChartExecutionResult(
                    status=False,
                    error_msg="RuntimeError: SVG string not generated."
                )
        
        except SyntaxError as syntax_error:
            # 语法错误，提供更详细的信息
            error_msg = f"SyntaxError: {str(syntax_error)}. Please check for escaped characters or invalid Python syntax."
            # 记录完整错误信息到日志
            if log_name:
                try:
                    import os
                    log_dir = os.path.dirname(log_name) if os.path.dirname(log_name) else '.'
                    error_log_path = os.path.join(log_dir, 'execution_errors.log')
                    with open(error_log_path, 'a') as f:
                        f.write(f"\n=== SyntaxError in {log_name} ===\n")
                        f.write(f"Error: {str(syntax_error)}\n")
                        f.write(f"Code snippet (last 20 lines):\n{code.split(chr(10))[-20:]}\n")
                except:
                    pass
            return ChartExecutionResult(status=False, error_msg=error_msg)
        except KeyError as key_error:
            # 键错误，可能是列名错误
            error_msg = f"KeyError: {str(key_error)}. Please verify column names match exactly with the schema (case-sensitive)."
            # 尝试提供列名建议
            try:
                # 从错误信息中提取列名
                import re
                col_match = re.search(r"['\"]([^'\"]+)['\"]", str(key_error))
                if col_match:
                    wrong_col = col_match.group(1)
                    # 从代码中查找可能的表名
                    for table_name, df in dfs.items():
                        if table_name in code:
                            suggestion = self._suggest_column_name(wrong_col, list(df.columns))
                            if suggestion:
                                error_msg += f" {suggestion}"
            except:
                pass
            return ChartExecutionResult(status=False, error_msg=error_msg)
        except ValueError as value_error:
            # ValueError，可能是标签/数据不匹配
            error_msg = f"ValueError: {str(value_error)}"
            if 'label' in str(value_error).lower() and 'bar' in str(value_error).lower():
                error_msg += ". Ensure the number of labels matches the number of data series when plotting multiple series."
            return ChartExecutionResult(status=False, error_msg=error_msg)
        except Exception as exception_error:
            exception_info = traceback.format_exception_only(
                type(exception_error), exception_error
            )
            # 提供更详细的错误信息
            error_msg = f"{type(exception_error).__name__}: {str(exception_error)}"
            # 记录完整错误信息到日志
            if log_name:
                try:
                    import os
                    log_dir = os.path.dirname(log_name) if os.path.dirname(log_name) else '.'
                    error_log_path = os.path.join(log_dir, 'execution_errors.log')
                    with open(error_log_path, 'a') as f:
                        f.write(f"\n=== {type(exception_error).__name__} in {log_name} ===\n")
                        f.write(f"Error: {str(exception_error)}\n")
                        f.write(f"Traceback:\n{traceback.format_exc()}\n")
                except:
                    pass
            return ChartExecutionResult(status=False, error_msg=error_msg)

