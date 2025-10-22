"""
系统状态检测模块
用于检测应用运行所需的各项依赖是否就绪
"""

import os
from typing import Dict, Tuple


def check_api_key() -> Tuple[bool, str]:
    """
    检查 DashScope API Key 是否已配置
    
    Returns:
        (is_ready, message): 状态和说明信息
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        return True, f"已配置 (密钥: {api_key[:8]}...)"
    else:
        return False, "未配置"


def check_knowledge_base() -> Tuple[bool, str]:
    """
    检查知识库索引是否已构建
    
    Returns:
        (is_ready, message): 状态和说明信息
    """
    index_path = "faiss_index"
    if os.path.exists(index_path) and os.path.isdir(index_path):
        # 检查索引文件是否存在
        index_file = os.path.join(index_path, "index.faiss")
        pkl_file = os.path.join(index_path, "index.pkl")
        if os.path.exists(index_file) and os.path.exists(pkl_file):
            # 获取文件大小
            index_size = os.path.getsize(index_file) / 1024  # KB
            return True, f"已构建 (索引大小: {index_size:.1f} KB)"
        else:
            return False, "索引文件不完整"
    else:
        return False, "索引不存在"


def check_knowledge_docs() -> Tuple[bool, str]:
    """
    检查知识库文档目录是否存在且包含文档
    
    Returns:
        (is_ready, message): 状态和说明信息
    """
    docs_path = "knowledge_base_docs"
    if os.path.exists(docs_path) and os.path.isdir(docs_path):
        # 统计 .txt 文件数量
        txt_files = [f for f in os.listdir(docs_path) if f.endswith('.txt')]
        if txt_files:
            return True, f"找到 {len(txt_files)} 个文档"
        else:
            return False, "目录为空"
    else:
        return False, "目录不存在"


def check_xrd_data() -> Tuple[bool, str]:
    """
    检查 XRD 示例数据是否存在
    
    Returns:
        (is_ready, message): 状态和说明信息
    """
    xrd_path = "experimental_data/xrd/simulated_xrd_data.txt"
    if os.path.exists(xrd_path):
        return True, "示例数据可用"
    else:
        return False, "示例数据缺失"


def check_model_data() -> Tuple[bool, str]:
    """
    检查性能预测模型所需的训练数据是否存在
    
    Returns:
        (is_ready, message): 状态和说明信息
    """
    data_path = "simulated_experimental_data.csv"
    if os.path.exists(data_path):
        # 检查文件大小
        file_size = os.path.getsize(data_path) / 1024  # KB
        return True, f"数据可用 ({file_size:.1f} KB)"
    else:
        return False, "训练数据缺失"


def check_conda_env() -> Tuple[bool, str]:
    """
    检查当前是否在正确的 Conda 环境中
    
    Returns:
        (is_ready, message): 状态和说明信息
    """
    conda_env = os.getenv("CONDA_DEFAULT_ENV")
    if conda_env:
        if conda_env == "mylangchain":
            return True, f"环境正确 ({conda_env})"
        else:
            return False, f"当前环境: {conda_env} (应为 mylangchain)"
    else:
        return False, "未检测到 Conda 环境"


def get_system_status() -> Dict[str, Tuple[bool, str]]:
    """
    获取所有系统依赖的状态
    
    Returns:
        dict: 各项检查的结果字典
    """
    return {
        "API密钥": check_api_key(),
        "知识库索引": check_knowledge_base(),
        "知识文档": check_knowledge_docs(),
        "XRD数据": check_xrd_data(),
        "训练数据": check_model_data(),
        "运行环境": check_conda_env(),
    }


def get_fix_commands() -> Dict[str, str]:
    """
    获取修复各项问题的命令提示
    
    Returns:
        dict: 修复命令字典
    """
    return {
        "API密钥": "export DASHSCOPE_API_KEY='your_api_key_here'",
        "知识库索引": "python build_knowledge_base.py",
        "知识文档": "请在 knowledge_base_docs/ 目录中添加 .txt 文档",
        "运行环境": "conda activate mylangchain",
    }


def get_overall_health() -> str:
    """
    获取系统整体健康状态
    
    Returns:
        str: "healthy", "warning", "error"
    """
    status = get_system_status()
    results = [v[0] for v in status.values()]
    
    # 核心依赖：API密钥和知识库索引
    core_deps = [status["API密钥"][0], status["知识库索引"][0]]
    
    if all(core_deps):
        if all(results):
            return "healthy"
        else:
            return "warning"
    else:
        return "error"
