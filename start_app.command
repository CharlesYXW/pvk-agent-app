#!/bin/bash
# AI助手 - 一键启动脚本 (macOS优化版)

# 切换到脚本所在目录，确保后续命令在正确的项目路径下执行
cd "$(dirname "$0")"

# --- 启动流程 ---
echo "正在准备环境以启动AI助手..."

# 关键步骤：初始化Conda环境
# 这使得脚本在新的终端窗口中也能找到并使用conda命令
# 检查conda命令是否存在
if ! command -v conda &> /dev/null; then
    echo "错误：未找到conda命令。请确保您已安装Anaconda或Miniconda。"
    read -p "按回车键退出..."
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "正在激活conda环境: mylangchain..."
conda activate mylangchain

# 检查激活是否成功
if [ $? -ne 0 ]; then
    echo "错误：激活conda环境 'mylangchain' 失败。"
    echo "请确保您已正确创建该环境。"
    read -p "按回车键退出..."
    exit 1
fi

echo "环境准备就绪，正在启动Streamlit应用..."
echo "您随时可以在此终端窗口按 Ctrl+C 来停止应用。"
echo "----------------------------------------"

# 运行Streamlit应用
streamlit run app.py --server.headless true