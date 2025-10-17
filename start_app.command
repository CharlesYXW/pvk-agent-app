#!/bin/bash
# AI助手 - 一键启动脚本 (macOS优化版)

# --- 配置 ---
PORT=8501

# --- 启动前检查 ---
echo "正在检查端口 ${PORT}..."
# 使用 lsof -ti :$PORT 来获取监听该端口的PID
PID=$(lsof -ti :${PORT})

if [ ! -z "$PID" ]; then
    echo "警告：端口 ${PORT} 已被进程 PID: ${PID} 占用。"
    # 询问用户是否终止进程
    read -p "是否需要终止该进程以启动新应用？(y/n): " answer
    case ${answer:0:1} in
        y|Y )
            echo "正在终止进程 PID: ${PID}..."
            kill -9 ${PID}
            if [ $? -eq 0 ]; then
                echo "进程已成功终止。"
                sleep 1 # 等待端口释放
            else
                echo "错误：无法终止进程。请手动关闭占用端口的应用。"
                read -p "按回车键退出..."
                exit 1
            fi
        ;;
        * )
            echo "操作已取消。无法启动新应用。"
            read -p "按回车键退出..."
            exit 1
        ;;
    esac
fi

# --- 环境与启动 ---
# 切换到脚本所在目录，确保后续命令在正确的项目路径下执行
cd "$(dirname "$0")"

echo "正在准备环境以启动AI助手..."

# 初始化Conda环境
if ! command -v conda &> /dev/null; then
    echo "错误：未找到conda命令。请确保您已安装Anaconda或Miniconda。"
    read -p "按回车键退出..."
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "正在激活conda环境: mylangchain..."
conda activate mylangchain

if [ $? -ne 0 ]; then
    echo "错误：激活conda环境 'mylangchain' 失败。"
    echo "请确保您已正确创建该环境。"
    read -p "按回车键退出..."
    exit 1
fi

echo "环境准备就绪，正在启动Streamlit应用..."
echo "应用将在您的默认浏览器中自动打开。"
echo "您随时可以在此终端窗口按 Ctrl+C 来停止应用。"
echo "----------------------------------------"

# 运行Streamlit应用。移除 --server.headless true 以便自动打开浏览器。
echo "" | streamlit run app.py --browser.gatherUsageStats=false

# 在脚本末尾添加暂停，以便在出错时查看日志
read -p "脚本执行完毕，按回车键退出..."