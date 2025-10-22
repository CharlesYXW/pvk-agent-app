import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# 在所有代码执行前，首先加载.env文件中的环境变量
load_dotenv()
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestRegressor
import itertools
import os
import arxiv
import datetime
import base64 # Added for base64 encoding of logo

# LangChain相关的库（仅用于检索）
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# DashScope官方SDK
from dashscope import Generation

# 系统状态检测模块
import system_status

# --- AI Persona Definition ---
JA_ASSISTANT_PERSONA = "你是晶澳科技（JA SOLAR）钙钛矿研究部的人工智能助手，名为‘晶澳智能助手’。你的任务是为用户提供光伏行业相关的专业支持。在所有回答中请保持这个身份和专业的语气。"
JA_ASSISTANT_INTRO = "您好！我是晶澳智能助手，专注于为钙钛矿光伏研究提供支持。如果您有任何关于钙钛矿技术、文献、实验数据分析等方面的问题，欢迎随时向我提问！"

# New persona for general knowledge with strict anti-hallucination rules
JA_ASSISTANT_GENERAL_KNOWLEDGE_PERSONA = """你是晶澳科技（JA SOLAR）钙钛矿研究部的人工智能助手，名为“晶澳智能助手”。
你的任务是为用户提供光伏行业相关的专业支持。
在回答问题时，你必须严格遵守以下规则：
1. 你可以结合你的通用知识进行回答。
2. **极其重要**: 在使用通用知识回答时，你必须明确说明这是一个行业内的普遍知识或公开信息，**绝对不能**将这些通用的技术、成果或数据归功于“晶澳科技”。不能捏造任何关于晶澳科技的信息。
3. 只有当你的知识库（如果提供了上下文）中明确提到了“晶澳科技”的具体成果时，你才能提及公司。
4. 保持专业、客观、严谨的语气。"""

# --- 核心功能逻辑 ---

def get_retriever():
    if not os.path.exists("faiss_index"): return None
    try:
        embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"加载知识库失败: {e}")
        return None

# 新增：使用DashScope SDK调用模型的辅助函数
def call_qwen_model(messages):
    try:
        response = Generation.call(
            model="qwen-flash",
            messages=messages,
            result_format="message",
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            return f"调用大模型时出错：{response.message}"
    except Exception as e:
        return f"调用大模型时发生异常: {e}"

def summarize_with_ai(summary_text):
    """使用Qwen模型总结论文摘要。"""
    prompt = f"请用简洁的中文总结以下学术论文的摘要，提炼其核心观点、方法和结论，以便快速了解其价值。不要超过三句话。摘要如下：\n\n{summary_text}"
    messages = [
        {"role": "system", "content": f"{JA_ASSISTANT_PERSONA} 在这个具体的任务里，你的角色是一个专门总结学术论文的专家。"},
        {"role": "user", "content": prompt}
    ]
    return call_qwen_model(messages)

@st.cache_data
def get_latest_papers(keywords, date_range="all_time", sort_by="Relevance"):
    """根据关键词、日期范围和排序方式从arXiv检索论文。"""
    if not keywords or not any(keywords):
        return [], "请输入至少一个关键词。"

    # 1. 构建日期查询字符串
    date_query_part = ""
    if date_range != "all_time":
        end_date = datetime.datetime.now(datetime.timezone.utc)
        if date_range == "last_month":
            start_date = end_date - datetime.timedelta(days=30)
        elif date_range == "last_3_months":
            start_date = end_date - datetime.timedelta(days=90)
        elif date_range == "last_year":
            start_date = end_date - datetime.timedelta(days=365)
        
        start_date_str = start_date.strftime("%Y%m%d%H%M")
        end_date_str = end_date.strftime("%Y%m%d%H%M")
        date_query_part = f" AND submittedDate:[{start_date_str} TO {end_date_str}]"

    # 2. 检索论文
    # 根据排序参数选择API的排序标准
    api_sort_criterion = arxiv.SortCriterion.Relevance
    if sort_by == "SubmittedDate":
        api_sort_criterion = arxiv.SortCriterion.SubmittedDate

    MAX_RESULTS_PER_KEYWORD = 10
    unique_papers = {}
    for keyword in keywords:
        try:
            full_query = f"({keyword}){date_query_part}"
            search = arxiv.Search(
                query=full_query, 
                max_results=MAX_RESULTS_PER_KEYWORD, 
                sort_by=api_sort_criterion # 使用选择的排序方式
            )
            for result in search.results():
                if result.entry_id not in unique_papers:
                    unique_papers[result.entry_id] = {
                        "entry_id": result.entry_id,
                        "title": result.title,
                        "authors": ', '.join(author.name for author in result.authors),
                        "pdf_url": result.pdf_url,
                        "summary": result.summary.replace('\n', ' '),
                        "published": result.published.strftime('%Y-%m-%d')
                    }
        except Exception as e:
            return [], f"检索时出错: {e}"
    
    if not unique_papers:
        return [], "在选定时间范围内，未找到与您关键词相关的新论文。"
        
    # 3. 对最终结果列表进行排序
    papers_list = list(unique_papers.values())
    if sort_by == "SubmittedDate":
        # 如果用户选择按最新发表排序，则对合并后的列表进行排序
        sorted_papers = sorted(papers_list, key=lambda p: p['published'], reverse=True)
    else:
        # 如果按相关度，则直接使用混合后的列表（顺序部分取决于API和合并过程）
        sorted_papers = papers_list
    
    return sorted_papers, None

def analyze_xrd_from_upload(uploaded_file):
    # ... (此函数不变)
    if uploaded_file is None: return None
    try:
        data = np.loadtxt(uploaded_file, comments="#", delimiter=",")
        angle, intensity = data[:, 0], data[:, 1]
    except Exception:
        st.error("文件解析失败。请确保是两列（角度, 强度）的CSV或TXT文件。")
        return None
    peaks, _ = find_peaks(intensity, height=np.mean(intensity), distance=10)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(angle, intensity, label="XRD Spectrum"), ax.plot(angle[peaks], intensity[peaks], "x", markersize=8, label="Detected Peaks")
    for i in peaks: ax.annotate(f"{angle[i]:.2f}°", (angle[i], intensity[i]), textcoords="offset points", xytext=(0,5), ha='center')
    ax.set_title("XRD Spectrum Analysis"), ax.set_xlabel("2-Theta Angle (°)"), ax.set_ylabel("Intensity (A.U.)")
    ax.legend(), ax.grid(True, linestyle='--', alpha=0.6)
    return fig

@st.cache_resource
def get_trained_model():
    # ... (此函数不变)
    if not os.path.exists("simulated_experimental_data.csv"): return None
    df = pd.read_csv("simulated_experimental_data.csv")
    features = ['spin_coating_rpm', 'annealing_temperature_C', 'additive_concentration_percent']
    target = 'efficiency_percent'
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[features], df[target])
    return model

def find_optimal_params(_model):
    # ... (此函数不变)
    features = ['spin_coating_rpm', 'annealing_temperature_C', 'additive_concentration_percent']
    param_grid = list(itertools.product(np.arange(2500, 5501, 500), np.arange(80, 121, 10), np.arange(0.5, 1.51, 0.2)))
    grid_df = pd.DataFrame(param_grid, columns=features)
    predicted_efficiencies = _model.predict(grid_df)
    best_index = np.argmax(predicted_efficiencies)
    return grid_df.iloc[best_index], predicted_efficiencies[best_index]

# --- Streamlit 应用界面 ---
st.set_page_config(
    page_title="晶澳研发智能助手",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
       .block-container {
            padding-top: 1.5rem !important;
        }
        /* Custom styling for the 'AI Summary' button using a more robust data-testid selector */
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton button {
            background-color: #4A90E2 !important; /* A medium-dark blue */
            border-color: #4A90E2 !important;
            color: white !important; /* White text for readability */
        }
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton button:hover {
            background-color: #357ABD !important; /* A slightly darker blue for hover */
            border-color: #357ABD !important;
            color: white !important;
        }
        /* 状态指示器样式 */
        .status-healthy { color: #28a745; font-weight: bold; }
        .status-warning { color: #ffc107; font-weight: bold; }
        .status-error { color: #dc3545; font-weight: bold; }
        /* 导航按钮样式优化 - 改为深蓝色主题 */
        .stButton > button {
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        /* 修改 primary 按钮颜色为深蓝色 */
        button[kind="primary"] {
            background-color: #1e3a8a !important;
            border-color: #1e3a8a !important;
        }
        button[kind="primary"]:hover {
            background-color: #1e40af !important;
            border-color: #1e40af !important;
        }
    </style>
    """,
    unsafe_allow_html=True)

# --- 系统状态面板 ---
def render_system_status_panel():
    """渲染系统状态提示面板"""
    status_dict = system_status.get_system_status()
    overall_health = system_status.get_overall_health()
    fix_commands = system_status.get_fix_commands()
    
    # 根据整体健康状态选择颜色和图标
    if overall_health == "healthy":
        status_color = "success"
        status_icon = "✅"
        status_text = "系统运行正常"
    elif overall_health == "warning":
        status_color = "warning"
        status_icon = "⚠️"
        status_text = "部分功能受限"
    else:
        status_color = "error"
        status_icon = "❌"
        status_text = "核心依赖缺失"
    
    with st.expander(f"{status_icon} 系统状态: {status_text}", expanded=(overall_health == "error")):
        cols = st.columns(3)
        
        for idx, (check_name, (is_ready, message)) in enumerate(status_dict.items()):
            col = cols[idx % 3]
            with col:
                status_emoji = "✅" if is_ready else "❌"
                st.markdown(f"**{status_emoji} {check_name}**")
                st.caption(message)
                
                # 如果未就绪且有修复命令，显示修复建议
                if not is_ready and check_name in fix_commands:
                    st.code(fix_commands[check_name], language="bash")
        
        # 添加健康检查按钮
        if st.button("🔄 刷新状态检查", use_container_width=True):
            st.rerun()

st.title("🔬 钙钛矿研发智能助手")

# 渲染系统状态面板
render_system_status_panel()

# --- 导航 ---
with st.sidebar:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(script_dir, "assets", "logo.png")
    st.image(logo_path, use_container_width=True)

    st.markdown("<h1 style='text-align: center; font-size: 24px; margin-bottom: 20px;'>功能导航</h1>", unsafe_allow_html=True)
    
    # 初始化页面状态
    if 'page' not in st.session_state: 
        st.session_state.page = "知识库问答"
    
    def set_page(page_name): 
        st.session_state.page = page_name
    
    # 定义页面配置（图标 + 名称 + 描述）
    pages = [
        {"icon": "💬", "name": "知识库问答", "desc": "基于内部文档的智能问答"},
        {"icon": "📰", "name": "文献检索", "desc": "追踪最新科研动态"},
        {"icon": "📈", "name": "XRD分析", "desc": "自动分析衍射图谱"},
        {"icon": "💡", "name": "性能预测", "desc": "AI预测材料性能"},
        {"icon": "🚀", "name": "实验优化", "desc": "寻找最佳参数组合"},
    ]
    
    # 渲染导航按钮
    for page_info in pages:
        is_current = st.session_state.page == page_info["name"]
        button_type = "primary" if is_current else "secondary"
        
        # 创建按钮容器
        btn_container = st.container()
        with btn_container:
            if st.button(
                f"{page_info['icon']} {page_info['name']}",
                on_click=set_page,
                args=(page_info["name"],),
                use_container_width=True,
                type=button_type
            ):
                pass
            if is_current:
                st.caption(f"📍 {page_info['desc']}")
    
    # 添加分隔线
    st.markdown("---")
    
    # 添加快捷操作
    st.markdown("### ⚡ 快捷操作")
    
    # 使用 tabs 组织两个功能
    tab1, tab2 = st.tabs(["🔑 设置API", "📚 重建索引"])
    
    with tab1:
        st.caption("配置 DashScope API 密钥")
        
        # 检查当前 API Key 状态
        current_key = os.getenv("DASHSCOPE_API_KEY")
        if current_key:
            st.success(f"✅ 已配置 (密钥: {current_key[:8]}...)")
        else:
            st.warning("⚠️ 未配置 API 密钥")
        
        # API Key 输入
        new_api_key = st.text_input(
            "输入新的 API 密钥",
            type="password",
            placeholder="sk-xxxxxxxxxx",
            key="api_key_input"
        )
        
        if st.button("💾 保存密钥", use_container_width=True, type="primary"):
            if new_api_key:
                try:
                    # 保存到 .env 文件
                    env_path = ".env"
                    env_content = ""
                    
                    # 读取现有 .env 内容（如果存在）
                    if os.path.exists(env_path):
                        with open(env_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            env_content = "".join([line for line in lines if not line.startswith("DASHSCOPE_API_KEY=")])
                    
                    # 添加新的 API Key
                    env_content += f"DASHSCOPE_API_KEY={new_api_key}\n"
                    
                    # 写入文件
                    with open(env_path, 'w', encoding='utf-8') as f:
                        f.write(env_content)
                    
                    # 更新当前环境变量
                    os.environ["DASHSCOPE_API_KEY"] = new_api_key
                    
                    st.success("✅ API 密钥保存成功！")
                    st.info("💡 提示：下次启动应用时会自动加载")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ 保存失败: {e}")
            else:
                st.warning("⚠️ 请输入有效的 API 密钥")
    
    with tab2:
        st.caption("重新构建知识库向量索引")
        
        # 显示当前知识库状态
        docs_path = "knowledge_base_docs"
        if os.path.exists(docs_path):
            txt_files = [f for f in os.listdir(docs_path) if f.endswith('.txt')]
            st.info(f"📄 当前文档数量: {len(txt_files)} 个")
        else:
            st.warning("⚠️ 知识库目录不存在")
        
        # 检查索引状态
        index_path = "faiss_index"
        if os.path.exists(index_path):
            index_file = os.path.join(index_path, "index.faiss")
            if os.path.exists(index_file):
                index_size = os.path.getsize(index_file) / 1024
                st.success(f"✅ 索引已存在 ({index_size:.1f} KB)")
        else:
            st.warning("⚠️ 索引未构建")
        
        if st.button("🔄 开始重建索引", use_container_width=True, type="primary"):
            if not os.path.exists(docs_path) or not os.listdir(docs_path):
                st.error("❌ 错误: knowledge_base_docs 目录为空或不存在")
            else:
                try:
                    with st.spinner("正在构建知识库索引，请稍候..."):
                        # 导入必要的模块
                        from langchain_community.document_loaders import DirectoryLoader, TextLoader
                        from langchain.text_splitter import RecursiveCharacterTextSplitter
                        from langchain_community.embeddings import SentenceTransformerEmbeddings
                        from langchain_community.vectorstores import FAISS
                        
                        # 1. 加载文档
                        loader = DirectoryLoader(
                            docs_path, 
                            glob="**/*.txt", 
                            loader_cls=TextLoader,
                            loader_kwargs={'encoding': 'utf-8'}
                        )
                        documents = loader.load()
                        
                        if not documents:
                            st.error("❌ 未找到任何文档")
                        else:
                            # 2. 文档分块
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000, 
                                chunk_overlap=200
                            )
                            docs = text_splitter.split_documents(documents)
                            
                            # 3. 生成向量并构建索引
                            embeddings = SentenceTransformerEmbeddings(
                                model_name="paraphrase-multilingual-MiniLM-L12-v2"
                            )
                            vectorstore = FAISS.from_documents(docs, embeddings)
                            
                            # 4. 保存索引
                            vectorstore.save_local(index_path)
                            
                            st.success(f"✅ 索引构建成功！")
                            st.info(f"📊 处理了 {len(documents)} 个文档，分割成 {len(docs)} 个块")
                            st.balloons()
                            
                            # 清除缓存的 retriever
                            if 'retriever' in st.session_state:
                                del st.session_state.retriever
                            
                except Exception as e:
                    st.error(f"❌ 构建失败: {e}")
                    st.code(str(e), language="python")

# --- 页面渲染 ---
if st.session_state.page == "知识库问答":
    # 页面头部卡片 - 改为沉稳的深蓝灰色
    st.markdown("""
    <div style='background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0;'>💬 智能知识库问答</h2>
        <p style='color: rgba(255,255,255,0.9); margin: 5px 0 0 0;'>基于内部知识文档，提供精准的问答能力。如果文档无相关信息，将由大模型提供通用回答。</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        if not os.getenv("DASHSCOPE_API_KEY"):
            st.error("错误：请先设置 DASHSCOPE_API_KEY 环境变量。")
            st.code("export DASHSCOPE_API_KEY='您的key'", language="shell")
        else:
            # 初始化会话状态和开场白
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": JA_ASSISTANT_INTRO}]

            # 创建一个容器来展示聊天记录
            chat_box = st.container(height=400)

            # 在容器中显示历史消息
            for message in st.session_state.messages:
                with chat_box.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # 聊天输入框
            if prompt := st.chat_input("请输入您的问题..."):
                # 将用户消息添加到历史并立即显示在容器中
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_box.chat_message("user"):
                    st.markdown(prompt)
                
                # 获取并显示助手的回复
                with chat_box.chat_message("assistant"):
                    with st.spinner("正在思考..."):
                        if 'retriever' not in st.session_state:
                            with st.spinner("正在初始化知识库，请稍候..."):
                                st.session_state.retriever = get_retriever()
                        retriever = st.session_state.retriever

                        if not retriever:
                            st.error("知识库索引未找到或加载失败！请检查faiss_index目录或运行 `build_knowledge_base.py`。")
                            st.stop()

                        # RAG逻辑...
                        with st.spinner("正在检索知识库并生成回答..."):
                            relevant_docs = retriever.get_relevant_documents(prompt)
                        use_rag = False
                        if relevant_docs:
                            context_string = "\n\n".join(doc.page_content for doc in relevant_docs)
                            validate_messages = [
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": f"仅根据以下上下文：\n\n{context_string}\n\n判断是否可以回答这个问题：'{prompt}'？请只回答'是'或'否'。"}
                            ]
                            validation_result = call_qwen_model(validate_messages)
                            if "是" in validation_result:
                                use_rag = True
                        
                        if use_rag:
                            st.info("✅ AI判断信息相关，将基于知识库回答...")
                            system_content = f"{JA_ASSISTANT_PERSONA} 请严格根据以下上下文来回答问题，回答时可以对信息进行总结和组织，但不要超出上下文范围:\n{context_string}"
                            messages = [{"role": "system", "content": system_content}, {"role": "user", "content": prompt}]
                            response = call_qwen_model(messages)
                        else:
                            st.warning("⚠️ AI判断知识库中无直接相关信息，将使用通用知识回答...")
                            messages = [{"role": "system", "content": JA_ASSISTANT_GENERAL_KNOWLEDGE_PERSONA}, {"role": "user", "content": prompt}]
                            response = call_qwen_model(messages)
                        
                        st.markdown(response)
                        # 将助手回复也添加到会话状态
                        st.session_state.messages.append({"role": "assistant", "content": response})

elif st.session_state.page == "文献检索":
    # 页面头部卡片 - 改为沉稳的深绿色
    st.markdown("""
    <div style='background: linear-gradient(135deg, #16a085 0%, #1abc9c 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0;'>📰 最新科研文献追踪</h2>
        <p style='color: rgba(255,255,255,0.9); margin: 5px 0 0 0;'>输入关键词，AI将自动从arXiv上检索最新的相关论文，并生成简报。</p>
    </div>
    """, unsafe_allow_html=True)

    # 初始化AI摘要的状态存储
    if 'ai_summaries' not in st.session_state:
        st.session_state.ai_summaries = {}
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None

    with st.container(border=True):
        # 使用列来布局输入框和选择器
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            keywords_input = st.text_input(
                "请输入关键词（多个请用英文逗号,隔开）:", 
                value="perovskite stability, CsPbI3",
                help="例如: perovskite solar cell, ETL, device stability"
            )
        with col2:
            date_range = st.selectbox(
                "时间范围",
                ("all_time", "last_month", "last_3_months", "last_year"),
                format_func=lambda x: {
                    "all_time": "所有时间",
                    "last_month": "最近一个月",
                    "last_3_months": "最近三个月",
                    "last_year": "最近一年"
                }.get(x),
            )
        with col3:
            sort_by = st.selectbox(
                "排序方式",
                ("Relevance", "SubmittedDate"),
                index=1, # 默认选择“最新发表”
                format_func=lambda x: {"Relevance": "相关度", "SubmittedDate": "最新发表"}.get(x)
            )

        if st.button("开始检索", use_container_width=True):
            # 清空之前的AI摘要和结果
            st.session_state.ai_summaries = {}
            st.session_state.search_results = None
            st.session_state.search_error = None
            if keywords_input:
                keywords_list = [keyword.strip() for keyword in keywords_input.split(',') if keyword.strip()]
                with st.spinner(f"正在从arXiv检索: {', '.join(keywords_list)}...\n请耐心等待，检索可能需要一些时间。"):
                    papers, error = get_latest_papers(keywords_list, date_range, sort_by)
                    # 将结果存储在session state中，以便在按钮点击后保留
                    st.session_state.search_results = papers
                    st.session_state.search_error = error
            else:
                st.warning("请输入关键词。")

    # 在主按钮逻辑外部渲染结果，以支持AI总结按钮的交互
    if st.session_state.search_results:
        papers = st.session_state.search_results
        st.success(f"检索完成！共找到 {len(papers)} 篇相关论文。")
        
        for i, paper in enumerate(papers):
            with st.expander(f"**{i+1}. {paper['title']}**", expanded=True):
                st.markdown(f"**发表日期:** {paper['published']} | **作者:** {paper['authors']}")
                st.markdown(f"**摘要:** {paper['summary']}")
                
                # 功能按钮 - 优化布局
                col1, col2, col3 = st.columns(3, gap="small")
                with col1:
                    st.link_button("阅读原文", paper['pdf_url'], use_container_width=True)
                with col2:
                    if st.button("AI总结", key=f"summarize_{paper['entry_id']}", use_container_width=True):
                        with st.spinner("AI正在阅读摘要，请稍候..."):
                            ai_summary = summarize_with_ai(paper['summary'])
                            st.session_state.ai_summaries[paper['entry_id']] = ai_summary
                with col3:
                    if st.button("深入研究", key=f"research_{paper['entry_id']}", use_container_width=True):
                        st.toast("该功能正在开发中...")

                # 如果存在AI总结，则显示它
                if paper['entry_id'] in st.session_state.ai_summaries:
                    st.info(f"{st.session_state.ai_summaries[paper['entry_id']]}")

    elif st.session_state.get('search_error'):
        st.error(st.session_state.search_error)
    # 只有在按钮被点击后，search_results才会被定义，所以需要检查
    elif st.session_state.get('search_results') is not None and not st.session_state.get('search_results'):
        st.warning("在选定时间范围内，未找到与您关键词相关的新论文。")

elif st.session_state.page == "XRD分析":
    # 页面头部卡片 - 改为沉稳的深蓝色
    st.markdown("""
    <div style='background: linear-gradient(135deg, #2980b9 0%, #3498db 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0;'>📈 XRD数据自动分析</h2>
        <p style='color: rgba(255,255,255,0.9); margin: 5px 0 0 0;'>上传您的原始XRD数据文件（txt或csv格式），AI将自动绘制图谱并识别主要衍射峰。</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "请在此处上传您的XRD数据文件", 
            type=["txt", "csv"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            with st.spinner("正在分析图谱..."):
                fig = analyze_xrd_from_upload(uploaded_file)
                if fig:
                    st.pyplot(fig)
                    st.success("图谱生成完毕！")

elif st.session_state.page == "性能预测":
    # 页面头部卡片 - 改为沉稳的深绿灰色
    st.markdown("""
    <div style='background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0;'>💡 材料性能预测</h2>
        <p style='color: rgba(255,255,255,0.9); margin: 5px 0 0 0;'>调整以下实验参数，AI模型将预测对应的光电转换效率。</p>
    </div>
    """, unsafe_allow_html=True)
    
    model = get_trained_model()
    if model:
        # 使用带边框的容器来组织UI
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                rpm = st.slider("旋涂速度 (rpm)", 2000, 6000, 4000, 100)
            with col2:
                temp = st.slider("退火温度 (°C)", 80, 140, 100, 5)
            with col3:
                conc = st.slider("添加剂浓度 (%)", 0.1, 2.0, 1.0, 0.1)
        
        # 将按钮和结果也放在一个容器中
        with st.container(border=True):
            if st.button("执行预测", use_container_width=True):
                new_params = pd.DataFrame({
                    'spin_coating_rpm': [rpm],
                    'annealing_temperature_C': [temp],
                    'additive_concentration_percent': [conc]
                })
                prediction = model.predict(new_params)
                st.metric(label="预测效率", value=f"{prediction[0]:.2f} %")
    else:
        st.error("数据文件 'simulated_experimental_data.csv' 不存在，无法进行预测。")

elif st.session_state.page == "实验优化":
    # 页面头部卡片 - 改为沉稳的深橙灰色
    st.markdown("""
    <div style='background: linear-gradient(135deg, #d35400 0%, #e67e22 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0;'>🚀 实验方案优化</h2>
        <p style='color: rgba(255,255,255,0.9); margin: 5px 0 0 0;'>AI将搜索多种参数组合，为您推荐能产生最高效率的'最佳实验方案'。</p>
    </div>
    """, unsafe_allow_html=True)
    
    model = get_trained_model()
    if model:
        with st.container(border=True):
            st.info("请注意：为保证快速响应，演示版的搜索空间较小。在获取更多真实数据后，可扩展搜索范围以获得更优结果。")
            if st.button("开始优化，寻找最佳参数", use_container_width=True):
                with st.spinner("正在进行网格搜索优化..."):
                    params, eff = find_optimal_params(model)
                    st.success("优化完成！")
                    st.metric(label="最高预测效率", value=f"{eff:.2f} %")
                    
                    st.write("AI推荐的最佳实验参数组合为:")
                    st.table(params)
    else:
        st.error("数据文件 'simulated_experimental_data.csv' 不存在，无法进行优化。")
