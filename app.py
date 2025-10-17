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

# LangChain相关的库（仅用于检索）
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# DashScope官方SDK
from dashscope import Generation

# --- 核心功能逻辑 ---

@st.cache_resource
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

@st.cache_data
def get_latest_papers(keywords):
    """根据关键词从arXiv检索最新的论文，返回结构化数据列表。"""
    if not keywords or not any(keywords):
        return [], "请输入至少一个关键词。"

    MAX_RESULTS_PER_KEYWORD = 3
    unique_papers = {}
    for query in keywords:
        try:
            search = arxiv.Search(query=query, max_results=MAX_RESULTS_PER_KEYWORD, sort_by=arxiv.SortCriterion.LastUpdatedDate)
            for result in search.results():
                if result.entry_id not in unique_papers:
                    unique_papers[result.entry_id] = {
                        "title": result.title,
                        "authors": ', '.join(author.name for author in result.authors),
                        "pdf_url": result.pdf_url,
                        "summary": result.summary.replace('\n', ' ')
                    }
        except Exception as e:
            return [], f"检索时出错: {e}"
    
    if not unique_papers:
        return [], "未找到与您关键词相关的新论文。"
        
    return list(unique_papers.values()), None

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
    </style>
    """,
    unsafe_allow_html=True)
st.title("🔬 钙钛矿研发智能助手")

# --- 导航 ---
with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", use_container_width=True)
    st.markdown("<h1 style='text-align: center; font-size: 24px;'>功能导航</h1>", unsafe_allow_html=True)
    if 'page' not in st.session_state: st.session_state.page = "知识库问答"
    def set_page(page_name): st.session_state.page = page_name
    st.button("知识库问答", on_click=set_page, args=("知识库问答",), use_container_width=True)
    st.button("文献检索", on_click=set_page, args=("文献检索",), use_container_width=True)
    st.button("XRD分析", on_click=set_page, args=("XRD分析",), use_container_width=True)
    st.button("性能预测", on_click=set_page, args=("性能预测",), use_container_width=True)
    st.button("实验优化", on_click=set_page, args=("实验优化",), use_container_width=True)

# --- 页面渲染 ---
if st.session_state.page == "知识库问答":
    st.header("💬 智能知识库问答 (RAG + Qwen)")
    st.markdown("基于内部知识文档，提供精准的问答能力。如果文档无相关信息，将由大模型提供通用回答。")

    with st.container(border=True):
        if not os.getenv("DASHSCOPE_API_KEY"):
            st.error("错误：请先设置 DASHSCOPE_API_KEY 环境变量。")
            st.code("export DASHSCOPE_API_KEY='您的key'", language="shell")
        else:
            # 初始化会话状态
            if "messages" not in st.session_state:
                st.session_state.messages = []

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
                        retriever = get_retriever()
                        if not retriever:
                            st.error("知识库索引未找到！请先运行 `build_knowledge_base.py` 来创建知识库。")
                            st.stop()

                        # RAG逻辑...
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
                            system_content = f"请仅根据以下上下文来回答问题，回答时可以对信息进行总结和组织，但不要超出上下文范围:\n{context_string}"
                            messages = [{"role": "system", "content": system_content}, {"role": "user", "content": prompt}]
                            response = call_qwen_model(messages)
                        else:
                            st.warning("⚠️ AI判断知识库中无直接相关信息，将使用通用知识回答...")
                            messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
                            response = call_qwen_model(messages)
                        
                        st.markdown(response)
                        # 将助手回复也添加到会话状态
                        st.session_state.messages.append({"role": "assistant", "content": response})

elif st.session_state.page == "文献检索":
    st.header("📰 最新科研文献追踪")
    st.markdown("输入关键词，AI将自动从arXiv上检索最新的相关论文，并生成简报。")

    with st.container(border=True):
        keywords_input = st.text_input(
            "请输入关键词（多个请用英文逗号,隔开）:", 
            value="perovskite stability, CsPbI3",
            help="例如: perovskite solar cell, ETL, device stability"
        )
        
        if st.button("开始检索", use_container_width=True):
            if keywords_input:
                keywords_list = [keyword.strip() for keyword in keywords_input.split(',') if keyword.strip()]
                with st.spinner(f"正在从arXiv检索: {', '.join(keywords_list)}..."):
                    papers, error = get_latest_papers(keywords_list)
                    if error:
                        st.error(error)
                    elif not papers:
                        st.warning("未找到与您关键词相关的新论文。")
                    else:
                        st.success(f"检索完成！共找到 {len(papers)} 篇相关论文。")
                        
                        for i, paper in enumerate(papers):
                            with st.expander(f"**{i+1}. {paper['title']}**", expanded=True):
                                st.markdown(f"**作者:** {paper['authors']}")
                                st.markdown(f"**摘要:** {paper['summary']}")
                                st.link_button("阅读原文 (PDF)", paper['pdf_url'])
            else:
                st.warning("请输入关键词。")

elif st.session_state.page == "XRD分析":
    st.header("📈 XRD数据自动分析")
    st.markdown("上传您的原始XRD数据文件（txt或csv格式），AI将自动绘制图谱并识别主要衍射峰。")

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
    st.header("💡 材料性能预测")
    st.markdown("调整以下实验参数，AI模型将预测对应的光电转换效率。")
    
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
    st.header("🚀 实验方案优化")
    st.markdown("AI将搜索多种参数组合，为您推荐能产生最高效率的‘最佳实验方案’。")
    
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
