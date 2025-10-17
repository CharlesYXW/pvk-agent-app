import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# 定义常量
DOCS_PATH = "knowledge_base_docs/"
INDEX_PATH = "faiss_index"

def build_knowledge_base():
    """
    加载文档，进行分块，生成向量，并构建FAISS索引。
    """
    # 1. 加载文档
    print("正在加载文档...")
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    if not documents:
        print("错误：在 'knowledge_base_docs' 文件夹中未找到任何文档。")
        return

    print(f"成功加载 {len(documents)} 篇文档。")

    # 2. 文档分块
    print("正在进行文档分块...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"文档被分割成 {len(docs)} 个块。")

    # 3. 生成向量并构建索引
    print("正在生成文本向量并构建FAISS索引...")
    # 使用一个强大的多语言模型，以获得更好的中文处理效果
    embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

    vectorstore = FAISS.from_documents(docs, embeddings)
    print("索引构建完成。")

    # 4. 保存索引到本地
    print(f"正在将索引保存到 '{INDEX_PATH}'...")
    vectorstore.save_local(INDEX_PATH)
    print("知识库构建完成并已成功保存。")

if __name__ == "__main__":
    build_knowledge_base()