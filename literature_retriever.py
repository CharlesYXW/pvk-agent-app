import os
import arxiv
import datetime
import time

# --- 配置区 ---
KEYWORDS = ["perovskite stability", "CsPbI3"]
MAX_RESULTS_PER_KEYWORD = 5 # 每个关键词检索的最大数量
OUTPUT_DIR = "research_briefings/"
RETRY_ATTEMPTS = 3  # 重试次数
RETRY_DELAY = 2     # 重试延迟（秒）
# --- 结束配置 ---

def search_arxiv_with_retry(query):
    """
    带重试机制的 arXiv 搜索函数。
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            print(f"正在搜索关键词: '{query}' (尝试 {attempt + 1}/{RETRY_ATTEMPTS})")
            
            # 配置 arxiv 的连接参数
            search = arxiv.Search(
                query=query,
                max_results=MAX_RESULTS_PER_KEYWORD,
                sort_by=arxiv.SortCriterion.LastUpdatedDate
            )
            
            # 尝试获取结果
            results = []
            for result in search.results():
                results.append(result)
            
            print(f"成功检索到 {len(results)} 篇论文")
            return results
            
        except Exception as e:
            print(f"检索失败: {str(e)}")
            if attempt < RETRY_ATTEMPTS - 1:
                print(f"等待 {RETRY_DELAY} 秒后重试...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"关键词 '{query}' 检索失败，已达最大重试次数")
                return []

def search_and_generate_briefing():
    """
    在arXiv上搜索关键词，并生成Markdown格式的科研简报。
    """
    print("开始检索 arXiv 上的最新论文...")
    unique_papers = {}

    for query in KEYWORDS:
        results = search_arxiv_with_retry(query)
        for result in results:
            # 通过论文ID去重
            if result.entry_id not in unique_papers:
                unique_papers[result.entry_id] = result
        
        # 在两次搜索之间添加延迟，避免 API 限制
        if query != KEYWORDS[-1]:
            time.sleep(2)

    if not unique_papers:
        print("未发现新的相关论文。")
        return

    print(f"检索到 {len(unique_papers)} 篇不重复的论文。正在生成简报...")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 生成Markdown报告
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    report_filename = os.path.join(OUTPUT_DIR, f"科研简报_{today_str}.md")

    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(f"# 科研简报 ({today_str})\n\n")
        f.write(f"本简报由AI助手自动生成，检索了以下关键词的最新研究：`{', '.join(KEYWORDS)}`。\n\n---\n\n")

        for i, paper in enumerate(unique_papers.values()):
            f.write(f"## {i+1}. {paper.title}\n\n")
            f.write(f"- **作者:** {', '.join(author.name for author in paper.authors)}\n")
            f.write(f"- **发布日期:** {paper.published.date()}\n")
            f.write(f"- **链接:** {paper.pdf_url}\n\n")
            f.write("**摘要:**\n")
            # 注意：当前版本直接使用原文摘要。未来可集成LLM进行深度总结。
            summary = paper.summary.replace('\n', ' ')
            f.write(f"> {summary}\n\n")
            f.write("---\n\n")

    print(f"科研简报已成功生成: '{report_filename}'")

if __name__ == "__main__":
    search_and_generate_briefing()
