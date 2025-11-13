"""
文献订阅管理模块
提供订阅管理、定时检索、更新通知等功能
"""

import os
import json
import datetime
import arxiv
from typing import List, Dict, Optional, Tuple
import time

# 配置
SUBSCRIPTIONS_FILE = "subscriptions.json"
SUBSCRIPTION_HISTORY_FILE = "subscription_history.json"
CHECK_INTERVAL_HOURS = 24  # 检查更新的时间间隔（小时）

# arXiv 检索配置
ARXIV_RETRY_ATTEMPTS = 3
ARXIV_RETRY_DELAY = 1
ARXIV_KEYWORD_DELAY = 0.5


class SubscriptionManager:
    """文献订阅管理器"""
    
    def __init__(self):
        self.subscriptions_file = SUBSCRIPTIONS_FILE
        self.history_file = SUBSCRIPTION_HISTORY_FILE
        self.subscriptions = self._load_subscriptions()
        self.history = self._load_history()
    
    def _load_subscriptions(self) -> Dict:
        """加载订阅配置"""
        if os.path.exists(self.subscriptions_file):
            try:
                with open(self.subscriptions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载订阅配置失败: {e}")
                return {}
        return {}
    
    def _save_subscriptions(self):
        """保存订阅配置"""
        try:
            with open(self.subscriptions_file, 'w', encoding='utf-8') as f:
                json.dump(self.subscriptions, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存订阅配置失败: {e}")
            return False
    
    def _load_history(self) -> Dict:
        """加载检索历史"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载历史记录失败: {e}")
                return {}
        return {}
    
    def _save_history(self):
        """保存检索历史"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存历史记录失败: {e}")
            return False
    
    def add_subscription(self, name: str, keywords: List[str], enabled: bool = True) -> bool:
        """添加订阅"""
        if not name or not keywords:
            return False
        
        # 生成唯一ID
        sub_id = f"sub_{int(datetime.datetime.now().timestamp())}"
        
        self.subscriptions[sub_id] = {
            "id": sub_id,
            "name": name,
            "keywords": keywords,
            "enabled": enabled,
            "created_at": datetime.datetime.now().isoformat(),
            "last_checked": None,
            "notification_count": 0
        }
        
        return self._save_subscriptions()
    
    def remove_subscription(self, sub_id: str) -> bool:
        """删除订阅"""
        if sub_id in self.subscriptions:
            del self.subscriptions[sub_id]
            return self._save_subscriptions()
        return False
    
    def update_subscription(self, sub_id: str, name: Optional[str] = None, 
                          keywords: Optional[List[str]] = None, 
                          enabled: Optional[bool] = None) -> bool:
        """更新订阅"""
        if sub_id not in self.subscriptions:
            return False
        
        if name is not None:
            self.subscriptions[sub_id]["name"] = name
        if keywords is not None:
            self.subscriptions[sub_id]["keywords"] = keywords
        if enabled is not None:
            self.subscriptions[sub_id]["enabled"] = enabled
        
        return self._save_subscriptions()
    
    def get_subscriptions(self, enabled_only: bool = False) -> List[Dict]:
        """获取所有订阅"""
        subs = list(self.subscriptions.values())
        if enabled_only:
            subs = [s for s in subs if s.get("enabled", True)]
        return subs
    
    def get_subscription(self, sub_id: str) -> Optional[Dict]:
        """获取单个订阅"""
        return self.subscriptions.get(sub_id)
    
    def check_for_updates(self, sub_id: str, days_back: int = 1) -> Tuple[List[Dict], Optional[str]]:
        """检查订阅的更新"""
        subscription = self.get_subscription(sub_id)
        if not subscription:
            return [], "订阅不存在"
        
        keywords = subscription["keywords"]
        papers = []
        
        # 计算时间范围
        end_date = datetime.datetime.now(datetime.timezone.utc)
        start_date = end_date - datetime.timedelta(days=days_back)
        
        # 检索论文
        for keyword_idx, keyword in enumerate(keywords):
            for attempt in range(ARXIV_RETRY_ATTEMPTS):
                try:
                    # 构建查询，只检索最近的论文
                    start_date_str = start_date.strftime("%Y%m%d%H%M")
                    end_date_str = end_date.strftime("%Y%m%d%H%M")
                    date_query = f" AND submittedDate:[{start_date_str} TO {end_date_str}]"
                    
                    search = arxiv.Search(
                        query=f"({keyword}){date_query}",
                        max_results=10,
                        sort_by=arxiv.SortCriterion.SubmittedDate
                    )
                    
                    for result in search.results():
                        # 确保 pdf_url 有效
                        pdf_url = str(result.pdf_url) if result.pdf_url else ""
                        if not pdf_url.startswith('http'):
                            pdf_url = f"https://arxiv.org/abs/{result.entry_id.split('/abs/')[-1]}"
                        
                        papers.append({
                            "entry_id": result.entry_id,
                            "title": result.title,
                            "authors": ', '.join(author.name for author in result.authors),
                            "pdf_url": pdf_url,
                            "summary": result.summary.replace('\n', ' '),
                            "published": result.published.strftime('%Y-%m-%d'),
                            "keyword": keyword  # 记录是哪个关键词检索到的
                        })
                    
                    break  # 成功后跳出重试循环
                    
                except Exception as e:
                    if attempt < ARXIV_RETRY_ATTEMPTS - 1:
                        time.sleep(ARXIV_RETRY_DELAY)
                    else:
                        return [], f"检索失败: {str(e)}"
            
            # 关键词间延迟
            if keyword_idx < len(keywords) - 1:
                time.sleep(ARXIV_KEYWORD_DELAY)
        
        # 去重（按 entry_id）
        unique_papers = {}
        for paper in papers:
            if paper["entry_id"] not in unique_papers:
                unique_papers[paper["entry_id"]] = paper
        
        papers = list(unique_papers.values())
        
        # 更新订阅的最后检查时间
        self.subscriptions[sub_id]["last_checked"] = datetime.datetime.now().isoformat()
        if papers:
            self.subscriptions[sub_id]["notification_count"] = self.subscriptions[sub_id].get("notification_count", 0) + len(papers)
        self._save_subscriptions()
        
        # 保存到历史记录
        if papers:
            if sub_id not in self.history:
                self.history[sub_id] = []
            
            self.history[sub_id].append({
                "check_time": datetime.datetime.now().isoformat(),
                "paper_count": len(papers),
                "papers": papers
            })
            self._save_history()
        
        return papers, None
    
    def get_update_history(self, sub_id: str, limit: int = 10) -> List[Dict]:
        """获取订阅的更新历史"""
        if sub_id not in self.history:
            return []
        
        history = self.history[sub_id]
        return history[-limit:] if limit > 0 else history
    
    def check_all_subscriptions(self, days_back: int = 1) -> Dict[str, Tuple[List[Dict], Optional[str]]]:
        """检查所有启用的订阅"""
        results = {}
        subscriptions = self.get_subscriptions(enabled_only=True)
        
        for sub in subscriptions:
            papers, error = self.check_for_updates(sub["id"], days_back)
            results[sub["id"]] = (papers, error)
        
        return results
    
    def get_statistics(self) -> Dict:
        """获取订阅统计信息"""
        total_subs = len(self.subscriptions)
        enabled_subs = len([s for s in self.subscriptions.values() if s.get("enabled", True)])
        total_papers = sum(len(h) for h in self.history.values())
        
        # 获取最近检查时间，过滤掉空字符串
        last_checks = [s.get("last_checked", "") for s in self.subscriptions.values()]
        valid_checks = [check for check in last_checks if check]  # 过滤空字符串
        last_check = max(valid_checks) if valid_checks else "从未检查"
        
        return {
            "total_subscriptions": total_subs,
            "enabled_subscriptions": enabled_subs,
            "total_papers_found": total_papers,
            "last_check": last_check
        }
    
    def get_unread_updates_count(self) -> int:
        """获取未读更新数量（最近一次检查的论文数）"""
        unread_count = 0
        for sub_id, history_list in self.history.items():
            if history_list:
                # 获取最近一次检查的论文数
                latest_check = history_list[-1]
                unread_count += latest_check.get('paper_count', 0)
        return unread_count
    
    def mark_as_read(self):
        """标记为已读（清空历史记录）"""
        # 注：这里不删除历史，只是为未来的已读/未读功能预留接口
        pass


def format_notification(subscription: Dict, papers: List[Dict]) -> str:
    """格式化通知消息"""
    if not papers:
        return f"订阅 '{subscription['name']}' 暂无新论文。"
    
    msg = f"📬 订阅 '{subscription['name']}' 发现 {len(papers)} 篇新论文：\n\n"
    
    for i, paper in enumerate(papers[:5], 1):  # 只显示前5篇
        msg += f"{i}. **{paper['title']}**\n"
        msg += f"   - 发表: {paper['published']}\n"
        msg += f"   - 作者: {paper['authors'][:100]}...\n\n"
    
    if len(papers) > 5:
        msg += f"... 还有 {len(papers) - 5} 篇论文\n"
    
    return msg


# 单例实例
_subscription_manager = None

def get_subscription_manager() -> SubscriptionManager:
    """获取订阅管理器单例"""
    global _subscription_manager
    if _subscription_manager is None:
        _subscription_manager = SubscriptionManager()
    return _subscription_manager
