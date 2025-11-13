#!/usr/bin/env python3
"""
文献订阅自动检查脚本
可以通过 cron 任务定时运行，自动检查所有订阅的更新
"""

import sys
import datetime
from literature_subscription import get_subscription_manager, format_notification

def main():
    """主函数：检查所有订阅并生成报告"""
    print("=" * 70)
    print(f"📬 文献订阅自动检查")
    print(f"🕐 检查时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    # 获取订阅管理器
    sub_manager = get_subscription_manager()
    
    # 获取所有启用的订阅
    subscriptions = sub_manager.get_subscriptions(enabled_only=True)
    
    if not subscriptions:
        print("ℹ️  当前没有启用的订阅。")
        return
    
    print(f"📋 找到 {len(subscriptions)} 个启用的订阅")
    print()
    
    # 检查所有订阅
    results = sub_manager.check_all_subscriptions(days_back=1)
    
    total_papers = 0
    notifications = []
    
    for sub in subscriptions:
        sub_id = sub['id']
        papers, error = results.get(sub_id, ([], "未检查"))
        
        print(f"🔍 [{sub['name']}]")
        print(f"   关键词: {', '.join(sub['keywords'])}")
        
        if error:
            print(f"   ❌ 错误: {error}")
        elif papers:
            print(f"   ✅ 发现 {len(papers)} 篇新论文")
            total_papers += len(papers)
            
            # 生成通知
            notification = format_notification(sub, papers)
            notifications.append(notification)
            
            # 显示前3篇
            for i, paper in enumerate(papers[:3], 1):
                print(f"      {i}. {paper['title'][:60]}...")
            
            if len(papers) > 3:
                print(f"      ... 还有 {len(papers) - 3} 篇")
        else:
            print(f"   ℹ️  暂无新论文")
        
        print()
    
    # 总结
    print("=" * 70)
    print(f"📊 检查完成")
    print(f"   - 检查订阅数: {len(subscriptions)}")
    print(f"   - 发现论文数: {total_papers}")
    print("=" * 70)
    
    # 如果有新论文，保存通知报告
    if notifications:
        report_file = f"subscription_report_{datetime.date.today()}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"文献订阅更新报告\n")
            f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            for notification in notifications:
                f.write(notification)
                f.write("\n" + "=" * 70 + "\n\n")
        
        print(f"\n📄 报告已保存: {report_file}")
        print("\n💡 提示：您可以通过邮件或其他方式发送此报告")
    
    return total_papers


if __name__ == "__main__":
    try:
        total_papers = main()
        sys.exit(0 if total_papers >= 0 else 1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
