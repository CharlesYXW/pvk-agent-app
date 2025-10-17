import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import itertools

# --- 配置区 ---
DATA_FILE = "simulated_experimental_data.csv"
# --- 结束配置 ---

def optimize_experimental_parameters():
    """
    在完整数据集上训练模型，并执行网格搜索以找到最大化效率的最佳参数。
    """
    print(f"正在从 '{DATA_FILE}' 加载数据...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"错误: 未在 '{DATA_FILE}' 找到数据文件。")
        return

    # --- 模型训练 ---
    print("正在完整数据集上训练随机森林模型...")
    features = ['spin_coating_rpm', 'annealing_temperature_C', 'additive_concentration_percent']
    target = 'efficiency_percent'
    X_full = df[features]
    y_full = df[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_full, y_full)
    print("模型训练完成。")
    print("-" * 30)

    # --- 网格搜索优化 ---
    print("开始通过网格搜索寻找最优参数...")

    # 定义参数的搜索空间
    spin_speeds = np.arange(2500, 5501, 100)       # 旋涂速度 (rpm)
    temperatures = np.arange(80, 121, 5)          # 退火温度 (C)
    concentrations = np.arange(0.5, 1.51, 0.1)    # 添加剂浓度 (%)

    # 创建所有可能的参数组合
    param_grid = list(itertools.product(spin_speeds, temperatures, concentrations))
    grid_df = pd.DataFrame(param_grid, columns=features)
    
    total_combinations = len(grid_df)
    print(f"将要搜索 {total_combinations} 种不同的参数组合...")

    # 为所有组合预测效率
    predicted_efficiencies = model.predict(grid_df)

    # 找到最佳结果的索引
    best_index = np.argmax(predicted_efficiencies)
    best_parameters = grid_df.iloc[best_index]
    best_predicted_efficiency = predicted_efficiencies[best_index]

    print("-" * 30)
    print("优化完成！")
    print(f"找到的最高预测效率: {best_predicted_efficiency:.2f} %")
    print("对应的最佳实验参数组合为:")
    print(best_parameters)
    print("-" * 30)


if __name__ == "__main__":
    optimize_experimental_parameters()
