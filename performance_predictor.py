import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- 配置区 ---
DATA_FILE = "simulated_experimental_data.csv"
# --- 结束配置 ---

def train_and_predict_efficiency():
    """
    加载实验数据，训练一个随机森林模型，并对新参数进行效率预测。
    """
    print(f"正在从 '{DATA_FILE}' 加载数据...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"错误: 未在 '{DATA_FILE}' 找到数据文件。")
        return

    # 定义特征 (X) 和目标 (y)
    features = ['spin_coating_rpm', 'annealing_temperature_C', 'additive_concentration_percent']
    target = 'efficiency_percent'

    X = df[features]
    y = df[target]

    # 将数据分为训练集和测试集（这是标准的机器学习实践）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("正在训练随机森林模型...")
    # 初始化并训练模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 在测试集上评估模型（可选，但能反映模型好坏）
    score = model.score(X_test, y_test)
    print(f"模型在测试数据上的 R^2 分数: {score:.3f}")
    print("-" * 30)

    # --- 使用模型进行预测 ---
    print("正在使用训练好的模型进行新一轮预测...")
    # 定义一组全新的、假设的实验参数
    new_parameters = {
        'spin_coating_rpm': [4100],
        'annealing_temperature_C': [99],
        'additive_concentration_percent': [1.0]
    }
    new_df = pd.DataFrame(new_parameters)

    print(f"为以下新参数预测效率:\n{new_df}")

    # 进行效率预测
    predicted_efficiency = model.predict(new_df)

    print("-" * 30)
    print(f"预测效率为: {predicted_efficiency[0]:.2f} %")
    print("-" * 30)


if __name__ == "__main__":
    train_and_predict_efficiency()
