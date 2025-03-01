import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 设置页面标题
st.set_page_config(page_title="线性回归正则化演示", layout="wide")

# 页面标题
st.title("线性回归过拟合问题与正则化方法演示")

# 侧边栏参数设置
st.sidebar.header("参数设置")

# 数据生成参数
n_samples = st.sidebar.slider("样本数量", 50, 500, 100)
n_features = st.sidebar.slider("特征数量", 10, 100, 50)
n_informative = st.sidebar.slider("有效特征数量", 5, 20, 10)
noise = st.sidebar.slider("噪声水平", 0.1, 5.0, 1.0)
test_size = st.sidebar.slider("测试集比例", 0.1, 0.5, 0.2)

# 正则化参数
alpha_ridge = st.sidebar.slider("岭回归正则化参数 (alpha)", 0.01, 10.0, 1.0, 0.01)
alpha_lasso = st.sidebar.slider("Lasso回归正则化参数 (alpha)", 0.01, 1.0, 0.1, 0.01)

# 生成模拟数据
@st.cache_data
def generate_data(n_samples, n_features, n_informative, noise, test_size):
    # 生成特征矩阵
    X = np.random.randn(n_samples, n_features)
    
    # 生成真实系数向量 - 只有n_informative个非零系数
    true_coef = np.zeros(n_features)
    true_coef[:n_informative] = np.random.randn(n_informative) * 5
    
    # 生成目标变量
    y = np.dot(X, true_coef) + np.random.normal(0, noise, size=n_samples)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, true_coef

# 生成数据
X_train, X_test, y_train, y_test, true_coef = generate_data(n_samples, n_features, n_informative, noise, test_size)

# 训练模型
def train_models(X_train, y_train, X_test, y_test, alpha_ridge, alpha_lasso):
    # 线性回归
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_train_pred = lr.predict(X_train)
    lr_test_pred = lr.predict(X_test)
    
    # 岭回归
    ridge = Ridge(alpha=alpha_ridge)
    ridge.fit(X_train, y_train)
    ridge_train_pred = ridge.predict(X_train)
    ridge_test_pred = ridge.predict(X_test)
    
    # Lasso回归
    lasso = Lasso(alpha=alpha_lasso, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_train_pred = lasso.predict(X_train)
    lasso_test_pred = lasso.predict(X_test)
    
    # 计算评估指标
    results = {
        "线性回归": {
            "系数": lr.coef_,
            "训练集R²": r2_score(y_train, lr_train_pred),
            "测试集R²": r2_score(y_test, lr_test_pred),
            "训练集MSE": mean_squared_error(y_train, lr_train_pred),
            "测试集MSE": mean_squared_error(y_test, lr_test_pred)
        },
        "岭回归": {
            "系数": ridge.coef_,
            "训练集R²": r2_score(y_train, ridge_train_pred),
            "测试集R²": r2_score(y_test, ridge_test_pred),
            "训练集MSE": mean_squared_error(y_train, ridge_train_pred),
            "测试集MSE": mean_squared_error(y_test, ridge_test_pred)
        },
        "Lasso回归": {
            "系数": lasso.coef_,
            "训练集R²": r2_score(y_train, lasso_train_pred),
            "测试集R²": r2_score(y_test, lasso_test_pred),
            "训练集MSE": mean_squared_error(y_train, lasso_train_pred),
            "测试集MSE": mean_squared_error(y_test, lasso_test_pred)
        }
    }
    
    return results

# 训练模型并获取结果
results = train_models(X_train, y_train, X_test, y_test, alpha_ridge, alpha_lasso)

# 显示模型性能
st.header("模型性能比较")

# 创建性能指标表格
performance_data = {
    "模型": ["线性回归", "岭回归", "Lasso回归"],
    "训练集R²": [results["线性回归"]["训练集R²"], results["岭回归"]["训练集R²"], results["Lasso回归"]["训练集R²"]],
    "测试集R²": [results["线性回归"]["测试集R²"], results["岭回归"]["测试集R²"], results["Lasso回归"]["测试集R²"]],
    "训练集MSE": [results["线性回归"]["训练集MSE"], results["岭回归"]["训练集MSE"], results["Lasso回归"]["训练集MSE"]],
    "测试集MSE": [results["线性回归"]["测试集MSE"], results["岭回归"]["测试集MSE"], results["Lasso回归"]["测试集MSE"]]
}

performance_df = pd.DataFrame(performance_data)
st.dataframe(performance_df)

# 可视化模型系数
st.header("模型系数比较")

col1, col2 = st.columns(2)

with col1:
    # 绘制系数对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 只显示前20个系数以便于可视化
    coef_display_count = min(20, n_features)
    
    x = np.arange(coef_display_count)
    width = 0.2
    
    ax.bar(x - width, true_coef[:coef_display_count], width, label='真实系数', color='green', alpha=0.7)
    ax.bar(x, results["线性回归"]["系数"][:coef_display_count], width, label='线性回归', color='blue', alpha=0.7)
    ax.bar(x + width, results["岭回归"]["系数"][:coef_display_count], width, label='岭回归', color='red', alpha=0.7)
    ax.bar(x + 2*width, results["Lasso回归"]["系数"][:coef_display_count], width, label='Lasso回归', color='purple', alpha=0.7)
    
    ax.set_xlabel('特征索引')
    ax.set_ylabel('系数值')
    ax.set_title('各模型系数对比 (前20个特征)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(coef_display_count)])
    ax.legend()
    
    st.pyplot(fig)

with col2:
    # 计算非零系数数量
    non_zero_counts = {
        "真实系数": np.sum(np.abs(true_coef) > 1e-10),
        "线性回归": np.sum(np.abs(results["线性回归"]["系数"]) > 1e-10),
        "岭回归": np.sum(np.abs(results["岭回归"]["系数"]) > 1e-10),
        "Lasso回归": np.sum(np.abs(results["Lasso回归"]["系数"]) > 1e-10)
    }
    
    # 绘制非零系数数量对比图
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = list(non_zero_counts.keys())
    counts = list(non_zero_counts.values())
    
    ax.bar(models, counts, color=['green', 'blue', 'red', 'purple'])
    ax.set_ylabel('非零系数数量')
    ax.set_title('各模型非零系数数量对比')
    
    # 在柱状图上显示具体数值
    for i, v in enumerate(counts):
        ax.text(i, v + 0.5, str(v), ha='center')
    
    st.pyplot(fig)

# 过拟合与正则化解释
st.header("过拟合问题与正则化解释")

st.markdown("""
### 过拟合问题
在高维数据中，线性回归容易出现过拟合问题。从上面的结果可以看出：
- 线性回归在**训练集**上表现很好，但在**测试集**上表现较差
- 线性回归估计了大量非零系数，而真实模型中只有少量特征是有效的

### 正则化方法
正则化通过对模型复杂度进行惩罚来减少过拟合：

#### 岭回归 (Ridge)
- 添加了L2正则化项：$\lambda \sum_{j=1}^{p} \beta_j^2$
- 将所有系数向零收缩，但不会使系数精确等于零
- 适用于多重共线性问题

#### Lasso回归
- 添加了L1正则化项：$\lambda \sum_{j=1}^{p} |\beta_j|$
- 可以将不重要的系数精确地压缩为零，实现特征选择
- 在高维稀疏数据中表现优异

### 结论
- 从上面的结果可以看出，正则化方法（岭回归和Lasso回归）在测试集上的表现优于普通线性回归
- Lasso回归能够识别出真正重要的特征，将大多数不相关特征的系数设为零
- 正则化参数α控制了正则化的强度，需要通过交叉验证等方法选择最优值
""")

# 添加交互式说明
st.header("交互式探索")
st.markdown("""
尝试调整侧边栏中的参数，观察不同设置下模型的表现：
- 增加特征数量或降低样本数量会加剧过拟合问题
- 调整正则化参数α可以观察其对模型性能和系数的影响
- 增加噪声水平会使模型拟合更加困难
""")

# 运行说明
st.sidebar.markdown("""
### 运行说明
1. 调整左侧参数
2. 观察右侧模型性能和系数变化
3. 理解正则化如何改善过拟合问题
""")

st.sidebar.info("提示：尝试增加特征数量或减少样本数量，观察过拟合现象如何变得更加明显。") 