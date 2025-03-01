import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 在文件开头导入matplotlib后添加字体设置
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']  # 宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 设置页面配置
st.set_page_config(page_title="多项式回归过拟合演示", layout="wide")

# 页面标题
st.title("多项式回归过拟合问题与正则化方法演示")

# 侧边栏参数设置
st.sidebar.header("参数设置")

# 数据生成参数
n_samples = st.sidebar.slider("样本数量", 20, 100, 30)
noise_level = st.sidebar.slider("噪声水平", 0.1, 3.0, 1.0)
test_size = st.sidebar.slider("测试集比例", 0.1, 0.5, 0.2)

# 模型参数
degree_low = st.sidebar.slider("低次多项式次数", 1, 5, 1)
degree_medium = st.sidebar.slider("中次多项式次数", 2, 10, 4)
degree_high = st.sidebar.slider("高次多项式次数", 5, 30, 15)

# 正则化参数
alpha_ridge = st.sidebar.slider("岭回归正则化参数 (alpha)", 0.0, 10.0, 1.0, 0.1)
alpha_lasso = st.sidebar.slider("Lasso回归正则化参数 (alpha)", 0.0, 1.0, 0.01, 0.01)

# 生成一元非线性数据
@st.cache_data
def generate_nonlinear_data(n_samples, noise_level, test_size):
    # 生成均匀分布的X
    X = np.linspace(0, 1, n_samples).reshape(-1, 1)
    
    # 生成真实的y（使用非线性函数）
    y_true = 20 + np.sin(4 * np.pi * X.ravel()) - 2 * X.ravel()**2
    
    # 添加噪声
    y = y_true + np.random.normal(0, noise_level, size=n_samples)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test, X, y_true

# 生成数据
X_train, X_test, y_train, y_test, X_full, y_true = generate_nonlinear_data(n_samples, noise_level, test_size)

# 定义函数来创建并训练多项式回归模型
def create_poly_model(degree, alpha=0, model_type='linear'):
    if model_type == 'linear':
        regressor = LinearRegression()
    elif model_type == 'ridge':
        regressor = Ridge(alpha=alpha)
    elif model_type == 'lasso':
        regressor = Lasso(alpha=alpha, max_iter=10000)
    
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('regressor', regressor)
    ])
    
    return model

# 训练模型并评估
def train_and_evaluate(degree, alpha=0, model_type='linear'):
    model = create_poly_model(degree, alpha, model_type)
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算误差
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    
    # 生成预测曲线的平滑点
    X_curve = np.linspace(0, 1, 100).reshape(-1, 1)
    y_curve = model.predict(X_curve)
    
    return train_error, test_error, X_curve, y_curve

# 训练三种不同次数的模型（无正则化）
train_error_low, test_error_low, X_curve_low, y_curve_low = train_and_evaluate(degree_low)
train_error_medium, test_error_medium, X_curve_medium, y_curve_medium = train_and_evaluate(degree_medium)
train_error_high, test_error_high, X_curve_high, y_curve_high = train_and_evaluate(degree_high)

# 训练正则化模型
train_error_ridge, test_error_ridge, X_curve_ridge, y_curve_ridge = train_and_evaluate(degree_high, alpha_ridge, 'ridge')
train_error_lasso, test_error_lasso, X_curve_lasso, y_curve_lasso = train_and_evaluate(degree_high, alpha_lasso, 'lasso')

# 创建可视化
st.header("多项式回归拟合效果展示")

# 设置三个图的布局
col1, col2, col3 = st.columns(3)

# 绘制低次多项式拟合
with col1:
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.scatter(X_train, y_train, color='blue', alpha=0.7, label='训练数据')
    ax1.plot(X_curve_low, y_curve_low, color='red', label=f'多项式拟合 (次数={degree_low})')
    ax1.set_title(f"多项式拟合 (次数 {degree_low})\n训练误差: {train_error_low:.2f}\n泛化误差: {test_error_low:.2f}")
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.text(0.05, 0.05, "欠拟合", transform=ax1.transAxes, fontsize=14, 
            bbox=dict(facecolor='white', alpha=0.8))
    st.pyplot(fig1)

# 绘制中次多项式拟合
with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.scatter(X_train, y_train, color='blue', alpha=0.7, label='训练数据')
    ax2.plot(X_curve_medium, y_curve_medium, color='red', label=f'多项式拟合 (次数={degree_medium})')
    ax2.set_title(f"多项式拟合 (次数 {degree_medium})\n训练误差: {train_error_medium:.2f}\n泛化误差: {test_error_medium:.2f}")
    ax2.set_xlabel('X')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.text(0.05, 0.05, "适当拟合", transform=ax2.transAxes, fontsize=14, 
            bbox=dict(facecolor='white', alpha=0.8))
    st.pyplot(fig2)

# 绘制高次多项式拟合
with col3:
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.scatter(X_train, y_train, color='blue', alpha=0.7, label='训练数据')
    ax3.plot(X_curve_high, y_curve_high, color='red', label=f'多项式拟合 (次数={degree_high})')
    ax3.set_title(f"多项式拟合 (次数 {degree_high})\n训练误差: {train_error_high:.2f}\n泛化误差: {test_error_high:.2f}")
    ax3.set_xlabel('X')
    ax3.set_ylabel('y')
    ax3.legend()
    ax3.text(0.05, 0.05, "过拟合", transform=ax3.transAxes, fontsize=14, 
            bbox=dict(facecolor='white', alpha=0.8))
    st.pyplot(fig3)

# 可视化训练误差和测试误差
st.header("训练误差与测试误差比较")

# 创建误差对比表格
error_data = {
    "模型": [f"多项式(次数={degree_low})", f"多项式(次数={degree_medium})", f"多项式(次数={degree_high})", 
             f"岭回归(次数={degree_high})", f"Lasso回归(次数={degree_high})"],
    "训练误差": [train_error_low, train_error_medium, train_error_high, train_error_ridge, train_error_lasso],
    "测试误差": [test_error_low, test_error_medium, test_error_high, test_error_ridge, test_error_lasso]
}
error_df = pd.DataFrame(error_data)
st.dataframe(error_df)

# 绘制误差vs多项式次数的关系
degree_range = range(1, 21)
train_errors = []
test_errors = []

for degree in degree_range:
    train_error, test_error, _, _ = train_and_evaluate(degree)
    train_errors.append(train_error)
    test_errors.append(test_error)

fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(degree_range, train_errors, 'o-', color='blue', label='训练误差')
ax4.plot(degree_range, test_errors, 'o-', color='red', label='测试误差')
ax4.set_xlabel('多项式次数')
ax4.set_ylabel('均方误差')
ax4.set_title('多项式次数与误差的关系')
ax4.legend()
ax4.grid(True)
st.pyplot(fig4)

# 正则化效果展示
st.header("正则化效果展示")

col1, col2 = st.columns(2)

# 绘制高次多项式的过拟合
with col1:
    fig5, ax5 = plt.subplots(figsize=(6, 5))
    ax5.scatter(X_train, y_train, color='blue', alpha=0.7, label='训练数据')
    ax5.plot(X_curve_high, y_curve_high, color='red', label=f'无正则化 (次数={degree_high})')
    ax5.set_title(f"高次多项式过拟合\n训练误差: {train_error_high:.2f}, 测试误差: {test_error_high:.2f}")
    ax5.set_xlabel('X')
    ax5.set_ylabel('y')
    ax5.legend()
    st.pyplot(fig5)

# 绘制正则化效果
with col2:
    fig6, ax6 = plt.subplots(figsize=(6, 5))
    ax6.scatter(X_train, y_train, color='blue', alpha=0.7, label='训练数据')
    ax6.plot(X_curve_ridge, y_curve_ridge, color='green', label=f'岭回归 (α={alpha_ridge})')
    ax6.plot(X_curve_lasso, y_curve_lasso, color='purple', label=f'Lasso回归 (α={alpha_lasso})')
    ax6.set_title(f"正则化改善过拟合\n岭回归测试误差: {test_error_ridge:.2f}\nLasso测试误差: {test_error_lasso:.2f}")
    ax6.set_xlabel('X')
    ax6.set_ylabel('y')
    ax6.legend()
    st.pyplot(fig6)

# 解释部分
st.header("多项式回归的过拟合与正则化")

st.markdown("""
### 过拟合问题解释

在多项式回归中，当多项式次数增加时，模型的复杂度也会随之增加：

- **低次多项式（如1次）**：模型过于简单，无法捕捉数据中的非线性关系，导致**欠拟合**
- **中等次数的多项式**：模型复杂度适中，能够较好地捕捉数据中的真实关系
- **高次多项式**：模型过于复杂，开始拟合数据中的噪声，导致**过拟合**

过拟合模型在训练数据上表现良好（训练误差低），但在测试数据上表现差（测试误差高）。

### 正则化方法改善过拟合

正则化通过对模型参数增加惩罚项来控制模型复杂度：

- **岭回归(Ridge)**：添加L2正则化项 $\\lambda\\sum_{j=0}^{p}\\beta_j^2$，收缩所有系数
- **Lasso回归**：添加L1正则化项 $\\lambda\\sum_{j=0}^{p}|\\beta_j|$，可以将某些系数精确设为零

正则化的强度由参数α控制，α越大，惩罚越强，模型越简单。

### 模型选择的原则

理想的模型应该在训练误差和测试误差之间取得平衡：
- 太简单的模型：训练误差和测试误差都很高
- 太复杂的模型：训练误差低但测试误差高
- 最佳模型：测试误差最低，表明模型具有良好的泛化能力
""")

# 交互说明
st.sidebar.markdown("""
### 使用说明

1. 调整左侧的参数设置
2. 观察不同次数多项式的拟合效果
3. 比较训练误差和测试误差
4. 尝试改变正则化参数α，观察其对高次多项式的影响

提示：
- 增加高次多项式的次数可以观察更明显的过拟合现象
- 调整噪声水平可以观察噪声对拟合的影响
- 调整岭回归和Lasso回归的α值，观察正则化的效果
""") 