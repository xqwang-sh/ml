import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib.colors import ListedColormap

def generate_data(n_samples=300, noise=0.2):
    """生成月牙形数据集"""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y

def plot_decision_boundary(X, y, model, title):
    """绘制模型的决策边界"""
    h = 0.02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Blues)
    
    # 绘制数据点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', 
                          cmap=plt.cm.Paired, s=70)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title, fontsize=15)
    
    return plt.gcf()

def nn_model(X, hidden_units, weights, biases, dropout_rate=0.0, is_training=True):
    """
    一个简单的神经网络模型，支持L2正则化和Dropout
    """
    # 第一层
    Z1 = np.dot(X, weights['W1']) + biases['b1']
    A1 = np.tanh(Z1)
    
    # Dropout
    if is_training and dropout_rate > 0:
        mask = np.random.rand(*A1.shape) > dropout_rate
        A1 = A1 * mask / (1 - dropout_rate)  # 缩放以保持期望值不变
    
    # 第二层
    Z2 = np.dot(A1, weights['W2']) + biases['b2']
    A2 = 1 / (1 + np.exp(-Z2))  # sigmoid
    
    return A2

def initialize_parameters(input_size, hidden_units, output_size, initialization="normal"):
    """
    初始化神经网络参数
    """
    if initialization == "normal":
        W1 = np.random.randn(input_size, hidden_units) * 0.01
        W2 = np.random.randn(hidden_units, output_size) * 0.01
    elif initialization == "xavier":
        W1 = np.random.randn(input_size, hidden_units) * np.sqrt(2 / (input_size + hidden_units))
        W2 = np.random.randn(hidden_units, output_size) * np.sqrt(2 / (hidden_units + output_size))
    elif initialization == "he":
        W1 = np.random.randn(input_size, hidden_units) * np.sqrt(2 / input_size)
        W2 = np.random.randn(hidden_units, output_size) * np.sqrt(2 / hidden_units)
    
    b1 = np.zeros((1, hidden_units))
    b2 = np.zeros((1, output_size))
    
    return {"W1": W1, "W2": W2}, {"b1": b1, "b2": b2}

def visualize_overfitting(X, y, models):
    """
    可视化不同复杂度模型的欠拟合、适合和过拟合
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, (title, model) in enumerate(models.items()):
        # 决策边界
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        axes[i].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Blues)
        
        # 绘制数据点
        scatter = axes[i].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', 
                                  cmap=plt.cm.Paired, s=50)
        
        axes[i].set_xlim(xx.min(), xx.max())
        axes[i].set_ylim(yy.min(), yy.max())
        axes[i].set_title(title, fontsize=14)
    
    plt.tight_layout()
    return fig

def visualize_l2_regularization(X, y, lambdas):
    """
    可视化不同L2正则化强度的效果
    """
    fig, axes = plt.subplots(1, len(lambdas), figsize=(20, 6))
    
    input_size = 2
    hidden_units = 50  # 复杂模型，容易过拟合
    output_size = 1
    
    for i, (title, lambda_val) in enumerate(lambdas.items()):
        weights, biases = initialize_parameters(input_size, hidden_units, output_size)
        
        # 使用高斯噪声扰动权重，模拟不同L2正则化的效果
        # L2正则化会使权重更接近0，所以我们通过缩小权重来模拟
        weights['W1'] = weights['W1'] / (1 + lambda_val)
        weights['W2'] = weights['W2'] / (1 + lambda_val)
        
        # 决策边界
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        def model(X):
            return nn_model(X, hidden_units, weights, biases)
        
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        axes[i].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Blues)
        
        # 绘制数据点
        scatter = axes[i].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', 
                                  cmap=plt.cm.Paired, s=50)
        
        axes[i].set_xlim(xx.min(), xx.max())
        axes[i].set_ylim(yy.min(), yy.max())
        axes[i].set_title(title, fontsize=14)
    
    plt.tight_layout()
    return fig

def visualize_dropout(X, y, dropout_rates):
    """
    可视化不同Dropout比率的效果
    """
    fig, axes = plt.subplots(1, len(dropout_rates), figsize=(20, 6))
    
    input_size = 2
    hidden_units = 50  # 复杂模型，容易过拟合
    output_size = 1
    
    # 初始化模型参数
    weights, biases = initialize_parameters(input_size, hidden_units, output_size)
    
    # 生成决策边界
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    for i, (title, dropout_rate) in enumerate(dropout_rates.items()):
        # 定义模型
        def model(X):
            return nn_model(X, hidden_units, weights, biases, dropout_rate=dropout_rate, is_training=False)
        
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        axes[i].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Blues)
        
        # 绘制数据点
        scatter = axes[i].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', 
                                  cmap=plt.cm.Paired, s=50)
        
        axes[i].set_xlim(xx.min(), xx.max())
        axes[i].set_ylim(yy.min(), yy.max())
        axes[i].set_title(title, fontsize=14)
    
    plt.tight_layout()
    return fig

def visualize_early_stopping():
    """
    可视化早停法的效果
    """
    epochs = np.arange(1, 101)
    train_loss = 1 / (1 + 0.1 * epochs) + 0.1 * np.random.randn(len(epochs))
    val_loss = 1 / (1 + 0.1 * epochs) + 0.3 / (1 + 0.01 * epochs) + 0.1 * np.random.randn(len(epochs))
    
    # 找到验证损失最小的点
    best_epoch = np.argmin(val_loss) + 1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='训练损失')
    ax.plot(epochs, val_loss, 'r-', linewidth=2, label='验证损失')
    
    # 标记早停点
    ax.axvline(x=best_epoch, color='g', linestyle='--', linewidth=2, 
               label=f'早停点 (轮次 {best_epoch})')
    
    # 标记过拟合区域
    ax.fill_between(epochs[best_epoch:], 0, 1, color='red', alpha=0.1, 
                    transform=ax.get_xaxis_transform())
    ax.text(best_epoch + 10, 0.95, '过拟合区域', fontsize=12, 
            transform=ax.get_xaxis_transform())
    
    ax.set_xlim([1, len(epochs)])
    ax.set_ylim([min(min(train_loss), min(val_loss))-0.1, max(max(train_loss), max(val_loss))+0.1])
    ax.set_xlabel('训练轮次', fontsize=12)
    ax.set_ylabel('损失', fontsize=12)
    ax.set_title('早停法示例', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    return fig

def visualize_batch_normalization():
    """
    可视化批量归一化的效果
    """
    # 生成正态分布的数据用于演示
    np.random.seed(42)
    data_raw = np.random.randn(1000, 2)
    data_shifted = data_raw + np.array([5, -5])  # 有偏移的数据
    
    # 对数据应用批量归一化
    data_bn = (data_shifted - data_shifted.mean(axis=0)) / data_shifted.std(axis=0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始数据分布
    axes[0].scatter(data_raw[:, 0], data_raw[:, 1], alpha=0.5)
    axes[0].set_xlim([-3, 8])
    axes[0].set_ylim([-8, 3])
    axes[0].set_title('原始数据分布', fontsize=14)
    axes[0].set_xlabel('特征 1', fontsize=12)
    axes[0].set_ylabel('特征 2', fontsize=12)
    axes[0].grid(alpha=0.3)
    
    # 有偏移的数据分布
    axes[1].scatter(data_shifted[:, 0], data_shifted[:, 1], alpha=0.5)
    axes[1].set_xlim([-3, 8])
    axes[1].set_ylim([-8, 3])
    axes[1].set_title('有偏移的数据分布', fontsize=14)
    axes[1].set_xlabel('特征 1', fontsize=12)
    axes[1].set_ylabel('特征 2', fontsize=12)
    axes[1].grid(alpha=0.3)
    
    # 批量归一化后的数据分布
    axes[2].scatter(data_bn[:, 0], data_bn[:, 1], alpha=0.5)
    axes[2].set_xlim([-3, 3])
    axes[2].set_ylim([-3, 3])
    axes[2].set_title('批量归一化后的数据分布', fontsize=14)
    axes[2].set_xlabel('特征 1', fontsize=12)
    axes[2].set_ylabel('特征 2', fontsize=12)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def show_regularization_page():
    """显示正则化方法页面"""
    st.markdown("<div class='sub-header'>正则化方法</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='explanation'>正则化是防止模型过拟合的重要技术。过拟合指模型在训练数据上表现极佳，但在新数据上表现不佳的现象。本页面展示了几种常用的正则化方法及其工作原理。</div>", unsafe_allow_html=True)
    
    # 欠拟合与过拟合
    st.markdown("<div class='sub-header'>欠拟合与过拟合</div>", unsafe_allow_html=True)
    
    st.markdown("""
    神经网络训练中常见的问题：
    
    - **欠拟合 (Underfitting)**：模型太简单，无法捕捉数据的复杂性
    - **适度拟合 (Good Fit)**：模型复杂度适中，能够很好地泛化到新数据
    - **过拟合 (Overfitting)**：模型太复杂，"记住"了训练数据的噪声
    """)
    
    # 生成示例数据
    X, y = generate_data(300, 0.2)
    
    # 定义不同复杂度的模型
    input_size = 2
    
    # 欠拟合模型（简单）
    hidden_units_simple = 2
    weights_simple, biases_simple = initialize_parameters(input_size, hidden_units_simple, 1)
    def model_simple(X):
        return nn_model(X, hidden_units_simple, weights_simple, biases_simple)
    
    # 适度拟合模型
    hidden_units_good = 8
    weights_good, biases_good = initialize_parameters(input_size, hidden_units_good, 1)
    def model_good(X):
        return nn_model(X, hidden_units_good, weights_good, biases_good)
    
    # 过拟合模型（复杂）
    hidden_units_complex = 100
    weights_complex, biases_complex = initialize_parameters(input_size, hidden_units_complex, 1)
    def model_complex(X):
        return nn_model(X, hidden_units_complex, weights_complex, biases_complex)
    
    models = {
        "欠拟合 (模型过于简单)": model_simple,
        "适度拟合 (模型复杂度适中)": model_good,
        "过拟合 (模型过于复杂)": model_complex
    }
    
    # 可视化不同复杂度的模型
    overfitting_fig = visualize_overfitting(X, y, models)
    st.pyplot(overfitting_fig)
    
    st.markdown("""
    **观察结果**：
    - **欠拟合模型**：决策边界太简单，无法正确分隔复杂的数据分布
    - **适度拟合模型**：决策边界平滑且正确分隔了数据
    - **过拟合模型**：决策边界非常复杂，捕捉了数据中的随机噪声
    """)
    
    # L2正则化
    st.markdown("<div class='sub-header'>L2正则化 (权重衰减)</div>", unsafe_allow_html=True)
    
    st.markdown("""
    L2正则化通过在损失函数中添加权重平方和项来惩罚大的权重值：
    
    $J_{regularized} = J + \\frac{\\lambda}{2m} \\sum_{l} \\sum_{i} \\sum_{j} (W^{[l]}_{ij})^2$
    
    其中，$\\lambda$ 是正则化强度参数，较大的 $\\lambda$ 会导致更小的权重。
    
    **L2正则化的优点**：
    - 减少过拟合
    - 使决策边界更平滑
    - 降低模型对小扰动的敏感性
    """)
    
    # 可视化不同L2正则化强度
    lambdas = {
        "无正则化 (λ=0)": 0,
        "弱正则化 (λ=0.1)": 0.1,
        "中等正则化 (λ=1)": 1.0,
        "强正则化 (λ=10)": 10.0
    }
    
    l2_reg_fig = visualize_l2_regularization(X, y, lambdas)
    st.pyplot(l2_reg_fig)
    
    st.markdown("""
    **观察结果**：
    - 随着正则化强度增加，决策边界变得更平滑
    - 过强的正则化可能导致欠拟合（如最右图）
    """)
    
    # Dropout正则化
    st.markdown("<div class='sub-header'>Dropout正则化</div>", unsafe_allow_html=True)
    
    st.markdown("""
    Dropout是一种在训练过程中随机"关闭"一部分神经元的技术：
    
    1. 训练时，每个神经元以概率 $p$ 被保留，以概率 $1-p$ 被临时关闭
    2. 测试时使用完整网络，但权重需要按保留率 $p$ 缩放
    
    **Dropout的优点**：
    - 防止神经元共适应
    - 类似于集成多个不同的网络
    - 迫使网络学习更鲁棒的特征
    """)
    
    # 可视化Dropout示意图
    st.image("https://miro.medium.com/max/1400/1*iWQzxhVlvadk6VAJjsgXgg.png", 
             caption="Dropout示意图 (图源: Towards Data Science)", use_column_width=True)
    
    # 可视化不同Dropout比率的效果
    dropout_rates = {
        "无Dropout (rate=0)": 0.0,
        "轻度Dropout (rate=0.2)": 0.2,
        "中度Dropout (rate=0.5)": 0.5,
        "高度Dropout (rate=0.8)": 0.8
    }
    
    dropout_fig = visualize_dropout(X, y, dropout_rates)
    st.pyplot(dropout_fig)
    
    st.markdown("""
    **观察结果**：
    - 适当的Dropout使决策边界更平滑，减少过拟合
    - 过高的Dropout率会使网络欠拟合，丢失太多信息
    - Dropout在测试阶段不启用，但训练时的Dropout会影响学习到的参数
    """)
    
    # 早停法
    st.markdown("<div class='sub-header'>早停法 (Early Stopping)</div>", unsafe_allow_html=True)
    
    st.markdown("""
    早停法通过监控验证集性能来决定何时停止训练：
    
    1. 在每个训练轮次后，评估模型在验证集上的性能
    2. 当验证误差开始上升时，停止训练并回退到验证误差最小的模型
    
    **早停法的优点**：
    - 简单易实现
    - 减少计算时间
    - 自动选择最佳训练轮次
    """)
    
    # 可视化早停法
    early_stopping_fig = visualize_early_stopping()
    st.pyplot(early_stopping_fig)
    
    st.markdown("""
    **观察结果**：
    - 随着训练进行，训练损失持续下降
    - 验证损失先降后升，表明模型开始过拟合
    - 绿色虚线标记了理想的停止点，此时验证损失最小
    """)
    
    # 批量归一化
    st.markdown("<div class='sub-header'>批量归一化 (Batch Normalization)</div>", unsafe_allow_html=True)
    
    st.markdown("""
    批量归一化是一种在每层输入上应用标准化的技术：
    
    1. 对每个小批量数据计算均值和方差
    2. 对输入进行归一化：$\\hat{x}^{(k)} = \\frac{x^{(k)} - \\mu_B}{\\sqrt{\\sigma^2_B + \\epsilon}}$
    3. 应用缩放和平移：$y^{(k)} = \\gamma \\hat{x}^{(k)} + \\beta$
    
    **批量归一化的优点**：
    - 加速训练收敛
    - 允许使用更高的学习率
    - 减少对初始化的敏感性
    - 具有轻微的正则化效果
    """)
    
    # 可视化批量归一化
    bn_fig = visualize_batch_normalization()
    st.pyplot(bn_fig)
    
    st.markdown("""
    **观察结果**：
    - 批量归一化将特征分布调整为均值为0、方差为1的标准正态分布
    - 无论原始数据如何分布，归一化后的数据分布相似
    - 这种标准化使得网络更容易学习，提高训练稳定性
    """)
    
    # 其他正则化方法
    st.markdown("<div class='sub-header'>其他正则化方法</div>", unsafe_allow_html=True)
    
    st.markdown("""
    除了上述方法外，还有许多其他正则化技术：
    
    - **数据增强 (Data Augmentation)**：通过变换已有数据创建新样本
      - 图像旋转、缩放、翻转、裁剪
      - 文本同义词替换、回译
    
    - **L1正则化**：向损失函数添加权重绝对值之和，促进稀疏性
      - $J_{regularized} = J + \\frac{\\lambda}{2m} \\sum_{l} \\sum_{i} \\sum_{j} |W^{[l]}_{ij}|$
    
    - **最大范数约束 (MaxNorm Constraint)**：限制权重向量的最大范数
    
    - **Label Smoothing**：软化目标标签，减少模型过度自信
    """)
    
    # 结论
    st.markdown("<div class='explanation'>选择合适的正则化方法对防止过拟合至关重要。通常，多种正则化技术组合使用效果最佳。在实践中，应根据数据量、模型复杂度和任务类型选择适当的正则化策略。</div>", unsafe_allow_html=True) 