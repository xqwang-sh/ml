import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def sigmoid(z):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-z))

def relu(z):
    """ReLU激活函数"""
    return np.maximum(0, z)

def tanh(z):
    """Tanh激活函数"""
    return np.tanh(z)

def plot_activation_functions():
    """绘制常见激活函数的图像"""
    z = np.linspace(-5, 5, 100)
    sigmoid_y = sigmoid(z)
    relu_y = relu(z)
    tanh_y = tanh(z)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z, sigmoid_y, label='Sigmoid', linewidth=2, color='#1E88E5')
    ax.plot(z, relu_y, label='ReLU', linewidth=2, color='#D81B60')
    ax.plot(z, tanh_y, label='Tanh', linewidth=2, color='#228B22')
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlim([-5, 5])
    ax.grid(alpha=0.3)
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('激活值', fontsize=12)
    ax.set_title('常见激活函数比较', fontsize=14)
    ax.legend(fontsize=12)
    
    return fig

def visualize_perceptron(w1, w2, b, activation_fn, x_range=(-10, 10), y_range=(-10, 10)):
    """可视化单个感知器的决策边界和输出"""
    x1 = np.linspace(x_range[0], x_range[1], 100)
    x2 = np.linspace(y_range[0], y_range[1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    # 计算z和激活值
    Z = w1 * X1 + w2 * X2 + b
    
    if activation_fn == "sigmoid":
        A = sigmoid(Z)
        fn = sigmoid
    elif activation_fn == "relu":
        A = relu(Z)
        fn = relu
    else:  # tanh
        A = tanh(Z)
        fn = tanh
    
    # 创建图
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制决策边界
    if activation_fn == "sigmoid":
        contour = axs[0].contourf(X1, X2, A, levels=50, cmap='Blues')
        threshold = 0.5
    elif activation_fn == "tanh":
        contour = axs[0].contourf(X1, X2, A, levels=50, cmap='PiYG')
        threshold = 0
    else:  # relu
        contour = axs[0].contourf(X1, X2, A, levels=50, cmap='Oranges')
        threshold = 0.5
    
    plt.colorbar(contour, ax=axs[0])
    
    # 绘制决策边界线
    if activation_fn == "sigmoid" or activation_fn == "tanh":
        # 仅对Sigmoid和Tanh绘制决策边界，因为它们有明确的阈值
        axs[0].contour(X1, X2, A, levels=[threshold], colors='red', linewidths=2)
    
    # 绘制权重向量
    if w1 != 0 or w2 != 0:  # 避免零向量
        scale = 3  # 缩放因子，使箭头更明显
        axs[0].arrow(0, 0, scale * w1, scale * w2, head_width=0.5, head_length=0.7, fc='black', ec='black')
        axs[0].text(scale * w1 * 1.1, scale * w2 * 1.1, f'w=({w1},{w2})', fontsize=12)
    
    axs[0].set_xlabel('特征 x₁', fontsize=12)
    axs[0].set_ylabel('特征 x₂', fontsize=12)
    axs[0].set_title(f'{activation_fn.capitalize()} 激活的感知器决策边界', fontsize=14)
    axs[0].grid(alpha=0.3)
    axs[0].set_xlim([x_range[0], x_range[1]])
    axs[0].set_ylim([y_range[0], y_range[1]])
    
    # 绘制3D表面图
    ax3d = plt.subplot(1, 2, 2, projection='3d')
    surf = ax3d.plot_surface(X1, X2, A, cmap='viridis', alpha=0.8, edgecolor='none')
    
    # 绘制z平面（对于有阈值的激活函数）
    if activation_fn == "sigmoid" or activation_fn == "tanh":
        xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 2), 
                            np.linspace(y_range[0], y_range[1], 2))
        z_plane = np.ones(xx.shape) * threshold
        ax3d.plot_surface(xx, yy, z_plane, color='red', alpha=0.3)
    
    plt.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)
    ax3d.set_xlabel('特征 x₁', fontsize=12)
    ax3d.set_ylabel('特征 x₂', fontsize=12)
    ax3d.set_zlabel('激活值', fontsize=12)
    ax3d.set_title(f'{activation_fn.capitalize()} 激活函数输出', fontsize=14)
    
    return fig

def plot_mlp_classification(layer1_size=3):
    """
    绘制多层感知器对XOR问题的分类效果
    """
    # 生成XOR问题的数据
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0]).reshape(-1, 1)
    
    # 设置权重（修改后的动态权重初始化）
    # 第一层
    W1 = np.array([
        [-7, -7],  # 神经元1
        [-7, 7],   # 神经元2
        [7, -7],   # 神经元3
        [5, 5],    # 新增神经元4
        [-5, 5]    # 新增神经元5
    ][:layer1_size])  # 根据选择的神经元数量切片
    
    b1 = np.array([3, 3, 3, 2, 2][:layer1_size]).reshape(-1, 1)  # 匹配隐藏层大小
    
    # 第二层
    W2 = np.ones((1, layer1_size))  # 自动匹配隐藏层神经元数量
    b2 = np.array([-2]).reshape(-1, 1)
    
    # 前向传播
    def forward(x, W1, b1, W2, b2):
        z1 = np.dot(W1, x.reshape(-1, 1)) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = sigmoid(z2)
        return a1, a2
    
    # 创建网格用于可视化
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 对网格点进行预测
    Z = np.zeros(grid_points.shape[0])
    A1_values = np.zeros((grid_points.shape[0], layer1_size))
    
    for i, point in enumerate(grid_points):
        a1, a2 = forward(point, W1, b1, W2, b2)
        Z[i] = a2[0, 0]
        A1_values[i] = a1.flatten()
    
    Z = Z.reshape(xx.shape)
    
    # 创建图形
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    
    # 绘制决策边界
    contour = axs[0].contourf(xx, yy, Z, levels=50, cmap='Blues', alpha=0.8)
    axs[0].contour(xx, yy, Z, levels=[0.5], colors='red', linewidths=2)
    
    # 绘制训练数据点
    axs[0].scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], c='#D81B60', 
                  marker='o', s=100, label='类别 0', edgecolors='k')
    axs[0].scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], c='#1E88E5', 
                  marker='^', s=100, label='类别 1', edgecolors='k')
    
    plt.colorbar(contour, ax=axs[0])
    axs[0].set_xlim([x_min, x_max])
    axs[0].set_ylim([y_min, y_max])
    axs[0].set_xlabel('特征 x₁', fontsize=12)
    axs[0].set_ylabel('特征 x₂', fontsize=12)
    axs[0].set_title('多层感知器对XOR问题的分类', fontsize=14)
    axs[0].legend(loc='upper right')
    axs[0].grid(alpha=0.3)
    
    # 可视化网络结构
    layer_sizes = [2, layer1_size, 1]
    layer_positions = [0, 1, 2]
    v_spacing = 0.25
    h_spacing = 1
    
    # 清空第二个子图
    axs[1].clear()
    axs[1].axis('off')
    
    # 绘制输入层
    layer_top = v_spacing * (layer_sizes[0] - 1) / 2
    for i in range(layer_sizes[0]):
        axs[1].scatter(0, layer_top - i * v_spacing, s=100, c='#D81B60', marker='o', edgecolors='k')
        axs[1].annotate(f'x{i+1}', xy=(0, layer_top - i * v_spacing), xytext=(-0.15, 0),
                       textcoords='offset points', ha='right', va='center', fontsize=12)
    
    # 绘制隐藏层
    layer_top = v_spacing * (layer_sizes[1] - 1) / 2
    for i in range(layer_sizes[1]):
        axs[1].scatter(h_spacing, layer_top - i * v_spacing, s=100, c='#FFC107', marker='o', edgecolors='k')
        
        # 绘制权重连接
        for j in range(layer_sizes[0]):
            weight = W1[i, j]
            color = 'green' if weight > 0 else 'red'
            linewidth = 1 + abs(weight) / 3
            alpha = np.minimum(0.5 + abs(weight) / 10, 0.95)
            axs[1].plot([0, h_spacing], 
                       [layer_top - j * v_spacing, layer_top - i * v_spacing], 
                       c=color, linewidth=linewidth, alpha=alpha)
        
        axs[1].annotate(f'a{i+1}', xy=(h_spacing, layer_top - i * v_spacing), xytext=(0.1, 0),
                       textcoords='offset points', ha='left', va='center', fontsize=12)
    
    # 绘制输出层
    layer_top = v_spacing * (layer_sizes[2] - 1) / 2
    for i in range(layer_sizes[2]):
        axs[1].scatter(2 * h_spacing, layer_top - i * v_spacing, s=100, c='#1E88E5', marker='o', edgecolors='k')
        
        # 绘制权重连接
        for j in range(layer_sizes[1]):
            weight = W2[i, j]
            color = 'green' if weight > 0 else 'red'
            linewidth = 1 + abs(weight) / 3
            alpha = np.minimum(0.5 + abs(weight) / 10, 0.95)
            axs[1].plot([h_spacing, 2 * h_spacing], 
                       [layer_top - j * v_spacing - (layer_sizes[1] - 1) * v_spacing / 2, layer_top - i * v_spacing], 
                       c=color, linewidth=linewidth, alpha=alpha)
        
        axs[1].annotate('y', xy=(2 * h_spacing, layer_top - i * v_spacing), xytext=(0.1, 0),
                       textcoords='offset points', ha='left', va='center', fontsize=12)
    
    # 添加图例
    green_patch = mpatches.Patch(color='green', label='正权重')
    red_patch = mpatches.Patch(color='red', label='负权重')
    axs[1].legend(handles=[green_patch, red_patch], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    axs[1].set_title('多层感知器网络结构', fontsize=14)
    
    plt.tight_layout()
    return fig

def show_perceptron_page():
    """显示感知器页面"""
    st.markdown("<div class='sub-header'>单层与多层感知器</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='explanation'>神经网络由基本单元——神经元（或感知器）组成。这些神经元接收输入信号，根据权重和偏置计算加权和，然后通过激活函数产生输出。通过将多个神经元组织成层级结构，神经网络可以学习复杂的非线性关系。</div>", unsafe_allow_html=True)
    
    # 激活函数部分
    st.markdown("<div class='sub-header'>激活函数</div>", unsafe_allow_html=True)
    st.markdown("激活函数引入非线性，是神经网络强大表达能力的关键。下图展示了三种常用的激活函数：")
    
    st.pyplot(plot_activation_functions())
    
    # 单个感知器可视化
    st.markdown("<div class='sub-header'>单个感知器的决策边界</div>", unsafe_allow_html=True)
    
    st.markdown("""
    感知器的基本计算过程如下：
    1. 计算输入的加权和：$z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$
    2. 将加权和传入激活函数：$a = \sigma(z)$
    3. 输出结果：$\hat{y} = a$
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("调整权重和偏置，观察它们如何影响感知器的决策边界：")
        w1 = st.slider("权重 w₁", -10.0, 10.0, 1.0, 0.1)
        w2 = st.slider("权重 w₂", -10.0, 10.0, 1.0, 0.1)
        b = st.slider("偏置 b", -10.0, 10.0, 0.0, 0.1)
    
    with col2:
        st.markdown("选择激活函数：")
        activation = st.radio(
            "激活函数",
            ["sigmoid", "relu", "tanh"],
            index=0
        )
    
    # 可视化单个感知器
    perceptron_fig = visualize_perceptron(w1, w2, b, activation)
    st.pyplot(perceptron_fig)
    
    # 数学表达式
    if activation == "sigmoid":
        formula = r"$a = \sigma(w_1 x_1 + w_2 x_2 + b) = \frac{1}{1 + e^{-(w_1 x_1 + w_2 x_2 + b)}}$"
    elif activation == "relu":
        formula = r"$a = \text{ReLU}(w_1 x_1 + w_2 x_2 + b) = \max(0, w_1 x_1 + w_2 x_2 + b)$"
    else:  # tanh
        formula = r"$a = \tanh(w_1 x_1 + w_2 x_2 + b) = \frac{e^{w_1 x_1 + w_2 x_2 + b} - e^{-(w_1 x_1 + w_2 x_2 + b)}}{e^{w_1 x_1 + w_2 x_2 + b} + e^{-(w_1 x_1 + w_2 x_2 + b)}}$"
    
    st.markdown(f"<div class='formula'>{formula}</div>", unsafe_allow_html=True)
    
    # 单层感知机局限性
    st.markdown("<div class='sub-header'>单层感知机的局限性</div>", unsafe_allow_html=True)
    
    st.markdown("""
    单层感知机只能解决**线性可分**的问题，无法处理如XOR（异或）这样的非线性问题。
    
    异或问题的特点：
    - 输入 (0,0) → 输出 0
    - 输入 (0,1) → 输出 1
    - 输入 (1,0) → 输出 1
    - 输入 (1,1) → 输出 0
    """)
    
    # 显示XOR问题图像
    x_values = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_values = np.array([0, 1, 1, 0])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_values[y_values == 0, 0], x_values[y_values == 0, 1], 
              c='#D81B60', marker='o', s=100, label='类别 0', edgecolors='k')
    ax.scatter(x_values[y_values == 1, 0], x_values[y_values == 1, 1], 
              c='#1E88E5', marker='^', s=100, label='类别 1', edgecolors='k')
    
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_xlabel('特征 x₁', fontsize=12)
    ax.set_ylabel('特征 x₂', fontsize=12)
    ax.set_title('XOR问题：无法用单个感知器解决', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)
    
    st.markdown("""
    你可以尝试调整上面的权重和偏置，但单个感知器无法找到一条直线将两种类别完全分开。这是因为XOR问题**不是线性可分的**。
    """)
    
    # 多层感知器部分
    st.markdown("<div class='sub-header'>多层感知器解决XOR问题</div>", unsafe_allow_html=True)
    
    st.markdown("""
    通过使用多层感知器（MLP），我们可以解决非线性问题，如XOR。
    
    MLP包含：
    - 输入层：接收原始特征
    - 隐藏层：提取复杂特征
    - 输出层：产生最终预测
    
    下图展示了一个解决XOR问题的多层感知器。通过添加隐藏层，网络能够学习复杂的决策边界。
    """)
    
    # 控制隐藏层神经元数量
    num_hidden = st.slider("隐藏层神经元数量", 2, 5, 3, 1)
    
    # 可视化多层感知器
    mlp_fig = plot_mlp_classification(num_hidden)
    st.pyplot(mlp_fig)
    
    st.markdown("""
    **观察结果**：
    - 多层感知器成功创建了复杂的决策边界，可以正确分类XOR问题
    - 红色线表示决策边界（输出为0.5的点）
    - 右图显示网络结构，绿色连接表示正权重，红色表示负权重
    """)
    
    st.markdown("""
    **前向传播计算**：
    1. 隐藏层: $\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}$
    2. 隐藏层激活: $\mathbf{a}^{[1]} = \sigma(\mathbf{z}^{[1]})$
    3. 输出层: $\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}$
    4. 输出层激活: $\mathbf{a}^{[2]} = \sigma(\mathbf{z}^{[2]})$
    """)
    
    # 添加结论
    st.markdown("<div class='explanation'>多层神经网络可以学习复杂的非线性关系，这使它们能够解决单层感知器无法处理的问题。通过添加更多层和神经元，模型的表达能力会进一步增强，但也增加了训练的复杂性。</div>", unsafe_allow_html=True) 