import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import io
import base64
from PIL import Image

def plot_exploding_gradient_example():
    """绘制一个梯度爆炸的简单示例"""
    # 创建一个简单的函数，当x接近某个值时梯度变得非常大
    x = np.linspace(-3, 3, 1000)
    # 函数f(x) = 1/x^2当x接近0时梯度爆炸
    y = 1 / (x**2 + 0.01)
    # 计算梯度 (使用有限差分近似)
    h = 0.001
    x_grad = x[1:-1]
    y_grad = (1 / ((x_grad + h)**2 + 0.01) - 1 / ((x_grad - h)**2 + 0.01)) / (2 * h)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制函数
    ax1.plot(x, y, 'b-', linewidth=2)
    ax1.set_ylim([0, 100])  # 限制y轴范围以便于观察
    ax1.set_title('函数 f(x) = 1/(x²+0.01)', fontsize=14)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # 绘制梯度
    ax2.plot(x_grad, y_grad, 'r-', linewidth=2)
    ax2.set_ylim([-1000, 1000])  # 限制y轴范围以便于观察
    ax2.set_title('函数的梯度 f\'(x)', fontsize=14)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('f\'(x)', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_weight_initialization_comparison():
    """比较不同权重初始化方法对梯度的影响"""
    # 设置随机种子以保证结果可重复
    np.random.seed(42)
    
    # 网络深度和每层宽度
    n_layers = 50
    layer_width = 100
    
    # 不同的初始化策略
    methods = {
        "标准正态分布": lambda n_in, n_out: np.random.randn(n_in, n_out),
        "Xavier/Glorot初始化": lambda n_in, n_out: np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out)),
        "He初始化": lambda n_in, n_out: np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in),
        "LeCun初始化": lambda n_in, n_out: np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
    }
    
    # 为每种方法计算前向传播时的激活值范数
    results = {}
    
    for name, init_func in methods.items():
        # 模拟输入数据
        X = np.random.randn(1, layer_width)  # 单个样本
        
        # 存储每一层的激活范数
        norm_per_layer = [np.linalg.norm(X)]
        
        # 前向传播
        a = X
        for i in range(n_layers):
            # 初始化权重
            W = init_func(layer_width, layer_width)
            # 前向传播
            a = np.tanh(np.dot(a, W))  # 使用tanh激活函数
            # 计算范数
            norm = np.linalg.norm(a)
            norm_per_layer.append(norm)
        
        results[name] = norm_per_layer
    
    # 绘制结果
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for name, norms in results.items():
        ax.plot(range(len(norms)), norms, 'o-', linewidth=2, label=name)
    
    ax.set_yscale('log')  # 使用对数刻度
    ax.set_title('不同权重初始化方法对激活值范数的影响', fontsize=16)
    ax.set_xlabel('网络层', fontsize=14)
    ax.set_ylabel('激活值范数 (对数尺度)', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    
    return fig

def plot_gradient_clipping_example():
    """可视化梯度裁剪效果"""
    # 生成网格
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # 定义一个具有大梯度区域的函数
    Z = 0.1 * (X**2 + Y**2) + 10 * np.exp(-((X-2)**2 + (Y-2)**2) / 0.5)
    
    # 计算梯度
    grad_x = 0.2 * X + 10 * (-2*(X-2)/0.5) * np.exp(-((X-2)**2 + (Y-2)**2) / 0.5)
    grad_y = 0.2 * Y + 10 * (-2*(Y-2)/0.5) * np.exp(-((X-2)**2 + (Y-2)**2) / 0.5)
    
    # 计算梯度范数
    grad_norm = np.sqrt(grad_x**2 + grad_y**2)
    
    # 应用梯度裁剪 (阈值为2.0)
    threshold = 2.0
    scale = np.minimum(1.0, threshold / (grad_norm + 1e-8))
    clipped_grad_x = grad_x * scale
    clipped_grad_y = grad_y * scale
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 绘制函数表面
    c1 = axes[0].contourf(X, Y, Z, 50, cmap='viridis')
    axes[0].set_title('函数表面', fontsize=14)
    plt.colorbar(c1, ax=axes[0])
    
    # 绘制原始梯度向量场
    skip = 5  # 跳过一些点以使向量场更清晰
    axes[1].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                 grad_x[::skip, ::skip], grad_y[::skip, ::skip],
                 grad_norm[::skip, ::skip], 
                 cmap='plasma', scale=30)
    axes[1].set_title('原始梯度场', fontsize=14)
    axes[1].set_xlim([-5, 5])
    axes[1].set_ylim([-5, 5])
    
    # 绘制裁剪后的梯度向量场
    c3 = axes[2].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                      clipped_grad_x[::skip, ::skip], clipped_grad_y[::skip, ::skip],
                      np.sqrt(clipped_grad_x[::skip, ::skip]**2 + clipped_grad_y[::skip, ::skip]**2), 
                      cmap='plasma', scale=30)
    axes[2].set_title(f'裁剪后的梯度场 (阈值={threshold})', fontsize=14)
    axes[2].set_xlim([-5, 5])
    axes[2].set_ylim([-5, 5])
    
    plt.tight_layout()
    return fig

def simulate_gradient_descent_with_clipping(clipping=False, learning_rate=0.1):
    """模拟带/不带梯度裁剪的梯度下降过程"""
    # 定义一个在某些区域具有大梯度的函数
    def func(x, y):
        return 0.1 * (x**2 + y**2) + 10 * np.exp(-((x-2)**2 + (y-2)**2) / 0.5)
    
    def grad(x, y):
        grad_x = 0.2 * x + 10 * (-2*(x-2)/0.5) * np.exp(-((x-2)**2 + (y-2)**2) / 0.5)
        grad_y = 0.2 * y + 10 * (-2*(y-2)/0.5) * np.exp(-((x-2)**2 + (y-2)**2) / 0.5)
        return np.array([grad_x, grad_y])
    
    # 初始位置
    position = np.array([3.0, 3.0])
    
    # 迭代次数
    n_iterations = 100
    
    # 保存轨迹
    trajectory = [position.copy()]
    
    # 梯度裁剪阈值
    threshold = 2.0
    
    # 执行梯度下降
    for _ in range(n_iterations):
        # 计算梯度
        g = grad(position[0], position[1])
        
        # 应用梯度裁剪（如果启用）
        if clipping:
            grad_norm = np.linalg.norm(g)
            if grad_norm > threshold:
                g = g * threshold / grad_norm
        
        # 更新位置
        position = position - learning_rate * g
        
        # 保存当前位置
        trajectory.append(position.copy())
    
    return np.array(trajectory)

def plot_gradient_descent_comparison():
    """比较带/不带梯度裁剪的梯度下降"""
    # 生成格点
    x = np.linspace(-1, 5, 100)
    y = np.linspace(-1, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # 定义函数
    Z = 0.1 * (X**2 + Y**2) + 10 * np.exp(-((X-2)**2 + (Y-2)**2) / 0.5)
    
    # 获取轨迹
    trajectory_without_clipping = simulate_gradient_descent_with_clipping(clipping=False, learning_rate=0.1)
    trajectory_with_clipping = simulate_gradient_descent_with_clipping(clipping=True, learning_rate=0.1)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # 绘制不带裁剪的梯度下降
    c1 = axes[0].contourf(X, Y, Z, 50, cmap='viridis')
    axes[0].plot(trajectory_without_clipping[:, 0], trajectory_without_clipping[:, 1], 'r-o', linewidth=2, markersize=4)
    axes[0].set_title('不带梯度裁剪的梯度下降', fontsize=14)
    axes[0].set_xlim([-1, 5])
    axes[0].set_ylim([-1, 5])
    plt.colorbar(c1, ax=axes[0])
    
    # 绘制带裁剪的梯度下降
    c2 = axes[1].contourf(X, Y, Z, 50, cmap='viridis')
    axes[1].plot(trajectory_with_clipping[:, 0], trajectory_with_clipping[:, 1], 'r-o', linewidth=2, markersize=4)
    axes[1].set_title('带梯度裁剪的梯度下降', fontsize=14)
    axes[1].set_xlim([-1, 5])
    axes[1].set_ylim([-1, 5])
    plt.colorbar(c2, ax=axes[1])
    
    plt.tight_layout()
    return fig

def show_exploding_gradient_page():
    st.markdown("<div class='sub-header'>梯度爆炸问题及解决方案</div>", unsafe_allow_html=True)
    
    # 问题说明
    with st.expander("什么是梯度爆炸问题?", expanded=True):
        st.markdown("""
        <div class='explanation'>
        <b>梯度爆炸</b>是深度神经网络训练中的另一个重要问题，与梯度消失相对，表现为梯度值异常增大。
        
        当网络很深时，在反向传播过程中，如果每一层的局部梯度大于1，梯度会通过链式法则从输出层向输入层传播时呈指数级增长，
        导致权重更新过大，这就是<b>梯度爆炸</b>问题。
        
        梯度爆炸会导致：
        - 训练不稳定，权重更新剧烈振荡
        - 模型参数溢出，出现NaN值
        - 无法收敛到最优解
        
        梯度爆炸在循环神经网络(RNN)中尤为常见，特别是在处理长序列时。
        </div>
        """, unsafe_allow_html=True)
    
    # 梯度爆炸示例
    st.markdown("<div class='sub-header'>梯度爆炸直观示例</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='explanation'>
    下图展示了一个简单函数的梯度爆炸示例。当自变量接近某个值时，函数的梯度变得异常大，导致优化算法在这些区域可能会"跳跃"到非常远的地方。
    </div>
    """, unsafe_allow_html=True)
    
    fig_explosion = plot_exploding_gradient_example()
    st.pyplot(fig_explosion)
    
    # 权重初始化比较
    st.markdown("<div class='sub-header'>权重初始化对梯度爆炸的影响</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='explanation'>
    不同的权重初始化方法对梯度传播有显著影响。下图展示了不同初始化方法在深度网络中激活值范数的变化：
    
    - <b>标准正态分布初始化</b>：激活值范数可能呈指数增长或消失
    - <b>Xavier/Glorot初始化</b>：为sigmoid/tanh激活函数设计，保持激活值范数相对稳定
    - <b>He初始化</b>：为ReLU激活函数设计，考虑了ReLU的特性
    - <b>LeCun初始化</b>：调整权重方差以保持前向传播信号的方差一致
    
    适当的初始化可以预防梯度爆炸，使网络训练更加稳定。
    </div>
    """, unsafe_allow_html=True)
    
    fig_initialization = plot_weight_initialization_comparison()
    st.pyplot(fig_initialization)
    
    # 梯度裁剪可视化
    st.markdown("<div class='sub-header'>梯度裁剪的工作原理</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='explanation'>
    <b>梯度裁剪</b>是解决梯度爆炸最直接有效的方法。它限制梯度的范数，确保更新步骤不会过大。
    
    下图对比了原始梯度和裁剪后的梯度：
    </div>
    """, unsafe_allow_html=True)
    
    fig_clipping = plot_gradient_clipping_example()
    st.pyplot(fig_clipping)
    
    # 梯度下降比较
    st.markdown("<div class='sub-header'>梯度裁剪对优化过程的影响</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='explanation'>
    下图对比了带梯度裁剪和不带梯度裁剪的梯度下降过程。观察在梯度大的区域，两种方法的行为差异：
    
    - 没有梯度裁剪：优化过程可能发散或在函数表面"跳跃"
    - 有梯度裁剪：优化过程更加平稳，逐步接近最小值
    </div>
    """, unsafe_allow_html=True)
    
    fig_descent = plot_gradient_descent_comparison()
    st.pyplot(fig_descent)
    
    # 解决方案
    st.markdown("<div class='sub-header'>梯度爆炸的解决方案</div>", unsafe_allow_html=True)
    
    solutions = st.tabs(["梯度裁剪", "权重初始化", "正则化", "层归一化", "残差连接", "激活函数选择"])
    
    with solutions[0]:
        st.markdown("""
        <div class='explanation'>
        <b>梯度裁剪 (Gradient Clipping)</b>
        
        梯度裁剪是处理梯度爆炸最直接的方法，特别是在RNN和LSTM模型中。
        
        <b>实现方式</b>：
        
        1. <b>基于范数裁剪</b>：
           ```python
           # 计算梯度的L2范数
           grad_norm = torch.norm(grad)
           # 如果梯度范数超过阈值，则缩放梯度
           if grad_norm > threshold:
               grad = threshold * grad / grad_norm
           ```
        
        2. <b>基于值裁剪</b>：
           ```python
           # 将每个梯度值限制在[-threshold, threshold]范围内
           grad = torch.clamp(grad, -threshold, threshold)
           ```
        
        <b>何时使用</b>：
        - RNN/LSTM模型训练
        - 当观察到损失函数突然变为NaN
        - 当训练过程不稳定，损失剧烈振荡
        
        几乎所有深度学习框架都内置了梯度裁剪功能，使用非常方便。
        </div>
        """, unsafe_allow_html=True)
    
    with solutions[1]:
        st.markdown("""
        <div class='explanation'>
        <b>合适的权重初始化</b>
        
        合理的权重初始化是防止梯度爆炸的关键预防措施。
        
        <b>常用初始化方法</b>：
        
        - <b>Xavier/Glorot初始化</b>：
          ```python
          # PyTorch实现
          nn.init.xavier_normal_(tensor)
          # TensorFlow/Keras实现
          tf.keras.initializers.GlorotNormal()
          ```
        
        - <b>He初始化</b>：
          ```python
          # PyTorch实现
          nn.init.kaiming_normal_(tensor)
          # TensorFlow/Keras实现
          tf.keras.initializers.HeNormal()
          ```
        
        - <b>正交初始化</b>（对RNN特别有效）：
          ```python
          # PyTorch实现
          nn.init.orthogonal_(tensor)
          # TensorFlow/Keras实现
          tf.keras.initializers.Orthogonal()
          ```
        
        合适的初始化可以使得网络在训练初期就处于一个相对稳定的状态，减少梯度爆炸和消失的风险。
        </div>
        """, unsafe_allow_html=True)
    
    with solutions[2]:
        st.markdown("""
        <div class='explanation'>
        <b>正则化技术</b>
        
        多种正则化技术可以间接帮助减轻梯度爆炸问题。
        
        <b>主要方法</b>：
        
        - <b>权重衰减 (L2正则化)</b>：
          - 通过在损失函数中添加权重的平方项来惩罚大权重
          - 保持权重较小，间接限制梯度大小
          ```python
          # PyTorch示例
          optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
          ```
        
        - <b>Dropout</b>：
          - 训练过程中随机关闭一定比例的神经元
          - 防止神经元共适应，创建更鲁棒的特征
          - 降低网络对特定路径的依赖
          ```python
          # PyTorch示例
          self.dropout = nn.Dropout(0.5)
          x = self.dropout(x)  # 应用在前向传播中
          ```
        
        正则化同时也是防止过拟合的重要技术，可以使模型更加鲁棒并具有更好的泛化能力。
        </div>
        """, unsafe_allow_html=True)
    
    with solutions[3]:
        st.markdown("""
        <div class='explanation'>
        <b>层归一化 (Layer Normalization)</b>
        
        与批归一化类似，层归一化也能稳定深度网络中的激活值分布，但它沿着特征维度而非批次维度进行归一化。
        
        <b>特点</b>：
        - 对每个样本单独进行归一化，而不依赖批次统计信息
        - 特别适合RNN等序列模型和批量大小小的情况
        - 可以与梯度裁剪结合使用以获得更稳定的训练
        
        <b>实现</b>：
        ```python
        # PyTorch示例
        self.layer_norm = nn.LayerNorm(hidden_size)
        h = self.layer_norm(h)  # 应用在RNN/Transformer层
        ```
        
        层归一化在Transformer架构中广泛使用，是现代NLP模型的关键组件。
        </div>
        """, unsafe_allow_html=True)
    
    with solutions[4]:
        st.markdown("""
        <div class='explanation'>
        <b>残差连接 (Residual Connections)</b>
        
        残差连接不仅有助于解决梯度消失问题，也能缓解梯度爆炸。通过提供直接的梯度流路径，残差连接可以稳定训练过程。
        
        <b>原理</b>：
        - 创建身份捷径：y = F(x) + x
        - 梯度可以直接通过跳跃连接传播，避免经过非线性激活函数
        - 即使某些层的梯度很大，整体梯度流仍然可以保持稳定
        
        <b>实现</b>：
        ```python
        # PyTorch残差块示例
        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            
            def forward(self, x):
                residual = x
                out = F.relu(self.conv1(x))
                out = self.conv2(out)
                out += residual  # 添加残差连接
                out = F.relu(out)
                return out
        ```
        
        残差连接已成为构建深度模型的标准技术，从CNN到Transformer都有广泛应用。
        </div>
        """, unsafe_allow_html=True)
    
    with solutions[5]:
        st.markdown("""
        <div class='explanation'>
        <b>激活函数选择</b>
        
        某些激活函数更容易导致梯度爆炸，而另一些则相对稳定。
        
        <b>更稳定的激活函数</b>：
        
        - <b>ELU (Exponential Linear Unit)</b>：
          - 负半轴有界，正半轴线性
          - 平滑过渡，导数连续
        
        - <b>Tanh</b>：
          - 输出范围有限 [-1, 1]
          - 在输入接近0时导数接近1，其他区域导数较小
        
        - <b>SELU (Scaled ELU)</b>：
          - 自归一化特性，保持激活均值和方差稳定
          - 可以防止深层网络中的梯度不稳定性
        
        <b>实现</b>：
        ```python
        # PyTorch示例
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.selu = nn.SELU()
        ```
        
        激活函数的选择应当与网络架构和任务特性相匹配，没有绝对最佳的选择。
        </div>
        """, unsafe_allow_html=True)
    
    # 实际应用
    st.markdown("<div class='sub-header'>实际应用中的梯度爆炸处理</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='explanation'>
    <b>梯度爆炸在不同模型中的处理</b>：
    
    1. <b>循环神经网络 (RNN/LSTM/GRU)</b>：
       - 梯度裁剪几乎是标准操作
       - 使用门控机制（如LSTM和GRU）控制信息流
       - 使用正交初始化和层归一化
    
    2. <b>深度CNN</b>：
       - 残差连接 + 批归一化
       - 合适的权重初始化（如He初始化）
       - 适当的学习率调度策略
    
    3. <b>Transformer</b>：
       - 层归一化和残差连接的结合
       - 梯度累积和梯度裁剪
       - 学习率预热（逐渐增加学习率）
    
    <b>检测和调试梯度爆炸</b>：
    - 监控损失值是否变为NaN或急剧增大
    - 跟踪梯度范数的变化趋势
    - 检查权重更新的幅度
    
    当训练不稳定时，应首先考虑降低学习率和应用梯度裁剪，这通常能解决大部分梯度爆炸问题。
    </div>
    """, unsafe_allow_html=True)
    
    # 代码示例
    with st.expander("PyTorch中实现梯度裁剪的代码示例"):
        st.code("""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # 定义模型
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练循环
        def train_with_gradient_clipping(model, train_loader, epochs=5, clip_value=1.0):
            for epoch in range(epochs):
                for data, target in train_loader:
                    # 前向传播
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 梯度裁剪 (基于范数)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    
                    # 更新参数
                    optimizer.step()
                    
                print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
        """, language="python")

if __name__ == "__main__":
    show_exploding_gradient_page() 