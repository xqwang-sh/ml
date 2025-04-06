import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import io
import base64
from PIL import Image

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Tanh激活函数"""
    return np.tanh(x)

def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU激活函数"""
    return np.maximum(alpha * x, x)

def elu(x, alpha=1.0):
    """ELU激活函数"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x, beta=1.0):
    """Swish激活函数"""
    return x * sigmoid(beta * x)

def plot_activation_functions():
    """绘制各种激活函数及其导数"""
    x = np.linspace(-10, 10, 1000)
    
    activations = {
        "Sigmoid": sigmoid,
        "Tanh": tanh,
        "ReLU": relu,
        "Leaky ReLU": leaky_relu,
        "ELU": elu,
        "Swish": swish
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('激活函数及其导数', fontsize=16)
    axes = axes.flatten()
    
    for i, (name, func) in enumerate(activations.items()):
        # 计算激活函数值
        y = func(x)
        
        # 计算导数（使用有限差分近似）
        h = 0.001
        x_deriv = x[1:-1]  # 排除边界点以避免越界
        y_deriv = (func(x_deriv + h) - func(x_deriv - h)) / (2 * h)
        
        # 绘制激活函数
        axes[i].plot(x, y, 'b-', linewidth=2, label=f'{name}函数')
        # 绘制导数
        axes[i].plot(x_deriv, y_deriv, 'r-', linewidth=2, label=f'{name}导数')
        
        axes[i].set_title(name)
        axes[i].grid(alpha=0.3)
        axes[i].legend()
        axes[i].set_ylim([-1.5, 2.0])  # 统一y轴范围
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[i].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整以适应标题
    return fig

def simulate_gradient_flow(n_layers=10, activation='sigmoid', batch_size=32):
    """
    模拟不同激活函数在深度网络中的梯度流动
    """
    # 设置随机种子以保证结果可重复
    np.random.seed(42)
    
    # 初始化网络参数
    weights = []
    activations = []
    gradients = []
    
    # 选择激活函数
    if activation == 'sigmoid':
        act_func = sigmoid
        # Sigmoid导数
        def act_grad(x):
            s = sigmoid(x)
            return s * (1 - s)
    elif activation == 'tanh':
        act_func = tanh
        # Tanh导数
        def act_grad(x):
            return 1 - np.tanh(x)**2
    elif activation == 'relu':
        act_func = relu
        # ReLU导数
        def act_grad(x):
            return np.where(x > 0, 1, 0)
    elif activation == 'leaky_relu':
        act_func = leaky_relu
        # Leaky ReLU导数
        def act_grad(x, alpha=0.01):
            return np.where(x > 0, 1, alpha)
    
    # 为每一层生成随机权重 (使用Xavier/Glorot初始化)
    input_size = 10  # 假设输入维度
    layer_sizes = [input_size] + [50] * n_layers + [1]  # 输入层、隐藏层、输出层
    
    for i in range(len(layer_sizes) - 1):
        # Xavier初始化
        scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
        W = np.random.normal(0, scale, (layer_sizes[i], layer_sizes[i+1]))
        weights.append(W)
    
    # 生成随机输入
    X = np.random.randn(batch_size, input_size)
    
    # 前向传播
    a = X
    activations.append(a)
    
    for W in weights:
        z = np.dot(a, W)
        a = act_func(z)
        activations.append(a)
    
    # 反向传播 (假设输出层误差为1)
    dout = np.ones_like(activations[-1])  # 输出层梯度
    gradients.append(dout)
    
    for i in reversed(range(len(weights))):
        # 当前层的激活值
        a = activations[i]
        # 下一层的加权输入
        z = np.dot(a, weights[i])
        # 激活函数导数
        dz = act_grad(z) * dout
        # 当前层的梯度
        da = np.dot(dz, weights[i].T)
        # 权重的梯度
        dW = np.dot(a.T, dz) / batch_size
        
        gradients.insert(0, da)  # 添加到梯度列表的前端
        dout = da  # 为下一层准备梯度
    
    # 计算每一层梯度的范数
    gradient_norms = [np.linalg.norm(g) / g.size for g in gradients]
    
    return gradient_norms

def plot_gradient_flow():
    """绘制不同激活函数的梯度流动情况"""
    n_layers = 20  # 层数
    activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
    labels = ['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, act in enumerate(activations):
        gradient_norms = simulate_gradient_flow(n_layers=n_layers, activation=act)
        # 只绘制隐藏层的梯度范数
        ax.semilogy(range(1, len(gradient_norms)-1), gradient_norms[1:-1], 
                   marker='o', label=labels[i])
    
    ax.set_title('不同激活函数的梯度流动对比', fontsize=16)
    ax.set_xlabel('网络层索引', fontsize=14)
    ax.set_ylabel('梯度范数 (对数尺度)', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    
    return fig

def plot_batchnorm_effect():
    """可视化批归一化对梯度流动的影响"""
    # 生成数据
    np.random.seed(42)
    n_layers = 50
    layer_indices = np.arange(1, n_layers + 1)
    
    # 模拟不同网络配置的梯度变化
    # 这里使用指数衰减来模拟梯度消失
    vanilla_gradients = 1.0 * (0.7 ** layer_indices)
    batchnorm_gradients = 1.0 * (0.95 ** layer_indices)
    resnet_gradients = 1.0 * (0.98 ** layer_indices)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.semilogy(layer_indices, vanilla_gradients, 'r-o', label='标准网络')
    ax.semilogy(layer_indices, batchnorm_gradients, 'g-o', label='带批归一化的网络')
    ax.semilogy(layer_indices, resnet_gradients, 'b-o', label='带残差连接的网络')
    
    ax.set_title('批归一化和残差连接对梯度流动的影响', fontsize=16)
    ax.set_xlabel('网络层索引', fontsize=14)
    ax.set_ylabel('梯度范数 (对数尺度)', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    
    return fig

def show_vanishing_gradient_page():
    st.markdown("<div class='sub-header'>梯度消失问题及解决方案</div>", unsafe_allow_html=True)
    
    # 问题说明
    with st.expander("什么是梯度消失问题?", expanded=True):
        st.markdown("""
        <div class='explanation'>
        <b>梯度消失</b>是深度神经网络训练中的一个关键问题，特别是在使用某些激活函数（如sigmoid或tanh）的深层网络中。
        
        当网络很深时，在反向传播过程中，梯度会通过链式法则从输出层向输入层传播。如果每一层的局部梯度小于1，
        那么梯度会随着层数的增加呈指数级减小，导致靠近输入层的权重几乎不会更新，这就是<b>梯度消失</b>问题。
        
        梯度消失会导致：
        - 深层网络训练困难或无法收敛
        - 训练过程缓慢
        - 网络无法学习长期依赖关系
        </div>
        """, unsafe_allow_html=True)
    
    # 激活函数可视化
    st.markdown("<div class='sub-header'>激活函数及其导数</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='explanation'>
    不同激活函数的导数特性直接影响梯度流动。观察下图中各激活函数的导数：
    - Sigmoid和Tanh的导数在输入较大或较小时接近于0，容易导致梯度消失
    - ReLU及其变种在正半轴导数为常数，有助于缓解梯度消失问题
    </div>
    """, unsafe_allow_html=True)
    
    fig_activations = plot_activation_functions()
    st.pyplot(fig_activations)
    
    # 梯度流动可视化
    st.markdown("<div class='sub-header'>不同激活函数的梯度流动</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='explanation'>
    下图展示了使用不同激活函数时，梯度在网络各层的传播情况。注意观察随着网络深度增加，梯度的变化趋势：
    - Sigmoid和Tanh：梯度随深度呈指数下降
    - ReLU和Leaky ReLU：梯度衰减较慢，能更好地传播到较深层
    </div>
    """, unsafe_allow_html=True)
    
    fig_gradient_flow = plot_gradient_flow()
    st.pyplot(fig_gradient_flow)
    
    # 解决方案
    st.markdown("<div class='sub-header'>梯度消失的解决方案</div>", unsafe_allow_html=True)
    
    solutions = st.tabs(["更好的激活函数", "批归一化", "残差连接", "LSTM/GRU", "权重初始化", "梯度裁剪"])
    
    with solutions[0]:
        st.markdown("""
        <div class='explanation'>
        <b>使用更好的激活函数</b>
        
        - <b>ReLU</b>：最常用的激活函数，在正半轴导数为1，有效缓解梯度消失，但可能导致"死亡ReLU"问题（神经元永久失活）
        - <b>Leaky ReLU</b>：在负半轴有一个小的正斜率(如0.01)，解决了"死亡ReLU"问题
        - <b>ELU (Exponential Linear Unit)</b>：负半轴为指数函数，结合了ReLU的优点并避免了死神经元问题
        - <b>SELU (Scaled ELU)</b>：自归一化特性，适合深度网络
        - <b>Swish</b>：x·sigmoid(x)，谷歌研究发现在某些任务上优于ReLU
        
        选择适当的激活函数可以显著减轻梯度消失问题。
        </div>
        """, unsafe_allow_html=True)
    
    with solutions[1]:
        st.markdown("""
        <div class='explanation'>
        <b>批归一化 (Batch Normalization)</b>
        
        批归一化是一种通过标准化每一层的输入来稳定和加速网络训练的技术。
        
        <b>工作原理</b>：
        1. 计算批次中每个特征的均值和方差
        2. 将特征标准化为均值0、方差1
        3. 通过可学习的缩放和偏移参数，使网络能够表示非零均值和非单位方差的数据分布
        
        <b>优势</b>：
        - 减缓梯度消失/爆炸
        - 允许使用更高的学习率，加速训练
        - 减少对精确初始化的依赖
        - 具有轻微正则化效果
        
        通常批归一化层放在全连接层或卷积层之后、激活函数之前。
        </div>
        """, unsafe_allow_html=True)
        
        fig_batchnorm = plot_batchnorm_effect()
        st.pyplot(fig_batchnorm)
    
    with solutions[2]:
        st.markdown("""
        <div class='explanation'>
        <b>残差连接 (Residual Connections)</b>
        
        残差网络(ResNet)通过添加"捷径连接"(shortcut connections)来解决梯度消失问题，使信号可以直接从一层传递到更深层。
        
        <b>工作原理</b>：
        - 不是直接学习函数映射 H(x)，而是学习残差 F(x) = H(x) - x
        - 原始输入 x 通过捷径直接添加到输出: H(x) = F(x) + x
        
        <b>优势</b>：
        - 梯度可以通过捷径连接直接流回较浅层，缓解梯度消失
        - 使训练非常深的网络成为可能 (ResNet-152, ResNet-1001等)
        - 不增加参数数量和计算复杂度
        
        残差连接已成为深度学习中的基本构建块，广泛应用于各种网络架构中。
        </div>
        """, unsafe_allow_html=True)
        
        # 残差连接示意图
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://production-media.paperswithcode.com/methods/resnet-e1548261771622.png", 
                    caption="残差块示意图", width=400)
    
    with solutions[3]:
        st.markdown("""
        <div class='explanation'>
        <b>LSTM和GRU</b>
        
        对于序列数据和时序问题，长短期记忆网络(LSTM)和门控循环单元(GRU)通过门控机制解决了RNN中的梯度消失问题。
        
        <b>LSTM特点</b>：
        - 包含三个门：输入门、遗忘门和输出门
        - 细胞状态作为信息高速公路，允许梯度无衰减传播
        - 可以学习长距离依赖关系
        
        <b>GRU特点</b>：
        - LSTM的简化版本，只有两个门：更新门和重置门
        - 性能通常与LSTM相当，但参数更少
        
        这些结构专为处理序列数据而设计，已在自然语言处理、语音识别等领域取得巨大成功。
        </div>
        """, unsafe_allow_html=True)
    
    with solutions[4]:
        st.markdown("""
        <div class='explanation'>
        <b>权重初始化策略</b>
        
        适当的权重初始化对于防止梯度消失至关重要。
        
        <b>常用初始化方法</b>：
        
        - <b>Xavier/Glorot初始化</b>：
          - 适用于tanh和sigmoid激活函数
          - 权重从均值为0、方差为2/(n_in + n_out)的分布中采样
          - 保持各层方差一致
        
        - <b>He初始化</b>：
          - 专为ReLU激活函数设计
          - 方差为2/n_in
          - 考虑到ReLU将约一半的激活设为0
        
        - <b>正交初始化</b>：
          - 使权重矩阵接近正交
          - 有助于保持梯度范数
          
        正确的初始化可以显著提高训练速度并改善收敛性。
        </div>
        """, unsafe_allow_html=True)
    
    with solutions[5]:
        st.markdown("""
        <div class='explanation'>
        <b>梯度裁剪</b>
        
        梯度裁剪是一种简单但有效的技术，可以防止梯度爆炸，并在某些情况下缓解梯度消失。
        
        <b>工作原理</b>：
        
        - <b>基于范数裁剪</b>：如果梯度的L2范数超过阈值，则按比例缩小
          ```python
          if ||g|| > threshold:
              g = threshold * g/||g||
          ```
        
        - <b>基于值裁剪</b>：将梯度值限制在特定范围内
          ```python
          g = np.clip(g, -threshold, threshold)
          ```
        
        梯度裁剪在RNN和LSTM等容易出现梯度爆炸的架构中特别有用，也可以稳定各种深度网络的训练。
        </div>
        """, unsafe_allow_html=True)
    
    # 实际应用案例
    st.markdown("<div class='sub-header'>实际应用中的梯度消失处理</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='explanation'>
    在实际深度学习应用中，通常会结合使用多种解决方案来处理梯度消失问题：
    
    - <b>ResNet</b>：结合残差连接、批归一化和ReLU激活函数
    - <b>Transformer</b>：使用层归一化、残差连接和多头注意力机制
    - <b>EfficientNet</b>：使用Swish激活函数、批归一化和复合缩放
    
    深度学习框架(如PyTorch, TensorFlow)提供了这些技术的简单实现，使开发人员能够轻松构建抗梯度消失的深度神经网络。
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_vanishing_gradient_page() 