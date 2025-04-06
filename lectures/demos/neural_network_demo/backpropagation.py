import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import io
import base64
from PIL import Image
import matplotlib.patches as patches
import time

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Sigmoid激活函数的导数"""
    s = sigmoid(x)
    return s * (1 - s)

def visualize_backpropagation():
    """创建反向传播过程的可视化"""
    # 创建一个简单的前馈神经网络结构
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制神经网络结构
    layer_sizes = [2, 3, 2, 1]  # 输入层、隐藏层1、隐藏层2、输出层
    layer_positions = [0, 1, 2, 3]
    
    # 绘制神经元
    neurons = {}
    for i, (layer_size, layer_pos) in enumerate(zip(layer_sizes, layer_positions)):
        neurons[i] = []
        for j in range(layer_size):
            y_pos = (layer_size - 1) / 2 - j
            neuron = plt.Circle((layer_pos, y_pos), 0.15, fill=True, color='skyblue')
            ax.add_patch(neuron)
            ax.text(layer_pos, y_pos, f"L{i}N{j}", ha='center', va='center', fontsize=8)
            neurons[i].append((layer_pos, y_pos))
    
    # 绘制连接
    for i in range(len(layer_sizes) - 1):
        for j, pos1 in enumerate(neurons[i]):
            for k, pos2 in enumerate(neurons[i + 1]):
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', alpha=0.3)
    
    # 设置图形参数
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-2, 2)
    ax.set_title("神经网络反向传播算法可视化", fontsize=14)
    ax.set_xticks(layer_positions)
    ax.set_xticklabels(['输入层', '隐藏层1', '隐藏层2', '输出层'])
    ax.set_yticks([])
    ax.axis('equal')
    
    return fig

def show_backpropagation_animation():
    """显示反向传播的动画过程"""
    # 创建示例数据
    np.random.seed(42)
    X = np.array([[0.1, 0.3]])  # 输入数据
    y = np.array([[0.7]])       # 目标值
    
    # 初始化权重
    W1 = np.random.randn(2, 3) * 0.1  # 输入层到隐藏层1
    b1 = np.zeros((1, 3))
    W2 = np.random.randn(3, 2) * 0.1  # 隐藏层1到隐藏层2
    b2 = np.zeros((1, 2))
    W3 = np.random.randn(2, 1) * 0.1  # 隐藏层2到输出层
    b3 = np.zeros((1, 1))
    
    # 前向传播
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)
    
    # 计算误差
    error = A3 - y
    
    # 反向传播
    dZ3 = error * sigmoid_derivative(Z3)
    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)
    
    dZ2 = np.dot(dZ3, W3.T) * sigmoid_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # 创建反向传播步骤的可视化
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('反向传播算法步骤', fontsize=16)
    
    # 第1步：计算输出层误差
    ax[0, 0].text(0.5, 0.5, '第1步：计算输出层误差\n' +
                 f'预测值：{A3[0][0]:.4f}\n' +
                 f'实际值：{y[0][0]:.4f}\n' +
                 f'误差：{error[0][0]:.4f}', 
                 ha='center', va='center', fontsize=12)
    ax[0, 0].axis('off')
    
    # 第2步：计算输出层梯度
    ax[0, 1].text(0.5, 0.5, '第2步：计算输出层梯度\n' +
                 f'dZ3 = 误差 * sigmoid导数(Z3)\n' +
                 f'dZ3 = {dZ3[0][0]:.4f}', 
                 ha='center', va='center', fontsize=12)
    ax[0, 1].axis('off')
    
    # 第3步：计算隐藏层梯度
    ax[1, 0].text(0.5, 0.5, '第3步：计算隐藏层梯度\n' +
                 f'dZ2 = dZ3 · W3.T * sigmoid导数(Z2)\n' +
                 f'dZ1 = dZ2 · W2.T * sigmoid导数(Z1)', 
                 ha='center', va='center', fontsize=12)
    ax[1, 0].axis('off')
    
    # 第4步：更新权重
    ax[1, 1].text(0.5, 0.5, '第4步：更新权重\n' +
                 f'dW3 = A2.T · dZ3\n' +
                 f'dW2 = A1.T · dZ2\n' +
                 f'dW1 = X.T · dZ1',
                 ha='center', va='center', fontsize=12)
    ax[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局以适应标题
    return fig

def create_interactive_backpropagation():
    """创建交互式反向传播演示"""
    # 创建网络结构
    st.markdown("### 交互式反向传播演示")
    st.markdown("""
    基于上面的反向传播步骤动画，现在您可以通过调整参数，自己尝试训练神经网络，
    观察不同参数设置如何影响网络训练效果。这将帮助您更好地理解反向传播算法的实际应用。
    """)
    
    # 用户可以选择的参数
    col1, col2, col3 = st.columns(3)
    with col1:
        learning_rate = st.slider("学习率", 0.01, 1.0, 0.1, 0.01)
    with col2:
        hidden_size = st.slider("隐藏层神经元数量", 2, 10, 2, 1)  # 默认值改为2以匹配动画演示
    with col3:
        iterations = st.slider("迭代次数", 1, 100, 10, 1)
    
    # 使用与动画相同的示例数据
    np.random.seed(42)
    # 创建一个包含多个类似动画中示例的数据集
    X = np.array([[0.5, 0.3], 
                   [0.4, 0.5],
                   [0.3, 0.4],
                   [0.7, 0.2],
                   [0.2, 0.6]])  # 5个样本，每个有2个特征
    y = np.array([[0.7], 
                   [0.8],
                   [0.6],
                   [0.8],
                   [0.5]])  # 目标值
    
    # 创建按钮开始训练
    if st.button("开始训练"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 初始化网络参数 - 使用与动画相似的初始值
        np.random.seed(42)
        W1 = np.random.randn(2, hidden_size) * 0.1
        b1 = np.zeros((1, hidden_size))
        W2 = np.random.randn(hidden_size, 1) * 0.1
        b2 = np.zeros((1, 1))
        
        st.markdown("#### 初始网络参数:")
        st.write(f"W1 = \n{W1}")
        st.write(f"b1 = \n{b1}")
        st.write(f"W2 = \n{W2}")
        st.write(f"b2 = \n{b2}")
        
        # 训练历史
        costs = []
        
        # 训练循环
        for i in range(iterations):
            # 前向传播
            Z1 = np.dot(X, W1) + b1
            A1 = sigmoid(Z1)
            Z2 = np.dot(A1, W2) + b2
            A2 = sigmoid(Z2)
            
            # 计算代价
            cost = -np.mean(y * np.log(A2 + 1e-8) + (1 - y) * np.log(1 - A2 + 1e-8))
            costs.append(cost)
            
            # 反向传播
            dZ2 = A2 - y
            dW2 = np.dot(A1.T, dZ2) / X.shape[0]
            db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]
            dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(Z1)
            dW1 = np.dot(X.T, dZ1) / X.shape[0]
            db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]
            
            # 更新参数
            W1 = W1 - learning_rate * dW1
            b1 = b1 - learning_rate * db1
            W2 = W2 - learning_rate * dW2
            b2 = b2 - learning_rate * db2
            
            # 更新进度条和状态
            progress = (i + 1) / iterations
            progress_bar.progress(progress)
            status_text.text(f"迭代 {i+1}/{iterations}，代价: {cost:.6f}")
        
        # 展示最终网络参数
        st.markdown("#### 训练后的网络参数:")
        st.write(f"W1 = \n{W1}")
        st.write(f"b1 = \n{b1}")
        st.write(f"W2 = \n{W2}")
        st.write(f"b2 = \n{b2}")
        
        # 绘制代价函数变化
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(costs)
        ax.set_title("代价函数变化")
        ax.set_xlabel("迭代次数")
        ax.set_ylabel("代价")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        # 测试结果
        Z1_test = np.dot(X, W1) + b1
        A1_test = sigmoid(Z1_test)
        Z2_test = np.dot(A1_test, W2) + b2
        A2_test = sigmoid(Z2_test)
        
        # 展示预测结果与实际值的比较
        results_df = pd.DataFrame({
            '输入特征1': X[:, 0],
            '输入特征2': X[:, 1],
            '目标值': y.flatten(),
            '预测值': A2_test.flatten(),
            '误差': (A2_test - y).flatten()
        })
        st.markdown("#### 预测结果:")
        st.dataframe(results_df)
        
        accuracy = np.mean(np.abs(A2_test - y) < 0.2)  # 误差小于0.2视为准确
        st.success(f"训练完成！准确率: {accuracy:.2%}")
        
        # 可视化决策过程
        st.markdown("#### 网络学习效果可视化:")
        st.markdown("下图展示了网络如何处理输入特征并产生预测值:")
        
        # 创建一个简单可视化，展示输入、隐藏层和输出
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 样本数量
        n_samples = min(3, X.shape[0])  # 仅可视化前3个样本
        
        for sample_idx in range(n_samples):
            # 计算该样本的前向传播
            x_sample = X[sample_idx:sample_idx+1]
            y_sample = y[sample_idx:sample_idx+1]
            
            z1_sample = np.dot(x_sample, W1) + b1
            a1_sample = sigmoid(z1_sample)
            z2_sample = np.dot(a1_sample, W2) + b2
            a2_sample = sigmoid(z2_sample)
            
            # 绘制该样本的网络图
            # 输入层
            input_x, input_y = 1, [3 - sample_idx*0.7, 1 - sample_idx*0.7]
            
            # 隐藏层
            hidden_x, hidden_y = 3, []
            for i in range(hidden_size):
                hidden_y.append(2 - i*1.5 - sample_idx*0.2)
            
            # 输出层
            output_x, output_y = 5, [2 - sample_idx*0.7]
            
            # 目标值
            target_x, target_y = 6.5, [2 - sample_idx*0.7]
            
            # 绘制神经元
            # 输入层
            for i, y_pos in enumerate(input_y):
                circle = plt.Circle((input_x, y_pos), 0.3, color='lightblue', alpha=0.6)
                ax.add_patch(circle)
                ax.text(input_x, y_pos, f"{x_sample[0][i]:.2f}", ha='center', va='center', fontsize=9)
            
            # 隐藏层
            for i, y_pos in enumerate(hidden_y):
                circle = plt.Circle((hidden_x, y_pos), 0.3, color='lightgreen', alpha=0.6)
                ax.add_patch(circle)
                ax.text(hidden_x, y_pos, f"{a1_sample[0][i]:.2f}", ha='center', va='center', fontsize=9)
            
            # 输出层
            for i, y_pos in enumerate(output_y):
                circle = plt.Circle((output_x, y_pos), 0.3, color='salmon', alpha=0.6)
                ax.add_patch(circle)
                ax.text(output_x, y_pos, f"{a2_sample[0][0]:.2f}", ha='center', va='center', fontsize=9)
            
            # 目标值
            for i, y_pos in enumerate(target_y):
                circle = plt.Circle((target_x, y_pos), 0.3, color='gold', alpha=0.6)
                ax.add_patch(circle)
                ax.text(target_x, y_pos, f"{y_sample[0][0]:.2f}", ha='center', va='center', fontsize=9)
            
            # 绘制连接
            # 输入层到隐藏层
            for i, i_y in enumerate(input_y):
                for j, h_y in enumerate(hidden_y):
                    alpha = min(1.0, abs(W1[i, j]) * 5)  # 权重大小影响线条透明度
                    ax.plot([input_x, hidden_x], [i_y, h_y], 'k-', alpha=alpha, linewidth=1+abs(W1[i, j])*3)
            
            # 隐藏层到输出层
            for i, h_y in enumerate(hidden_y):
                for j, o_y in enumerate(output_y):
                    alpha = min(1.0, abs(W2[i, j]) * 5)  # 权重大小影响线条透明度
                    ax.plot([hidden_x, output_x], [h_y, o_y], 'k-', alpha=alpha, linewidth=1+abs(W2[i, j])*3)
            
            # 输出层到目标值的误差
            ax.plot([output_x, target_x], [output_y[0], target_y[0]], 'r--', alpha=0.7)
            ax.text((output_x + target_x)/2, target_y[0]-0.4, f"误差={a2_sample[0][0]-y_sample[0][0]:.2f}", 
                   color='red', ha='center', fontsize=8)
        
        # 设置图形参数
        ax.set_xlim(0, 7.5)
        ax.set_ylim(-1, 4)
        ax.text(1, 3.5, "输入层", ha='center')
        ax.text(3, 3.5, "隐藏层", ha='center')
        ax.text(5, 3.5, "输出层", ha='center')
        ax.text(6.5, 3.5, "目标值", ha='center')
        ax.axis('off')
        
        st.pyplot(fig)
        
        st.markdown("""
        **说明：** 上图展示了训练后的神经网络如何处理输入数据。线条的粗细表示权重的大小，
        透明度也与权重相关。这直观地展示了网络学习到的模式。
        """)
        
        st.info("""
        **练习:** 
        1. 尝试调整学习率，观察收敛速度的变化
        2. 增加或减少隐藏层神经元数量，观察模型拟合能力的变化
        3. 增加迭代次数，观察最终误差是否继续下降
        """)

def create_detailed_backprop_animation():
    """创建详细的反向传播动画，展示从输入到输出，再到反向更新的完整过程"""
    st.markdown("### 反向传播算法步骤详解")
    st.markdown("""
    下面我们将通过一个简单的例子，详细展示反向传播算法的工作流程。首先我们会看到算法的每个步骤，
    然后您可以在交互式演示中尝试不同的参数设置，观察它们如何影响网络的学习过程。
    """)
    
    # 创建一个简单的神经网络示例
    np.random.seed(42)
    # 简单的示例数据
    x = np.array([0.5, 0.3])  # 输入数据
    y = np.array([0.7])       # 目标值
    
    # 初始化权重（使用固定值便于演示）
    W1 = np.array([[0.2, 0.3], 
                   [0.1, 0.4]])  # 输入层到隐藏层
    b1 = np.array([0.1, 0.2])    # 隐藏层偏置
    
    W2 = np.array([[0.3], 
                   [0.5]])       # 隐藏层到输出层
    b2 = np.array([0.1])         # 输出层偏置
    
    learning_rate = 0.1  # 学习率
    
    # 创建展示进度的容器
    animation_container = st.container()
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # 创建按钮开始动画
    if st.button("开始动画演示"):
        # 步骤1: 前向传播 - 输入层到隐藏层
        with animation_container:
            st.markdown("#### 步骤1: 前向传播 - 输入层到隐藏层")
            
            # 计算隐藏层的输入(Z1)和激活值(A1)
            Z1 = np.dot(x, W1) + b1
            A1 = sigmoid(Z1)
            
            # 可视化输入层到隐藏层
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制神经元
            # 输入层
            input_x, input_y = 1, [3, 1]
            for i, y_pos in enumerate(input_y):
                circle = plt.Circle((input_x, y_pos), 0.4, color='lightblue', alpha=0.8)
                ax.add_patch(circle)
                ax.text(input_x, y_pos, f"x{i+1}={x[i]:.2f}", ha='center', va='center')
            
            # 隐藏层
            hidden_x, hidden_y = 3, [3, 1]
            for i, y_pos in enumerate(hidden_y):
                circle = plt.Circle((hidden_x, y_pos), 0.4, color='lightgreen', alpha=0.8)
                ax.add_patch(circle)
                ax.text(hidden_x, y_pos, f"h{i+1}\nZ={Z1[i]:.2f}\nA={A1[i]:.2f}", ha='center', va='center', fontsize=9)
            
            # 绘制连接和权重
            for i, i_y in enumerate(input_y):
                for j, h_y in enumerate(hidden_y):
                    ax.arrow(input_x+0.4, i_y, hidden_x-input_x-0.8, h_y-i_y, 
                            head_width=0.1, head_length=0.1, fc='black', ec='black', alpha=0.6)
                    ax.text((input_x+hidden_x)/2, (i_y+h_y)/2, f"W={W1[i,j]:.2f}", 
                           ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
            
            # 绘制计算过程
            ax.text(2, 5, f"Z1 = X·W1 + b1\nZ1[0] = {x[0]:.2f}×{W1[0,0]:.2f} + {x[1]:.2f}×{W1[1,0]:.2f} + {b1[0]:.2f} = {Z1[0]:.4f}\n"+
                      f"Z1[1] = {x[0]:.2f}×{W1[0,1]:.2f} + {x[1]:.2f}×{W1[1,1]:.2f} + {b1[1]:.2f} = {Z1[1]:.4f}\n\n"+
                      f"A1 = sigmoid(Z1)\nA1[0] = sigmoid({Z1[0]:.4f}) = {A1[0]:.4f}\nA1[1] = sigmoid({Z1[1]:.4f}) = {A1[1]:.4f}",
                   bbox=dict(facecolor='lightyellow', alpha=0.9), fontsize=10)
            
            # 设置图形参数
            ax.set_xlim(0, 6)
            ax.set_ylim(0, 6)
            ax.axis('off')
            
            st.pyplot(fig)
            progress_bar.progress(0.2)
            status_text.text("步骤1: 前向传播 - 输入层到隐藏层完成")
            time.sleep(1)  # 暂停1秒
        
        # 步骤2: 前向传播 - 隐藏层到输出层
        with animation_container:
            st.markdown("#### 步骤2: 前向传播 - 隐藏层到输出层")
            
            # 计算输出层的输入(Z2)和激活值(A2)
            Z2 = np.dot(A1, W2) + b2
            A2 = sigmoid(Z2)
            
            # 可视化隐藏层到输出层
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制神经元
            # 隐藏层
            hidden_x, hidden_y = 1, [3, 1]
            for i, y_pos in enumerate(hidden_y):
                circle = plt.Circle((hidden_x, y_pos), 0.4, color='lightgreen', alpha=0.8)
                ax.add_patch(circle)
                ax.text(hidden_x, y_pos, f"h{i+1}={A1[i]:.2f}", ha='center', va='center')
            
            # 输出层
            output_x, output_y = 3, [2]
            for i, y_pos in enumerate(output_y):
                circle = plt.Circle((output_x, y_pos), 0.4, color='salmon', alpha=0.8)
                ax.add_patch(circle)
                ax.text(output_x, y_pos, f"ŷ\nZ={Z2[0]:.2f}\nA={A2[0]:.2f}", ha='center', va='center', fontsize=9)
            
            # 目标值
            target_x, target_y = 5, [2]
            for i, y_pos in enumerate(target_y):
                circle = plt.Circle((target_x, y_pos), 0.4, color='gold', alpha=0.8)
                ax.add_patch(circle)
                ax.text(target_x, y_pos, f"y={y[0]:.2f}", ha='center', va='center')
            
            # 绘制连接和权重
            for i, h_y in enumerate(hidden_y):
                for j, o_y in enumerate(output_y):
                    ax.arrow(hidden_x+0.4, h_y, output_x-hidden_x-0.8, o_y-h_y, 
                            head_width=0.1, head_length=0.1, fc='black', ec='black', alpha=0.6)
                    ax.text((hidden_x+output_x)/2, (h_y+o_y)/2, f"W={W2[i,j]:.2f}", 
                           ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
            
            # 误差
            error = A2 - y
            ax.arrow(output_x+0.4, output_y[0], target_x-output_x-0.8, target_y[0]-output_y[0], 
                    head_width=0.1, head_length=0.1, fc='red', ec='red')
            ax.text((output_x+target_x)/2, output_y[0]-0.5, f"误差={error[0]:.4f}", 
                   ha='center', va='center', color='red', fontsize=12)
            
            # 绘制计算过程
            ax.text(2, 5, f"Z2 = A1·W2 + b2\nZ2 = {A1[0]:.4f}×{W2[0,0]:.2f} + {A1[1]:.4f}×{W2[1,0]:.2f} + {b2[0]:.2f} = {Z2[0]:.4f}\n\n"+
                     f"A2 = sigmoid(Z2)\nA2 = sigmoid({Z2[0]:.4f}) = {A2[0]:.4f}\n\n"+
                     f"误差 = A2 - y = {A2[0]:.4f} - {y[0]:.2f} = {error[0]:.4f}",
                   bbox=dict(facecolor='lightyellow', alpha=0.9), fontsize=10)
            
            # 设置图形参数
            ax.set_xlim(0, 6)
            ax.set_ylim(0, 6)
            ax.set_title("前向传播: 隐藏层到输出层", fontsize=14)
            ax.axis('off')
            
            st.pyplot(fig)
            progress_bar.progress(0.4)
            status_text.text("步骤2: 前向传播 - 隐藏层到输出层完成")
            time.sleep(1)  # 暂停1秒
        
        # 步骤3: 反向传播 - 计算输出层梯度
        with animation_container:
            st.markdown("#### 步骤3: 反向传播 - 计算输出层梯度")
            
            # 计算输出层梯度
            dZ2 = error * sigmoid_derivative(Z2)
            dW2 = np.outer(A1, dZ2)
            db2 = dZ2.copy()
            
            # 可视化输出层梯度
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制神经元
            # 隐藏层
            hidden_x, hidden_y = 1, [3, 1]
            for i, y_pos in enumerate(hidden_y):
                circle = plt.Circle((hidden_x, y_pos), 0.4, color='lightgreen', alpha=0.8)
                ax.add_patch(circle)
                ax.text(hidden_x, y_pos, f"h{i+1}={A1[i]:.2f}", ha='center', va='center')
            
            # 输出层
            output_x, output_y = 3, [2]
            for i, y_pos in enumerate(output_y):
                circle = plt.Circle((output_x, y_pos), 0.4, color='salmon', alpha=0.8)
                ax.add_patch(circle)
                ax.text(output_x, y_pos, f"ŷ={A2[0]:.2f}\ndZ={dZ2[0]:.4f}", ha='center', va='center', fontsize=9)
            
            # 绘制连接和权重梯度
            for i, h_y in enumerate(hidden_y):
                for j, o_y in enumerate(output_y):
                    ax.arrow(output_x-0.4, o_y, hidden_x-output_x+0.8, h_y-o_y, 
                            head_width=0.1, head_length=0.1, fc='red', ec='red', linestyle='--', alpha=0.6)
                    ax.text((hidden_x+output_x)/2, (h_y+o_y)/2+0.3, f"dW={dW2[i,j]:.4f}", 
                           ha='center', va='center', color='red', bbox=dict(facecolor='white', alpha=0.7))
            
            # 绘制计算过程
            ax.text(2, 5, f"dZ2 = 误差 * sigmoid'(Z2)\ndZ2 = {error[0]:.4f} * sigmoid'({Z2[0]:.4f}) = {dZ2[0]:.4f}\n\n"+
                     f"dW2 = A1.T · dZ2\ndW2[0,0] = {A1[0]:.4f} * {dZ2[0]:.4f} = {dW2[0,0]:.4f}\ndW2[1,0] = {A1[1]:.4f} * {dZ2[0]:.4f} = {dW2[1,0]:.4f}\n\n"+
                     f"db2 = dZ2 = {db2[0]:.4f}",
                   bbox=dict(facecolor='lightyellow', alpha=0.9), fontsize=10)
            
            # 设置图形参数
            ax.set_xlim(0, 6)
            ax.set_ylim(0, 6)
            ax.axis('off')
            
            st.pyplot(fig)
            progress_bar.progress(0.6)
            status_text.text("步骤3: 反向传播 - 计算输出层梯度完成")
            time.sleep(1)  # 暂停1秒
        
        # 步骤4: 反向传播 - 计算隐藏层梯度
        with animation_container:
            st.markdown("#### 步骤4: 反向传播 - 计算隐藏层梯度")
            
            # 计算隐藏层梯度
            dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(Z1)
            dW1 = np.outer(x, dZ1)
            db1 = dZ1.copy()
            
            # 可视化隐藏层梯度
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制神经元
            # 输入层
            input_x, input_y = 1, [3, 1]
            for i, y_pos in enumerate(input_y):
                circle = plt.Circle((input_x, y_pos), 0.4, color='lightblue', alpha=0.8)
                ax.add_patch(circle)
                ax.text(input_x, y_pos, f"x{i+1}={x[i]:.2f}", ha='center', va='center')
            
            # 隐藏层
            hidden_x, hidden_y = 3, [3, 1]
            for i, y_pos in enumerate(hidden_y):
                circle = plt.Circle((hidden_x, y_pos), 0.4, color='lightgreen', alpha=0.8)
                ax.add_patch(circle)
                ax.text(hidden_x, y_pos, f"h{i+1}={A1[i]:.2f}\ndZ={dZ1[i]:.4f}", ha='center', va='center', fontsize=9)
            
            # 绘制连接和权重梯度
            for i, i_y in enumerate(input_y):
                for j, h_y in enumerate(hidden_y):
                    ax.arrow(hidden_x-0.4, h_y, input_x-hidden_x+0.8, i_y-h_y, 
                            head_width=0.1, head_length=0.1, fc='red', ec='red', linestyle='--', alpha=0.6)
                    ax.text((input_x+hidden_x)/2, (i_y+h_y)/2+0.3, f"dW={dW1[i,j]:.4f}", 
                           ha='center', va='center', color='red', bbox=dict(facecolor='white', alpha=0.7))
            
            # 绘制计算过程
            ax.text(2, 5, f"dZ1 = dZ2 · W2.T * sigmoid'(Z1)\ndZ1[0] = {dZ2[0]:.4f} * {W2[0,0]:.2f} * sigmoid'({Z1[0]:.4f}) = {dZ1[0]:.4f}\ndZ1[1] = {dZ2[0]:.4f} * {W2[1,0]:.2f} * sigmoid'({Z1[1]:.4f}) = {dZ1[1]:.4f}\n\n"+
                     f"dW1 = X.T · dZ1\ndW1[0,0] = {x[0]:.2f} * {dZ1[0]:.4f} = {dW1[0,0]:.4f}\ndW1[0,1] = {x[0]:.2f} * {dZ1[1]:.4f} = {dW1[0,1]:.4f}\ndW1[1,0] = {x[1]:.2f} * {dZ1[0]:.4f} = {dW1[1,0]:.4f}\ndW1[1,1] = {x[1]:.2f} * {dZ1[1]:.4f} = {dW1[1,1]:.4f}\n\n"+
                     f"db1[0] = dZ1[0] = {db1[0]:.4f}\ndb1[1] = dZ1[1] = {db1[1]:.4f}",
                   bbox=dict(facecolor='lightyellow', alpha=0.9), fontsize=9)
            
            # 设置图形参数
            ax.set_xlim(0, 6)
            ax.set_ylim(0, 6)
            ax.axis('off')
            
            st.pyplot(fig)
            progress_bar.progress(0.8)
            status_text.text("步骤4: 反向传播 - 计算隐藏层梯度完成")
            time.sleep(1)  # 暂停1秒
        
        # 步骤5: 更新权重
        with animation_container:
            st.markdown("#### 步骤5: 更新网络权重和偏置")
            
            # 使用梯度下降更新权重
            W1_new = W1 - learning_rate * dW1
            b1_new = b1 - learning_rate * db1
            W2_new = W2 - learning_rate * dW2
            b2_new = b2 - learning_rate * db2
            
            # 可视化权重更新
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 显示更新表格
            data = {
                '参数': ['W1[0,0]', 'W1[0,1]', 'W1[1,0]', 'W1[1,1]', 'b1[0]', 'b1[1]', 'W2[0,0]', 'W2[1,0]', 'b2[0]'],
                '旧值': [W1[0,0], W1[0,1], W1[1,0], W1[1,1], b1[0], b1[1], W2[0,0], W2[1,0], b2[0]],
                '梯度': [dW1[0,0], dW1[0,1], dW1[1,0], dW1[1,1], db1[0], db1[1], dW2[0,0], dW2[1,0], db2[0]],
                '更新量': [learning_rate * dW1[0,0], learning_rate * dW1[0,1], learning_rate * dW1[1,0], learning_rate * dW1[1,1], 
                       learning_rate * db1[0], learning_rate * db1[1], learning_rate * dW2[0,0], learning_rate * dW2[1,0], learning_rate * db2[0]],
                '新值': [W1_new[0,0], W1_new[0,1], W1_new[1,0], W1_new[1,1], b1_new[0], b1_new[1], W2_new[0,0], W2_new[1,0], b2_new[0]]
            }
            
            ax.axis('tight')
            ax.axis('off')
            
            # 修复表格数据处理
            table_data = []
            for i in range(len(data['参数'])):
                row = []
                for key in list(data.keys()):  # 转换为列表，然后遍历
                    value = data[key][i]
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(value)
                table_data.append(row)
            
            table = ax.table(cellText=table_data,
                           colLabels=list(data.keys()),  # 转换为列表
                           loc='center',
                           cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # 设置更新公式
            ax.text(0.5, 0.9, f"权重更新公式: W_new = W_old - learning_rate * dW\n学习率 = {learning_rate}", 
                   ha='center', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(facecolor='lightyellow', alpha=0.9))
            
            # 再次前向传播计算新误差
            Z1_new = np.dot(x, W1_new) + b1_new
            A1_new = sigmoid(Z1_new)
            Z2_new = np.dot(A1_new, W2_new) + b2_new
            A2_new = sigmoid(Z2_new)
            new_error = A2_new - y
            
            # 修复误差比较部分
            ax.text(0.5, 0.1, f"更新权重后的结果:\n预测值: {A2[0]:.4f} → {A2_new[0]:.4f}\n误差: {error[0]:.4f} → {new_error[0]:.4f}", 
                   ha='center', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(facecolor='lightgreen', alpha=0.9))
            
            
            st.pyplot(fig)
            progress_bar.progress(1.0)
            status_text.text("步骤5: 更新网络权重和偏置")
        
        st.success("反向传播动画演示完成！")

def show_backpropagation_page():
    """显示反向传播算法页面"""
    st.markdown("<div class='sub-header'>神经网络反向传播算法</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='explanation'>
    反向传播算法（Backpropagation）是训练神经网络的核心算法，它通过计算损失函数相对于网络参数的梯度，并沿着梯度的反方向更新参数，使网络逐步优化。
    </div>
    """, unsafe_allow_html=True)
    
    # 反向传播的原理
    st.markdown("### 反向传播的基本原理")
    st.markdown("""
    1. **前向传播**：输入数据从输入层传递到隐藏层，再传递到输出层，得到预测值。
    2. **计算误差**：将预测值与实际值进行比较，计算误差。
    3. **反向传播**：从输出层开始，向后计算每一层参数的梯度。
    4. **参数更新**：使用梯度下降法更新网络参数。
    """)
    
    # 公式说明
    st.markdown("### 关键公式")
    st.markdown("""
    <div class='formula'>
    输出层误差: δ<sup>L</sup> = ∇<sub>a</sub>C ⊙ σ'(z<sup>L</sup>)
    </div>
    
    <div class='formula'>
    隐藏层误差: δ<sup>l</sup> = ((W<sup>l+1</sup>)<sup>T</sup>δ<sup>l+1</sup>) ⊙ σ'(z<sup>l</sup>)
    </div>
    
    <div class='formula'>
    参数梯度: ∇<sub>W</sub>C = a<sup>l-1</sup>(δ<sup>l</sup>)<sup>T</sup>
    </div>
    
    <div class='formula'>
    偏置梯度: ∇<sub>b</sub>C = δ<sup>l</sup>
    </div>
    """, unsafe_allow_html=True)
    
    # 神经网络结构可视化
    st.markdown("### 神经网络结构")
    fig = visualize_backpropagation()
    st.pyplot(fig)
    
    # 添加详细的反向传播可视化动画
    st.markdown("### 反向传播算法步骤详解")
    st.markdown("""
    下面我们将通过一个简单的例子，详细展示反向传播算法的工作流程。首先我们会看到算法的每个步骤，
    然后您可以在交互式演示中尝试不同的参数设置，观察它们如何影响网络的学习过程。
    """)
    create_detailed_backprop_animation()
    
    # 交互式演示
    create_interactive_backpropagation()
    
    # 补充说明
    st.markdown("### 常见问题与解决方法")
    st.markdown("""
    1. **梯度消失**：在深层网络中，梯度可能变得非常小，导致参数几乎不更新。
       - 解决方法：使用ReLU等激活函数，添加残差连接，使用批归一化等。
    
    2. **梯度爆炸**：梯度值变得非常大，导致参数更新过度。
       - 解决方法：梯度裁剪，权重正则化等。
    
    3. **局部最小值**：梯度下降可能陷入局部最优解。
       - 解决方法：使用动量方法，自适应学习率等优化算法。
    """)
    
    st.markdown("### 总结")
    st.markdown("""
    反向传播算法是神经网络学习的核心机制，通过它，网络能够自动调整参数以减小预测误差。理解反向传播的工作原理
    对于掌握深度学习至关重要。以上的步骤动画和交互式演示展示了算法的基本工作流程，希望能帮助您更好地理解这一重要概念。
    
    在实际应用中，我们通常使用深度学习框架（如TensorFlow、PyTorch）来自动实现反向传播，但理解其原理有助于我们
    设计更有效的网络结构和选择合适的优化策略。
    """) 