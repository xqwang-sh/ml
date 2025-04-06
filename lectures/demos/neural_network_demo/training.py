import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
import base64
from PIL import Image
import os
from matplotlib.lines import Line2D
from sklearn.datasets import make_moons

def cost_function(x, y, params):
    """
    简单的二次代价函数 J = (y - f(x))^2
    """
    m, b = params
    y_pred = m * x + b
    return ((y - y_pred) ** 2).mean()

def gradient_function(x, y, params):
    """
    代价函数对参数的梯度
    """
    m, b = params
    y_pred = m * x + b
    
    # 梯度计算
    dm = -2 * x.dot(y - y_pred) / len(x)
    db = -2 * np.sum(y - y_pred) / len(x)
    
    return np.array([dm, db])

def animate_gradient_descent(x, y, learning_rate=0.1, num_iterations=20):
    """
    创建梯度下降动画
    """
    # 初始参数
    params = np.array([0.0, 0.0])  # 初始 m, b
    
    # 保存每次迭代的参数和代价
    all_params = [params.copy()]
    all_costs = [cost_function(x, y, params)]
    
    # 运行梯度下降
    for _ in range(num_iterations):
        grads = gradient_function(x, y, params)
        params = params - learning_rate * grads
        
        all_params.append(params.copy())
        all_costs.append(cost_function(x, y, params))
    
    # 创建网格用于绘制代价函数轮廓
    m_range = np.linspace(-3, 3, 50)
    b_range = np.linspace(-3, 3, 50)
    M, B = np.meshgrid(m_range, b_range)
    Z = np.zeros(M.shape)
    
    # 计算每个网格点的代价
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Z[i, j] = cost_function(x, y, [M[i, j], B[i, j]])
    
    # 创建3D图形 - 将布局从横向(1,3)改为竖向(3,1)
    fig = plt.figure(figsize=(10, 18))
    
    # 代价函数轮廓图
    ax1 = fig.add_subplot(3, 1, 1)
    contour = ax1.contourf(M, B, Z, 50, cmap='viridis', alpha=0.8)
    
    # 先画一条完整路径，以便用户可以立即看到
    m_values = [p[0] for p in all_params]
    b_values = [p[1] for p in all_params]
    ax1.plot(m_values, b_values, 'r-', linewidth=1, alpha=0.5)
    
    # 然后为动画准备空路径
    path_line, = ax1.plot([], [], 'r-', linewidth=2)
    path_point, = ax1.plot([], [], 'ro', markersize=8)
    
    ax1.set_xlabel('斜率 (m)', fontsize=12)
    ax1.set_ylabel('截距 (b)', fontsize=12)
    ax1.set_title('代价函数轮廓与梯度下降路径', fontsize=14)
    plt.colorbar(contour, ax=ax1)
    
    # 代价函数3D表面
    ax2 = fig.add_subplot(3, 1, 2, projection='3d')
    surface = ax2.plot_surface(M, B, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    
    # 为3D图添加完整路径
    z_values = all_costs
    ax2.plot(m_values, b_values, z_values, 'r-', linewidth=1, alpha=0.5)
    
    # 然后为动画准备空路径
    path_line_3d, = ax2.plot([], [], [], 'r-', linewidth=2)
    path_point_3d, = ax2.plot([], [], [], 'ro', markersize=8)
    
    ax2.set_xlabel('斜率 (m)', fontsize=12)
    ax2.set_ylabel('截距 (b)', fontsize=12)
    ax2.set_zlabel('代价 J(m,b)', fontsize=12)
    ax2.set_title('代价函数3D表面', fontsize=14)
    
    # 数据拟合图
    ax3 = fig.add_subplot(3, 1, 3)
    scatter = ax3.scatter(x, y, color='blue', s=50, alpha=0.8)
    line, = ax3.plot([], [], 'r-', linewidth=2)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    ax3.set_title('数据与模型拟合', fontsize=14)
    ax3.grid(alpha=0.3)
    
    # 限制轴范围
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    margin = 0.2
    ax3.set_xlim([x_min - margin, x_max + margin])
    ax3.set_ylim([y_min - margin, y_max + margin])
    
    fig.tight_layout()
    
    # 初始化函数
    def init():
        path_line.set_data([], [])
        path_point.set_data([], [])
        path_line_3d.set_data([], [])
        path_line_3d.set_3d_properties([])
        path_point_3d.set_data([], [])
        path_point_3d.set_3d_properties([])
        line.set_data([], [])
        return path_line, path_point, path_line_3d, path_point_3d, line
    
    # 更新函数
    def update(frame):
        # 更新轮廓图中的路径
        m_values = [p[0] for p in all_params[:frame+1]]
        b_values = [p[1] for p in all_params[:frame+1]]
        
        path_line.set_data(m_values, b_values)
        path_point.set_data(m_values[-1], b_values[-1])
        
        # 更新3D图中的路径
        z_values = [all_costs[i] for i in range(frame+1)]
        path_line_3d.set_data(m_values, b_values)
        path_line_3d.set_3d_properties(z_values)
        path_point_3d.set_data([m_values[-1]], [b_values[-1]])
        path_point_3d.set_3d_properties([z_values[-1]])
        
        # 更新拟合线
        m, b = all_params[frame]
        x_line = np.array([x_min - margin, x_max + margin])
        y_line = m * x_line + b
        line.set_data(x_line, y_line)
        
        return path_line, path_point, path_line_3d, path_point_3d, line
    
    ani = FuncAnimation(fig, update, frames=num_iterations+1, init_func=init, blit=True, interval=500)
    
    # 保存为GIF时使用正确的参数
    buf = io.BytesIO()
    # 使用临时文件路径而不是直接保存到BytesIO
    temp_path = 'temp_animation.gif'
    ani.save(temp_path, writer='pillow', fps=2, dpi=80)
    
    # 读取临时文件到BytesIO
    with open(temp_path, 'rb') as f:
        buf.write(f.read())
    
    # 删除临时文件
    os.remove(temp_path)
    
    buf.seek(0)
    
    return buf

# --- 将 sigmoid 和 predict 定义移到顶层 --- 
def sigmoid(z):
    """ Sigmoid 激活函数 (带数值稳定性) """
    z = np.clip(z, -500, 500) # 防止 overflow/underflow
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """ Sigmoid 函数的导数 """
    s = sigmoid(z)
    return s * (1 - s)

def predict(X_pred, W1, b1, W2, b2):
    """ 使用训练好的参数进行预测 """
    Z1 = np.dot(X_pred, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A2
# ---------------------------------------------

def simple_neural_network(X, y, hidden_size=4, learning_rate=0.5, epochs=15):
    """
    实现一个简单的神经网络训练过程，并记录训练历史
    """
    np.random.seed(42)  # 保证结果可重复
    
    input_size = X.shape[1]
    output_size = 1
    
    # 初始化参数
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))
    
    # 保存训练历史
    history = {
        'loss': [],
        'params': []
    }
    # 存储初始参数
    history['params'].append({
        'W1': W1.copy(), 'b1': b1.copy(), 'W2': W2.copy(), 'b2': b2.copy()
    })
    
    # 训练循环 (现在调用顶层的 sigmoid 和 sigmoid_derivative)
    for i in range(epochs):
        # 前向传播
        Z1 = np.dot(X, W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)
        
        # 计算代价 (添加 epsilon 防止 log(0))
        cost = -np.mean(y * np.log(A2 + 1e-8) + (1 - y) * np.log(1 - A2 + 1e-8))
        history['loss'].append(cost)
        
        # 反向传播
        dZ2 = A2 - y
        dW2 = np.dot(A1.T, dZ2) / X.shape[0]
        db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]
        dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(Z1) # 使用顶层的 sigmoid_derivative
        dW1 = np.dot(X.T, dZ1) / X.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]
        
        # 更新参数
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        
        # 保存当前参数
        history['params'].append({
            'W1': W1.copy(),
            'b1': b1.copy(), 
            'W2': W2.copy(),
            'b2': b2.copy()
        })
    
    return history

def plot_neural_network_training(X, y, history):
    """
    绘制神经网络训练过程
    """
    epochs = len(history['loss'])  # 训练轮次数
    params_count = len(history['params'])  # 参数集数量 (初始参数 + 每轮训练后参数)
    
    # 创建图形 - 将布局从横向(1,3)改为竖向(3,1)
    fig = plt.figure(figsize=(10, 18))
    
    # 绘制损失函数变化
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(range(epochs), history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('训练轮次', fontsize=12)
    ax1.set_ylabel('损失', fontsize=12)
    ax1.set_title('训练过程中的损失变化', fontsize=14)
    ax1.grid(alpha=0.3)
    
    # 绘制决策边界变化
    ax2 = fig.add_subplot(3, 1, 2)
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 初始决策边界
    W1_init = history['params'][0]['W1']
    b1_init = history['params'][0]['b1']
    W2_init = history['params'][0]['W2']
    b2_init = history['params'][0]['b2']
    
    # 现在调用顶层的 predict 函数
    Z_init = predict(grid_points, W1_init, b1_init, W2_init, b2_init)
    Z_init = Z_init.reshape(xx.shape)
    
    # 最终决策边界
    W1_final = history['params'][-1]['W1']
    b1_final = history['params'][-1]['b1']
    W2_final = history['params'][-1]['W2']
    b2_final = history['params'][-1]['b2']
    
    # 现在调用顶层的 predict 函数
    Z_final = predict(grid_points, W1_final, b1_final, W2_final, b2_final)
    Z_final = Z_final.reshape(xx.shape)
    
    # 绘制数据点
    ax2.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], c='#D81B60', 
               marker='o', s=100, label='类别 0', edgecolors='k', zorder=10)
    ax2.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], c='#1E88E5', 
               marker='^', s=100, label='类别 1', edgecolors='k', zorder=10)
               
    # 绘制填充区域
    contour = ax2.contourf(xx, yy, Z_final, levels=50, cmap='Blues', alpha=0.3, zorder=1)
    
    # 绘制初始边界
    init_line = ax2.contour(xx, yy, Z_init, levels=[0.5], colors=['darkred'], 
                           linestyles='--', linewidths=3, zorder=5)
                           
    # 绘制最终边界
    final_line = ax2.contour(xx, yy, Z_final, levels=[0.5], colors=['darkgreen'], 
                            linestyles='-', linewidths=3, zorder=5)
    
    # 添加手动图例
    legend_elements = [
        Line2D([0], [0], color='darkred', linestyle='--', lw=3, label='初始边界'),
        Line2D([0], [0], color='darkgreen', linestyle='-', lw=3, label='最终边界')
    ]
    
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    ax2.set_xlabel('特征 x1', fontsize=12)
    ax2.set_ylabel('特征 x2', fontsize=12)
    ax2.set_title('决策边界变化', fontsize=14)
    
    # 合并图例
    handles, labels = ax2.get_legend_handles_labels()
    init_handles = init_line.collections[0].get_paths() if init_line.collections else []
    final_handles = final_line.collections[0].get_paths() if final_line.collections else []
    valid_legend_elements = []
    if init_handles:
        valid_legend_elements.append(legend_elements[0])
    if final_handles:
        valid_legend_elements.append(legend_elements[1])
        
    ax2.legend(handles=valid_legend_elements + handles, loc='upper right', fontsize=10)
    
    ax2.grid(alpha=0.3)
    
    # 绘制参数变化
    ax3 = fig.add_subplot(3, 1, 3)
    
    # 跟踪几个主要参数的变化 - 修改以匹配 loss 的长度
    # 使用 history['params'][1:] 跳过初始参数，只使用训练后的参数
    w11 = [hist['W1'][0, 0] for hist in history['params'][1:]]  # 跳过初始参数
    w12 = [hist['W1'][0, 1] for hist in history['params'][1:]]
    w21 = [hist['W2'][0, 0] for hist in history['params'][1:]]
    w22 = [hist['W2'][1, 0] for hist in history['params'][1:]]
    
    ax3.plot(range(epochs), w11, 'r-', label='W1[0,0]', linewidth=2)
    ax3.plot(range(epochs), w12, 'g-', label='W1[0,1]', linewidth=2)
    ax3.plot(range(epochs), w21, 'b-', label='W2[0,0]', linewidth=2)
    ax3.plot(range(epochs), w22, 'c-', label='W2[1,0]', linewidth=2)
    
    ax3.set_xlabel('训练轮次', fontsize=12)
    ax3.set_ylabel('参数值', fontsize=12)
    ax3.set_title('参数变化', fontsize=14)
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_animated_gif(buf):
    """
    从buffer创建可在Streamlit中显示的GIF图像
    """
    # 直接返回BytesIO对象
    return buf

def create_optimization_algorithms_comparison():
    """
    创建优化算法比较图
    """
    # 设置随机种子
    np.random.seed(42)
    
    # 创建一个非凸函数
    def complex_function(x):
        return 0.1 * x**4 - 0.5 * x**3 - 0.2 * x**2 + 2 * x
    
    # 定义各种优化算法
    def sgd(x, lr=0.1):
        return x - lr * (0.4 * x**3 - 1.5 * x**2 - 0.4 * x + 2)
    
    def momentum(x, v, lr=0.1, beta=0.9):
        v_new = beta * v - lr * (0.4 * x**3 - 1.5 * x**2 - 0.4 * x + 2)
        x_new = x + v_new
        return x_new, v_new
    
    def adagrad(x, G, lr=0.1, epsilon=1e-8):
        grad = 0.4 * x**3 - 1.5 * x**2 - 0.4 * x + 2
        G_new = G + grad**2
        x_new = x - lr * grad / (np.sqrt(G_new) + epsilon)
        return x_new, G_new
    
    def rmsprop(x, G, lr=0.1, beta=0.9, epsilon=1e-8):
        grad = 0.4 * x**3 - 1.5 * x**2 - 0.4 * x + 2
        G_new = beta * G + (1 - beta) * grad**2
        x_new = x - lr * grad / (np.sqrt(G_new) + epsilon)
        return x_new, G_new
    
    def adam(x, m, v, t, lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        grad = 0.4 * x**3 - 1.5 * x**2 - 0.4 * x + 2
        m_new = beta1 * m + (1 - beta1) * grad
        v_new = beta2 * v + (1 - beta2) * grad**2
        
        # 偏差修正
        m_hat = m_new / (1 - beta1**t)
        v_hat = v_new / (1 - beta2**t)
        
        x_new = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        return x_new, m_new, v_new
    
    # 创建图形 - 将布局从横向(2,3)改为竖向(6,1)
    x = np.linspace(-3, 3, 1000)
    y_func = complex_function(x)
    
    fig, axs = plt.subplots(6, 1, figsize=(10, 24))
    axs = axs.flatten()
    
    # 绘制目标函数
    for i in range(6):
        axs[i].plot(x, y_func, 'b-', alpha=0.3)
        axs[i].set_xlim([-3, 3])
        axs[i].set_ylim([min(y_func)-1, max(y_func)+1])
        axs[i].grid(alpha=0.3)
    
    # 优化起点
    start_x = 2.5
    
    # SGD
    x_sgd = start_x
    x_history_sgd = [x_sgd]
    
    for _ in range(20):
        x_sgd = sgd(x_sgd)
        x_history_sgd.append(x_sgd)
    
    axs[0].scatter(x_history_sgd, [complex_function(x_val) for x_val in x_history_sgd], c='r', s=50)
    axs[0].plot(x_history_sgd, [complex_function(x_val) for x_val in x_history_sgd], 'r-')
    axs[0].set_title('随机梯度下降 (SGD)', fontsize=14)
    
    # Momentum
    x_momentum = start_x
    v_momentum = 0
    x_history_momentum = [x_momentum]
    
    for _ in range(20):
        x_momentum, v_momentum = momentum(x_momentum, v_momentum)
        x_history_momentum.append(x_momentum)
    
    axs[1].scatter(x_history_momentum, [complex_function(x_val) for x_val in x_history_momentum], c='g', s=50)
    axs[1].plot(x_history_momentum, [complex_function(x_val) for x_val in x_history_momentum], 'g-')
    axs[1].set_title('动量法 (Momentum)', fontsize=14)
    
    # AdaGrad
    x_adagrad = start_x
    G_adagrad = 0
    x_history_adagrad = [x_adagrad]
    
    for _ in range(20):
        x_adagrad, G_adagrad = adagrad(x_adagrad, G_adagrad)
        x_history_adagrad.append(x_adagrad)
    
    axs[2].scatter(x_history_adagrad, [complex_function(x_val) for x_val in x_history_adagrad], c='m', s=50)
    axs[2].plot(x_history_adagrad, [complex_function(x_val) for x_val in x_history_adagrad], 'm-')
    axs[2].set_title('AdaGrad', fontsize=14)
    
    # RMSProp
    x_rmsprop = start_x
    G_rmsprop = 0
    x_history_rmsprop = [x_rmsprop]
    
    for _ in range(20):
        x_rmsprop, G_rmsprop = rmsprop(x_rmsprop, G_rmsprop)
        x_history_rmsprop.append(x_rmsprop)
    
    axs[3].scatter(x_history_rmsprop, [complex_function(x_val) for x_val in x_history_rmsprop], c='c', s=50)
    axs[3].plot(x_history_rmsprop, [complex_function(x_val) for x_val in x_history_rmsprop], 'c-')
    axs[3].set_title('RMSProp', fontsize=14)
    
    # Adam
    x_adam = start_x
    m_adam = 0
    v_adam = 0
    x_history_adam = [x_adam]
    
    for t in range(1, 21):
        x_adam, m_adam, v_adam = adam(x_adam, m_adam, v_adam, t)
        x_history_adam.append(x_adam)
    
    axs[4].scatter(x_history_adam, [complex_function(x_val) for x_val in x_history_adam], c='y', s=50)
    axs[4].plot(x_history_adam, [complex_function(x_val) for x_val in x_history_adam], 'y-')
    axs[4].set_title('Adam', fontsize=14)
    
    # 算法比较
    axs[5].plot(x_history_sgd, [complex_function(x_val) for x_val in x_history_sgd], 'r-', label='SGD')
    axs[5].plot(x_history_momentum, [complex_function(x_val) for x_val in x_history_momentum], 'g-', label='Momentum')
    axs[5].plot(x_history_adagrad, [complex_function(x_val) for x_val in x_history_adagrad], 'm-', label='AdaGrad')
    axs[5].plot(x_history_rmsprop, [complex_function(x_val) for x_val in x_history_rmsprop], 'c-', label='RMSProp')
    axs[5].plot(x_history_adam, [complex_function(x_val) for x_val in x_history_adam], 'y-', label='Adam')
    axs[5].set_title('优化算法比较', fontsize=14)
    axs[5].legend()
    
    plt.tight_layout()
    return fig

def show_training_page():
    """显示参数训练算法可视化页面"""
    st.markdown("<div class='sub-header'>参数训练算法可视化</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='explanation'>神经网络的训练过程是通过优化算法调整网络参数（权重和偏置），使代价函数最小化。这个页面展示了几种常见的优化方法及其工作原理。</div>", unsafe_allow_html=True)
    
    # 梯度下降可视化
    st.markdown("<div class='sub-header'>梯度下降</div>", unsafe_allow_html=True)
    
    st.markdown("""
    梯度下降是训练神经网络的基础优化算法。其核心思想是沿着损失函数的负梯度方向更新参数，以最小化损失函数。
    """)
    
    # 添加神经网络参数更新公式的详细说明
    st.markdown("""
    参数更新公式：模型参数 = 模型参数 - 学习率 × 梯度
    
    其中：
    - 模型参数：神经网络中需要学习的权重和偏置
    - 学习率：控制每次更新的步长大小
    - 梯度：损失函数对参数的导数，指示参数更新方向
    """)
    
    st.markdown("下面的动画展示了梯度下降如何优化线性回归模型的参数：")
    
    # 生成简单数据集
    np.random.seed(42)
    x_lin = np.random.rand(50) * 4 - 2
    y_lin = 1.5 * x_lin - 0.5 + np.random.randn(50) * 0.2
    
    # 控制学习率和迭代次数
    col1, col2 = st.columns(2)
    with col1:
        gd_learning_rate = st.slider("梯度下降学习率", 0.01, 0.5, 0.1, 0.01, key="gd_lr")
    with col2:
        gd_iterations = st.slider("梯度下降迭代次数", 5, 30, 15, 1, key="gd_iter")
    
    # 生成动画
    gif_buffer = animate_gradient_descent(x_lin, y_lin, gd_learning_rate, gd_iterations)
    gif_data = create_animated_gif(gif_buffer)
    
    # 显示动画
    st.image(gif_data, caption="梯度下降优化过程", use_column_width=True)
    
    st.markdown("""
    **观察点**：
    - 左图：参数空间中的代价函数轮廓，红色线表示参数优化路径
    - 中图：代价函数的3D表面，可以看到参数如何向最小值点移动
    - 右图：数据点和拟合直线，直线随着参数更新而变化
    """)
    
    # 反向传播
    st.markdown("<div class='sub-header'>反向传播算法</div>", unsafe_allow_html=True)
    
    st.markdown("""
    反向传播是训练神经网络的核心算法，它通过链式法则高效计算梯度。
    """)
    
    st.markdown("""
    **算法步骤：**
    
    1. **前向传播**：从输入层到输出层依次计算每层的激活值和输出
    
    2. **计算输出层误差**：计算预测值与真实值之间的误差
    
    3. **反向传播误差**：将误差从输出层反向传播到各个隐藏层
    
    4. **计算梯度**：计算损失函数对各层权重和偏置的梯度
    
    5. **更新参数**：使用梯度下降法更新网络中的权重和偏置参数
    """)
    
    # 生成 Moons 数据集
    np.random.seed(42)
    X, y = make_moons(n_samples=200, noise=0.20, random_state=42)
    y = y.reshape(-1, 1) # 确保 y 是列向量
    
    st.markdown("以下可视化展示了一个神经网络在 Moons 数据集上的训练过程：")
    
    col1, col2 = st.columns(2)
    with col1:
        nn_learning_rate = st.slider("神经网络学习率", 0.1, 1.0, 0.5, 0.1, key="nn_lr")
    with col2:
        # 增加训练轮次的范围和默认值
        nn_epochs = st.slider("训练轮次", 10, 200, 100, 10, key="nn_epochs")
    
    # 训练简单神经网络
    history = simple_neural_network(X, y, hidden_size=4, learning_rate=nn_learning_rate, epochs=nn_epochs)
    
    # 绘制训练过程
    training_fig = plot_neural_network_training(X, y, history)
    st.pyplot(training_fig)
    
    st.markdown("""
    **图表说明**：
    - 左图：训练过程中损失函数的变化
    - 中图：决策边界的变化（红色虚线是初始边界，绿色实线是最终边界）
    - 右图：部分网络参数随训练进行的变化
    """)
    
    # 高级优化算法
    st.markdown("<div class='sub-header'>高级优化算法</div>", unsafe_allow_html=True)
    
    st.markdown("""
    除了基本的梯度下降算法，还有许多改进版本可以加速训练和提高性能：
    
    - **动量法 (Momentum)**：累积过去梯度，加速收敛和穿过局部最小值
    - **AdaGrad**：自适应学习率，为不同参数调整更新速度
    - **RMSProp**：解决AdaGrad学习率递减问题，使用移动平均累积梯度平方
    - **Adam**：结合动量和自适应学习率的方法，自动调整每个参数的学习率
    """)
    
    # 显示优化算法比较图
    optimization_fig = create_optimization_algorithms_comparison()
    st.pyplot(optimization_fig)
    
    st.markdown("""
    **比较结果**：
    - SGD：容易在曲折地形中震荡
    - Momentum：可以加速收敛，克服局部最小值
    - AdaGrad：在早期步骤较大，后期逐渐变小
    - RMSProp：解决AdaGrad学习率递减过快问题
    - Adam：通常表现最佳，结合了动量和自适应学习率的优点
    
    在右下角的比较图中，我们可以看到各算法收敛速度的差异。
    """)
    
    # 结束语
    st.markdown("<div class='explanation'>选择合适的优化算法对神经网络训练至关重要。不同的问题可能需要不同的优化策略。理解这些算法的工作原理有助于更好地设计和调试神经网络。</div>", unsafe_allow_html=True) 