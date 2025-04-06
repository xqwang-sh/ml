import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.datasets import make_moons

# --- 从 training.py 复制过来的必要函数 ---

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def simple_neural_network(X, y, hidden_size=4, learning_rate=0.5, epochs=15):
    """
    实现一个简单的神经网络训练过程，并记录训练历史
    (简化版，只为了获取 history)
    """
    np.random.seed(42)
    input_size = X.shape[1]
    output_size = 1
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))
    history = {'loss': [], 'params': []}
    
    # 存储初始参数
    history['params'].append({
        'W1': W1.copy(), 'b1': b1.copy(), 'W2': W2.copy(), 'b2': b2.copy()
    })

    for i in range(epochs):
        Z1 = np.dot(X, W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)
        cost = -np.mean(y * np.log(A2 + 1e-8) + (1 - y) * np.log(1 - A2 + 1e-8))
        history['loss'].append(cost)
        dZ2 = A2 - y
        dW2 = np.dot(A1.T, dZ2) / X.shape[0]
        db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]
        dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(Z1)
        dW1 = np.dot(X.T, dZ1) / X.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        history['params'].append({
            'W1': W1.copy(), 'b1': b1.copy(), 'W2': W2.copy(), 'b2': b2.copy()
        })
    return history

def plot_neural_network_training(X, y, history):
    """
    绘制神经网络训练过程 (清理调试代码)
    """
    epochs = len(history['loss'])
    fig = plt.figure(figsize=(8, 6))
    ax2 = fig.add_subplot(1, 1, 1)
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    W1_init = history['params'][0]['W1']
    b1_init = history['params'][0]['b1']
    W2_init = history['params'][0]['W2']
    b2_init = history['params'][0]['b2']
    
    def predict(X_pred, W1, b1, W2, b2):
        Z1 = np.dot(X_pred, W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)
        return A2

    Z_init = predict(grid_points, W1_init, b1_init, W2_init, b2_init)
    Z_init = Z_init.reshape(xx.shape)
    
    W1_final = history['params'][-1]['W1']
    b1_final = history['params'][-1]['b1']
    W2_final = history['params'][-1]['W2']
    b2_final = history['params'][-1]['b2']
    
    Z_final = predict(grid_points, W1_final, b1_final, W2_final, b2_final)
    Z_final = Z_final.reshape(xx.shape)
    
    ax2.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], c='#D81B60', 
               marker='o', s=100, label='类别 0', edgecolors='k', zorder=10)
    ax2.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], c='#1E88E5', 
               marker='^', s=100, label='类别 1', edgecolors='k', zorder=10)
               
    contour = ax2.contourf(xx, yy, Z_final, levels=50, cmap='Blues', alpha=0.3, zorder=1)
    
    init_line = ax2.contour(xx, yy, Z_init, levels=[0.5], colors=['darkred'], 
                                     linestyles='--', linewidths=3, zorder=5)
    final_line = ax2.contour(xx, yy, Z_final, levels=[0.5], colors=['darkgreen'], 
                                      linestyles='-', linewidths=3, zorder=5)
    
    legend_elements = [
        Line2D([0], [0], color='darkred', linestyle='--', lw=3, label='初始边界 (0.5)'),
        Line2D([0], [0], color='darkgreen', linestyle='-', lw=3, label='最终边界 (0.5)')
    ]
    
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    ax2.set_xlabel('特征 x₁', fontsize=12)
    ax2.set_ylabel('特征 x₂', fontsize=12)
    ax2.set_title('决策边界 (Moons 数据集)', fontsize=14)
    
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
    return fig

# --- 主调试逻辑 ---
if __name__ == "__main__":
    # 使用 Moons 数据集
    np.random.seed(42)
    X_data, y_data = make_moons(n_samples=200, noise=0.20, random_state=42)
    y_data = y_data.reshape(-1, 1)

    print("Using Moons dataset for debugging.")

    # 设置参数
    nn_learning_rate = 0.5
    nn_epochs = 100
    hidden_size = 4

    print(f"Running NN training with lr={nn_learning_rate}, epochs={nn_epochs}, hidden_size={hidden_size}")

    # 训练模型获取 history
    history_data = simple_neural_network(X_data, y_data, hidden_size=hidden_size,
                                       learning_rate=nn_learning_rate, epochs=nn_epochs)

    print(f"Training complete. History has {len(history_data['params'])} parameter sets.")

    # 绘制图形
    fig = plot_neural_network_training(X_data, y_data, history_data)

    # 显示图形
    plt.show() 