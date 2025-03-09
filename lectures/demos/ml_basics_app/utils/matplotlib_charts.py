"""
使用Matplotlib创建各种图表的模块，替代SVG渲染
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def create_sigmoid_chart():
    """使用Matplotlib创建Sigmoid函数的图像"""
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 生成数据
    x = np.linspace(-6, 6, 100)
    y = 1 / (1 + np.exp(-x))
    
    # 绘制Sigmoid曲线
    ax.plot(x, y, 'k-', linewidth=2, label='Sigmoid函数: σ(z) = 1/(1+e^(-z))')
    
    # 添加标记点，显示σ(0) = 0.5
    ax.plot(0, 0.5, 'ro', markersize=8)
    ax.annotate('z=0, σ(z)=0.5', xy=(0, 0.5), xytext=(1, 0.4),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # 添加水平渐近线
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # 添加垂直线 z=0
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.3)
    
    # 添加z值标记
    for z_val, label_text in [(-4, 'z << 0\nσ(z) ≈ 0'), (4, 'z >> 0\nσ(z) ≈ 1')]:
        ax.annotate(label_text, 
                   xy=(z_val, 1/(1+np.exp(-z_val))),
                   xytext=(z_val, 1/(1+np.exp(-z_val)) + 0.2 if z_val > 0 else 1/(1+np.exp(-z_val)) - 0.2),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1))
    
    # 添加网格和样式
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('σ(z)', fontsize=12)
    ax.set_title('Sigmoid函数', fontsize=14)
    ax.legend(loc='lower right')
    
    # 填充决策区域背景色
    ax.fill_between(x, 0, y, where=(x > 0), color='#ffcccc', alpha=0.3, label='预测为正类 (y=1)')
    ax.fill_between(x, 0, y, where=(x < 0), color='#ccccff', alpha=0.3, label='预测为负类 (y=0)')
    
    ax.legend(loc='best')
    plt.tight_layout()
    
    return fig

def create_svm_concept_chart():
    """使用Matplotlib创建SVM概念的图像"""
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 生成两类数据点
    np.random.seed(42)
    
    # 第一类 - 红色
    X1 = np.random.randn(20, 2) - np.array([2, 2])
    # 第二类 - 蓝色
    X2 = np.random.randn(20, 2) + np.array([2, 2])
    
    # 绘制散点图
    ax.scatter(X1[:, 0], X1[:, 1], color='#FF6B6B', s=50, edgecolor='k', label='类别 1')
    ax.scatter(X2[:, 0], X2[:, 1], color='#4ECDC4', s=50, edgecolor='k', label='类别 2')
    
    # 绘制分离超平面
    xx = np.linspace(-5, 5, 10)
    w = np.array([1, 1])
    b = 0
    
    # 决策边界 w⋅x + b = 0
    yy = -w[0]/w[1] * xx - b/w[1]
    
    # 绘制决策边界
    ax.plot(xx, yy, 'k-', label='决策边界')
    
    # 绘制间隔
    margin = 1 / np.linalg.norm(w)
    yy_neg = yy - margin
    yy_pos = yy + margin
    ax.plot(xx, yy_neg, 'k--', label='间隔边界')
    ax.plot(xx, yy_pos, 'k--')
    
    # 标记一些支持向量
    sv_indices = [0, 3, 5, 12, 15]
    sv_X = np.vstack([X1[sv_indices[:3]], X2[sv_indices[3:]]])
    ax.scatter(sv_X[:, 0], sv_X[:, 1], s=120, facecolors='none', 
               edgecolors='green', linewidth=2, label='支持向量')
    
    # 添加填充区域表示不同类别
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    # 决策面及间隔的区域
    margin_pts = np.c_[xx, yy_neg]
    upper_pts = np.array([[xx[0], ymin], [xx[-1], ymin], [xx[-1], yy_neg[-1]], [xx[0], yy_neg[0]]])
    lower_pts = np.array([[xx[0], ymax], [xx[-1], ymax], [xx[-1], yy_pos[-1]], [xx[0], yy_pos[0]]])
    
    ax.fill(upper_pts[:, 0], upper_pts[:, 1], '#FF6B6B', alpha=0.2)
    ax.fill(lower_pts[:, 0], lower_pts[:, 1], '#4ECDC4', alpha=0.2)
    
    # 设置坐标轴和标题
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('特征 1', fontsize=12)
    ax.set_ylabel('特征 2', fontsize=12)
    ax.set_title('支持向量机(SVM)概念图示', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 添加训练目标标注
    ax.text(0, ymax-0.5, '最大化间隔', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    plt.tight_layout()
    return fig

def create_model_comparison_chart():
    """使用Matplotlib创建模型比较图像"""
    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 1. 线性可分数据
    X1 = np.random.randn(100, 2) - np.array([2, 2])
    X2 = np.random.randn(100, 2) + np.array([2, 2])
    X_linear = np.vstack([X1, X2])
    y_linear = np.hstack([np.zeros(100), np.ones(100)])
    
    # 2. 非线性数据 (同心圆)
    X1 = np.random.randn(100, 2)
    r1 = np.random.uniform(0, 1, 100)
    X1 = X1 / np.linalg.norm(X1, axis=1).reshape(-1, 1) * r1.reshape(-1, 1)
    
    X2 = np.random.randn(100, 2)
    r2 = np.random.uniform(2, 3, 100)
    X2 = X2 / np.linalg.norm(X2, axis=1).reshape(-1, 1) * r2.reshape(-1, 1)
    
    X_nonlinear = np.vstack([X1, X2])
    y_nonlinear = np.hstack([np.zeros(100), np.ones(100)])
    
    # 绘制线性数据场景
    axes[0, 0].scatter(X_linear[y_linear==0, 0], X_linear[y_linear==0, 1], color='#4ECDC4', edgecolor='k', alpha=0.7)
    axes[0, 0].scatter(X_linear[y_linear==1, 0], X_linear[y_linear==1, 1], color='#FF6B6B', edgecolor='k', alpha=0.7)
    axes[0, 0].set_title('线性可分数据')
    axes[0, 0].set_xlabel('特征1')
    axes[0, 0].set_ylabel('特征2')
    axes[0, 0].text(-5, 5, '逻辑回归适用：✓\nSVM适用：✓', fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # 绘制线性模型决策边界
    xx = np.linspace(-6, 6, 50)
    yy = -xx  # 简化的线性决策边界
    axes[0, 0].plot(xx, yy, 'k-', label='决策边界')
    axes[0, 0].legend(loc='lower right')
    
    # 绘制非线性数据场景
    axes[0, 1].scatter(X_nonlinear[y_nonlinear==0, 0], X_nonlinear[y_nonlinear==0, 1], color='#4ECDC4', edgecolor='k', alpha=0.7)
    axes[0, 1].scatter(X_nonlinear[y_nonlinear==1, 0], X_nonlinear[y_nonlinear==1, 1], color='#FF6B6B', edgecolor='k', alpha=0.7)
    axes[0, 1].set_title('非线性数据 (同心圆)')
    axes[0, 1].set_xlabel('特征1')
    axes[0, 1].set_ylabel('特征2')
    axes[0, 1].text(-3, 3, '逻辑回归适用：✗\nSVM适用：✓', fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # 绘制非线性决策边界 (圆形)
    theta = np.linspace(0, 2*np.pi, 100)
    r = 1.5  # 两类之间的分界圆半径
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    axes[0, 1].plot(x, y, 'k-', label='决策边界 (核SVM)')
    axes[0, 1].plot([0, 3], [0, 3], 'r--', label='线性模型 (无法分类)')
    axes[0, 1].legend(loc='lower right')
    
    # 创建理论比较图表
    comparison_data = [
        ['输出', '概率 (0-1之间)', '类别标签/距离'],
        ['决策边界', '线性', '线性或非线性 (取决于核函数)'],
        ['优化目标', '最大化似然', '最大化边界'],
        ['处理大数据', '高效', '较慢 (特别是非线性核)'],
        ['处理高维稀疏数据', '一般', '优秀'],
        ['可解释性', '好', '线性核好，非线性核差'],
        ['过拟合处理', 'L1/L2正则化', '软间隔SVM (C参数)']
    ]
    
    # 绘制比较表格
    table_ax = axes[1, 0]
    table_ax.axis('tight')
    table_ax.axis('off')
    table = table_ax.table(
        cellText=comparison_data[1:],
        colLabels=['特性', '逻辑回归', 'SVM'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table_ax.set_title('算法特性比较', y=1.05)
    
    # 优缺点总结
    pros_cons = axes[1, 1]
    pros_cons.axis('off')
    
    lr_pros = "• 输出为概率值，易于解释\n• 训练速度快，适合大数据集\n• 特征重要性易于获取\n• 实现简单，易于部署"
    lr_cons = "• 难以处理非线性关系\n• 表示能力受限于线性\n• 高度共线性特征会影响性能\n• 特征数量远大于样本数时容易过拟合"
    
    svm_pros = "• 通过核技巧处理非线性关系\n• 适合高维数据\n• 对异常值和噪声有一定鲁棒性\n• 小样本学习效果好"
    svm_cons = "• 训练大数据集计算开销大\n• 核SVM难以解释\n• 参数选择敏感\n• 概率输出需要特别处理"
    
    pros_cons.text(0.05, 0.9, "逻辑回归优点:", fontweight='bold')
    pros_cons.text(0.05, 0.7, lr_pros)
    pros_cons.text(0.05, 0.5, "逻辑回归缺点:", fontweight='bold')
    pros_cons.text(0.05, 0.3, lr_cons)
    
    pros_cons.text(0.55, 0.9, "SVM优点:", fontweight='bold')
    pros_cons.text(0.55, 0.7, svm_pros)
    pros_cons.text(0.55, 0.5, "SVM缺点:", fontweight='bold')
    pros_cons.text(0.55, 0.3, svm_cons)
    
    pros_cons.set_title('优缺点总结')
    
    plt.tight_layout()
    return fig

def create_learning_path_chart():
    """使用Matplotlib创建学习路径图像"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 隐藏坐标轴
    ax.axis('off')
    
    # 定义节点位置
    nodes = {
        'start': (0.5, 0.95),
        'theory': (0.5, 0.8),
        'visualization': (0.5, 0.65),
        'basic_practice': (0.3, 0.5),
        'advanced_practice': (0.7, 0.5),
        'bias_variance': (0.5, 0.35),
        'regularization': (0.3, 0.2),
        'model_selection': (0.7, 0.2),
        'end': (0.5, 0.05)
    }
    
    # 定义节点标签
    node_labels = {
        'start': '开始学习',
        'theory': '理论基础',
        'visualization': '可视化理解',
        'basic_practice': '基础实践',
        'advanced_practice': '进阶实践',
        'bias_variance': '偏差-方差权衡',
        'regularization': '正则化技术',
        'model_selection': '模型选择',
        'end': '掌握分类算法'
    }
    
    # 定义节点颜色 - 使用灰色和黑色调
    node_colors = {
        'start': '#4A4A4A',  # 深灰色
        'theory': '#606060',  # 灰色
        'visualization': '#707070',  # 浅灰色
        'basic_practice': '#555555',  # 中等灰色
        'advanced_practice': '#666666',  # 浅灰色
        'bias_variance': '#777777',  # 更浅灰色
        'regularization': '#505050',  # 深灰色
        'model_selection': '#606060',  # 灰色
        'end': '#333333'  # 黑色
    }
    
    # 绘制节点和标签
    for node, (x, y) in nodes.items():
        circle = plt.Circle((x, y), 0.05, color=node_colors[node], alpha=0.8)
        ax.add_patch(circle)
        
        # 节点标签
        plt.text(x, y-0.05, node_labels[node], ha='center', fontsize=10, fontweight='bold')
        
        # 节点内容说明
        if node == 'theory':
            plt.text(x, y+0.03, '逻辑回归与SVM基本原理', ha='center', fontsize=8)
        elif node == 'visualization':
            plt.text(x, y+0.03, '决策边界、参数效果可视化', ha='center', fontsize=8)
        elif node == 'basic_practice':
            plt.text(x, y+0.03, '基础分类任务实战', ha='center', fontsize=8)
        elif node == 'advanced_practice':
            plt.text(x, y+0.03, '复杂数据处理、特征工程', ha='center', fontsize=8)
        elif node == 'bias_variance':
            plt.text(x, y+0.03, '理解和处理过拟合与欠拟合', ha='center', fontsize=8)
        elif node == 'regularization':
            plt.text(x, y+0.03, 'L1/L2正则化、软间隔SVM', ha='center', fontsize=8)
        elif node == 'model_selection':
            plt.text(x, y+0.03, '超参数调优、交叉验证', ha='center', fontsize=8)
    
    # 绘制连接线和箭头
    ax.annotate('', xy=nodes['theory'], xytext=nodes['start'],
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    ax.annotate('', xy=nodes['visualization'], xytext=nodes['theory'],
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    ax.annotate('', xy=nodes['basic_practice'], xytext=nodes['visualization'],
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    ax.annotate('', xy=nodes['advanced_practice'], xytext=nodes['visualization'],
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    ax.annotate('', xy=nodes['bias_variance'], xytext=nodes['basic_practice'],
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    ax.annotate('', xy=nodes['bias_variance'], xytext=nodes['advanced_practice'],
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    ax.annotate('', xy=nodes['regularization'], xytext=nodes['bias_variance'],
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    ax.annotate('', xy=nodes['model_selection'], xytext=nodes['bias_variance'],
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    ax.annotate('', xy=nodes['end'], xytext=nodes['regularization'],
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    ax.annotate('', xy=nodes['end'], xytext=nodes['model_selection'],
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    # 添加标题
    plt.title('分类算法学习路径图', fontsize=14, fontweight='bold', y=1.02)
    
    # 添加说明文本
    plt.figtext(0.5, 0.01, '提示：路径不必严格按顺序，可根据个人学习进度灵活调整', 
                ha='center', fontsize=10, style='italic')
    
    return fig 