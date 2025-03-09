import numpy as np
import streamlit as st
from typing import List, Dict, Tuple, Optional, Callable
import io
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import base64
from sklearn.datasets import make_classification
import os
import math
import uuid

def fig_to_svg(fig: Figure) -> str:
    """将Matplotlib图形转换为SVG字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    svg_content = buf.getvalue().decode('utf-8')
    buf.close()
    
    # 确保SVG内容正确，去除可能导致问题的XML声明
    if '<?xml' in svg_content:
        svg_content = svg_content[svg_content.find('<svg'):]
    
    return svg_content

def create_sigmoid_svg() -> str:
    """创建Sigmoid函数的SVG图像"""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(-10, 10, 100)
    y = 1 / (1 + np.exp(-x))
    ax.plot(x, y)
    ax.grid(True)
    ax.set_xlabel("z")
    ax.set_ylabel("$\\sigma(z)$")
    ax.set_title("Sigmoid函数: $\\sigma(z) = \\frac{1}{1 + e^{-z}}$")
    
    return fig_to_svg(fig)

def create_bias_variance_svg() -> str:
    """创建偏差-方差权衡图的SVG图像"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_complexity = np.linspace(1, 10, 100)
    bias = 10 / model_complexity
    variance = 0.1 * model_complexity
    total_error = bias + variance + 1
    
    ax.plot(model_complexity, bias, 'b-', label='偏差')
    ax.plot(model_complexity, variance, 'r-', label='方差')
    ax.plot(model_complexity, total_error, 'g-', label='总误差')
    ax.axvline(x=np.sqrt(10/0.1), color='k', linestyle='--', label='最优复杂度')
    
    ax.set_xlabel('模型复杂度')
    ax.set_ylabel('误差')
    ax.set_title('偏差-方差权衡')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_svg(fig)

def create_dataset_split_svg() -> str:
    """创建数据集划分示意图的SVG图像"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 创建一个表示整个数据集的矩形
    ax.add_patch(plt.Rectangle((0, 0), 100, 1, fc='lightgray', ec='black'))
    
    # 划分区域
    ax.add_patch(plt.Rectangle((0, 0), 70, 1, fc='lightblue', ec='black'))
    ax.add_patch(plt.Rectangle((70, 0), 15, 1, fc='lightgreen', ec='black'))
    ax.add_patch(plt.Rectangle((85, 0), 15, 1, fc='salmon', ec='black'))
    
    # 添加标签
    ax.text(35, 0.5, '训练集 (70%)', ha='center', va='center', fontsize=12)
    ax.text(77.5, 0.5, '验证集\n(15%)', ha='center', va='center', fontsize=12)
    ax.text(92.5, 0.5, '测试集\n(15%)', ha='center', va='center', fontsize=12)
    
    # 设置轴
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('数据集划分示意图', fontsize=14)
    
    return fig_to_svg(fig)

def create_overfitting_svg(polynomial_degrees: Optional[List[int]] = None) -> str:
    """
    创建过拟合与欠拟合示意图
    
    Args:
        polynomial_degrees: 多项式度数列表，默认为[1, 3, 15]
    """
    if polynomial_degrees is None:
        polynomial_degrees = [1, 3, 15]
    
    titles = ["欠拟合", "适当拟合", "过拟合"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 生成示例数据
    np.random.seed(42)
    x = np.linspace(0, 1, 30)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, 30)
    x_plot = np.linspace(0, 1, 100)
    y_real = np.sin(2 * np.pi * x_plot)
    
    for i, (ax, degree, title) in enumerate(zip(axes, polynomial_degrees, titles)):
        # 多项式拟合
        coeffs = np.polyfit(x, y, degree)
        y_poly = np.polyval(coeffs, x_plot)
        
        ax.scatter(x, y, color='blue', label='数据点')
        ax.plot(x_plot, y_real, 'g--', label='真实函数')
        ax.plot(x_plot, y_poly, 'r-', label=f'{degree}次多项式')
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    return fig_to_svg(fig)

def create_interactive_overfitting_svg(degree: int) -> str:
    """
    创建用于交互式演示的过拟合SVG图
    
    Args:
        degree: 多项式度数
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 生成示例数据
    np.random.seed(42)
    x = np.linspace(0, 1, 30)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, 30)
    x_plot = np.linspace(0, 1, 100)
    y_real = np.sin(2 * np.pi * x_plot)
    
    # 多项式拟合
    coeffs = np.polyfit(x, y, degree)
    y_poly = np.polyval(coeffs, x_plot)
    
    ax.scatter(x, y, color='blue', label='数据点')
    ax.plot(x_plot, y_real, 'g--', label='真实函数')
    ax.plot(x_plot, y_poly, 'r-', label=f'{degree}次多项式')
    ax.set_title(f'多项式拟合(度数={degree})')
    ax.legend()
    ax.set_ylim(-1.5, 1.5)
    
    return fig_to_svg(fig)

def create_svm_hyperplane_svg() -> str:
    """创建SVM超平面与支持向量示意图的SVG"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 生成数据
    np.random.seed(0)
    X = np.random.randn(40, 2)
    y = np.array([1] * 20 + [-1] * 20)
    
    # 调整数据使其更明显地线性可分
    margin = 1
    X[:20] += margin
    X[20:] -= margin
    
    # 绘制数据点
    ax.scatter(X[:20, 0], X[:20, 1], color='red', marker='o', label='类别 +1')
    ax.scatter(X[20:, 0], X[20:, 1], color='blue', marker='x', label='类别 -1')
    
    # 绘制分割超平面
    w = np.array([1, 1])
    b = 0
    xx = np.linspace(-3, 3, 100)
    yy = -w[0]/w[1] * xx - b/w[1]
    ax.plot(xx, yy, 'k-', label='决策边界')
    
    # 绘制边距线
    margin = 1/np.linalg.norm(w)
    yy_up = -w[0]/w[1] * xx - (b - 1)/w[1]
    yy_down = -w[0]/w[1] * xx - (b + 1)/w[1]
    ax.plot(xx, yy_up, 'k--', label='边距边界')
    ax.plot(xx, yy_down, 'k--')
    
    # 标记支持向量
    sv_indices = [18, 19, 20, 21]
    ax.scatter(X[sv_indices, 0], X[sv_indices, 1], s=100, 
               linewidth=1, facecolors='none', edgecolors='k',
               label='支持向量')
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.set_title('支持向量机的超平面与支持向量')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_svg(fig)

def create_interactive_regularization_svg(regularization_strength: float, regularization_type: str = 'L2') -> str:
    """
    创建用于演示正则化效果的SVG图
    
    Args:
        regularization_strength: 正则化强度
        regularization_type: 正则化类型 ('L1' 或 'L2')
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 生成数据
    np.random.seed(42)
    x = np.linspace(0, 1, 30)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, 30)
    x_plot = np.linspace(0, 1, 100)
    y_real = np.sin(2 * np.pi * x_plot)
    
    # 多项式拟合
    X = np.vander(x, 15, increasing=True)  # 创建15次多项式的范德蒙德矩阵
    X_plot = np.vander(x_plot, 15, increasing=True)
    
    # 没有正则化的拟合
    coeffs_no_reg = np.linalg.lstsq(X, y, rcond=None)[0]
    y_no_reg = X_plot @ coeffs_no_reg
    
    # 有正则化的拟合
    if regularization_type == 'L2':
        # 岭回归 (L2正则化)
        I = np.eye(X.shape[1])
        coeffs_reg = np.linalg.solve(X.T @ X + regularization_strength * I, X.T @ y)
    else:  # L1正则化 (简化版)
        # 这不是真正的L1实现，但能展示效果
        coeffs_reg = np.linalg.lstsq(X, y, rcond=None)[0]
        sign_mask = np.sign(coeffs_reg)
        coeffs_reg = np.maximum(np.abs(coeffs_reg) - regularization_strength, 0) * sign_mask
    
    y_reg = X_plot @ coeffs_reg
    
    # 绘图
    ax.scatter(x, y, color='blue', label='数据点')
    ax.plot(x_plot, y_real, 'g--', label='真实函数')
    ax.plot(x_plot, y_no_reg, 'r-', label='无正则化')
    ax.plot(x_plot, y_reg, 'c-', label=f'{regularization_type}正则化 (λ={regularization_strength:.4f})')
    
    # 设置
    ax.set_title(f'{regularization_type}正则化效果演示')
    ax.legend()
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)
    
    return fig_to_svg(fig)

def create_cross_validation_svg(k: int = 5) -> str:
    """
    创建k折交叉验证示意图的SVG
    
    Args:
        k: 折数，默认为5
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#a1dab4', '#41b6c4', '#2c7fb8', '#253494', '#f7fcf5']
    
    for i in range(k):
        # 每行显示一次折叠，突出显示当前的验证集
        for j in range(k):
            if j == i:
                # 验证集
                ax.add_patch(plt.Rectangle((j/k, i/k), 1/k, 1/k, 
                                          fc='salmon', ec='black'))
                ax.text((j+0.5)/k, (i+0.5)/k, "验证", 
                       ha='center', va='center')
            else:
                # 训练集
                ax.add_patch(plt.Rectangle((j/k, i/k), 1/k, 1/k, 
                                          fc='lightblue', ec='black'))
                ax.text((j+0.5)/k, (i+0.5)/k, "训练", 
                       ha='center', va='center')
    
    # 设置坐标轴
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([(i+0.5)/k for i in range(k)])
    ax.set_yticklabels([f"第{i+1}折" for i in range(k)])
    ax.set_title("k折交叉验证示意图")
    
    return fig_to_svg(fig)

def create_learning_curve_svg() -> str:
    """创建学习曲线示意图的SVG"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 模拟学习曲线数据
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # 高偏差模型（两条曲线都较高但接近）
    train_high_bias = 0.7 - 0.1 * np.sqrt(train_sizes)
    valid_high_bias = 0.65 - 0.1 * np.sqrt(train_sizes)
    
    # 高方差模型（训练误差低，验证误差高）
    train_high_var = 0.2 + 0.1 * np.sqrt(train_sizes)
    valid_high_var = 0.8 - 0.2 * np.sqrt(train_sizes)
    
    # 绘制两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 高偏差模型
    ax1.plot(train_sizes, train_high_bias, 'o-', color='blue', label='训练误差')
    ax1.plot(train_sizes, valid_high_bias, 'o-', color='red', label='验证误差')
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('训练集大小')
    ax1.set_ylabel('错误率')
    ax1.set_title('高偏差模型(欠拟合)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 高方差模型
    ax2.plot(train_sizes, train_high_var, 'o-', color='blue', label='训练误差')
    ax2.plot(train_sizes, valid_high_var, 'o-', color='red', label='验证误差')
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('训练集大小')
    ax2.set_ylabel('错误率')
    ax2.set_title('高方差模型(过拟合)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig_to_svg(fig)

def create_roc_curve_svg() -> str:
    """创建ROC曲线示意图的SVG"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制随机猜测的对角线
    ax.plot([0, 1], [0, 1], 'k--', label='随机猜测 (AUC = 0.5)')
    
    # 绘制几个不同性能的ROC曲线
    # 优秀模型
    fpr_excellent = np.linspace(0, 1, 100)
    tpr_excellent = 1 - np.exp(-5 * fpr_excellent)
    ax.plot(fpr_excellent, tpr_excellent, 'g-', label='优秀模型 (AUC ≈ 0.95)')
    
    # 良好模型
    fpr_good = np.linspace(0, 1, 100)
    tpr_good = 1 - np.exp(-2 * fpr_good)
    ax.plot(fpr_good, tpr_good, 'b-', label='良好模型 (AUC ≈ 0.8)')
    
    # 一般模型
    fpr_fair = np.linspace(0, 1, 100)
    tpr_fair = fpr_fair + 0.2 * (1 - fpr_fair) * fpr_fair
    tpr_fair = np.minimum(tpr_fair, 1)
    ax.plot(fpr_fair, tpr_fair, 'y-', label='一般模型 (AUC ≈ 0.6)')
    
    # 设置图表属性
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('假阳性率 (FPR)')
    ax.set_ylabel('真阳性率 (TPR)')
    ax.set_title('ROC曲线比较')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    return fig_to_svg(fig)

def generate_logistic_function_svg():
    """生成逻辑函数的SVG图形"""
    # 设定大小和样式
    width = 600
    height = 300
    padding = 40
    
    # 为防止ID冲突，使用UUID生成唯一ID
    unique_id = str(uuid.uuid4()).replace('-', '')
    
    # 创建SVG字符串
    svg = f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" class="st-svg">
        <defs>
            <marker id="arrowhead-{unique_id}" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
            </marker>
        </defs>
        
        <!-- 坐标系 -->
        <line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-{unique_id})"/>
        <line x1="{padding}" y1="{height-padding}" x2="{padding}" y2="{padding}" stroke="#333" stroke-width="2" marker-end="url(#arrowhead-{unique_id})"/>
        
        <!-- 坐标轴标签 -->
        <text x="{width-padding+10}" y="{height-padding+5}" font-size="14" text-anchor="start">z</text>
        <text x="{padding-5}" y="{padding-10}" font-size="14" text-anchor="end">σ(z)</text>
        
        <!-- 刻度 -->
        <line x1="{padding}" y1="{height-padding-100}" x2="{padding-5}" y2="{height-padding-100}" stroke="#333" stroke-width="1"/>
        <text x="{padding-10}" y="{height-padding-100+5}" font-size="12" text-anchor="end">0.5</text>
        
        <line x1="{padding}" y1="{height-padding-200}" x2="{padding-5}" y2="{height-padding-200}" stroke="#333" stroke-width="1"/>
        <text x="{padding-10}" y="{height-padding-200+5}" font-size="12" text-anchor="end">1.0</text>
        
        <!-- 绘制sigmoid函数 -->
        <path d="M {padding} {height-padding-5} '''
    
    # 计算sigmoid曲线的点
    x_range = np.linspace(-6, 6, 100)
    for x in x_range:
        # 计算sigmoid函数值 1/(1+e^(-x))
        y = 1 / (1 + math.exp(-x))
        # 缩放到SVG坐标系
        svg_x = padding + (x + 6) * (width - 2*padding) / 12
        svg_y = height - padding - y * 200
        svg += f"L {svg_x} {svg_y} "
    
    svg += f'''" stroke="#1E88E5" stroke-width="3" fill="none"/>
        
        <!-- 中心点 -->
        <circle cx="{padding + 6*(width-2*padding)/12}" cy="{height-padding-100}" r="4" fill="red"/>
        <text x="{padding + 6*(width-2*padding)/12 + 10}" y="{height-padding-100-10}" font-size="12" text-anchor="start">z=0, σ(z)=0.5</text>
        
        <!-- 公式 -->
        <text x="{width/2}" y="{padding/2}" font-size="16" text-anchor="middle" font-style="italic">σ(z) = 1/(1+e^(-z))</text>
    </svg>
    '''
    
    return svg

def create_matplotlib_sigmoid():
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

def create_matplotlib_svm_concept():
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
    ax.scatter(X1[:, 0], X1[:, 1], color='red', s=50, edgecolor='k', label='类别 1')
    ax.scatter(X2[:, 0], X2[:, 1], color='blue', s=50, edgecolor='k', label='类别 2')
    
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
    
    ax.fill(upper_pts[:, 0], upper_pts[:, 1], 'red', alpha=0.2)
    ax.fill(lower_pts[:, 0], lower_pts[:, 1], 'blue', alpha=0.2)
    
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

def create_matplotlib_model_comparison():
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
    axes[0, 0].scatter(X_linear[y_linear==0, 0], X_linear[y_linear==0, 1], color='blue', edgecolor='k', alpha=0.7)
    axes[0, 0].scatter(X_linear[y_linear==1, 0], X_linear[y_linear==1, 1], color='red', edgecolor='k', alpha=0.7)
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
    axes[0, 1].scatter(X_nonlinear[y_nonlinear==0, 0], X_nonlinear[y_nonlinear==0, 1], color='blue', edgecolor='k', alpha=0.7)
    axes[0, 1].scatter(X_nonlinear[y_nonlinear==1, 0], X_nonlinear[y_nonlinear==1, 1], color='red', edgecolor='k', alpha=0.7)
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

def create_matplotlib_learning_path():
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
    
    # 定义节点颜色
    node_colors = {
        'start': '#4CAF50',  # 绿色
        'theory': '#FF9800',  # 橙色
        'visualization': '#2196F3',  # 蓝色
        'basic_practice': '#9C27B0',  # 紫色
        'advanced_practice': '#E91E63',  # 粉色
        'bias_variance': '#00BCD4',  # 青色
        'regularization': '#F44336',  # 红色
        'model_selection': '#673AB7',  # 深紫色
        'end': '#4CAF50'  # 绿色
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

def render_svg(svg_string):
    """在Streamlit中渲染SVG图形"""
    # 防止空值或非字符串类型
    if svg_string is None:
        st.warning("提供的SVG内容为空")
        return
    
    if not isinstance(svg_string, str):
        st.warning(f"无效的SVG内容类型: {type(svg_string)}")
        return
    
    # 确保svg_string是包含完整SVG元素的字符串
    svg_string = svg_string.strip()
    
    try:
        # 确保SVG内容有效
        if not svg_string.startswith('<svg'):
            # 尝试查找SVG标签
            svg_start = svg_string.find('<svg')
            if svg_start >= 0:
                svg_string = svg_string[svg_start:]
            else:
                st.warning("无法识别有效的SVG内容")
                st.code(svg_string[:100] + "..." if len(svg_string) > 100 else svg_string)
                return
        
        # 查找并确保SVG结束标签存在
        if '</svg>' not in svg_string:
            st.warning("SVG内容缺少结束标签")
            st.code(svg_string[:100] + "..." if len(svg_string) > 100 else svg_string)
            return
            
        # 使用HTML组件而不是markdown来渲染SVG，避免转义问题
        from streamlit.components.v1 import html
        
        # 使用div包装并添加SVG类，保证良好的样式应用
        wrapped_svg = f'''
        <div style="margin: 10px auto; max-width: 100%; overflow-x: auto;">
            {svg_string}
        </div>
        '''
        
        # 使用HTML组件进行渲染
        html(wrapped_svg, height=500, scrolling=True)
    except Exception as e:
        st.error(f"渲染SVG时出错: {str(e)}")
        st.code(svg_string[:100] + "..." if len(svg_string) > 100 else svg_string, language="xml") 