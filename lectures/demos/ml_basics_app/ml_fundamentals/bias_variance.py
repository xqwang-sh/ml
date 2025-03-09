"""
偏差-方差权衡的交互式演示模块
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from utils.svg_generator import create_bias_variance_svg

def show_bias_variance_demo():
    """显示偏差-方差权衡的交互式演示"""
    
    st.subheader("偏差-方差权衡")
    
    st.markdown("""
    模型的预测误差可以分解为三个关键组成部分：
    
    1. **偏差(Bias)** - 模型预测值与真实值的平均差异。高偏差模型通常过于简单，无法捕捉数据中的复杂模式，导致欠拟合。
    
    2. **方差(Variance)** - 对不同训练集的敏感度。高方差模型对训练数据中的微小变化非常敏感，容易过拟合。
    
    3. **不可约误差** - 数据本身的噪声，无法通过任何模型消除。
    
    总误差 = 偏差² + 方差 + 不可约误差
    
    模型复杂度增加时，偏差通常会减少，而方差会增加。模型设计的挑战是找到平衡点，即**最小化总误差**。
    """)
    
    # 显示偏差-方差权衡图
    st.markdown(create_bias_variance_svg(), unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 关键概念解析")
        st.markdown("""
        **高偏差(欠拟合)特征：**
        - 训练误差高
        - 验证误差高
        - 训练误差 ≈ 验证误差
        - 模型过于简单
        
        **高方差(过拟合)特征：**
        - 训练误差低
        - 验证误差高
        - 训练误差 << 验证误差
        - 模型过于复杂
        
        **最佳模型：**
        - 训练误差适中
        - 验证误差低
        - 训练误差与验证误差接近
        """)
    
    # 交互式演示
    st.markdown("### 交互式演示：偏差-方差模拟")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        此演示模拟不同模型复杂度对偏差和方差的影响。
        
        调整参数观察偏差、方差和总误差的变化：
        """)
        
        bias_factor = st.slider("偏差因子", min_value=1.0, max_value=20.0, value=10.0, step=0.5,
                               help="较大的值意味着简单模型的偏差更高")
        
        variance_factor = st.slider("方差因子", min_value=0.01, max_value=0.5, value=0.1, step=0.01,
                                  help="较大的值意味着复杂模型的方差增长更快")
        
        noise = st.slider("不可约误差", min_value=0.0, max_value=3.0, value=1.0, step=0.1,
                         help="数据固有的噪声水平")
    
    with col2:
        # 动态创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_complexity = np.linspace(1, 10, 100)
        bias = bias_factor / model_complexity
        variance = variance_factor * model_complexity
        total_error = bias + variance + noise
        
        # 找到最优复杂度
        optimal_complexity = np.sqrt(bias_factor / variance_factor)
        min_error_idx = np.argmin(total_error)
        min_error_complexity = model_complexity[min_error_idx]
        
        ax.plot(model_complexity, bias, 'b-', label='偏差')
        ax.plot(model_complexity, variance, 'r-', label='方差')
        ax.plot(model_complexity, total_error, 'g-', label='总误差')
        ax.axvline(x=optimal_complexity, color='k', linestyle='--', label='理论最优复杂度')
        
        ax.set_xlabel('模型复杂度')
        ax.set_ylabel('误差')
        ax.set_title('偏差-方差权衡')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.metric("理论最优复杂度", f"{optimal_complexity:.2f}")
    
    # 现实示例
    st.markdown("### 现实中的偏差-方差权衡")
    
    st.markdown("""
    **不同模型的偏差-方差特性：**
    
    | 模型 | 偏差 | 方差 | 适用场景 |
    |------|------|------|----------|
    | 线性回归 | 高 | 低 | 数据关系接近线性，特征少 |
    | 决策树 | 低 | 高 | 非线性关系，特征交互强 |
    | 随机森林 | 中 | 中 | 平衡偏差和方差，广泛适用 |
    | 支持向量机 | 可调 | 可调 | 通过核函数和惩罚参数调节 |
    | 神经网络 | 可调 | 可调 | 复杂模式，大数据集 |
    
    **如何找到平衡点：**
    
    1. **交叉验证**：使用k折交叉验证评估不同复杂度模型
    
    2. **学习曲线**：分析训练集大小与误差的关系
    
    3. **验证曲线**：分析模型参数与训练/验证误差的关系
    
    4. **集成方法**：结合多个模型以平衡偏差和方差
    """)
    
    # 实战练习提示
    st.info("""
    **实验任务**：调整偏差因子和方差因子，观察最优复杂度的变化。
    
    **思考问题**：在现实机器学习应用中，我们如何估计模型的偏差和方差？什么情况下应该关注降低偏差，什么情况下应该关注降低方差？
    """) 