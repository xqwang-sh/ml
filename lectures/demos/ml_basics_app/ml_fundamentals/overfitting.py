"""
过拟合与欠拟合的交互式演示模块
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from utils.svg_generator import create_overfitting_svg, create_interactive_overfitting_svg

def show_overfitting_demo():
    """显示过拟合与欠拟合的交互式演示"""
    
    st.subheader("过拟合与欠拟合")
    
    st.markdown("""
    **过拟合(Overfitting)**是指模型在训练数据上表现良好，但在新数据上表现不佳的现象。过拟合的模型"记住"了训练数据中的噪声和随机波动，而不是学习到数据的真实模式。
    
    **欠拟合(Underfitting)**是指模型既不能很好地拟合训练数据，也不能很好地泛化到新数据的情况。欠拟合通常是由于模型过于简单，无法捕捉数据中的复杂模式所致。
    
    下图直观地展示了欠拟合、适当拟合和过拟合：
    """)
    
    # 显示过拟合vs欠拟合示意图
    st.markdown(create_overfitting_svg(), unsafe_allow_html=True)
    
    # 交互式演示
    st.markdown("### 交互式演示：多项式拟合")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        尝试改变多项式的阶数，观察模型如何拟合数据。
        
        - **较低的阶数**可能导致欠拟合
        - **适中的阶数**通常能较好地拟合数据
        - **较高的阶数**可能导致过拟合
        """)
        
        degree = st.slider("多项式阶数", min_value=1, max_value=20, value=3, step=1)
        
        noise_level = st.slider("噪声水平", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
        
        sample_points = st.slider("样本点数量", min_value=10, max_value=100, value=30, step=5)
    
    with col2:
        # 生成并显示交互式多项式拟合图
        if 'random_seed' not in st.session_state:
            st.session_state.random_seed = 42
        
        if st.button("生成新数据"):
            st.session_state.random_seed = np.random.randint(0, 1000)
        
        # 动态创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 生成示例数据
        np.random.seed(st.session_state.random_seed)
        x = np.linspace(0, 1, sample_points)
        y_true = np.sin(2 * np.pi * x)
        y = y_true + np.random.normal(0, noise_level, sample_points)
        x_plot = np.linspace(0, 1, 100)
        y_real = np.sin(2 * np.pi * x_plot)
        
        # 多项式拟合
        coeffs = np.polyfit(x, y, degree)
        y_poly = np.polyval(coeffs, x_plot)
        
        # 计算误差
        train_error = np.mean((np.polyval(coeffs, x) - y)**2)
        
        # 生成测试数据
        np.random.seed(st.session_state.random_seed + 1)  # 不同的随机种子
        x_test = np.random.uniform(0, 1, 30)
        y_test = np.sin(2 * np.pi * x_test) + np.random.normal(0, noise_level, 30)
        test_error = np.mean((np.polyval(coeffs, x_test) - y_test)**2)
        
        ax.scatter(x, y, color='blue', label='训练数据')
        ax.plot(x_plot, y_real, 'g--', label='真实函数')
        ax.plot(x_plot, y_poly, 'r-', label=f'{degree}次多项式')
        ax.set_title(f'多项式拟合(度数={degree})')
        ax.legend()
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("训练集误差", f"{train_error:.4f}")
        with col2:
            st.metric("测试集误差", f"{test_error:.4f}")
            
    # 讨论部分
    st.markdown("### 如何防止过拟合？")
    
    st.markdown("""
    1. **收集更多数据**：增加训练样本通常有助于减少过拟合
    
    2. **特征选择**：减少不相关特征，保留最重要的特征
    
    3. **交叉验证**：使用K折交叉验证来调整模型参数
    
    4. **正则化**：向模型添加惩罚项以减少复杂性
    
    5. **提前停止**：在训练误差继续减少但验证误差开始增加时停止训练
    
    6. **集成方法**：如随机森林，通过结合多个模型减少方差
    
    7. **适当的模型选择**：根据问题复杂度选择适当复杂度的模型
    """)
    
    # 实战练习提示
    st.info("""
    **实验任务**：尝试不同的多项式阶数，观察训练误差和测试误差的变化趋势。找出在本例中最佳的多项式阶数是多少？
    
    **思考问题**：当多项式阶数增加时，训练误差和测试误差如何变化？这与偏差-方差权衡有什么关系？
    """) 