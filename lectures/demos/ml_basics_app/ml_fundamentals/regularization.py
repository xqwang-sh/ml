"""
正则化技术的交互式演示模块
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from utils.svg_generator import create_interactive_regularization_svg

def show_regularization_demo():
    """显示正则化技术的交互式演示"""
    
    st.subheader("正则化技术")
    
    st.markdown("""
    **正则化**是一种防止过拟合的技术，通过向损失函数添加惩罚项来限制模型参数的大小，从而降低模型复杂度。
    
    ### 为什么需要正则化？
    
    - 防止模型过度拟合训练数据
    - 提高模型泛化能力
    - 处理高维数据中的特征共线性
    - 在有大量特征但少量样本的情况下尤其重要
    
    ### 常用的正则化方法：
    
    1. **L1正则化(Lasso)**：向损失函数添加参数绝对值之和的惩罚
       - 损失函数：$L(w) + \\lambda \\sum_{i=1}^{n} |w_i|$
       - 特点：倾向于产生稀疏解（许多参数为零），可用于特征选择
       
    2. **L2正则化(Ridge)**：向损失函数添加参数平方和的惩罚
       - 损失函数：$L(w) + \\lambda \\sum_{i=1}^{n} w_i^2$
       - 特点：惩罚较大的参数，使所有参数值变小但不为零
       
    3. **弹性网络(Elastic Net)**：结合L1和L2正则化
       - 损失函数：$L(w) + \\lambda_1 \\sum_{i=1}^{n} |w_i| + \\lambda_2 \\sum_{i=1}^{n} w_i^2$
       - 特点：结合了Lasso和Ridge的优点
    """)
    
    # 交互式演示
    st.markdown("### 交互式演示：正则化效果")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        reg_type = st.radio(
            "正则化类型",
            ["L1 (Lasso)", "L2 (Ridge)"]
        )
        
        reg_strength = st.slider(
            "正则化强度 (λ)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01
        )
        
        st.markdown("""
        尝试调整正则化强度，观察模型拟合的变化：
        
        - **λ = 0**: 没有正则化，可能过拟合
        - **较小的λ**: 轻微正则化
        - **较大的λ**: 强正则化，可能欠拟合
        """)
    
    with col2:
        # 显示正则化效果的SVG图
        reg_type_param = "L1" if reg_type == "L1 (Lasso)" else "L2"
        st.markdown(create_interactive_regularization_svg(reg_strength, reg_type_param), unsafe_allow_html=True)
    
    # 系数变化演示
    st.markdown("### 系数变化演示")
    
    # 生成一些相关的特征数据
    np.random.seed(42)
    n_samples, n_features = 50, 10
    X = np.random.randn(n_samples, n_features)
    # 添加一些相关性
    X[:, 5:] = X[:, :5] + np.random.randn(n_samples, 5) * 0.5
    y = 3*X[:, 0] + 1.5*X[:, 1] + np.random.randn(n_samples) * 1.5
    
    # 训练不同正则化强度的模型
    alphas = [0, 0.01, 0.1, 0.5, 1.0, 10.0]
    
    ridge_coefs = []
    lasso_coefs = []
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)
        ridge_coefs.append(ridge.coef_)
        
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, y)
        lasso_coefs.append(lasso.coef_)
    
    # 绘制系数变化图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.set_title("Ridge (L2)系数随正则化强度变化")
    ax1.set_xscale('log')
    for i in range(n_features):
        ax1.plot(alphas, [coef[i] for coef in ridge_coefs], 'o-', label=f"特征 {i+1}")
    ax1.set_xlabel('正则化强度 (alpha)')
    ax1.set_ylabel('系数值')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Lasso (L1)系数随正则化强度变化")
    ax2.set_xscale('log')
    for i in range(n_features):
        ax2.plot(alphas, [coef[i] for coef in lasso_coefs], 'o-', label=f"特征 {i+1}")
    ax2.set_xlabel('正则化强度 (alpha)')
    ax2.set_ylabel('系数值')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("解释系数变化图"):
        st.markdown("""
        **观察要点：**
        
        1. **Ridge (L2)正则化**：
           - 随着正则化强度增加，所有系数逐渐变小，但通常不会变为绝对零
           - 系数减小的速度相对平缓
           - 保留了所有特征的影响
        
        2. **Lasso (L1)正则化**：
           - 随着正则化强度增加，许多系数变为精确的零
           - 系数变化更加剧烈，呈现出"特征选择"效果
           - 在高正则化强度下，只保留最重要的几个特征
        
        这说明了为什么Lasso常用于特征选择，而Ridge则用于处理多重共线性。
        """)
    
    # 案例研究
    st.markdown("### 实际应用案例")
    
    st.markdown("""
    #### 1. 高维数据中的特征选择
    
    在基因表达分析中，我们可能有成千上万个基因特征，但只有少数几十个样本。使用L1正则化可以自动选择最相关的基因，降低过拟合风险。
    
    #### 2. 图像重建与压缩感知
    
    在MRI图像重建中，L1正则化可以从少量测量中恢复完整图像，利用了医学图像在某些变换域中的稀疏性。
    
    #### 3. 金融市场预测
    
    在预测股票价格时，有大量可能的预测变量。L2正则化可以稳定模型，减少市场噪声的影响，提高预测稳定性。
    
    #### 4. 推荐系统
    
    在协同过滤中，正则化可以防止模型对某些活跃用户或热门物品过度拟合，提高推荐质量。
    """)
    
    # 实战练习提示
    st.info("""
    **实验任务**：
    1. 尝试不同的正则化强度，观察模型拟合曲线的变化
    2. 观察L1和L2正则化如何不同地影响模型系数
    
    **思考问题**：
    1. 什么情况下应该选择L1正则化而不是L2正则化？
    2. 如何在实际应用中确定最佳的正则化强度？
    """) 