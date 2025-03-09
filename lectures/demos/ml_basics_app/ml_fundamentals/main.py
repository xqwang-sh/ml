"""
机器学习基础模块的主要入口点
"""

import streamlit as st
from utils.fonts import configure_matplotlib_fonts, get_svg_style

# 导入子模块
from ml_fundamentals.overfitting import show_overfitting_demo
from ml_fundamentals.bias_variance import show_bias_variance_demo
from ml_fundamentals.regularization import show_regularization_demo
from ml_fundamentals.model_evaluation import show_model_evaluation_demo
from ml_fundamentals.exercises import show_ml_exercises

def show_ml_fundamentals():
    """显示机器学习基础知识"""
    
    # 配置字体
    configure_matplotlib_fonts()
    
    # 添加SVG字体样式
    st.markdown(get_svg_style(), unsafe_allow_html=True)
    
    st.header("机器学习基础概念")
    
    # 创建选项卡
    topic = st.radio(
        "选择主题:",
        ["过拟合与欠拟合", "偏差-方差权衡", "正则化技术", "模型评估与超参数", "交互式练习"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # 显示选择的主题内容
    if topic == "过拟合与欠拟合":
        show_overfitting_demo()
    elif topic == "偏差-方差权衡":
        show_bias_variance_demo()
    elif topic == "正则化技术":
        show_regularization_demo()
    elif topic == "模型评估与超参数":
        show_model_evaluation_demo()
    elif topic == "交互式练习":
        show_ml_exercises() 