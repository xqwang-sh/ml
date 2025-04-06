import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import time
from PIL import Image
import io
import base64

# 设置绘图风格
plt.style.use('ggplot')
# 根据操作系统设置不同的字体
import platform
# 获取操作系统类型
system = platform.system()
# 设置 matplotlib 字体
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 使用黑体
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Songti SC']  # Mac 使用宋体
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # Linux 使用文泉驿正黑
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 设置页面配置
st.set_page_config(
    page_title="神经网络演示",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #004D40;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .explanation {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .formula {
        background-color: #E3F2FD;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown("<div class='main-header'>神经网络交互式学习</div>", unsafe_allow_html=True)

# 侧边栏导航
st.sidebar.title("导航")
pages = [
    "⚙️ 神经网络训练算法可视化",
    "🔄 神经网络反向传播算法可视化",
    "📉 梯度消失问题及解决方案",
    "📈 梯度爆炸问题及解决方案"
]
selection = st.sidebar.radio("选择一个部分:", pages)

# 添加侧边栏学习资源和关于信息
with st.sidebar.expander("学习资源"):
    st.markdown("""
    - [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning)
    - [Deep Learning Specialization](https://www.deeplearning.ai/)
    - [PyTorch 教程](https://pytorch.org/tutorials/)
    - [TensorFlow 教程](https://www.tensorflow.org/tutorials)
    """)

with st.sidebar.expander("关于"):
    st.markdown("""
    此应用程序旨在帮助学生理解神经网络的训练过程、梯度问题及其解决方案。
    
    通过交互式可视化，您可以深入了解神经网络的工作原理及常见问题的解决方法。
    
    版本: 2.0.0
    """)

# 根据选择加载不同页面的内容
if selection == "⚙️ 神经网络训练算法可视化":
    from training import show_training_page
    show_training_page()
elif selection == "🔄 神经网络反向传播算法可视化":
    from backpropagation import show_backpropagation_page
    show_backpropagation_page()
elif selection == "📉 梯度消失问题及解决方案":
    from vanishing_gradient import show_vanishing_gradient_page
    show_vanishing_gradient_page()
elif selection == "📈 梯度爆炸问题及解决方案":
    from exploding_gradient import show_exploding_gradient_page
    show_exploding_gradient_page() 