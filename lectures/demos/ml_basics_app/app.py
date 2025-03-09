import streamlit as st
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置页面标题和配置
st.set_page_config(
    page_title="机器学习基础",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "机器学习基础 教学应用"
    }
)

# 导入样式工具
from utils.styles import apply_modern_style, create_card

# 应用现代风格样式
apply_modern_style()

# 导入机器学习基础模块
from ml_fundamentals import show_ml_fundamentals

def main():
    st.title("机器学习基础")
    
    st.markdown("""
    这个应用程序涵盖了机器学习的核心基础概念，包括:
    
    - 过拟合与欠拟合
    - 偏差-方差权衡
    - 正则化技术
    - 模型评估与超参数
    
    每个主题都配有交互式演示，帮助你直观理解这些重要概念。
    """)
    
    # 显示机器学习基础内容
    show_ml_fundamentals()

if __name__ == "__main__":
    main() 