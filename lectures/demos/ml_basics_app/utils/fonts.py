import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st

def get_system_font():
    """根据操作系统选择合适的字体"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统常见中文字体
        font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        # macOS系统常见中文字体
        font_candidates = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Hiragino Sans GB', 'Apple LiGothic']
    else:  # Linux或其他系统
        # Linux系统常见中文字体
        font_candidates = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'Noto Sans CJK SC', 'DejaVu Sans']
    
    # 检查哪些字体实际可用
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 选择第一个可用的字体
    for font in font_candidates:
        if font in available_fonts:
            return font
    
    # 如果没有找到匹配的字体，返回默认字体
    return 'sans-serif'

def configure_matplotlib_fonts():
    """配置Matplotlib使用系统字体"""
    font = get_system_font()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def get_svg_style():
    """获取SVG样式，包括合适的字体设置"""
    font = get_system_font()
    return f"""
    <style>
        .st-svg {{
            font-family: {font}, 'DejaVu Sans', 'Arial Unicode MS', sans-serif;
        }}
        .st-svg text {{
            font-family: {font}, 'DejaVu Sans', 'Arial Unicode MS', sans-serif;
        }}
    </style>
    """ 