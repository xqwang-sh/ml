import streamlit as st

def apply_modern_style():
    """应用清新的投影友好型Streamlit样式"""
    st.markdown("""
    <style>
        /* 全局主题颜色 - 清新投影友好型 */
        :root {
            --primary-color: #3498db;
            --secondary-color: #2ecc71;
            --background-color: #ffffff;
            --second-background-color: #f0f8ff;
            --text-color: #2c3e50;
            --light-text-color: #7f8c8d;
            --border-color: #d4e6f1;
            --shadow-color: rgba(0, 0, 0, 0.05);
            --font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;
        }
        
        /* 全局字体和背景 */
        html, body, [class*="css"] {
            font-family: var(--font-family);
            color: var(--text-color);
            -webkit-font-smoothing: antialiased;
        }
        
        /* 标题样式 */
        h1, h2, h3, h4, h5, h6 {
            font-family: var(--font-family);
            font-weight: 600;
            color: var(--text-color);
            letter-spacing: -0.01em;
        }
        
        h1 {
            font-size: 2.6rem;
            color: #2980b9;
            border-bottom: 2px solid #d4e6f1;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }
        
        h2 {
            font-size: 2rem;
            margin-top: 2rem;
            font-weight: 500;
            color: #3498db;
        }
        
        h3 {
            font-size: 1.6rem;
            margin-top: 1.5rem;
            font-weight: 500;
            color: #2c3e50;
        }
        
        /* 段落文本 */
        p {
            color: var(--text-color);
            font-size: 1.1rem;
            line-height: 1.6;
        }
        
        /* 卡片容器 */
        .stCard {
            border-radius: 8px;
            box-shadow: 0 1px 3px var(--shadow-color);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background-color: var(--background-color);
            transition: all 0.2s ease;
            border: 1px solid var(--border-color);
        }
        
        .stCard:hover {
            box-shadow: 0 2px 6px var(--shadow-color);
        }
        
        /* 侧边栏样式 */
        .css-1d391kg, .css-12oz5g7 {
            background-color: var(--second-background-color);
        }
        
        .sidebar .sidebar-content {
            background-color: var(--second-background-color);
        }
        
        /* 按钮样式 - 清新风格 */
        .stButton>button {
            border-radius: 6px;
            font-weight: 500;
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 0.6rem 1.4rem;
            transition: all 0.2s ease;
            font-size: 1rem;
        }
        
        .stButton>button:hover {
            background-color: #2980b9;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        /* Checkbox 和 Radio 样式 */
        .stCheckbox>div>label, .stRadio>div>label {
            font-weight: 500;
            color: var(--text-color);
            font-size: 1.05rem;
        }
        
        /* 选项卡样式 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 2.6rem;
            white-space: pre-wrap;
            background-color: #e8f4fc;
            border-radius: 6px;
            padding: 0 1.2rem;
            font-size: 1rem;
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            background-color: var(--primary-color) !important;
            color: white !important;
        }
        
        /* SVG样式 */
        .st-svg {
            max-width: 100%;
            height: auto;
            margin: 1rem 0;
        }
        
        /* 滚动条样式 */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f0f8ff;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #a9cce3;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #7fb3d5;
        }
        
        /* 代码块样式 */
        code {
            padding: 0.2em 0.4em;
            margin: 0;
            font-size: 90%;
            background-color: #eaf2f8;
            border-radius: 4px;
            font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            color: #e74c3c;
        }
        
        pre {
            background-color: #f7fbff;
            border-radius: 8px;
            padding: 16px;
            overflow: auto;
            border: 1px solid #d4e6f1;
        }
        
        /* 表格样式 */
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #d4e6f1;
        }
        
        thead th {
            background-color: #e8f4fc;
            color: #2c3e50;
            padding: 0.8rem 1.2rem;
            text-align: left;
            font-weight: 500;
            border-bottom: 1px solid #d4e6f1;
            font-size: 1.05rem;
        }
        
        tbody tr:nth-child(even) {
            background-color: #f7fbff;
        }
        
        tbody td {
            padding: 0.8rem 1.2rem;
            border-bottom: 1px solid #d4e6f1;
            color: #34495e;
            font-size: 1rem;
        }
        
        /* Form 元素样式 */
        .stForm {
            background-color: #f7fbff;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #d4e6f1;
            margin-bottom: 1.5rem;
        }
        
        /* Radio 按钮组样式强化 */
        .stRadio > div {
            background-color: #f7fbff;
            padding: 0.8rem;
            border-radius: 8px;
            border: 1px solid #d4e6f1;
        }
        
        .stRadio > div > div > label {
            font-size: 1.05rem;
            padding: 0.5rem 0;
        }
        
        /* 响应式调整 */
        @media (max-width: 768px) {
            h1 {
                font-size: 2.2rem;
            }
            
            h2 {
                font-size: 1.8rem;
            }
            
            h3 {
                font-size: 1.4rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def create_card(title, content):
    """创建一个清新风格的styled card"""
    return f"""
    <div class="stCard">
        <h3>{title}</h3>
        <div>{content}</div>
    </div>
    """ 