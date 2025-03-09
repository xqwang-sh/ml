import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
import pandas as pd
import time

def show_exercises_page():
    """显示练习页面"""
    st.markdown("<div class='sub-header'>神经网络练习</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='explanation'>本页面提供了一系列练习题，帮助巩固对神经网络基本概念和原理的理解。包括选择题和交互式实践。</div>", unsafe_allow_html=True)
    
    # 选择练习类型
    exercise_type = st.selectbox(
        "选择练习类型",
        ["选择题", "神经网络调参练习", "梯度下降可视化练习", "正则化效果练习"]
    )
    
    if exercise_type == "选择题":
        show_quiz()
    elif exercise_type == "神经网络调参练习":
        show_neural_network_exercise()
    elif exercise_type == "梯度下降可视化练习":
        show_gradient_descent_exercise()
    else:  # 正则化效果练习
        show_regularization_exercise()

def show_quiz():
    """显示选择题"""
    st.markdown("<div class='sub-header'>神经网络基础知识选择题</div>", unsafe_allow_html=True)
    
    # 初始化session state
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    
    # 问题列表
    questions = [
        {
            "question": "1. 单层感知机的主要局限性是什么？",
            "options": [
                "A. 训练速度太慢",
                "B. 只能解决线性可分问题",
                "C. 需要太多计算资源",
                "D. 容易过拟合"
            ],
            "answer": "B. 只能解决线性可分问题"
        },
        {
            "question": "2. 以下哪个激活函数可能导致'神经元死亡'问题？",
            "options": [
                "A. Sigmoid",
                "B. Tanh",
                "C. ReLU",
                "D. Softmax"
            ],
            "answer": "C. ReLU"
        },
        {
            "question": "3. 反向传播算法主要用于计算什么？",
            "options": [
                "A. 损失函数值",
                "B. 网络的输出",
                "C. 参数的梯度",
                "D. 激活函数"
            ],
            "answer": "C. 参数的梯度"
        },
        {
            "question": "4. 神经网络中的梯度消失问题最常见于使用哪种激活函数时？",
            "options": [
                "A. ReLU",
                "B. Leaky ReLU",
                "C. Sigmoid",
                "D. ELU"
            ],
            "answer": "C. Sigmoid"
        },
        {
            "question": "5. 多层感知器能够解决XOR问题的关键原因是什么？",
            "options": [
                "A. 使用了更多参数",
                "B. 使用了非线性激活函数",
                "C. 训练时间更长",
                "D. 使用了更复杂的损失函数"
            ],
            "answer": "B. 使用了非线性激活函数"
        },
        {
            "question": "6. 以下哪种正则化方法通过随机'关闭'神经元来防止过拟合？",
            "options": [
                "A. L1正则化",
                "B. L2正则化",
                "C. Dropout",
                "D. 早停法"
            ],
            "answer": "C. Dropout"
        },
        {
            "question": "7. 神经网络中的批量归一化(Batch Normalization)主要作用是什么？",
            "options": [
                "A. 加速训练收敛",
                "B. 减少内存使用",
                "C. 简化网络结构",
                "D. 减少训练样本需求"
            ],
            "answer": "A. 加速训练收敛"
        },
        {
            "question": "8. 在选择优化算法时，哪种算法通常被认为结合了动量和自适应学习率的优点？",
            "options": [
                "A. SGD",
                "B. RMSProp",
                "C. Adagrad",
                "D. Adam"
            ],
            "answer": "D. Adam"
        },
        {
            "question": "9. 深度神经网络相比浅层网络的主要优势是什么？",
            "options": [
                "A. 训练速度更快",
                "B. 参数更少",
                "C. 具有层次化特征学习能力",
                "D. 不会过拟合"
            ],
            "answer": "C. 具有层次化特征学习能力"
        },
        {
            "question": "10. 'He初始化'方法最适合与哪种激活函数一起使用？",
            "options": [
                "A. Sigmoid",
                "B. Tanh",
                "C. ReLU",
                "D. Softmax"
            ],
            "answer": "C. ReLU"
        }
    ]
    
    # 显示问题
    for i, q in enumerate(questions):
        st.markdown(f"**{q['question']}**")
        
        key = f"q{i}"
        answer = st.radio(
            "选择答案：",
            q['options'],
            key=key,
            index=None
        )
        
        if key in st.session_state.quiz_answers:
            if st.session_state.quiz_submitted:
                if st.session_state.quiz_answers[key] == q['answer']:
                    st.success("✓ 回答正确！")
                else:
                    st.error(f"✗ 回答错误。正确答案是：{q['answer']}")
        
        st.session_state.quiz_answers[key] = answer
        
        st.markdown("---")
    
    # 提交按钮
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("提交答案"):
            st.session_state.quiz_submitted = True
            
            # 计算得分
            score = 0
            for i, q in enumerate(questions):
                key = f"q{i}"
                if key in st.session_state.quiz_answers:
                    if st.session_state.quiz_answers[key] == q['answer']:
                        score += 1
            
            st.session_state.quiz_score = score
            st.rerun()
    
    with col2:
        if st.session_state.quiz_submitted:
            st.markdown(f"**得分: {st.session_state.quiz_score}/{len(questions)}**")
            
            if st.session_state.quiz_score == len(questions):
                st.balloons()
                st.success("恭喜你全部答对！")
            elif st.session_state.quiz_score >= len(questions) * 0.7:
                st.success("干得不错！你对神经网络有很好的理解。")
            else:
                st.info("继续努力！建议再回顾相关概念。")

def neural_network(X, hidden_layers, activation):
    """
    简单的神经网络前向传播
    """
    np.random.seed(42)
    
    # 输入层
    n_samples, n_features = X.shape
    layer_sizes = [n_features] + hidden_layers + [1]
    
    # 初始化权重和偏置
    weights = []
    biases = []
    
    for i in range(len(layer_sizes) - 1):
        w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
        b = np.zeros((1, layer_sizes[i+1]))
        weights.append(w)
        biases.append(b)
    
    # 前向传播
    activations = [X]
    
    for i in range(len(weights)):
        Z = np.dot(activations[-1], weights[i]) + biases[i]
        
        if activation == "sigmoid":
            A = 1 / (1 + np.exp(-Z))
        elif activation == "tanh":
            A = np.tanh(Z)
        elif activation == "relu":
            A = np.maximum(0, Z)
        
        activations.append(A)
    
    return activations[-1]

def show_neural_network_exercise():
    """
    神经网络调参练习
    """
    st.markdown("<div class='sub-header'>神经网络调参练习</div>", unsafe_allow_html=True)
    
    st.markdown("""
    在本练习中，您将通过调整神经网络的结构和参数，尝试解决不同的分类问题。
    观察不同设置如何影响模型的决策边界。
    
    **目标**：通过调整网络结构（层数、每层神经元数量）和激活函数，使模型能够正确分类所选数据集。
    """)
    
    # 选择数据集
    dataset_type = st.selectbox(
        "选择数据集",
        ["月牙形数据", "圆形数据"],
        key="dataset_select"
    )
    
    # 生成数据
    np.random.seed(42)
    if dataset_type == "月牙形数据":
        X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
        dataset_desc = "月牙形数据集由两个相互缠绕的半圆组成，是测试非线性分类能力的经典数据集。"
    else:  # 圆形数据
        X, y = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=42)
        dataset_desc = "圆形数据集由一个内圈和一个外圈组成，需要模型能够学习圆形决策边界。"
    
    st.markdown(f"**数据集描述**：{dataset_desc}")
    
    # 显示数据
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X[y==0, 0], X[y==0, 1], c='#1E88E5', s=50, edgecolors='k', label='类别 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='#D81B60', s=50, edgecolors='k', label='类别 1')
    ax.set_xlabel('特征1', fontsize=12)
    ax.set_ylabel('特征2', fontsize=12)
    ax.set_title('数据集可视化', fontsize=14)
    ax.legend()
    st.pyplot(fig)
    
    # 神经网络参数设置
    st.markdown("### 调整神经网络参数")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_layers = st.slider("隐藏层数量", 1, 3, 1)
        
        hidden_layers = []
        for i in range(num_layers):
            neurons = st.slider(f"第 {i+1} 层神经元数量", 2, 20, 5)
            hidden_layers.append(neurons)
        
        activation = st.selectbox(
            "激活函数",
            ["sigmoid", "tanh", "relu"]
        )
    
    with col2:
        st.markdown("### 网络结构")
        
        # 显示网络结构
        layer_desc = "输入层(2) → "
        for i, neurons in enumerate(hidden_layers):
            layer_desc += f"隐藏层{i+1}({neurons}) → "
        layer_desc += "输出层(1)"
        
        st.markdown(f"""
        **网络结构**：{layer_desc}
        
        **激活函数**：{activation}
        
        **参数总量**：{sum([2 * hidden_layers[0] + 1] + [hidden_layers[i] * hidden_layers[i+1] + hidden_layers[i+1] for i in range(num_layers-1)]) + hidden_layers[-1] + 1}
        """)
    
    # 绘制决策边界函数
    def plot_decision_boundary(X, model, hidden_layers, activation, title):
        h = 0.02  # 网格步长
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # 生成预测
        Z = model(np.c_[xx.ravel(), yy.ravel()], hidden_layers, activation)
        Z = Z.reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制决策边界
        ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Blues)
        
        # 绘制数据点
        ax.scatter(X[y==0, 0], X[y==0, 1], c='#1E88E5', s=50, edgecolors='k', label='类别 0')
        ax.scatter(X[y==1, 0], X[y==1, 1], c='#D81B60', s=50, edgecolors='k', label='类别 1')
        ax.set_xlabel('特征1', fontsize=12)
        ax.set_ylabel('特征2', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        
        return fig
    
    # 运行模型并绘制决策边界
    if st.button("生成决策边界"):
        with st.spinner("模型训练中..."):
            fig = plot_decision_boundary(X, neural_network, hidden_layers, activation, 
                                       f"神经网络决策边界 ({num_layers}层, {activation}激活函数)")
            st.pyplot(fig)
    
    # 挑战任务
    st.markdown("### 挑战任务")
    
    challenges = {
        "月牙形数据": {
            "easy": "使用一个至少有5个神经元的隐藏层和任意激活函数，生成能够分隔两个月牙的决策边界。",
            "medium": "使用tanh激活函数和2层隐藏层，每层不超过8个神经元，生成平滑的决策边界。",
            "hard": "使用ReLU激活函数和最少数量的神经元（总数<20），正确分类所有点。"
        },
        "圆形数据": {
            "easy": "使用一个至少有10个神经元的隐藏层，生成近似圆形的决策边界。",
            "medium": "使用tanh激活函数和2层隐藏层，每层不超过6个神经元，分隔内外圆。",
            "hard": "设计一个总神经元数<15的网络，生成完美的圆形决策边界。"
        }
    }
    
    challenge_level = st.selectbox(
        "选择挑战难度",
        ["简单", "中等", "困难"]
    )
    
    if challenge_level == "简单":
        st.info(f"**简单挑战**：{challenges[dataset_type]['easy']}")
    elif challenge_level == "中等":
        st.warning(f"**中等挑战**：{challenges[dataset_type]['medium']}")
    else:
        st.error(f"**困难挑战**：{challenges[dataset_type]['hard']}")

def show_gradient_descent_exercise():
    """
    梯度下降可视化练习的简化版本
    """
    st.markdown("<div class='sub-header'>梯度下降可视化练习</div>", unsafe_allow_html=True)
    
    st.markdown("""
    在本练习中，您将通过调整梯度下降的参数，观察优化过程如何受到影响。
    
    本练习使用简单的线性回归模型，目标是找到合适的权重和偏置，使模型最佳拟合数据。
    """)
    
    # 生成简单的线性数据
    np.random.seed(42)
    x = np.random.rand(20) * 4 - 2
    true_w, true_b = 2.5, -1.0
    y = true_w * x + true_b + np.random.randn(20) * 0.5
    
    # 显示数据
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, c='blue', s=50, alpha=0.8)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('数据集', fontsize=14)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    # 梯度下降参数
    st.markdown("### 参数设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.slider("学习率", 0.01, 1.0, 0.1, 0.01)
        num_iterations = st.slider("迭代次数", 1, 20, 10)
        
    with col2:
        initial_w = st.slider("初始权重 (w)", -5.0, 5.0, 0.0, 0.1)
        initial_b = st.slider("初始偏置 (b)", -5.0, 5.0, 0.0, 0.1)
    
    # 定义损失函数和梯度函数
    def loss_function(w, b, x, y):
        """简单的二次损失函数"""
        y_pred = w * x + b
        return ((y - y_pred) ** 2).mean()
    
    def gradient(w, b, x, y):
        """损失函数的梯度"""
        y_pred = w * x + b
        dw = -2 * (y - y_pred).dot(x) / len(x)
        db = -2 * (y - y_pred).sum() / len(x)
        return dw, db
    
    # 梯度下降
    if st.button("运行梯度下降"):
        # 初始化参数
        w = initial_w
        b = initial_b
        
        # 保存历史
        w_history = [w]
        b_history = [b]
        loss_history = [loss_function(w, b, x, y)]
        
        # 执行梯度下降
        for i in range(num_iterations):
            dw, db = gradient(w, b, x, y)
            w = w - learning_rate * dw
            b = b - learning_rate * db
            
            w_history.append(w)
            b_history.append(b)
            loss_history.append(loss_function(w, b, x, y))
        
        # 绘制损失变化图
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(range(len(loss_history)), loss_history, 'b-', linewidth=2)
        ax1.set_xlabel('迭代次数', fontsize=12)
        ax1.set_ylabel('损失', fontsize=12)
        ax1.set_title('损失函数变化', fontsize=14)
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
        
        # 绘制拟合结果
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.scatter(x, y, c='blue', s=50, alpha=0.8, label='数据点')
        
        x_line = np.array([x.min() - 0.5, x.max() + 0.5])
        
        # 初始线
        y_line_init = initial_w * x_line + initial_b
        ax2.plot(x_line, y_line_init, 'r--', linewidth=2, 
                label=f'初始: y = {initial_w:.2f}x + {initial_b:.2f}')
        
        # 最终线
        y_line_final = w_history[-1] * x_line + b_history[-1]
        ax2.plot(x_line, y_line_final, 'g-', linewidth=2, 
                label=f'最终: y = {w_history[-1]:.2f}x + {b_history[-1]:.2f}')
        
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('y', fontsize=12)
        ax2.set_title('回归线拟合结果', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        
        # 显示结果
        st.success(f"""
        **梯度下降结果**：
        - 初始参数: w = {initial_w:.4f}, b = {initial_b:.4f}
        - 最终参数: w = {w_history[-1]:.4f}, b = {b_history[-1]:.4f}
        - 初始损失: {loss_history[0]:.6f}
        - 最终损失: {loss_history[-1]:.6f}
        """)

def show_regularization_exercise():
    """
    正则化效果练习的简化版本
    """
    st.markdown("<div class='sub-header'>正则化效果练习</div>", unsafe_allow_html=True)
    
    st.markdown("""
    在本练习中，您将探索正则化如何影响多项式回归模型的拟合效果。通过调整多项式次数和正则化强度，
    观察模型如何在欠拟合和过拟合之间取得平衡。
    """)
    
    # 生成非线性数据
    np.random.seed(42)
    x = np.linspace(-3, 3, 50)
    y_true = 1.5 * np.sin(x) + 0.5 * x**2
    y_noisy = y_true + np.random.normal(0, 1, size=len(x))
    
    # 分割训练集和测试集
    indices = np.random.permutation(len(x))
    train_idx, test_idx = indices[:35], indices[35:]
    
    x_train, y_train = x[train_idx], y_noisy[train_idx]
    x_test, y_test = x[test_idx], y_noisy[test_idx]
    
    # 可视化数据
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y_true, 'g-', linewidth=2, label='真实函数')
    ax.scatter(x_train, y_train, c='blue', s=50, alpha=0.8, label='训练集')
    ax.scatter(x_test, y_test, c='red', s=50, alpha=0.8, label='测试集')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('数据集', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    # 模型参数设置
    st.markdown("### 模型设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        poly_degree = st.slider("多项式次数", 1, 10, 3)
        reg_strength = st.slider("正则化强度", 0.0, 1.0, 0.0, 0.01)
    
    # 训练模型
    if st.button("拟合模型"):
        # 创建多项式特征
        X_train = np.vstack([x_train**i for i in range(1, poly_degree+1)]).T
        X_test = np.vstack([x_test**i for i in range(1, poly_degree+1)]).T
        
        # 添加截距项
        X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
        
        # 带正则化的最小二乘法
        n_features = X_train_bias.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0  # 不正则化截距
        
        # 解析解
        XTX = X_train_bias.T.dot(X_train_bias)
        XTX_reg = XTX + reg_strength * I
        XTy = X_train_bias.T.dot(y_train)
        weights = np.linalg.solve(XTX_reg, XTy)
        
        # 预测
        y_train_pred = X_train_bias.dot(weights)
        y_test_pred = X_test_bias.dot(weights)
        
        # 计算MSE
        train_mse = np.mean((y_train - y_train_pred)**2)
        test_mse = np.mean((y_test - y_test_pred)**2)
        
        # 绘制拟合曲线
        x_plot = np.linspace(min(x), max(x), 1000)
        X_plot = np.vstack([x_plot**i for i in range(1, poly_degree+1)]).T
        X_plot_bias = np.hstack([np.ones((X_plot.shape[0], 1)), X_plot])
        y_plot_pred = X_plot_bias.dot(weights)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y_true, 'g-', linewidth=2, label='真实函数')
        ax.scatter(x_train, y_train, c='blue', s=30, alpha=0.6, label='训练集')
        ax.scatter(x_test, y_test, c='red', s=30, alpha=0.6, label='测试集')
        ax.plot(x_plot, y_plot_pred, 'k-', linewidth=2, 
               label=f'模型预测 (阶数={poly_degree}, 正则化={reg_strength})')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title('多项式回归拟合', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        # 显示系数
        coef_fig, coef_ax = plt.subplots(figsize=(10, 6))
        coef_labels = ['截距'] + [f'x^{i}' for i in range(1, poly_degree+1)]
        coef_ax.bar(coef_labels, weights, color='blue', alpha=0.7)
        coef_ax.set_xlabel('系数', fontsize=12)
        coef_ax.set_ylabel('值', fontsize=12)
        coef_ax.set_title('模型系数', fontsize=14)
        plt.xticks(rotation=45)
        st.pyplot(coef_fig)
        
        # 显示评估指标
        st.success(f"""
        **模型评估**：
        - 训练集MSE：{train_mse:.6f}
        - 测试集MSE：{test_mse:.6f}
        - 训练/测试误差比：{train_mse/test_mse if test_mse > 0 else 'N/A':.4f}
        
        **过拟合判断**：
        {
            "模型表现良好，没有明显过拟合。" if test_mse < 1.5 * train_mse else
            "模型可能存在轻微过拟合。" if test_mse < 3 * train_mse else
            "模型存在明显过拟合，建议增强正则化或降低模型复杂度。"
        }
        """)
    
    # 学习曲线
    st.markdown("### 关于正则化的建议")
    
    st.info("""
    **过拟合的处理方法**：
    1. 减少模型复杂度（降低多项式次数）
    2. 增加正则化强度
    3. 获取更多训练数据
    
    **欠拟合的处理方法**：
    1. 增加模型复杂度（提高多项式次数）
    2. 减少正则化强度
    3. 添加更多相关特征
    
    尝试找到一组参数，使测试集误差最小！
    """) 