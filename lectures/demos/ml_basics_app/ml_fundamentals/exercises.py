"""
机器学习基础的练习模块
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from utils.svg_generator import create_learning_curve_svg

def show_ml_exercises():
    """显示机器学习基础的练习"""
    
    st.subheader("机器学习基础实践练习")
    
    st.markdown("""
    本节提供交互式练习，帮助您巩固对机器学习基础概念的理解。通过实验不同的参数设置，
    您可以直观地观察过拟合、欠拟合、正则化等概念的效果。
    """)
    
    # 选择练习类型
    exercise_type = st.selectbox(
        "选择练习类型",
        ["过拟合与欠拟合实验", "正则化效果实验", "学习曲线分析练习"]
    )
    
    if exercise_type == "过拟合与欠拟合实验":
        show_overfitting_exercise()
    elif exercise_type == "正则化效果实验":
        show_regularization_exercise()
    elif exercise_type == "学习曲线分析练习":
        show_learning_curve_exercise()

def show_overfitting_exercise():
    """过拟合与欠拟合实验"""
    
    st.markdown("## 过拟合与欠拟合实验")
    
    st.markdown("""
    在本练习中，您将使用多项式回归模型拟合数据，并通过调整多项式阶数观察模型如何从欠拟合过渡到过拟合。
    
    ### 任务目标
    
    1. 生成带有噪声的数据
    2. 使用不同阶数的多项式模型拟合数据
    3. 观察训练误差和测试误差的变化
    4. 确定最佳的多项式阶数
    """)
    
    # 实验设置
    col1, col2 = st.columns([1, 1])
    
    with col1:
        noise = st.slider("噪声水平", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
        n_samples = st.slider("样本数量", min_value=20, max_value=200, value=50, step=10)
        test_size = st.slider("测试集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        max_degree = st.slider("最大多项式阶数", min_value=1, max_value=20, value=10, step=1)
    
    with col2:
        st.markdown("""
        ### 参数说明
        
        - **噪声水平**: 控制数据中随机噪声的大小
        - **样本数量**: 生成的总样本点数
        - **测试集比例**: 划分给测试集的数据比例
        - **最大多项式阶数**: 尝试的最高多项式阶数
        
        点击下方按钮生成数据并运行实验:
        """)
        
        run_experiment = st.button("运行实验", key="overfitting_experiment")
    
    if run_experiment:
        # 生成数据
        np.random.seed(42)
        X = np.sort(np.random.rand(n_samples) * 5 - 2.5)
        y_true = X**2 + X - 1  # 真实函数是二次函数
        y = y_true + np.random.normal(0, noise, size=X.shape)
        
        # 重塑数据
        X = X.reshape(-1, 1)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # 不同阶数的多项式模型
        degrees = range(1, max_degree + 1)
        train_errors = []
        test_errors = []
        
        for degree in degrees:
            # 创建多项式回归模型
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('linear', LinearRegression())
            ])
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 计算误差
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            
            train_errors.append(train_mse)
            test_errors.append(test_mse)
        
        # 找出最佳阶数
        best_degree = degrees[np.argmin(test_errors)]
        
        # 绘制数据和多项式拟合曲线
        st.markdown("### 数据拟合可视化")
        
        # 选择展示的三个阶数
        if max_degree <= 3:
            display_degrees = list(range(1, max_degree + 1))
        else:
            display_degrees = [1, best_degree, max_degree]
            display_degrees.sort()
        
        fig, axes = plt.subplots(1, len(display_degrees), figsize=(15, 5))
        if len(display_degrees) == 1:
            axes = [axes]
        
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        
        for i, degree in enumerate(display_degrees):
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('linear', LinearRegression())
            ])
            model.fit(X_train, y_train)
            y_plot = model.predict(X_plot)
            
            axes[i].scatter(X_train, y_train, color='blue', alpha=0.7, label='训练数据')
            axes[i].scatter(X_test, y_test, color='red', alpha=0.7, label='测试数据')
            axes[i].plot(X_plot, y_plot, color='green', label=f'{degree}次多项式')
            axes[i].set_title(f'{degree}次多项式拟合')
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('y')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 绘制误差曲线
        st.markdown("### 误差分析")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(degrees, train_errors, 'o-', color='blue', label='训练误差')
        ax.plot(degrees, test_errors, 'o-', color='red', label='测试误差')
        ax.axvline(x=best_degree, color='green', linestyle='--', 
                  label=f'最佳阶数 = {best_degree}')
        
        ax.set_xlabel('多项式阶数')
        ax.set_ylabel('均方误差 (MSE)')
        ax.set_title('训练误差 vs 测试误差')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # 实验结论
        st.markdown("### 实验结论")
        
        st.success(f"**最佳多项式阶数**: {best_degree}")
        
        if best_degree <= 2:
            st.info("模型选择了一个简单的多项式，这可能接近真实的数据生成过程（二次函数）。")
        elif best_degree <= 4:
            st.info("模型选择了一个适度复杂的多项式，能够平衡拟合和泛化。")
        else:
            st.warning("模型选择了一个较复杂的多项式，可能存在过拟合风险。请考虑增加训练数据或引入正则化。")
        
        st.markdown("""
        #### 观察到的现象
        
        - **低阶多项式**（阶数< 2）: 可能欠拟合，无法捕捉数据的曲率
        - **中阶多项式**（阶数 2-4）: 通常能较好地拟合数据，又不过拟合
        - **高阶多项式**（阶数 > 4）: 容易过拟合，在训练数据上表现很好，但在测试数据上表现较差
        
        #### 思考问题
        
        1. 噪声水平如何影响最佳多项式阶数的选择？
        2. 样本数量如何影响过拟合的可能性？
        3. 如果真实函数是二次函数，为什么最佳阶数可能不是2？
        """)

def show_regularization_exercise():
    """正则化效果实验"""
    
    st.markdown("## 正则化效果实验")
    
    st.markdown("""
    在本练习中，您将比较不同类型和强度的正则化对高维回归问题的影响。
    
    ### 任务目标
    
    1. 生成具有一些无关特征和噪声的高维数据
    2. 使用不同正则化方法（无正则化、L1、L2）训练模型
    3. 观察模型性能和特征系数的变化
    4. 体验正则化如何提高模型的泛化能力
    """)
    
    # 实验设置
    col1, col2 = st.columns([1, 1])
    
    with col1:
        n_features = st.slider("特征数量", min_value=10, max_value=100, value=50, step=10)
        n_informative = st.slider("有效特征数量", min_value=5, max_value=20, value=10, step=1)
        noise = st.slider("噪声水平", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        
        alpha_values = st.multiselect(
            "选择正则化强度 (alpha)",
            options=[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            default=[0.0, 0.1, 1.0, 10.0]
        )
        
        if not alpha_values:
            alpha_values = [0.0, 0.1, 1.0, 10.0]  # 默认值
    
    with col2:
        st.markdown("""
        ### 参数说明
        
        - **特征数量**: 数据集中的总特征数
        - **有效特征数量**: 真正与目标变量相关的特征数
        - **噪声水平**: 数据中的随机噪声大小
        - **正则化强度 (alpha)**: 控制正则化惩罚的强度
          - **0.0**: 无正则化
          - **较小值**: 轻微正则化
          - **较大值**: 强正则化
        
        点击下方按钮生成数据并运行实验:
        """)
        
        run_experiment = st.button("运行实验", key="regularization_experiment")
    
    if run_experiment:
        # 生成数据
        np.random.seed(42)
        X, y, coef = make_regression(
            n_samples=100,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            coef=True,
            random_state=42
        )
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练不同正则化的模型
        models = {
            "无正则化": LinearRegression,
            "L1 (Lasso)": Lasso,
            "L2 (Ridge)": Ridge
        }
        
        results = []
        all_coefs = {}
        
        for name, model_class in models.items():
            if name == "无正则化":
                model = model_class()
                model.fit(X_train_scaled, y_train)
                
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                results.append({
                    "模型": name,
                    "alpha": 0.0,
                    "训练MSE": train_mse,
                    "测试MSE": test_mse,
                    "非零系数数量": np.sum(model.coef_ != 0)
                })
                
                all_coefs[name] = {"alpha": 0.0, "coef": model.coef_}
            else:
                for alpha in alpha_values:
                    if alpha == 0.0 and name != "无正则化":
                        continue  # 跳过，因为已经有无正则化的模型
                        
                    model = model_class(alpha=alpha)
                    model.fit(X_train_scaled, y_train)
                    
                    train_pred = model.predict(X_train_scaled)
                    test_pred = model.predict(X_test_scaled)
                    
                    train_mse = mean_squared_error(y_train, train_pred)
                    test_mse = mean_squared_error(y_test, test_pred)
                    
                    results.append({
                        "模型": f"{name} (alpha={alpha})",
                        "alpha": alpha,
                        "训练MSE": train_mse,
                        "测试MSE": test_mse,
                        "非零系数数量": np.sum(model.coef_ != 0)
                    })
                    
                    all_coefs[f"{name} (alpha={alpha})"] = {"alpha": alpha, "coef": model.coef_}
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 找出最佳模型
        best_model = results_df.loc[results_df["测试MSE"].idxmin()]
        
        # 显示结果表
        st.markdown("### 不同正则化模型的性能比较")
        
        styled_df = results_df.style.highlight_min(subset=["测试MSE"], color="lightgreen").format({
            "训练MSE": "{:.4f}",
            "测试MSE": "{:.4f}"
        })
        
        st.dataframe(styled_df)
        
        # 绘制性能比较图
        st.markdown("### 性能可视化")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MSE比较
        l1_results = results_df[results_df["模型"].str.contains("L1")]
        l2_results = results_df[results_df["模型"].str.contains("L2")]
        
        if not l1_results.empty:
            ax1.plot(l1_results["alpha"], l1_results["训练MSE"], 'o-', color='blue', label='L1 训练MSE')
            ax1.plot(l1_results["alpha"], l1_results["测试MSE"], 'o-', color='red', label='L1 测试MSE')
        
        if not l2_results.empty:
            ax1.plot(l2_results["alpha"], l2_results["训练MSE"], 's--', color='darkblue', label='L2 训练MSE')
            ax1.plot(l2_results["alpha"], l2_results["测试MSE"], 's--', color='darkred', label='L2 测试MSE')
        
        ax1.set_xscale('log')
        ax1.set_xlabel('正则化强度 (alpha)')
        ax1.set_ylabel('均方误差 (MSE)')
        ax1.set_title('不同正则化强度的性能比较')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 非零系数数量
        if not l1_results.empty:
            ax2.plot(l1_results["alpha"], l1_results["非零系数数量"], 'o-', color='blue', label='L1 (Lasso)')
        
        if not l2_results.empty:
            ax2.plot(l2_results["alpha"], l2_results["非零系数数量"], 's--', color='red', label='L2 (Ridge)')
        
        ax2.axhline(y=n_informative, color='green', linestyle='--', 
                   label=f'真实有效特征数量 ({n_informative})')
        
        ax2.set_xscale('log')
        ax2.set_xlabel('正则化强度 (alpha)')
        ax2.set_ylabel('非零系数数量')
        ax2.set_title('特征选择效果')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 系数变化可视化
        st.markdown("### 系数变化可视化")
        
        # 选择一些有代表性的模型进行比较
        selected_models = ["无正则化"]
        if "L1 (Lasso) (alpha=1.0)" in all_coefs:
            selected_models.append("L1 (Lasso) (alpha=1.0)")
        if "L2 (Ridge) (alpha=1.0)" in all_coefs:
            selected_models.append("L2 (Ridge) (alpha=1.0)")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 只显示前20个系数，避免图表过于拥挤
        display_features = min(20, n_features)
        x = np.arange(display_features)
        width = 0.25
        
        for i, model_name in enumerate(selected_models):
            coefs = all_coefs[model_name]["coef"][:display_features]
            ax.bar(x + (i - 1) * width, coefs, width, label=model_name)
        
        ax.set_xlabel('特征索引')
        ax.set_ylabel('系数值')
        ax.set_title('不同正则化方法的系数比较 (前20个特征)')
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(display_features)])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # 实验结论
        st.markdown("### 实验结论")
        
        st.success(f"**最佳模型**: {best_model['模型']} (测试MSE = {best_model['测试MSE']:.4f})")
        
        st.markdown(f"""
        #### 观察到的现象
        
        1. **特征选择**:
           - L1正则化(Lasso)有明显的特征选择能力，随着alpha增加，系数变为0
           - L2正则化(Ridge)保留了所有特征，但减小了系数的大小
           
        2. **性能比较**:
           - 无正则化模型: {'容易过拟合，测试误差较高' if results_df.loc[0, '测试MSE'] > best_model['测试MSE'] else '表现还不错，但可能缺乏泛化能力'}
           - L1正则化: {'在合适的alpha值下能够选出真正相关的特征，提高泛化能力' if 'L1' in best_model['模型'] else '在本例中表现不如其他模型'}
           - L2正则化: {'在合适的alpha值下能够控制所有系数的大小，提高泛化能力' if 'L2' in best_model['模型'] else '在本例中表现不如其他模型'}
        
        3. **最佳正则化强度**:
           - 本实验中，最佳正则化强度为 {best_model['alpha'] if 'alpha' in best_model else 'N/A'}
           - 此时非零系数数量: {best_model['非零系数数量']} (真实有效特征数量: {n_informative})
        """)
        
        st.markdown("""
        #### 思考问题
        
        1. 为什么L1正则化可以实现特征选择，而L2正则化不能？
        2. 在什么情况下您会选择L1而不是L2正则化，反之亦然？
        3. 如果特征之间存在强相关性，不同的正则化方法会有什么不同的表现？
        """)

def show_learning_curve_exercise():
    """显示学习曲线分析练习"""
    
    st.markdown("### 学习曲线分析练习")
    
    st.markdown("""
    学习曲线是诊断模型偏差和方差问题的有力工具。通过观察训练误差和验证误差随训练集大小的变化，
    我们可以判断模型是否存在欠拟合或过拟合问题。
    
    **典型的学习曲线模式：**
    """)
    
    # 显示学习曲线示例图
    st.markdown(create_learning_curve_svg(), unsafe_allow_html=True)
    
    st.markdown("""
    **练习目标：**
    1. 生成二分类数据集
    2. 训练逻辑回归模型，调整正则化参数
    3. 绘制学习曲线并分析模型是否存在偏差或方差问题
    4. 尝试不同的正则化设置，观察学习曲线的变化
    """)
    
    # 练习参数设置
    col1, col2 = st.columns([1, 1])
    
    with col1:
        n_features = st.slider("特征数量", min_value=10, max_value=50, value=20, step=5)
        n_informative = st.slider("有信息特征数量", min_value=2, max_value=10, value=5, step=1)
        
        c_param = st.select_slider(
            "正则化强度 (C)",
            options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            value=1.0
        )
        
        penalty = st.radio(
            "正则化类型",
            ["l1", "l2"]
        )
    
    with col2:
        st.markdown("""
        ### 参数说明
        
        - **特征数量**: 数据集中的总特征数
        - **有信息特征数量**: 真正与目标变量相关的特征数
        - **正则化强度 (C)**: 控制正则化惩罚的强度
          - **小C值**: 强正则化（更简单的模型）
          - **大C值**: 弱正则化（更复杂的模型）
        - **正则化类型**:
          - **L1**: 倾向于产生稀疏解
          - **L2**: 倾向于小权重值
        
        点击下方按钮生成数据并分析学习曲线:
        """)
        
        run_experiment = st.button("运行实验", key="learning_curve_experiment")
    
    if run_experiment:
        # 生成数据
        np.random.seed(42)
        X, y = make_classification(
            n_samples=500,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_features - n_informative - 5,
            n_classes=2,
            class_sep=1.5,
            random_state=42
        )
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 创建并训练模型
        model = LogisticRegression(C=c_param, penalty=penalty, solver='liblinear', random_state=42)
        model.fit(X_train, y_train)
        
        # 评估模型
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        # 计算学习曲线
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, valid_scores = learning_curve(
            model, X_train, y_train, train_sizes=train_sizes, cv=5, scoring='accuracy'
        )
        
        # 计算平均值和标准差
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        valid_mean = np.mean(valid_scores, axis=1)
        valid_std = np.std(valid_scores, axis=1)
        
        # 展示模型性能
        st.markdown("### 模型性能")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("训练集准确率", f"{train_acc:.4f}")
        with col2:
            st.metric("测试集准确率", f"{test_acc:.4f}")
        
        # 计算更多指标
        precision = precision_score(y_test, test_pred)
        recall = recall_score(y_test, test_pred)
        f1 = f1_score(y_test, test_pred)
        
        st.markdown("#### 详细评估指标")
        metrics_df = pd.DataFrame({
            "指标": ["准确率", "精确率", "召回率", "F1分数"],
            "值": [test_acc, precision, recall, f1]
        })
        
        st.dataframe(metrics_df.style.format({"值": "{:.4f}"}))
        
        # 绘制学习曲线
        st.markdown("### 学习曲线分析")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.grid(True, alpha=0.3)
        ax.fill_between(train_sizes, train_mean - train_std,
                       train_mean + train_std, alpha=0.1, color='blue')
        ax.fill_between(train_sizes, valid_mean - valid_std,
                       valid_mean + valid_std, alpha=0.1, color='red')
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='训练准确率')
        ax.plot(train_sizes, valid_mean, 'o-', color='red', label='交叉验证准确率')
        ax.set_xlabel('训练样本数量')
        ax.set_ylabel('准确率')
        ax.set_title(f'学习曲线 (LogisticRegression, C={c_param}, penalty={penalty})')
        ax.legend(loc='best')
        
        st.pyplot(fig)
        
        # 诊断分析
        gap = np.mean(train_mean - valid_mean)
        final_train = train_mean[-1]
        final_valid = valid_mean[-1]
        
        if final_train > 0.9 and gap > 0.1:
            diagnosis = "高方差(过拟合)"
            symptom = "模型在训练集上表现很好，但验证集性能显著较低"
            suggestion = "尝试增加正则化强度（减小C值），收集更多数据，或减少特征数量"
        elif final_train < 0.8 and gap < 0.1:
            diagnosis = "高偏差(欠拟合)"
            symptom = "训练和验证性能都不理想，且接近"
            suggestion = "尝试减少正则化强度（增大C值），使用更复杂的模型，或添加更多特征"
        else:
            diagnosis = "平衡的模型"
            symptom = "模型有合理的训练-验证性能权衡"
            suggestion = "当前参数设置很好，可以微调以进一步改进"
        
        # 系数分析
        non_zero_coefs = np.sum(model.coef_ != 0)
        
        # 结果分析
        st.markdown("### 模型诊断")
        
        st.info(f"""
        **诊断结果**: {diagnosis}
        
        **观察到的症状**: {symptom}
        
        **建议**: {suggestion}
        
        **模型复杂度分析**:
        - 使用特征数量: {n_features}
        - 有信息特征数量: {n_informative}
        - 非零系数数量: {non_zero_coefs}
        - C值(正则化强度倒数): {c_param}
        
        **学习曲线特征**:
        - 训练集和验证集性能差距: {gap:.4f}
        - 训练集最终准确率: {final_train:.4f}
        - 验证集最终准确率: {final_valid:.4f}
        """)
        
        # 动手实验建议
        st.markdown("### 下一步实验建议")
        
        if diagnosis == "高方差(过拟合)":
            st.success("""
            **尝试这些参数来减少过拟合**:
            - 减小C值 (增加正则化): 尝试 0.1, 0.01
            - 如果使用L2正则化，考虑切换到L1以获得更稀疏的模型
            """)
        elif diagnosis == "高偏差(欠拟合)":
            st.success("""
            **尝试这些参数来减少欠拟合**:
            - 增大C值 (减少正则化): 尝试 10, 100
            - 如果当前使用L1正则化，考虑切换到L2以保留更多特征
            """)
        else:
            st.success("""
            **当前模型设置良好! 您可以尝试**:
            - 小幅调整C值以查看是否能进一步改善
            - 尝试特征工程或不同的特征选择方法
            - 考虑尝试更复杂的模型类型
            """)
        
        st.markdown("""
        #### 思考问题
        
        1. 学习曲线的收敛性如何帮助判断是否需要更多训练数据？
        2. 如何通过学习曲线确定模型的最佳复杂度？
        3. 对于当前的数据集，什么样的正则化设置最合适，为什么？
        """) 