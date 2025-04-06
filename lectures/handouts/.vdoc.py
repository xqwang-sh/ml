# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| include: false
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import tensorflow as tf
import keras
from keras import layers

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Songti SC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 导入警告处理
import warnings
warnings.filterwarnings('ignore')
#
#
#
#
#
#| label: load-data
#| warning: false

# 读取训练集和测试集
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# 为数据集添加标记
train_data['train'] = 1
test_data['train'] = 0
test_data['Survived'] = np.nan

# 合并数据集
all_data = pd.concat([train_data, test_data], axis=0)

print(f"训练集样本数: {train_data.shape[0]}")
print(f"测试集样本数: {test_data.shape[0]}")
print(f"总样本数: {all_data.shape[0]}")
#
#
#
#
#
#| label: head-data

# 查看训练集前5行
train_data.head()
#
#
#
#
#
#
#
#| label: info-data

# 查看数据基本信息
train_data.info()
#
#
#
#
#
#| label: describe-data

# 统计描述
train_data.describe()
#
#
#
#
#
#| label: missing-values

# 检查缺失值
print("缺失值统计:")
all_data.isnull().sum()
#
#
#
#
#
#
#
#
#
#
#
#
#| label: survival-rate

# 分析生存率
survival_rate = train_data['Survived'].mean() * 100
print(f"总体生存率: {survival_rate:.2f}%")
#
#
#
#
#
#
#
#| label: visualization
#| fig-width: 12
#| fig-height: 10

# 设置图形大小
plt.figure(figsize=(15, 12))

# 1. 性别与生存率
plt.subplot(2, 2, 1)
sns.countplot(x='Sex', hue='Survived', data=train_data)
plt.title('性别与生存状况')
plt.xlabel('性别')
plt.ylabel('人数')

# 2. 船票等级与生存率
plt.subplot(2, 2, 2)
sns.countplot(x='Pclass', hue='Survived', data=train_data)
plt.title('船票等级与生存状况')
plt.xlabel('船票等级')
plt.ylabel('人数')

# 3. 年龄分布与生存情况
plt.subplot(2, 2, 3)
sns.histplot(data=train_data, x='Age', hue='Survived', multiple='stack', bins=20)
plt.title('年龄分布与生存状况')
plt.xlabel('年龄')
plt.ylabel('人数')

# 4. 家庭规模与生存情况
plt.subplot(2, 2, 4)
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
sns.countplot(x='FamilySize', hue='Survived', data=train_data)
plt.title('家庭规模与生存状况')
plt.xlabel('家庭规模')
plt.ylabel('人数')

plt.tight_layout()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: extract-title

# 提取姓名中的头衔
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# 统计头衔分布
print("头衔分布:")
all_data['Title'].value_counts()
#
#
#
#
#
#| label: combine-title

# 合并稀有头衔
rare_titles = ['Capt', 'Col', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'Countess']
all_data['Title'] = all_data['Title'].replace(rare_titles, 'Rare')
all_data['Title'] = all_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')

# 合并后的头衔分布
print("合并后的头衔分布:")
all_data['Title'].value_counts()
#
#
#
#
#
#
#
#| label: family-size

# 创建家庭规模特征
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1

# 创建是否独自一人特征
all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)

# 查看是否独自一人的分布
print("是否独自一人的分布:")
all_data['IsAlone'].value_counts()
#
#
#
#
#
#
#
#| label: cabin-info

# 从Cabin提取信息：是否有Cabin记录
all_data['HasCabin'] = (~all_data['Cabin'].isnull()).astype(int)

print("Cabin记录情况分布:")
all_data['HasCabin'].value_counts()
#
#
#
#
#
#
#
#
#
#| label: age-imputation

# Age缺失值填充（按Pclass和Sex分组的中位数）
age_imputer = all_data.groupby(['Pclass', 'Sex'])['Age'].transform('median')
all_data['Age'] = all_data['Age'].fillna(age_imputer)
#
#
#
#
#
#
#
#| label: embarked-imputation

# Embarked缺失值用众数填充
most_common_embarked = all_data['Embarked'].mode()[0]
all_data['Embarked'] = all_data['Embarked'].fillna(most_common_embarked)
#
#
#
#
#
#
#
#| label: fare-imputation

# Fare缺失值用Pclass中位数填充
fare_imputer = all_data.groupby('Pclass')['Fare'].transform('median')
all_data['Fare'] = all_data['Fare'].fillna(fare_imputer)

# 检查缺失值是否都已处理
print("缺失值处理后的统计:")
all_data[['Age', 'Embarked', 'Fare']].isnull().sum()
#
#
#
#
#
#
#
#| label: categorical-to-numerical

# 类别特征转换为数值
all_data['Sex'] = all_data['Sex'].map({'male': 0, 'female': 1})
all_data['Embarked'] = all_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
all_data['Title'] = all_data['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
#
#
#
#
#
#| label: drop-features

# 删除不需要的特征
all_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

# 查看处理后的数据结构
all_data.head()
#
#
#
#
#
#
#
#| label: prepare-train-test

# 准备训练和测试数据
train_data = all_data[all_data['train'] == 1].drop('train', axis=1)
test_data = all_data[all_data['train'] == 0].drop(['train', 'Survived'], axis=1)

X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived'].astype(int)

print(f"训练特征形状: {X_train.shape}")
print(f"训练标签形状: {y_train.shape}")
print(f"测试特征形状: {test_data.shape}")
#
#
#
#
#
#
#
#| label: train-evaluate-function

# 模型训练与评估函数
def train_and_evaluate(model, X, y, cv=5, model_name="模型"):
    """训练模型并评估性能"""
    # 交叉验证评估
    accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
    recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    roc_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    # 输出评估结果
    print(f"{model_name}交叉验证结果：")
    print(f"准确率: {accuracy.mean():.4f} (±{accuracy.std():.4f})")
    print(f"精确率: {precision.mean():.4f} (±{precision.std():.4f})")
    print(f"召回率: {recall.mean():.4f} (±{recall.std():.4f})")
    print(f"F1分数: {f1.mean():.4f} (±{f1.std():.4f})")
    print(f"ROC AUC: {roc_auc.mean():.4f} (±{roc_auc.std():.4f})")
    
    # 在全部训练数据上拟合模型
    model.fit(X, y)
    return model
#
#
#
#
#
#| label: logistic-regression

# 逻辑回归模型
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_fitted = train_and_evaluate(logreg_model, X_train, y_train, model_name="逻辑回归")

# 查看特征系数
logreg_coef = pd.DataFrame(
    logreg_fitted.coef_[0],
    index=X_train.columns,
    columns=['系数']
).sort_values('系数', ascending=False)

print("\n逻辑回归系数（特征重要性）:")
logreg_coef
#
#
#
#
#
#| label: decision-tree
#| code-fold: true

# 决策树模型
# 超参数网格搜索
param_grid = {
    'max_depth': range(1, 20),
    'min_samples_split': range(2, 20)
}

dt_model = DecisionTreeClassifier(random_state=42)
dt_grid = GridSearchCV(dt_model, param_grid, cv=5, scoring='accuracy')
dt_grid.fit(X_train, y_train)

print(f"决策树最佳参数: {dt_grid.best_params_}")
print(f"最佳交叉验证分数: {dt_grid.best_score_:.4f}")

# 可视化max_depth对模型性能的影响
plt.figure(figsize=(10, 6))
max_depths = range(1, 20)
test_scores = []

for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=depth, 
                              min_samples_split=dt_grid.best_params_['min_samples_split'],
                              random_state=42)
    dt.fit(X_train, y_train)
    # 使用交叉验证来评估
    score = cross_val_score(dt, X_train, y_train, cv=5, scoring='accuracy').mean()
    test_scores.append(score)

plt.plot(max_depths, test_scores)
plt.xlabel('max_depth (最大深度)')
plt.ylabel('准确率')
plt.title('决策树性能随最大深度的变化')
plt.grid(True)

# 使用最佳参数训练模型并评估
dt_fitted = train_and_evaluate(dt_grid.best_estimator_, X_train, y_train, model_name="决策树")

# 特征重要性
dt_importance = pd.DataFrame(
    dt_fitted.feature_importances_,
    index=X_train.columns,
    columns=['重要性']
).sort_values('重要性', ascending=False)

print("\n决策树特征重要性:")
dt_importance
#
#
#
#
#
#| label: random-forest
#| code-fold: true

# 随机森林模型
# 超参数网格搜索
param_grid = {
    'n_estimators': [10, 30, 50, 70, 100, 200],
    'max_depth': [3, 5, 7, 10, 15, 20, None]
}

rf_model = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print(f"随机森林最佳参数: {rf_grid.best_params_}")
print(f"最佳交叉验证分数: {rf_grid.best_score_:.4f}")

# 可视化n_estimators对模型性能的影响
plt.figure(figsize=(10, 6))
n_estimators = [10, 30, 50, 70, 100, 200]
test_scores = []

for n in n_estimators:
    rf = RandomForestClassifier(n_estimators=n,
                              max_depth=rf_grid.best_params_['max_depth'],
                              random_state=42)
    # 使用交叉验证来评估
    score = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy').mean()
    test_scores.append(score)

plt.plot(n_estimators, test_scores)
plt.xlabel('n_estimators (树的数量)')
plt.ylabel('准确率')
plt.title('随机森林性能随树数量的变化')
plt.grid(True)

# 可视化max_depth对模型性能的影响
plt.figure(figsize=(10, 6))
max_depths = [3, 5, 7, 10, 15, 20]
test_scores = []

for depth in max_depths:
    rf = RandomForestClassifier(n_estimators=rf_grid.best_params_['n_estimators'],
                              max_depth=depth,
                              random_state=42)
    # 使用交叉验证来评估
    score = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy').mean()
    test_scores.append(score)

plt.plot(max_depths, test_scores)
plt.xlabel('max_depth (最大深度)')
plt.ylabel('准确率')
plt.title('随机森林性能随最大深度的变化')
plt.grid(True)

# 使用最佳参数训练模型并评估
rf_fitted = train_and_evaluate(rf_grid.best_estimator_, X_train, y_train, model_name="随机森林")

# 特征重要性
rf_importance = pd.DataFrame(
    rf_fitted.feature_importances_,
    index=X_train.columns,
    columns=['重要性']
).sort_values('重要性', ascending=False)

print("\n随机森林特征重要性:")
rf_importance
#
#
#
#
#
#| label: xgboost
#| code-fold: true

# XGBoost模型
# 超参数网格搜索
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
}

xgb_model = xgb.XGBClassifier(random_state=42)
xgb_grid = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train, y_train)

print(f"XGBoost最佳参数: {xgb_grid.best_params_}")
print(f"最佳交叉验证分数: {xgb_grid.best_score_:.4f}")

# 可视化max_depth对模型性能的影响
plt.figure(figsize=(10, 6))
max_depths = [3, 5, 7, 9, 11]
test_scores = []

for depth in max_depths:
    xgb_model = xgb.XGBClassifier(
        n_estimators=xgb_grid.best_params_['n_estimators'],
        max_depth=depth,
        learning_rate=xgb_grid.best_params_['learning_rate'],
        random_state=42
    )
    # 使用交叉验证来评估
    score = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy').mean()
    test_scores.append(score)

plt.plot(max_depths, test_scores)
plt.xlabel('max_depth (最大深度)')
plt.ylabel('准确率')
plt.title('XGBoost性能随最大深度的变化')
plt.grid(True)

# 可视化learning_rate对模型性能的影响
plt.figure(figsize=(10, 6))
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
test_scores = []

for lr in learning_rates:
    xgb_model = xgb.XGBClassifier(
        n_estimators=xgb_grid.best_params_['n_estimators'],
        max_depth=xgb_grid.best_params_['max_depth'],
        learning_rate=lr,
        random_state=42
    )
    # 使用交叉验证来评估
    score = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy').mean()
    test_scores.append(score)

plt.plot(learning_rates, test_scores)
plt.xlabel('learning_rate (学习率)')
plt.ylabel('准确率')
plt.title('XGBoost性能随学习率的变化')
plt.grid(True)

# 使用最佳参数训练模型并评估
xgb_fitted = train_and_evaluate(xgb_grid.best_estimator_, X_train, y_train, model_name="XGBoost")

# 特征重要性
xgb_importance = pd.DataFrame(
    xgb_fitted.feature_importances_,
    index=X_train.columns,
    columns=['重要性']
).sort_values('重要性', ascending=False)

print("\nXGBoost特征重要性:")
xgb_importance
#
#
#
#
#
#
#
#
#
#| label: nn-standardization

# 对特征进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
test_data_scaled = scaler.transform(test_data)

# 划分训练集和验证集
X_train_nn, X_val, y_train_nn, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

print(f"神经网络训练集形状: {X_train_nn.shape}")
print(f"神经网络验证集形状: {X_val.shape}")
#
#
#
#
#
#
#
#| label: nn-simple

# 定义简单的全连接网络
def create_simple_nn():
    model = keras.Sequential([
        layers.Dense(8, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# 创建并训练简单神经网络
simple_nn = create_simple_nn()
simple_nn.summary()  # 显示模型结构

# 训练模型
simple_history = simple_nn.fit(
    X_train_nn, y_train_nn,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=0
)

# 评估性能
simple_train_loss, simple_train_acc = simple_nn.evaluate(X_train_nn, y_train_nn, verbose=0)
simple_val_loss, simple_val_acc = simple_nn.evaluate(X_val, y_val, verbose=0)

print(f"简单网络 - 训练集准确率: {simple_train_acc:.4f}")
print(f"简单网络 - 验证集准确率: {simple_val_acc:.4f}")
#
#
#
#
#
#| label: nn-simple-curves
#| fig-width: 12
#| fig-height: 5

# 绘制简单网络的学习曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(simple_history.history['accuracy'])
plt.plot(simple_history.history['val_accuracy'])
plt.title('简单网络准确率')
plt.ylabel('准确率')
plt.xlabel('轮次')
plt.legend(['训练集', '验证集'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(simple_history.history['loss'])
plt.plot(simple_history.history['val_loss'])
plt.title('简单网络损失')
plt.ylabel('损失')
plt.xlabel('轮次')
plt.legend(['训练集', '验证集'], loc='upper right')

plt.tight_layout()
#
#
#
#
#
#
#
#| label: nn-medium

# 定义中等复杂度的网络
def create_medium_nn():
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# 创建并训练中等复杂度神经网络
medium_nn = create_medium_nn()
medium_nn.summary()  # 显示模型结构

# 训练模型
medium_history = medium_nn.fit(
    X_train_nn, y_train_nn,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=0
)

# 评估性能
medium_train_loss, medium_train_acc = medium_nn.evaluate(X_train_nn, y_train_nn, verbose=0)
medium_val_loss, medium_val_acc = medium_nn.evaluate(X_val, y_val, verbose=0)

print(f"中等网络 - 训练集准确率: {medium_train_acc:.4f}")
print(f"中等网络 - 验证集准确率: {medium_val_acc:.4f}")
#
#
#
#
#
#| label: nn-medium-curves
#| fig-width: 12
#| fig-height: 5

# 绘制中等网络的学习曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(medium_history.history['accuracy'])
plt.plot(medium_history.history['val_accuracy'])
plt.title('中等网络准确率')
plt.ylabel('准确率')
plt.xlabel('轮次')
plt.legend(['训练集', '验证集'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(medium_history.history['loss'])
plt.plot(medium_history.history['val_loss'])
plt.title('中等网络损失')
plt.ylabel('损失')
plt.xlabel('轮次')
plt.legend(['训练集', '验证集'], loc='upper right')

plt.tight_layout()
#
#
#
#
#
#
#
#| label: nn-complex

# 定义复杂网络
def create_complex_nn():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# 创建并训练复杂神经网络
complex_nn = create_complex_nn()
complex_nn.summary()  # 显示模型结构

# 训练模型
complex_history = complex_nn.fit(
    X_train_nn, y_train_nn,
    epochs=100,  # 增加训练轮次以观察过拟合
    batch_size=16,  # 减小批量大小
    validation_data=(X_val, y_val),
    verbose=0
)

# 评估性能
complex_train_loss, complex_train_acc = complex_nn.evaluate(X_train_nn, y_train_nn, verbose=0)
complex_val_loss, complex_val_acc = complex_nn.evaluate(X_val, y_val, verbose=0)

print(f"复杂网络 - 训练集准确率: {complex_train_acc:.4f}")
print(f"复杂网络 - 验证集准确率: {complex_val_acc:.4f}")
#
#
#
#
#
#| label: nn-complex-curves
#| fig-width: 12
#| fig-height: 5

# 绘制复杂网络的学习曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(complex_history.history['accuracy'])
plt.plot(complex_history.history['val_accuracy'])
plt.title('复杂网络准确率')
plt.ylabel('准确率')
plt.xlabel('轮次')
plt.legend(['训练集', '验证集'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(complex_history.history['loss'])
plt.plot(complex_history.history['val_loss'])
plt.title('复杂网络损失')
plt.ylabel('损失')
plt.xlabel('轮次')
plt.legend(['训练集', '验证集'], loc='upper right')

plt.tight_layout()
#
#
#
#
#
#
#
#| label: nn-regularized

# 定义带有正则化的复杂网络
def create_regularized_nn():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(8, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# 创建并训练带有正则化的神经网络
regularized_nn = create_regularized_nn()
regularized_nn.summary()  # 显示模型结构

# 添加早停策略
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 训练模型
regularized_history = regularized_nn.fit(
    X_train_nn, y_train_nn,
    epochs=100,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=0
)

# 评估性能
regularized_train_loss, regularized_train_acc = regularized_nn.evaluate(X_train_nn, y_train_nn, verbose=0)
regularized_val_loss, regularized_val_acc = regularized_nn.evaluate(X_val, y_val, verbose=0)

print(f"正则化网络 - 训练集准确率: {regularized_train_acc:.4f}")
print(f"正则化网络 - 验证集准确率: {regularized_val_acc:.4f}")
#
#
#
#
#
#| label: nn-regularized-curves
#| fig-width: 12
#| fig-height: 5

# 绘制正则化网络的学习曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(regularized_history.history['accuracy'])
plt.plot(regularized_history.history['val_accuracy'])
plt.title('正则化网络准确率')
plt.ylabel('准确率')
plt.xlabel('轮次')
plt.legend(['训练集', '验证集'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(regularized_history.history['loss'])
plt.plot(regularized_history.history['val_loss'])
plt.title('正则化网络损失')
plt.ylabel('损失')
plt.xlabel('轮次')
plt.legend(['训练集', '验证集'], loc='upper right')

plt.tight_layout()
#
#
#
#
#
#
#
#| label: nn-comparison
#| fig-width: 12
#| fig-height: 6

# 比较不同网络的性能
nn_models = ['简单网络', '中等网络', '复杂网络', '正则化网络']
train_accuracy = [simple_train_acc, medium_train_acc, complex_train_acc, regularized_train_acc]
val_accuracy = [simple_val_acc, medium_val_acc, complex_val_acc, regularized_val_acc]

plt.figure(figsize=(12, 6))
x = np.arange(len(nn_models))
width = 0.35

plt.bar(x - width/2, train_accuracy, width, label='训练集准确率')
plt.bar(x + width/2, val_accuracy, width, label='验证集准确率')

plt.ylabel('准确率')
plt.title('不同神经网络模型的性能比较')
plt.xticks(x, nn_models)
plt.legend()

# 显示差距
for i in range(len(nn_models)):
    gap = train_accuracy[i] - val_accuracy[i]
    plt.text(i, 0.5, f'差距: {gap:.4f}', ha='center')

plt.tight_layout()
#
#
#
#
#
#
#
#| label: final-nn-model

# 使用正则化网络作为最终模型
nn_model = regularized_nn

# 在完整训练集上重新训练
X_full_scaled = scaler.transform(X_train)
nn_model.fit(X_full_scaled, y_train, epochs=50, batch_size=16, verbose=0)

# 生成最终预测
nn_pred = (nn_model.predict(X_full_scaled) > 0.5).astype(int).flatten()
nn_accuracy = accuracy_score(y_train, nn_pred)
print(f"最终神经网络在训练集上的准确率: {nn_accuracy:.4f}")

# 将正则化网络用于测试集预测
test_pred_nn = (nn_model.predict(test_data_scaled) > 0.5).astype(int).flatten()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: model-comparison

# 在整个训练集上再次训练各模型
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(**dt_grid.best_params_, random_state=42)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(**rf_grid.best_params_, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = xgb.XGBClassifier(**xgb_grid.best_params_, random_state=42)
xgb_model.fit(X_train, y_train)

# 在训练集上的预测
y_pred_logreg = logreg_model.predict(X_train)
y_pred_dt = dt_model.predict(X_train)
y_pred_rf = rf_model.predict(X_train)
y_pred_xgb = xgb_model.predict(X_train)
y_pred_nn = (nn_model.predict(X_train_scaled) > 0.5).astype(int).flatten()

# 计算各种评估指标
models = ['逻辑回归', '决策树', '随机森林', 'XGBoost', '神经网络']
predictions = [y_pred_logreg, y_pred_dt, y_pred_rf, y_pred_xgb, y_pred_nn]

metrics_df = pd.DataFrame(index=models, columns=['准确率', '精确率', '召回率', 'F1分数'])

for i, model_name in enumerate(models):
    acc = accuracy_score(y_train, predictions[i])
    prec = precision_score(y_train, predictions[i])
    rec = recall_score(y_train, predictions[i])
    f1 = f1_score(y_train, predictions[i])
    
    metrics_df.loc[model_name] = [acc, prec, rec, f1]

print("各模型在训练集上的性能比较:")
metrics_df
#
#
#
#
#
#| label: model-comparison-plot
#| fig-width: 12
#| fig-height: 6

# 绘制性能比较图
plt.figure(figsize=(12, 6))
metrics_df.plot(kind='bar', figsize=(12, 6))
plt.title('各模型性能比较')
plt.ylabel('得分')
plt.xlabel('模型')
plt.legend(loc='lower right')
plt.tight_layout()
#
#
#
#
#
#
#
#| label: test-prediction

# 为所有模型生成预测
test_pred_logreg = logreg_model.predict(test_data)
test_pred_dt = dt_model.predict(test_data)
test_pred_rf = rf_model.predict(test_data)
test_pred_xgb = xgb_model.predict(test_data)
test_pred_nn = (nn_model.predict(test_data_scaled) > 0.5).astype(int).flatten()

# 使用随机森林的预测作为最终结果（可以根据交叉验证结果选择最佳模型）
final_predictions = test_pred_rf

# 创建提交文件
submission = pd.DataFrame({
    'PassengerId': pd.read_csv("data/test.csv")['PassengerId'],
    'Survived': final_predictions
})

submission.to_csv('titanic_submission.csv', index=False)
print("提交文件已保存为 'titanic_submission.csv'")
#
#
#
#
#
#| label: prediction-comparison

# 各模型预测结果的比较
test_predictions = pd.DataFrame({
    'PassengerId': pd.read_csv("data/test.csv")['PassengerId'],
    '逻辑回归': test_pred_logreg,
    '决策树': test_pred_dt,
    '随机森林': test_pred_rf,
    'XGBoost': test_pred_xgb,
    '神经网络': test_pred_nn
})

# 检查不同模型预测的一致性
agreement = test_predictions.iloc[:, 1:].sum(axis=1)
print("模型预测一致性统计:")
print(agreement.value_counts())

print("\n各模型预测示例 (前10行):")
test_predictions.head(10)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
