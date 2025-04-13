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
#| include: false
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Songti SC']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 导入警告处理
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子，确保结果可复现
np.random.seed(42)
#
#
#
#
#
#
#
#| label: load-data
#| warning: false

# 模拟获取沪深300成分股数据
# 实际应用中可使用akshare或tushare接口获取实时数据
def get_sample_stock_data():
    # 这里使用预设数据代替API调用，简化演示
    sample_stocks = [
        "600000", "600036", "601318", "600519", "000858", 
        "000651", "601166", "000333", "002475", "600276", 
        "601888", "300750", "600900", "601899", "601628",
        "600048", "600887", "600050", "600030", "601398"
    ]
    
    # 模拟行业分类
    industry_map = {
        "600000": "银行", "600036": "银行", "601318": "保险", 
        "600519": "白酒", "000858": "食品饮料", "000651": "家电",
        "601166": "银行", "000333": "家电", "002475": "软件服务",
        "600276": "医药", "601888": "旅游", "300750": "电子设备",
        "600900": "公用事业", "601899": "有色金属", "601628": "保险",
        "600048": "房地产", "600887": "食品饮料", "600050": "电信运营",
        "600030": "证券", "601398": "银行"
    }
    
    # 创建股票及行业映射DataFrame
    stock_data = pd.DataFrame({
        '股票代码': sample_stocks,
        '行业': [industry_map[code] for code in sample_stocks]
    })
    
    return stock_data, industry_map

# 获取股票数据
stock_data, industry_map = get_sample_stock_data()
stock_list = stock_data['股票代码'].tolist()

print(f"获取到 {len(stock_list)} 只股票")
stock_data.head(10)
#
#
#
#
#
#| label: generate-returns

# 模拟生成股票收益率数据
def generate_sample_returns(stock_list, days=252, seed=42):
    """
    生成模拟的股票日收益率数据
    - stock_list: 股票代码列表
    - days: 交易日数量(约一年)
    - seed: 随机种子
    """
    np.random.seed(seed)
    
    # 生成日期索引(从今天往前推一年)
    import datetime
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, periods=days)
    
    # 创建空的DataFrame用于存储收益率
    returns_matrix = pd.DataFrame(index=date_range)
    
    # 行业相关性(同行业股票走势可能更相似)
    industries = [industry_map[code] for code in stock_list]
    unique_industries = list(set(industries))
    industry_factors = {ind: np.random.normal(0, 0.01, days) for ind in unique_industries}
    
    # 市场共同因子(影响所有股票)
    market_factor = np.random.normal(0.0005, 0.01, days)
    
    # 为每只股票生成日收益率数据
    for code in stock_list:
        # 获取行业因子
        industry = industry_map[code]
        industry_factor = industry_factors[industry]
        
        # 生成特定股票噪声
        stock_specific = np.random.normal(0, 0.015, days)
        
        # 组合因子生成最终收益率(市场因子+行业因子+个股因素)
        returns = market_factor + industry_factor * 0.7 + stock_specific
        
        # 添加到DataFrame
        returns_matrix[code] = returns
    
    return returns_matrix

# 生成收益率数据
returns_matrix = generate_sample_returns(stock_list)

print(f"收益率矩阵形状: {returns_matrix.shape}")
returns_matrix.head()
#
#
#
#
#
#
#
#| label: eda
#| fig-width: 12
#| fig-height: 8

# 绘制部分股票的收益率走势
plt.figure(figsize=(12, 6))
returns_matrix.iloc[:, :5].cumsum().plot()
plt.title('部分股票的累积收益率走势')
plt.xlabel('日期')
plt.ylabel('累积收益率')
plt.grid(True)
plt.legend()
plt.show()

# 绘制收益率相关性热力图
plt.figure(figsize=(10, 8))
corr_matrix = returns_matrix.corr()
sns.heatmap(corr_matrix, cmap='viridis', annot=False)
plt.title('股票收益率相关性热力图')
plt.tight_layout()
plt.show()

# 按行业分组计算平均收益率
industry_returns = {}
for industry in set(industry_map.values()):
    # 获取该行业的股票
    industry_stocks = [code for code in stock_list if industry_map[code] == industry]
    if industry_stocks:
        # 计算该行业的平均日收益率
        industry_returns[industry] = returns_matrix[industry_stocks].mean(axis=1)

# 转换为DataFrame并计算累积收益
industry_returns_df = pd.DataFrame(industry_returns)
industry_cumulative = industry_returns_df.cumsum()

# 绘制行业累积收益对比
plt.figure(figsize=(12, 6))
industry_cumulative.plot()
plt.title('不同行业的累积收益率对比')
plt.xlabel('日期')
plt.ylabel('累积收益率')
plt.grid(True)
plt.legend()
plt.show()
#
#
#
#
#
#
#
#| label: preprocessing

# 转置矩阵，使每一行代表一只股票，每一列代表一个交易日
# 这样每只股票成为一个样本点，其收益率序列为特征
X = returns_matrix.T.values

# 使用StandardScaler进行特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"原始数据形状: {X.shape}")
print(f"标准化后数据形状: {X_scaled.shape}")
#
#
#
#
#
#
#
#
#
#| label: kmeans-k-selection
#| fig-width: 12
#| fig-height: 10

# 设置要尝试的K值范围
k_range = range(2, 11)

# --- 肘部法则 ---
wcss = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# --- 轮廓系数 ---
silhouette_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# --- 戴维斯-布尔丁指数 ---
dbi_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = davies_bouldin_score(X_scaled, labels)
    dbi_scores.append(score)

# 可视化三种评估方法
plt.figure(figsize=(15, 10))

# 绘制肘部图
plt.subplot(2, 2, 1)
plt.plot(k_range, wcss, marker='o', linestyle='-')
plt.title('肘部法则 - 确定最佳K值')
plt.xlabel('簇数量 (K)')
plt.ylabel('WCSS (簇内平方和)')
plt.xticks(k_range)
plt.grid(True)

# 绘制轮廓系数图
plt.subplot(2, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o', linestyle='-')
plt.title('轮廓系数 - 确定最佳K值')
plt.xlabel('簇数量 (K)')
plt.ylabel('平均轮廓系数')
plt.xticks(k_range)
plt.grid(True)

# 绘制戴维斯-布尔丁指数图
plt.subplot(2, 2, 3)
plt.plot(k_range, dbi_scores, marker='o', linestyle='-')
plt.title('戴维斯-布尔丁指数 - 确定最佳K值')
plt.xlabel('簇数量 (K)')
plt.ylabel('戴维斯-布尔丁指数(越低越好)')
plt.xticks(k_range)
plt.grid(True)

plt.tight_layout()
plt.show()

# 打印最佳K值的指标
best_k_silhouette = k_range[np.argmax(silhouette_scores)]
best_k_dbi = k_range[np.argmin(dbi_scores)]

print(f"根据轮廓系数，最佳K值为 {best_k_silhouette}")
print(f"根据戴维斯-布尔丁指数，最佳K值为 {best_k_dbi}")

# 选择最佳K值(基于上述指标)
optimal_k = best_k_silhouette
print(f"选择的最佳K值为: {optimal_k}")
#
#
#
#
#
#
#
#| label: kmeans-clustering

# 使用最佳K值创建并训练K-Means模型
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 将聚类标签与股票数据关联
stock_clusters_kmeans = stock_data.copy()
stock_clusters_kmeans['聚类标签'] = kmeans_labels

# 查看聚类结果
print("K-Means聚类结果:")
print(stock_clusters_kmeans.head(10))
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
#| label: dbscan-param-selection
#| fig-width: 10
#| fig-height: 6

# 使用K-距离图选择eps参数
min_pts = 4  # 通常，MinPts ≥ 维度 + 1，这里的维度很高，但我们使用较小值便于示例
nbrs = NearestNeighbors(n_neighbors=min_pts).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)

# 获取第k个最近邻的距离
k_distances = np.sort(distances[:, min_pts-1], axis=0)

# 绘制K-距离图
plt.figure(figsize=(10, 6))
plt.plot(range(len(k_distances)), k_distances)
plt.title(f'K-距离图 (k={min_pts})')
plt.xlabel('数据点 (按距离排序)')
plt.ylabel(f'到第{min_pts}个最近邻的距离')
plt.grid(True)
plt.show()

# 根据K-距离图，选择一个合适的eps值
# 通常在曲线拐点处选择
eps = 1.5  # 这里是基于上面的K-距离图选择的示例值
print(f"选择的DBSCAN参数: eps={eps}, min_samples={min_pts}")
#
#
#
#
#
#| label: dbscan-clustering

# 应用DBSCAN算法
dbscan = DBSCAN(eps=eps, min_samples=min_pts)
dbscan_labels = dbscan.fit_predict(X_scaled)

# 计算找到的簇数量(不包括噪声点，标记为-1)
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"DBSCAN识别出的簇数量: {n_clusters}")
print(f"识别出的噪声点数量: {n_noise}")

# 将聚类标签与股票数据关联
stock_clusters_dbscan = stock_data.copy()
stock_clusters_dbscan['聚类标签'] = dbscan_labels

# 查看聚类结果
print("\nDBSCAN聚类结果 (噪声点标记为-1):")
print(stock_clusters_dbscan)
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
#| label: clustering-visualization
#| fig-width: 15
#| fig-height: 12

# 使用PCA进行降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 创建画布
plt.figure(figsize=(15, 12))

# 可视化K-Means聚类结果
plt.subplot(2, 1, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', 
                     s=80, alpha=0.8, edgecolors='w')

# 添加簇中心
centers = kmeans.cluster_centers_
centers_pca = pca.transform(centers)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, alpha=0.8)

# 添加股票代码标签
for i, stock in enumerate(stock_list):
    plt.annotate(stock, (X_pca[i, 0], X_pca[i, 1]), fontsize=8)

plt.title('K-Means聚类结果可视化 (PCA降维)')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.colorbar(scatter, label='簇标签')
plt.grid(True)

# 可视化DBSCAN聚类结果
plt.subplot(2, 1, 2)
# 用不同颜色表示不同簇
unique_labels = set(dbscan_labels)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:  # 黑色表示噪声点
        col = [0, 0, 0, 1]
    
    mask = (dbscan_labels == k)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[col], s=80, 
               alpha=0.8, edgecolors='w', label=f'簇 {k}' if k != -1 else '噪声')
    
    # 为每个簇添加标签
    for i, stock in enumerate(stock_list):
        if dbscan_labels[i] == k:
            plt.annotate(stock, (X_pca[i, 0], X_pca[i, 1]), fontsize=8)

plt.title('DBSCAN聚类结果可视化 (PCA降维)')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.legend(loc='best')
plt.grid(True)

plt.tight_layout()
plt.show()

# 展示PCA解释的方差比例
explained_variance = pca.explained_variance_ratio_
print(f"PCA前两个主成分解释的方差比例: {explained_variance}")
print(f"总解释方差: {sum(explained_variance):.4f}")
#
#
#
#
#
#
#
#| label: industry-analysis
#| fig-width: 14
#| fig-height: 10

# 1. K-Means聚类的行业分布
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
plt.title('K-Means各簇的行业分布')

for cluster in range(optimal_k):
    cluster_stocks = stock_clusters_kmeans[stock_clusters_kmeans['聚类标签'] == cluster]
    industry_counts = cluster_stocks['行业'].value_counts()
    
    plt.subplot(2, optimal_k, cluster+1)
    industry_counts.plot(kind='bar')
    plt.title(f'簇 {cluster} (共{len(cluster_stocks)}只股票)')
    plt.xlabel('行业')
    plt.ylabel('股票数量')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 2. DBSCAN聚类的行业分布
# 只分析非噪声簇
non_noise_clusters = [label for label in set(dbscan_labels) if label != -1]
n_valid_clusters = len(non_noise_clusters)

if n_valid_clusters > 0:
    plt.figure(figsize=(14, 5*n_valid_clusters))
    plt.suptitle('DBSCAN各簇的行业分布')
    
    for i, cluster in enumerate(non_noise_clusters):
        cluster_stocks = stock_clusters_dbscan[stock_clusters_dbscan['聚类标签'] == cluster]
        industry_counts = cluster_stocks['行业'].value_counts()
        
        plt.subplot(n_valid_clusters, 1, i+1)
        industry_counts.plot(kind='bar')
        plt.title(f'簇 {cluster} (共{len(cluster_stocks)}只股票)')
        plt.xlabel('行业')
        plt.ylabel('股票数量')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
else:
    print("DBSCAN未找到有效簇，所有点都被视为噪声")

# 噪声点的行业分布
noise_stocks = stock_clusters_dbscan[stock_clusters_dbscan['聚类标签'] == -1]
if not noise_stocks.empty:
    plt.figure(figsize=(10, 5))
    noise_industry_counts = noise_stocks['行业'].value_counts()
    noise_industry_counts.plot(kind='bar')
    plt.title(f'DBSCAN噪声点的行业分布 (共{len(noise_stocks)}只股票)')
    plt.xlabel('行业')
    plt.ylabel('股票数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
#
#
#
#
#
#
#
#| label: algorithm-comparison

# 为两种算法计算评估指标
evaluation_metrics = pd.DataFrame(columns=['轮廓系数', '戴维斯-布尔丁指数', '簇数量', '处理噪声能力'])

# 计算K-Means指标
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels) if len(set(kmeans_labels)) > 1 else np.nan
kmeans_dbi = davies_bouldin_score(X_scaled, kmeans_labels) if len(set(kmeans_labels)) > 1 else np.nan

# 计算DBSCAN指标(排除噪声点)
if n_noise < len(X_scaled) and len(set(dbscan_labels) - {-1}) > 1:
    # 获取非噪声点的索引
    non_noise_mask = dbscan_labels != -1
    dbscan_silhouette = silhouette_score(X_scaled[non_noise_mask], 
                                        dbscan_labels[non_noise_mask]) if np.sum(non_noise_mask) > 1 else np.nan
    dbscan_dbi = davies_bouldin_score(X_scaled[non_noise_mask], 
                                     dbscan_labels[non_noise_mask]) if np.sum(non_noise_mask) > 1 else np.nan
else:
    dbscan_silhouette = np.nan
    dbscan_dbi = np.nan

# 填充评估指标表
evaluation_metrics.loc['K-Means'] = [kmeans_silhouette, kmeans_dbi, optimal_k, '不处理噪声']
evaluation_metrics.loc['DBSCAN'] = [dbscan_silhouette, dbscan_dbi, n_clusters, f'识别{n_noise}个噪声点']

print("聚类算法性能比较:")
print(evaluation_metrics)

# 算法比较总结
print("\n算法优缺点对比:")
print("K-Means:")
print("  优点: 简单高效，簇内紧凑")
print("  缺点: 需要预先指定K值，只能识别凸形簇，对异常值敏感")
print("\nDBSCAN:")
print("  优点: 可发现任意形状的簇，能自动识别噪声点，不需要预先指定簇数量")
print("  缺点: 参数选择难度大，对高维稀疏数据效果较差")
#
#
#
#
#
#
#
#
#
#| label: time-period-analysis
#| fig-width: 12
#| fig-height: 8

# 按季度划分时间段
def split_by_quarter(returns_matrix):
    """将收益率数据按季度划分"""
    # 确保索引是日期类型
    returns_matrix.index = pd.to_datetime(returns_matrix.index)
    
    # 获取数据的起止时间
    start_date = returns_matrix.index.min()
    end_date = returns_matrix.index.max()
    
    # 构建季度划分
    quarters = pd.date_range(start=start_date, end=end_date, freq='Q')
    quarters = [q.strftime('%Y-Q%q') for q in quarters]
    
    # 划分数据
    quarter_data = {}
    current = start_date
    
    for i, quarter_end in enumerate(pd.date_range(start=start_date, end=end_date, freq='Q')):
        quarter_label = quarter_end.strftime('%Y-Q%q')
        quarter_data[quarter_label] = returns_matrix.loc[current:quarter_end]
        current = quarter_end + pd.Timedelta(days=1)
    
    return quarter_data

# 按季度划分数据
quarters_data = split_by_quarter(returns_matrix)

print(f"数据已划分为 {len(quarters_data)} 个季度")
for quarter, data in quarters_data.items():
    print(f"{quarter}: {data.shape[0]} 个交易日")

# 计算各簇在各季度的表现
def calculate_cluster_performance(quarters_data, stock_clusters, n_clusters):
    """计算各簇在各时间段的表现"""
    results = {}
    
    for quarter, quarter_data in quarters_data.items():
        # 初始化该季度结果
        quarter_results = pd.DataFrame(
            index=range(n_clusters),
            columns=['平均收益率', '波动性', '夏普比率']
        )
        
        for cluster in range(n_clusters):
            # 获取该簇的股票
            cluster_stocks = stock_clusters[stock_clusters['聚类标签'] == cluster]['股票代码'].tolist()
            
            if cluster_stocks:  # 确保簇非空
                # 计算该簇在该季度的表现
                cluster_returns = quarter_data[cluster_stocks]
                
                # 平均日收益率
                avg_return = cluster_returns.mean(axis=1).mean()
                
                # 波动性(收益率标准差)
                volatility = cluster_returns.mean(axis=1).std()
                
                # 夏普比率(简化版，假设无风险收益率为0)
                sharpe = avg_return / volatility if volatility != 0 else 0
                
                quarter_results.loc[cluster] = [avg_return, volatility, sharpe]
            else:
                quarter_results.loc[cluster] = [np.nan, np.nan, np.nan]
        
        results[quarter] = quarter_results
    
    return results

# 计算K-Means聚类的各簇表现
kmeans_performance = calculate_cluster_performance(quarters_data, stock_clusters_kmeans, optimal_k)

# 可视化K-Means聚类的各簇季度表现
plt.figure(figsize=(12, 8))

# 提取各簇在各季度的平均收益率
quarters = list(kmeans_performance.keys())
clusters = [f'簇{i}' for i in range(optimal_k)]

# 创建收益率数据框
returns_data = pd.DataFrame(index=clusters)
for quarter in quarters:
    returns_data[quarter] = kmeans_performance[quarter]['平均收益率'].values

# 绘制季度收益率对比图
returns_data.T.plot(kind='bar', figsize=(12, 6))
plt.title('各簇在不同季度的平均日收益率')
plt.xlabel('季度')
plt.ylabel('平均日收益率')
plt.legend(title='簇')
plt.grid(True, axis='y')
plt.show()

# 热力图：更直观地展示板块轮动
plt.figure(figsize=(10, 6))
sns.heatmap(returns_data.T, annot=True, cmap='RdYlGn', center=0, fmt='.4f')
plt.title('板块轮动热力图 (季度平均日收益率)')
plt.tight_layout()
plt.show()
#
#
#
#
#
#
#
#| label: cluster-characteristics

# 分析每个簇的特征
def analyze_cluster_features(stock_clusters, returns_matrix, industry_map, performance_results):
    """深入分析每个簇的特征"""
    n_clusters = len(set(stock_clusters['聚类标签']))
    features = {}
    
    for cluster in range(n_clusters):
        # 获取该簇的股票
        cluster_stocks = stock_clusters[stock_clusters['聚类标签'] == cluster]
        stock_codes = cluster_stocks['股票代码'].tolist()
        
        if not stock_codes:  # 跳过空簇
            continue
            
        # 1. 行业分布
        industry_counts = cluster_stocks['行业'].value_counts()
        top_industries = industry_counts.head(2).index.tolist()
        
        # 2. 收益率特征
        cluster_returns = returns_matrix[stock_codes]
        avg_return = cluster_returns.mean(axis=1).mean() * 252  # 年化
        volatility = cluster_returns.mean(axis=1).std() * np.sqrt(252)  # 年化
        sharpe = avg_return / volatility if volatility != 0 else 0
        
        # 3. 季度表现
        # 获取最近两个季度
        quarters = list(performance_results.keys())
        if len(quarters) >= 2:
            latest_quarter, previous_quarter = quarters[-1], quarters[-2]
            latest_return = performance_results[latest_quarter].loc[cluster, '平均收益率']
            previous_return = performance_results[previous_quarter].loc[cluster, '平均收益率']
            momentum = (latest_return - previous_return) > 0
        else:
            momentum = None
        
        # 整合特征
        features[cluster] = {
            '股票数量': len(stock_codes),
            '主要行业': top_industries,
            '年化收益率': avg_return,
            '年化波动率': volatility,
            '夏普比率': sharpe,
            '近期动量': '上升' if momentum else '下降' if momentum is not None else '未知',
            '代表股票': stock_codes[:5]  # 列出前5只股票
        }
    
    return features

# 分析K-Means聚类的簇特征
kmeans_features = analyze_cluster_features(
    stock_clusters_kmeans, returns_matrix, industry_map, kmeans_performance
)

# 打印各簇的特征概况
print("K-Means聚类各簇特征概况:")
for cluster, features in kmeans_features.items():
    print(f"\n簇 {cluster}:")
    for key, value in features.items():
        print(f"  {key}: {value}")
#
#
#
#
#
#
#
#| label: investment-strategy

# 生成投资策略建议
def generate_investment_strategy(cluster_features, performance_results):
    """根据聚类分析结果生成投资策略建议"""
    # 获取最近两个季度
    quarters = list(performance_results.keys())
    if len(quarters) < 2:
        return "季度数据不足，无法生成策略建议"
    
    latest_quarter, previous_quarter = quarters[-1], quarters[-2]
    
    # 分析各簇的动量和反转特征
    momentum_clusters = []
    reversal_clusters = []
    
    for cluster, features in cluster_features.items():
        # 计算收益率变化
        latest_return = performance_results[latest_quarter].loc[cluster, '平均收益率']
        previous_return = performance_results[previous_quarter].loc[cluster, '平均收益率']
        return_change = latest_return - previous_return
        
        # 判断趋势
        if return_change > 0:
            momentum_clusters.append((cluster, return_change, features))
        else:
            reversal_clusters.append((cluster, return_change, features))
    
    # 排序
    momentum_clusters.sort(key=lambda x: x[1], reverse=True)
    reversal_clusters.sort(key=lambda x: x[1])
    
    # 生成策略建议
    strategies = {}
    
    # 1. 动量策略
    if momentum_clusters:
        top_momentum = momentum_clusters[0]
        strategies["动量策略"] = {
            "描述": "投资近期表现持续改善的板块",
            "推荐簇": top_momentum[0],
            "表现改善": f"{top_momentum[1]:.4f}",
            "主要行业": top_momentum[2]['主要行业'],
            "代表股票": top_momentum[2]['代表股票']
        }
    
    # 2. 反转策略
    if reversal_clusters:
        top_reversal = reversal_clusters[0]
        strategies["反转策略"] = {
            "描述": "投资近期表现转弱但可能反弹的板块",
            "推荐簇": top_reversal[0],
            "表现下降": f"{top_reversal[1]:.4f}",
            "主要行业": top_reversal[2]['主要行业'],
            "代表股票": top_reversal[2]['代表股票']
        }
    
    # 3. 高夏普比率策略
    high_sharpe_cluster = max(cluster_features.items(), key=lambda x: x[1]['夏普比率'])
    strategies["高夏普比率策略"] = {
        "描述": "投资风险调整收益最高的板块",
        "推荐簇": high_sharpe_cluster[0],
        "夏普比率": f"{high_sharpe_cluster[1]['夏普比率']:.4f}",
        "主要行业": high_sharpe_cluster[1]['主要行业'],
        "代表股票": high_sharpe_cluster[1]['代表股票']
    }
    
    return strategies

# 生成K-Means聚类的投资策略建议
kmeans_strategies = generate_investment_strategy(kmeans_features, kmeans_performance)

# 打印投资策略建议
print("基于K-Means聚类的投资策略建议:")
for strategy_name, details in kmeans_strategies.items():
    print(f"\n{strategy_name}:")
    for key, value in details.items():
        print(f"  {key}: {value}")
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
