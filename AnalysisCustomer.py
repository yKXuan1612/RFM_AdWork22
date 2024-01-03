#%% import các thư viện cần thiết
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sqlalchemy import create_engine
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer

warnings.filterwarnings("ignore")

#%%
sv = 'MSI-XUANKY\\HTTT1'
db = 'K21411_GROUP4'
username = 'xuanky'
pwd = '1'

# tạo chuỗi kết nối
conn_str = f'mssql+pyodbc://{username}:{pwd}@{sv}/{db}?driver=SQL+Server'
engine = create_engine(conn_str)
querry = 'SELECT * FROM [GROUP4].[FactSales]'
df = pd.read_sql_query(querry, engine)

# conn = pyodbc.connect(conn_str)

# cursor = conn.cursor()
#%%
df.describe()
df.info()
#%%
unique_count = df['CustomerID'].nunique()

#%%
plt.boxplot(df['OrderQty'],labels=['Order Quantity'], showfliers=True)
plt.show()

#%%
quantities_outliers = df[df['OrderQty']>20].copy()
print(quantities_outliers)

#%%
count_less_than_20 = len(df[df['OrderQty'] < 20])
#%%
df['InvoiceDate'] = pd.to_datetime(df['OrderDate'], format='%m/%d/%Y %H:%M')
most_recentpurchaseDate = max(df['InvoiceDate'])
most_recentpurchaseDate = most_recentpurchaseDate + pd.DateOffset(days=1)
df['Distance'] = most_recentpurchaseDate - df['InvoiceDate']

#%%
df.info()
#%%
fig = plt.figure(4, figsize=(10,8))
plt.boxplot(df['Distance'])
plt.show()

#%%
specific_cus = df[df['CustomerID'] == 29825].copy()

#%%
Monetary = df.groupby('CustomerID').LineTotal.sum().reset_index(name='Monetary')
Frequency = df.groupby('CustomerID')['SalesOrderNumber'].nunique().reset_index(name='Frequency')
Recency = df.groupby('CustomerID')['Distance'].min().reset_index(name='Recency')

#%% plot frequency
fig = plt.figure(4, figsize=(10,8))
plt.boxplot(Frequency['Frequency'])
plt.show()

#%% plot recency
fig = plt.figure(4, figsize=(10,8))
plt.boxplot(Recency['Recency'])
plt.show()

#%% plot monetary
fig = plt.figure(4, figsize=(10,8))
plt.boxplot(Monetary['Monetary'])
plt.show()
#%%
Monetary.sum()
#%%
Monetary.describe()
#%%
RFM = Monetary.merge(Frequency, how='inner', on='CustomerID')
RFM = RFM.merge(Recency, how='inner', on='CustomerID')
#%%
RFM.isnull().sum()
scaler = StandardScaler() # lưu ý phải định nghĩa thằng StandardScaler ra thì mới dùng được
# #%% xử lí outlier monetary
# q1 = RFM['Monetary'].quantile(0.25)
# q3 = RFM['Monetary'].quantile(0.75)
#
# IQR = q3 - q1
# RFM = RFM[(RFM['Monetary'] >=q1 - 1.5*IQR) & (RFM['Monetary'] <= q3 + 1.5*IQR)]
#
# #%% xử lí outlier Frequency
# q1 = RFM['Frequency'].quantile(0.25)
# q3 = RFM['Frequency'].quantile(0.75)
#
# IQR = q3 - q1
# RFM = RFM[(RFM['Frequency'] >=q1 - 1.5*IQR) & (RFM['Frequency'] <= q3 + 1.5*IQR)]
#
# #%% xử lí outlier Recency
# q1 = RFM['Recency'].quantile(0.25)
# q3 = RFM['Recency'].quantile(0.75)
#
# IQR = q3 - q1
# RFM = RFM[(RFM['Recency'] >=q1 - 1.5*IQR) & (RFM['Recency'] <= q3 + 1.5*IQR)]
#%%
RFM.info()
#%% Tách ra khách hàng cá nhân và KH doanh nghiệp
RFM_in = RFM[RFM['CustomerID'] < 20778]
RFM_resell = RFM[RFM['CustomerID'] >= 20778]
#%%
fig = plt.figure(4, figsize=(10,8))
plt.boxplot(RFM_resell['Monetary'])
plt.show()

# #%%
# fig = plt.figure(4, figsize=(10,8))
# plt.boxplot(RFM['Frequency'])
# plt.show()
#
# #%%
# fig = plt.figure(4, figsize=(10,8))
# plt.boxplot(RFM['Recency'].dt.days)
# plt.show()
#%%
RFM_resell_cluster = RFM_resell[['Monetary', 'Recency', 'Frequency']]
RFM_resell_cluster['Recency'] = RFM_resell_cluster['Recency'].dt.days
RFM_resell_scaled = scaler.fit_transform(RFM_resell_cluster) # scale 3 giá trị
#%%
RFM_in_cluster = RFM_in[['Monetary', 'Recency', 'Frequency']]
RFM_in_cluster['Recency'] = RFM_in_cluster['Recency'].dt.days
RFM_in_scaled = scaler.fit_transform(RFM_in_cluster) # scale 3 giá trị

#%% tìm K bằng elbow cho in
X = RFM_in_scaled

# Phạm vi giá trị K bạn muốn kiểm tra
k_values = range(1, 11)
sse = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Vẽ đồ thị Elbow
plt.plot(k_values, sse, marker='o')
plt.xlabel('Số cụm (K)')
plt.ylabel('SSE (Sum of Squared Errors)')
plt.title('Elbow Method để chọn K tối ưu')
plt.show()

# => nhận thấy K = 5 là tối ưu

#%% tìm K bằng silhoutte cho in

k_values = range(2, 11)
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Vẽ đồ thị Silhouette Score
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Số cụm (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score để chọn K tối ưu')
for i, sil_score in enumerate(silhouette_scores):
    plt.text(i+2, sil_score, f'{sil_score:.4f}', ha='center', va='bottom')
plt.show()

#%% tìm K bằng elbow cho resell
X_resll = RFM_resell_scaled

# Phạm vi giá trị K bạn muốn kiểm tra
k_values = range(1, 11)
sse = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_resll)
    sse.append(kmeans.inertia_)

# Vẽ đồ thị Elbow
plt.plot(k_values, sse, marker='o')
plt.xlabel('Số cụm (K)')
plt.ylabel('SSE (Sum of Squared Errors)')
plt.title('Elbow Method để chọn K tối ưu')
plt.show()

# => nhận thấy K = 5 là tối ưu

#%% tìm K bằng silhoutte cho resell

k_values = range(2, 11)
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_resll)
    silhouette_avg = silhouette_score(X_resll, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Vẽ đồ thị Silhouette Score
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Số cụm (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score để chọn K tối ưu')
for i, sil_score in enumerate(silhouette_scores):
    plt.text(i+2, sil_score, f'{sil_score:.4f}', ha='center', va='bottom')
plt.show()
# ==> 2 nhóm doanh nghiệp là tối ưu
#%%
linked = linkage(RFM_in_scaled, method='ward')  # Có thể thử các phương pháp khác như 'complete', 'single',
# 'average'

# Vẽ dendrogram để xác định số cụm phù hợp
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.axhline(y=60, color='r', linestyle='--')
plt.show()

#%%
# from scipy.cluster.hierarchy import cut_tree
# num_clusters = 5  # Đặt số cụm dựa trên đồ thị dendrogram
#
# # Chia cụm sử dụng cut_tree
# clusters_in = cut_tree(linked, n_clusters=num_clusters)
# #%%
# RFM_in['Cluster'] = clusters_in.flatten()
# #%%
# cluster_counts_in = RFM_in['Cluster'].value_counts()
# #%%
# silhouette_avg = silhouette_score(RFM_in_scaled, clusters_in.flatten())

#%%
# fig, ax = plt.subplots(3, 2, figsize=(15, 12))
#
# for idx, k in enumerate([2, 3, 4, 5, 6, 7]):
#     km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=42)
#
#     # Calculate subplot indices
#     q, mod = divmod(idx, 2)
#
#     visualize = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q, mod])
#     visualize.fit(RFM_cluster_scaled)
#
# plt.show()

#%% chia cho IN
kmeans_in = KMeans(n_clusters=5, random_state=42)
kmeans_in.fit(RFM_in_scaled)

#%%
RFM_in['Cluster'] = kmeans_in.labels_
# print(RFM.head(10))

#%%
cluster_counts_in = RFM_in['Cluster'].value_counts()
#%%
inertia_score = kmeans_in.inertia_
print(inertia_score)
#%%
sns.boxplot(x=RFM_in['Cluster'], y=RFM_in['Monetary'])
plt.show()
#%%
kmeans_resell = KMeans(n_clusters= 4,random_state= 43 )
kmeans_resell.fit(RFM_resell_scaled)

#%%
RFM_resell['Cluster'] = kmeans_resell.labels_
cluster_counts_resell = RFM_resell['Cluster'].value_counts()

#%%
inertia_values = []
possible_k_values = range(1, 11)  # Thử nghiệm từ 1 đến 10 cụm

for k in possible_k_values:
    kmeans = KMeans(n_clusters=k, random_state=43)
    kmeans.fit(RFM_resell_scaled)
    inertia_values.append(kmeans.inertia_)

# Vẽ biểu đồ elbow method
plt.plot(possible_k_values, inertia_values, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

#%%
sns.boxplot(x=RFM_resell['Cluster'], y=RFM_resell['Monetary'])
plt.show()

#%%
RFM_in['Recency Rank'] = RFM_in['Recency'].rank(ascending=True)
RFM_in['Frequency Rank'] = RFM_in['Frequency'].rank(ascending=False)
RFM_in['Monetary Rank'] = RFM_in['Monetary'].rank(ascending=False)

#%%
MinMaxScale = MinMaxScaler()
RFM_in_scaled_min = MinMaxScale.fit_transform(RFM_in_cluster)
#%%
X_in_min = RFM_in_scaled_min
k_values = range(2, 11)
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_in_min)
    silhouette_avg = silhouette_score(X_in_min, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Vẽ đồ thị Silhouette Score
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Số cụm (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score để chọn K tối ưu')
for i, sil_score in enumerate(silhouette_scores):
    plt.text(i+2, sil_score, f'{sil_score:.4f}', ha='center', va='bottom')
plt.show()
#%%
kmeans_in_min = KMeans(n_clusters=2, random_state=44)
kmeans_in_min.fit(RFM_in_scaled_min)

RFM_in['Cluster'] = kmeans_in_min.labels_
cluster_counts_in = RFM_in['Cluster'].value_counts()

#%%
sns.boxplot(x=RFM_in['Cluster'], y=RFM_in['Frequency'])
plt.show()

#%%
RFM_resell_scaled_min = MinMaxScale.fit_transform(RFM_resell_cluster)

#%%
X_resell_min = RFM_resell_scaled_min
k_values = range(2, 11)
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_resell_min)
    silhouette_avg = silhouette_score(X_resell_min, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Vẽ đồ thị Silhouette Score
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Số cụm (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score để chọn K tối ưu')
for i, sil_score in enumerate(silhouette_scores):
    plt.text(i+2, sil_score, f'{sil_score:.4f}', ha='center', va='bottom')
plt.show()
#%%
kmeans_resell_min = KMeans(n_clusters=3, random_state=47)
kmeans_resell_min.fit(RFM_resell_scaled_min)

RFM_resell['Cluster'] = kmeans_resell_min.labels_
count_clusters_resell = RFM_resell['Cluster'].value_counts()
#%%
sns.boxplot(x=RFM_resell['Cluster'], y=RFM_resell['Recency'].dt.days)
plt.show()
#%%
RFM_in_subset = RFM_in.iloc[:, 0:5]
#%%
RFM_in_subset.to_csv('RFM_Individuals.csv', index=False)
RFM_resell.to_csv('RFM_Reseller.csv', index=False)
#%%
# RFM.describe()
# #%%
# eps_values = np.arange(0.1, 2.1, 0.1)
# min_samples_values = np.arange(200, 301, 5)
# param_grid = {'eps': eps_values,
#               'min_samples': min_samples_values}
#
# best_silhouette_score = -1
# best_params = None
#
# # Lặp qua tất cả các tham số trong lưới
# for params in ParameterGrid(param_grid):
#     dbscan = DBSCAN(**params)
#     labels = dbscan.fit_predict(RFM_cluster_scaled)
#
#     # Đánh giá chất lượng phân cụm bằng silhouette score
#     silhouette = silhouette_score(RFM_cluster_scaled, labels)
#
#     # Kiểm tra và cập nhật giá trị tối ưu
#     if silhouette > best_silhouette_score:
#         best_silhouette_score = silhouette
#         best_params = params
#
# print(f"Best Silhouette Score: {best_silhouette_score}")
# print(f"Best Parameters: {best_params}")
#
# #%% - bắt đầu cluster từ đây
# dbscan = DBSCAN(eps=0.687, min_samples=3900)
#
# # Áp dụng DBSCAN lên dữ liệu và lấy nhãn của từng điểm dữ liệu
# labels_dbscan = dbscan.fit_predict(RFM_cluster_scaled)
#
# # Kiểm tra số lượng nhóm
# n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
# print(f"Số lượng nhóm của DBSCAN: {n_clusters_dbscan}")
#
# # Đánh giá chất lượng phân cụm bằng silhouette score
# silhouette_dbscan = silhouette_score(RFM_cluster_scaled, labels_dbscan)
# print(f"Silhouette Score của DBSCAN: {silhouette_dbscan}")
#
# #%%
# labels_dbscan = dbscan.fit_predict(RFM_cluster_scaled)
#
# # Gán nhãn cho dữ liệu
# RFM_labeled_DB = RFM_cluster.copy()
# RFM_labeled_DB['ClusterLabel'] = labels_dbscan
#
# #%%
# cluster_counts_BD = RFM_labeled_DB['ClusterLabel'].value_counts()
#
# #%%
# sns.boxplot(x=RFM_labeled_DB['ClusterLabel'], y=RFM_labeled_DB['Monetary'])
# plt.show()
#
# #%%
# sns.boxplot(x=RFM_labeled_DB['ClusterLabel'], y=RFM_labeled_DB['Recency'])
# plt.show()
#
# #%%
# sns.boxplot(x=RFM_labeled_DB['ClusterLabel'], y=RFM_labeled_DB['Frequency'])
# plt.show()
# #%%
# RFM_DB_0 = RFM_labeled_DB[RFM_labeled_DB['ClusterLabel'] ==0] # lấy những thằng có label = 0 ra
# RFM_DB_0_cluster = RFM_DB_0.drop(['ClusterLabel'], axis=1) # drop cột
# RFM_DB_0_cluster_scaled = scaler.fit_transform(RFM_DB_0_cluster)
#
# #%% làm lại lần nữa với cluster 0
# dbscan = DBSCAN(eps=0.687, min_samples=1700)
#
# # Áp dụng DBSCAN lên dữ liệu và lấy nhãn của từng điểm dữ liệu
# labels_dbscan = dbscan.fit_predict(RFM_DB_0_cluster_scaled)
#
# # Kiểm tra số lượng nhóm
# n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
# print(f"Số lượng nhóm của DBSCAN: {n_clusters_dbscan}")
#
# # Đánh giá chất lượng phân cụm bằng silhouette score
# silhouette_dbscan = silhouette_score(RFM_DB_0_cluster, labels_dbscan)
# print(f"Silhouette Score của DBSCAN: {silhouette_dbscan}")
#
# #%%
# labels_dbscan = dbscan.fit_predict(RFM_DB_0_cluster_scaled)
#
# # Gán nhãn cho dữ liệu - và đếm lần 2
# RFM_DB_0_cluster['ClusterLabel'] = labels_dbscan
# RFM_DB_0_cluster['ClusterLabel'] = RFM_DB_0_cluster['ClusterLabel'].apply(lambda x: 3 if x ==0 else 2)
# cluster_counts_BD_0 = RFM_DB_0_cluster['ClusterLabel'].value_counts() # đếm lần 2
#
# #%% phân cụm lần 3
# RFM_DB_0_0_cluster = RFM_DB_0_cluster[RFM_DB_0_cluster['ClusterLabel']==3] # tách tiếp
# RFM_DB_0_0_scaled = scaler.fit_transform(RFM_DB_0_0_cluster.drop('ClusterLabel', axis=1))
#
# #%% chia nhỏ lần 3
# dbscan = DBSCAN(eps=0.9, min_samples=3000)
#
# # Áp dụng DBSCAN lên dữ liệu và lấy nhãn của từng điểm dữ liệu
# labels_dbscan = dbscan.fit_predict(RFM_DB_0_0_scaled)
#
# # Kiểm tra số lượng nhóm
# n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
# print(f"Số lượng nhóm của DBSCAN: {n_clusters_dbscan}")
#
# # Đánh giá chất lượng phân cụm bằng silhouette score
# silhouette_dbscan = silhouette_score(RFM_DB_0_0_scaled, labels_dbscan)
# print(f"Silhouette Score của DBSCAN: {silhouette_dbscan}")
#
# #%%
# labels_dbscan = dbscan.fit_predict(RFM_DB_0_0_scaled)
#
# # Gán nhãn cho dữ liệu
# RFM_DB_0_0_cluster['ClusterLabel'] = labels_dbscan
# RFM_DB_0_0_cluster['ClusterLabel'] = RFM_DB_0_0_cluster['ClusterLabel'].apply(lambda x:3 if x ==
#                                                                                             -1
# else 4)
# cluster_counts_BD_0_0 = RFM_DB_0_0_cluster['ClusterLabel'].value_counts()
#
#
#
# #%%
#
#
#
# #
# # #%%
# # silhouette_scores_hierarchical = []
# #
# # # Thử nghiệm từ K=2 đến K=20
# # for k in range(2, 10):
# #     kmeans = KMeans(n_clusters=k, random_state=0)
# #     labels_kmeans = kmeans.fit_predict(RFM_DB_0_0_scaled)
# #     silhouette_avg = silhouette_score(RFM_DB_0_0_cluster, labels_kmeans)
# #     silhouette_scores_hierarchical.append(silhouette_avg)
# #
# # # Vẽ biểu đồ đường
# # plt.figure(figsize=(10, 6))
# # plt.plot(range(2, 10), silhouette_scores_hierarchical, marker='o', linestyle='-', color='b')
# # plt.title('Silhouette Score for different values of K (Kmeans Clustering)')
# # plt.xlabel('Number of Clusters (K)')
# # plt.ylabel('Silhouette Score')
# # plt.grid(True)
# # plt.show()
# #
# # #%%
# # kmeans_0_0 = KMeans(n_clusters=6, random_state=0)
# # kmeans_0_0.fit_predict(RFM_DB_0_0_cluster)
# # RFM_DB_0_0_cluster['Cluster'] = kmeans_0_0
# # custer_counts_0_0 = RFM_DB_0_0_cluster['Cluster'].value_counts()
# #
# # #%%
# # RFM_scale_withID = RFM.copy()
# #
# # # Chuyển đổi cột Recency thành số ngày nếu cần
# # RFM_scale_withID['Recency'] = RFM_scale_withID['Recency'].dt.days
# #
# # # Lựa chọn các cột RFM cần scale
# # columns_to_scale = ['Monetary', 'Frequency', 'Recency']
# # data_to_scale = RFM_scale_withID[columns_to_scale]
# #
# # scaled_data = scaler.fit_transform(data_to_scale)
# #
# # RFM_scale_withID[columns_to_scale] = scaled_data
# # #%%
# # RFM_scale_withID['CustomerID'] = RFM_scale_withID['CustomerID'].astype(int)
# #
# # # Kiểm tra và giới hạn phạm vi giá trị của CustomerID
# # min_customer_id = RFM_scale_withID['CustomerID'].min()
# # max_customer_id = RFM_scale_withID['CustomerID'].max()
# #
# # #%%
# # min_customer_id = RFM_scale_withID['CustomerID'].min()
# # max_customer_id = RFM_scale_withID['CustomerID'].max()
# # valid_range = range(min_customer_id, max_customer_id+1)
# # RFM_filtered = RFM_scale_withID[RFM_scale_withID['CustomerID'].isin(valid_range)]
# # linked_filtered = linkage(RFM_filtered.drop('CustomerID', axis=1), 'ward')
# #
# # # Vẽ biểu đồ dendrogram
# # plt.figure(figsize=(12, 8))
# # dendrogram(linked_filtered,
# #            orientation='top',
# #            labels=RFM_filtered['CustomerID'],
# #            distance_sort='descending',
# #            show_leaf_counts=True)
# # plt.title('Hierarchical Clustering Dendrogram')
# # plt.xlabel('Samples')
# # plt.ylabel('Distance')
# # plt.show()
# #
# # #%%
# # from scipy.spatial.distance import pdist, squareform
# #
# # # Tính ma trận khoảng cách
# # distances = pdist(RFM_filtered.drop('CustomerID', axis=1), metric='euclidean')
# #
# # # Chuyển đổi ma trận khoảng cách thành ma trận vuông
# # distance_matrix = squareform(distances)
# #
# # # Vẽ biểu đồ dendrogram
# # plt.figure(figsize=(12, 8))
# # dendrogram(linkage(distance_matrix, method='ward'),
# #            orientation='top',
# #            labels=RFM_filtered['CustomerID'],
# #            distance_sort='descending',
# #            show_leaf_counts=True)
# # plt.title('Hierarchical Clustering Dendrogram')
# # plt.xlabel('Samples')
# # plt.ylabel('Distance')
# # plt.show()
# #
# # #%%
# # data_for_heatmap = RFM_filtered[['Recency', 'Frequency', 'Monetary']]
# #
# # # Tính ma trận tương quan
# # correlation_matrix = data_for_heatmap.corr()
# #
# # # Vẽ biểu đồ heatmap
# # plt.figure(figsize=(10, 8))
# # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# # plt.title('Heatmap - Tương quan giữa R, F, và M')
# # plt.show()
# #
# # #%%
# # RFM['CustomerID'].describe()
# # #%%
# # # RFM_copy.info()
# # #%%
# # RFM_copy = RFM.copy()
# # RFM_copy['Recency'] = RFM_copy['Recency'].dt.days
# # #%%
# # RFM_copy_cluster = RFM_copy[['Monetary','Frequency', 'Recency']]
# # linked = linkage(RFM_copy_cluster.reset_index(drop=True), method='ward')
# #
# # plt.figure(figsize=(12, 8))
# # dendrogram(linked,
# #            orientation='top',
# #            distance_sort='descending',
# #            show_leaf_counts=True)
# # plt.title('Hierarchical Clustering Dendrogram')
# # plt.xlabel('Samples')
# # plt.ylabel('Distance')
# # plt.show()
# #
# # #%%
# # silhouette_scores_hierarchical = []
# #
# # # Thử nghiệm từ K=2 đến K=20
# # for k in range(2, 16):
# #     hierarchical = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
# #     labels_hierarchical = hierarchical.fit_predict(RFM_cluster_scaled)
# #     silhouette_avg = silhouette_score(RFM_cluster_scaled, labels_hierarchical)
# #     silhouette_scores_hierarchical.append(silhouette_avg)
# #
# # # Vẽ biểu đồ đường
# # plt.figure(figsize=(10, 6))
# # plt.plot(range(2, 16), silhouette_scores_hierarchical, marker='o', linestyle='-', color='b')
# # plt.title('Silhouette Score for different values of K (Hierarchical Clustering)')
# # plt.xlabel('Number of Clusters (K)')
# # plt.ylabel('Silhouette Score')
# # plt.grid(True)
# # plt.show()
# #
# # #%%
# # cluster = AgglomerativeClustering(n_clusters=2,affinity='euclidean', linkage='ward')
# # cluster.fit_predict(RFM_cluster_scaled)
# #
# # #%%
# # cl = cluster.fit_predict(RFM_cluster_scaled)
# # sc = silhouette_score(RFM_cluster_scaled,cl)
# # print(sc)
# #
# # #%%
# # RFM_labeled_HAC = RFM_cluster.copy()
# # RFM_labeled_HAC['ClusterLabel'] = cl
# #
# # #%%
# # cluster_counts_HAC = RFM_labeled_HAC['ClusterLabel'].value_counts()
# #
# # #%%
# # sns.boxplot(x=RFM_labeled_HAC['ClusterLabel'], y=RFM_labeled_HAC['Monetary'])
# # plt.show()
# #
# # #%%
# # sns.boxplot(x=RFM_labeled_HAC['ClusterLabel'],y=RFM_labeled_HAC['Recency'])
# # plt.show()
# # #%%
# # sns.boxplot(x=RFM_labeled_HAC['ClusterLabel'],y=RFM_labeled_HAC['Frequency'])
# # plt.show()
# #
# # #%%
# # RFM_labeled_HAC_0 = RFM_labeled_HAC[RFM_labeled_HAC['ClusterLabel'] == 0]
# # RFM_scale_0 = RFM_labeled_HAC_0.drop(['ClusterLabel'], axis=1)
# # RFM_scale_0 = scaler.fit_transform(RFM_scale_0)
# # #%%
# # silhouette_scores_hierarchical = []
# #
# # # Thử nghiệm từ K=2 đến K=20
# # for k in range(2, 5):
# #     hierarchical = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
# #     labels_hierarchical = hierarchical.fit_predict(RFM_scale_0)
# #     silhouette_avg = silhouette_score(RFM_scale_0, labels_hierarchical)
# #     silhouette_scores_hierarchical.append(silhouette_avg)
# #
# # # Vẽ biểu đồ đường
# # plt.figure(figsize=(10, 6))
# # plt.plot(range(2, 16), silhouette_scores_hierarchical, marker='o', linestyle='-', color='b')
# # plt.title('Silhouette Score for different values of K (Hierarchical Clustering)')
# # plt.xlabel('Number of Clusters (K)')
# # plt.ylabel('Silhouette Score')
# # plt.grid(True)
# # plt.show()
# #
# # #%%
# # cluster_0 = AgglomerativeClustering(n_clusters=7,affinity='euclidean', linkage='ward')
# # cluster_0.fit_predict(RFM_scale_0)
# # RFM_labeled_HAC_0['ClusterLabel'] = cluster_0.labels_
# #
# # #%%
# # cluster_0_labels_count = RFM_labeled_HAC_0['ClusterLabel'].value_counts()
# # #%%
# # kmeans = KMeans(n_clusters=2, random_state=0)
# # labels_kmeans=kmeans.fit_predict(RFM_cluster_scaled)
# #
# #
# # #%%
# # fig = plt.figure(figsize=(10, 8))
# # ax = fig.add_subplot(111, projection='3d')
# #
# # # Lấy nhãn của từng mẫu
# # labels = cluster.labels_
# #
# # # Vẽ các điểm dữ liệu theo nhóm
# # for label in set(labels):
# #     group = RFM_cluster[labels == label]
# #     ax.scatter(group['Recency'], group['Frequency'], group['Monetary'], label=f'Cluster {label}')
# #
# # # Đặt tên cho trục và biểu đồ
# # ax.set_xlabel('Recency')
# # ax.set_ylabel('Frequency')
# # ax.set_zlabel('Monetary')
# # ax.set_title('3D Scatter Plot of Clusters')
# #
# # # Hiển thị chú thích nhãn
# # ax.legend()
# #
# # # Hiển thị biểu đồ
# # plt.show()
# #
# # #%%
# # RFM_labeled_Kmeans = RFM_cluster.copy()
# # RFM_labeled_Kmeans['Cluster'] = labels_kmeans
# # sc = silhouette_score(RFM_cluster_scaled,labels_kmeans)
# # print(sc)
# # #%%
# # cluster_counts_Kmeans  = RFM_labeled_Kmeans['Cluster'].value_counts()
# # #%%
# # # Tạo biểu đồ 3D
# # fig = plt.figure(figsize=(12, 10))
# # ax = fig.add_subplot(111, projection='3d')
# #
# # # Vẽ các điểm dữ liệu theo nhóm
# # for label in set(labels_kmeans):
# #     group = RFM_labeled_Kmeans[labels_kmeans == label]
# #     ax.scatter(group['Recency'], group['Frequency'], group['Monetary'], label=f'Cluster {label}')
# #
# # # Đặt tên cho trục và biểu đồ
# # ax.set_xlabel('Recency')
# # ax.set_ylabel('Frequency')
# # ax.set_zlabel('Monetary')
# # ax.set_title('3D Scatter Plot of K-Means Clusters')
# #
# # # Hiển thị chú thích nhãn
# # ax.legend()
# #
# # # Hiển thị biểu đồ
# # plt.show()
# #
# # #%%
# # model = KMeans()
# #
# # # Tạo visualizer để xem Elbow Method
# # visualizer = KElbowVisualizer(model, k=(2, 20), metric='silhouette', timings=False)
# #
# # # Fit dữ liệu vào visualizer
# # visualizer.fit(RFM_cluster_scaled)
# #
# # # Hiển thị biểu đồ
# # visualizer.show()
# # #%%
# # silhouette_scores = []
# #
# # # Thử nghiệm từ K=2 đến K=20
# # for k in range(2, 21):
# #     kmeans = KMeans(n_clusters=k, random_state=0)
# #     labels_kmeans = kmeans.fit_predict(RFM_cluster_scaled)
# #     silhouette_avg = silhouette_score(RFM_cluster_scaled, labels_kmeans)
# #     silhouette_scores.append(silhouette_avg)
# #
# # # Vẽ biểu đồ đường
# # plt.figure(figsize=(10, 6))
# # plt.plot(range(2, 21), silhouette_scores, marker='o', linestyle='-', color='b')
# # plt.title('Silhouette Score for different values of K')
# # plt.xlabel('Number of Clusters (K)')
# # plt.ylabel('Silhouette Score')
# # plt.grid(True)
# # plt.show()
#
# #%%