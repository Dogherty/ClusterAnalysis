import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# Generating synthetic data with make_blobs
# Генерація синтетичних даних за допомогою make_blobs
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

# Estimating optimal bandwidth for MeanShift algorithm
# Оцінка оптимальної ширини вікна для алгоритму MeanShift
bandwidth = estimate_bandwidth(X)

# Clustering data using MeanShift algorithm
# Кластеризація даних за допомогою алгоритму MeanShift
ms = MeanShift(bandwidth=bandwidth)
ms.fit(X)

# Getting cluster labels and cluster centers
# Отримання міток кластерів та центрів кластерів
labels = ms.labels_
clustercenters = ms.cluster_centers_
nclusters = len(clustercenters)

# Visualizing initial data points and cluster centers found by MeanShift algorithm
# Візуалізація початкових точок даних та центрів кластерів, знайдених алгоритмом MeanShift
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1])
plt.title('Starting points')
plt.subplot(122)
plt.scatter(clustercenters[:, 0], clustercenters[:, 1], color='r')
plt.title('Cluster centers')

# Calculating silhouette scores for different numbers of clusters using KMeans algorithm
# Обчислення оцінок силуету для різної кількості кластерів за допомогою алгоритму KMeans
scores = []
for n in range(2, 16):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
    scores.append(score)

# Visualizing silhouette scores depending on the number of clusters
# Візуалізація оцінок силуету в залежності від кількості кластерів
plt.figure(figsize=(6, 6))
plt.bar(range(2, 16), scores)
plt.xlabel('Number of clusters')
plt.ylabel('Score')
plt.title('Bar chart score(number of clusters)')

# Determining the optimal number of clusters as the one with the maximum silhouette score
# Визначення оптимальної кількості кластерів як тієї, що має максимальну оцінку силуету
optimal_n_clusters = scores.index(max(scores)) + 2

# Performing clustering again using the optimal number of clusters
# Повторне кластеризування з використанням оптимальної кількості кластерів
kmeans = KMeans(n_clusters=optimal_n_clusters)
kmeans.fit(X)

# Visualizing clustered data and clustering areas using KMeans
# Візуалізація кластеризованих даних та областей кластеризації за допомогою алгоритму KMeans
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='r')
plt.title('Clustered data with clustering areas')

plt.show()
