import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics


def hierarchical_clustering(data, variables, method='ward'):
    # Standardize the variables
    data_std = StandardScaler().fit_transform(data[variables])

    # Compute the linkage matrix
    linkage_matrix = linkage(data_std, method=method)

    # Plot the dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix)
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.savefig('/home/kaarlahti/PycharmProjects/kirovsk_230516/img/dendrogram.png', dpi=300)
    plt.show()

    # Plot the silhouette method and find optimal cluster number
    silhouette_scores = []
    for i in range(2, 11):  # starts from 2 because a minimum of 2 clusters is needed
        hc = AgglomerativeClustering(n_clusters=i, linkage=method, metric='euclidean')
        hc.fit(data_std)
        silhouette_scores.append(metrics.silhouette_score(data_std, hc.labels_))

    plt.figure(figsize=(12, 8))
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.title('Silhouette Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig('/home/kaarlahti/PycharmProjects/kirovsk_230516/img/silhouette.png', dpi=300)
    plt.show()

    # Find the optimal number of clusters
    optimal_clusters = silhouette_scores.index(
        max(silhouette_scores)) + 2  # +2 because index starts from 0 and we started range from 2

    print("Optimal number of clusters: ", optimal_clusters)

    # Fit the model again with the optimal number of clusters and attach labels to original data
    hc_optimal = AgglomerativeClustering(n_clusters=optimal_clusters, linkage=method, metric='euclidean')
    hc_optimal.fit(data_std)

    data['hcl_cluster'] = hc_optimal.labels_
    return data
