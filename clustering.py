import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, estimate_bandwidth
from sklearn import metrics
from sklearn.cluster import DBSCAN, MeanShift
import seaborn as sns
from sklearn.metrics import silhouette_score


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
        max(silhouette_scores)) + 2  # +2 because index starts from 0, and we started range from 2

    print("Optimal number of clusters: ", optimal_clusters)

    # Fit the model again with the optimal number of clusters and attach labels to original data
    hc_optimal = AgglomerativeClustering(n_clusters=optimal_clusters, linkage=method, metric='euclidean')
    hc_optimal.fit(data_std)

    data['hcl_cluster'] = hc_optimal.labels_
    return data


def kmeans_clustering(df, variables, max_clusters=10):
    """
    Perform KMeans clustering on 'df' using 'variables'.
    - df: pandas DataFrame, contains the data to cluster
    - variables: list of strings, column names of the variables to use in clustering
    - max_clusters: int, the maximum number of clusters to consider.

    Return df with an extra column 'Cluster_Labels' with cluster assignments.
    """

    # Standardize
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[variables] = scaler.fit_transform(df[variables])

    # Prepare models with different numbers of clusters
    models = [KMeans(n_clusters=k, n_init='auto',
                     random_state=42).fit(df_scaled[variables]) for k in range(1, max_clusters + 1)]

    # Compute silhouette scores
    silhouette_scores = [metrics.silhouette_score(df_scaled[variables], model.labels_) for model in models[1:]]

    # Plot Silhouette scores
    plt.figure(figsize=(12, 8))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    plt.savefig('/home/kaarlahti/PycharmProjects/kirovsk_230516/img/kmeans_silhouette_scores.png', dpi=300)
    plt.close()

    # Choose optimal number of clusters
    optimal_k = np.argmax(silhouette_scores) + 2  # Adding 2 because k starts from 2
    print(f"Optimal number of clusters: {optimal_k}")

    # Final KMeans clustering model with optimal number of clusters
    final_model = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42)
    df['kcl_cluster'] = final_model.fit_predict(df_scaled[variables])

    return df


def dbscan_clustering(data, variables, eps=0.5, min_samples=5):
    """
    Function to perform DBSCAN clustering.

    Parameters:
    data: DataFrame containing the data
    variables: list of variables to consider in the clustering
    eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    """

    # Standardize the data
    df = data.copy()
    df[variables] = StandardScaler().fit_transform(df[variables])

    # Fit the DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df[variables])

    # Add the labels to the data
    df['dbscan_cluster'] = labels

    # Create a pairplot
    sns.pairplot(df, vars=variables, hue="dbscan_cluster", plot_kws={'alpha': 0.5})
    plt.savefig("/home/kaarlahti/PycharmProjects/kirovsk_230516/img/dbscan_pairplot.png", dpi=300)

    # Return the data with the cluster labels
    return df


def meanshift_clustering(data, variables):
    """
    Function to perform Mean Shift clustering.

    Parameters:
    data: DataFrame containing the data
    variables: list of variables to consider in the clustering
    bandwidth: Bandwidth used in the RBF kernel. If not given, it is estimated using sklearn's estimate_bandwidth function.
    """

    # Standardize the data
    df = data.copy()
    df[variables] = StandardScaler().fit_transform(df[variables])

    # Fit the MeanShift model
    meanshift = MeanShift(bandwidth=estimate_bandwidth(df[variables]))
    labels = meanshift.fit_predict(df[variables])

    # Add the labels to the data
    df['ms_cluster'] = labels

    # Create a pairplot
    sns.pairplot(df, vars=variables, hue="ms_cluster", plot_kws={'alpha': 0.5})
    plt.savefig("/home/kaarlahti/PycharmProjects/kirovsk_230516/img/meanshift_pairplot.png", dpi=300)

    # Return the data with the cluster labels
    return df


def spectral_clustering_auto(data, variables, max_clusters):
    """
    Function to perform Spectral Clustering with automatic cluster number determination.

    Parameters:
    data: DataFrame containing the data
    variables: list of variables to consider in the clustering
    max_clusters: Maximum number of clusters to consider for silhouette scores
    """

    # Standardize the data
    df = data.copy()
    df[variables] = StandardScaler().fit_transform(df[variables])

    # Placeholder for maximum silhouette score and optimal number of clusters
    max_sil_score = -1
    optimal_clusters = 0

    # Compute silhouette scores for different numbers of clusters
    for n_clusters in range(2, max_clusters + 1):
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
        labels = spectral.fit_predict(df[variables])
        sil_score = silhouette_score(df[variables], labels)

        # If the silhouette score for the current clustering is higher than the current maximum,
        # update the maximum silhouette score and optimal number of clusters
        if sil_score > max_sil_score:
            max_sil_score = sil_score
            optimal_clusters = n_clusters

    # Fit the SpectralClustering model with optimal number of clusters
    spectral = SpectralClustering(n_clusters=optimal_clusters, affinity='nearest_neighbors')
    labels = spectral.fit_predict(df[variables])

    # Add the labels to the data
    df['scl_clusters'] = labels

    # Create a pairplot
    sns.pairplot(df, vars=variables, hue="scl_clusters", plot_kws={'alpha': 0.5})
    plt.savefig("/home/kaarlahti/PycharmProjects/kirovsk_230516/img/spectral_pairplot.png", dpi=300)

    print(f'Optimal number of clusters: {optimal_clusters}')

    # Return the data with the cluster labels
    return df


