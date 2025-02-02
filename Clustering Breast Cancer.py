import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # Import libraries for data manipulation, visualization, and machine learning
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Import the dataset and preprocessing, clustering, and evaluation tools
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    # Optional: Set plot style for better visuals
    sns.set(style="whitegrid")
    return (
        AgglomerativeClustering,
        KMeans,
        StandardScaler,
        TSNE,
        load_breast_cancer,
        mo,
        np,
        pd,
        plt,
        silhouette_score,
        sns,
    )


@app.cell
def _(load_breast_cancer, pd):
    # Load the Breast Cancer dataset
    data = load_breast_cancer()

    # Convert the dataset into a pandas DataFrame for easier exploration
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target  # The target labels (0: malignant, 1: benign)

    # Display the first few rows and the shape of the dataset
    print("Dataset shape:", df.shape)
    return data, df


@app.cell
def _(StandardScaler, data, df):
    # Select feature columns (exclude the target)
    features = data.feature_names
    X = df[features].values

    # Standardize the features to have mean 0 and variance 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, features, scaler


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **silhouette score** is a measure of how well a data point fits into its assigned cluster versus how close it is to the next nearest cluster. It is a measure of clustering quality. 

    If s is close to 1, the point is well-clustered. If s is close to 0, the point is on the decision boundary between two clusters. If s is -1, the point is likely in the wrong cluster.

    \[
        s = \frac{b - a}{\max(a, b)}
    \]

    where   
        • $a$ = Average distance to points in own cluster (cohesion).  
        • $b$ = Average distance to points in the nearest other cluster (separation).
    """
    )
    return


@app.cell
def _(KMeans, X_scaled, mo, silhouette_score):
    # Set the number of clusters; since the dataset has two types (malignant, benign), we choose 2
    n_clusters = 2

    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # Evaluate the clustering performance with the silhouette score
    silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
    print("Silhouette Score for KMeans:", silhouette_kmeans)

    mo.md("""
    **K-Means Clustering** is a centroid-based clustering algorithm that partitions data into K groups, where K is a user-defined number of clusters. 
    """)
    return kmeans, kmeans_labels, n_clusters, silhouette_kmeans


@app.cell
def _(AgglomerativeClustering, X_scaled, mo, n_clusters, silhouette_score):
    # Initialize and fit Agglomerative Clustering with the same number of clusters
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    agglo_labels = agglo.fit_predict(X_scaled)

    # Evaluate the clustering performance with the silhouette score
    silhouette_agglo = silhouette_score(X_scaled, agglo_labels)
    print("Silhouette Score for Agglomerative Clustering:", silhouette_agglo)

    mo.md("""
    **Agglomerative clustering** is a hierarchical clustering method that starts with each data point as its own cluster and then merges clusters step by step until only one cluster remains. 
    """)
    return agglo, agglo_labels, silhouette_agglo


@app.cell
def _(TSNE, X_scaled, agglo_labels, kmeans_labels, mo, plt):
    # Initialize and run t-SNE for dimensionality reduction to 2 components
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    # Create side-by-side scatter plots for the two clustering methods
    plt.figure(figsize=(12, 6))

    # Plot for KMeans clusters
    plt.subplot(1, 2, 1)
    plt.scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=kmeans_labels, cmap="viridis", alpha=0.7
    )
    plt.title("t-SNE Visualization (KMeans Clusters)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar()

    # Plot for Agglomerative Clustering clusters
    plt.subplot(1, 2, 2)
    plt.scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=agglo_labels, cmap="viridis", alpha=0.7
    )
    plt.title("t-SNE Visualization (Agglomerative Clustering)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    mo.md("""
    ***t-SNE** is a dimensionality reduction technique used to visualize high-dimensional data in 2D or 3D space. It computes the probability that two points are neighbors in the high-dimensional space, maps points to low dimensions (while preserving local similarities), and uses gradient descent to optimize the arrangement of points.
    """)
    return X_tsne, tsne


@app.cell
def _(X_tsne, df, plt):
    # Visualize the t-SNE embedding with true labels for comparison
    plt.figure(figsize=(6, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['target'], cmap='coolwarm', alpha=0.7)
    plt.title("t-SNE Visualization (True Labels)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['Malignant', 'Benign'])
    plt.show()
    return (cbar,)


if __name__ == "__main__":
    app.run()
