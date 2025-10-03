import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_data(path="customers.csv"):
    df = pd.read_csv(path)
    return df

def preprocess(df, use_gender=False):
    # Drop CustomerID
    df2 = df.drop(columns=["CustomerID"])
    if not use_gender:
        df2 = df2.drop(columns=["Gender"])
    else:
        # encode gender
        df2["Gender"] = df2["Gender"].map({"Male": 1, "Female": 0})
    return df2

def scale_data(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def find_elbow(Xs, k_range=range(1, 11)):
    wcss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(Xs)
        wcss.append(kmeans.inertia_)
    # plot
    plt.figure()
    plt.plot(list(k_range), wcss, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS (inertia)")
    plt.title("Elbow Method for Optimal k")
    plt.show()
    return wcss

def fit_kmeans(Xs, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(Xs)
    return kmeans, labels

def plot_pca_clusters(Xs, labels, kmeans=None):
    # reduce to 2D via PCA
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=Xp[:,0], y=Xp[:,1], hue=labels, palette="Set1", s=50)
    if kmeans is not None:
        # project cluster centers also
        centers = kmeans.cluster_centers_
        centers_p = pca.transform(centers)
        plt.scatter(centers_p[:,0], centers_p[:,1], c='black', s=200, marker='X', label="centroids")
    plt.title("Clusters visualized via PCA (2D)")
    plt.legend()
    plt.show()

def evaluate_silhouette(Xs, labels):
    score = silhouette_score(Xs, labels)
    return score

def main():
    df = load_data("customers.csv")
    print("Data head:\n", df.head())
    df2 = preprocess(df, use_gender=False)
    Xs, scaler = scale_data(df2)
    
    # find the elbow
    _ = find_elbow(Xs, k_range=range(1, 11))
    # Suppose we pick k = 5 (from elbow)
    k = 5
    kmeans, labels = fit_kmeans(Xs, k)
    print("Silhouette Score (k = {}): {:.4f}".format(k, evaluate_silhouette(Xs, labels)))
    
    # attach labels back to original df
    df["Cluster"] = labels
    # save to CSV
    df.to_csv("mall_customers_with_clusters.csv", index=False)
    print("Saved clustered data to mall_customers_with_clusters.csv")
    
    # visualize
    plot_pca_clusters(Xs, labels, kmeans=kmeans)

if __name__ == "__main__":
    main()
