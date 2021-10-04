import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy import cluster

def cluster_glove(someSemPath, height):
    X = pd.read_csv(someSemPath).set_index("video")
    Z = cluster.hierarchy.ward(X)
    cutree = cluster.hierarchy.cut_tree(Z, height=height)
    clusterLabels = [int(x[0]) for x in cutree]
    clusters_len = len(list(set(clusterLabels)))
    
    if  1 < clusters_len < len(X.index):
        return X.index, silhouette_score(X, cutree.ravel(), metric='euclidean'), clusterLabels, clusters_len
    else:
        return False, False, False, False
        
    

        
        