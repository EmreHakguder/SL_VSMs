import pandas as pd, os, json
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
    
def merge_cluster_data(outPath):
    languages = ["ASL", "BSL"]
    dims = ["50d", "100d", "200d", "300d"]
    masterPath = "results/clustering/signPairs_byCluster/"
    
    clustered_df = pd.DataFrame(columns=["language", "dim", "height", "clusterID", "signPair",
                                         "semSim", "HS_sim", "LOC_sim", "MOV_sim", "ENTIRE_sim"])

    for language in languages:
        for dim in dims:
            list_height_path = masterPath+language+"/"+dim+"/"
            heights = [x.split("_")[3] for x in os.listdir(list_height_path) if not x.startswith(".")]

            for heit in heights:
                path = list_height_path+language+"_"+dim+"_heightheight_"+str(heit)+"_signPairs_byCluster.json"
                print(path)

                with open(path, "r") as read_file:
                    js_file = json.load(read_file)

                    for clusterID in js_file: 
                        for pair in js_file[clusterID]:                        
                            signPair = pair[0] + " + " + pair[1]
                            clustered_df = clustered_df.append({"language":language,
                                                                "dim":dim, 
                                                                "height":heit, 
                                                                "clusterID":clusterID,
                                                                "signPair":signPair,
                                                                "semSim":None,
                                                                "HS_sim":None,
                                                                "LOC_sim":None,
                                                                "MOV_sim":None,
                                                                "ENTIRE_sim":None}, ignore_index=True)

    clustered_df.to_csv(outPath, index=False)
    
    return clustered_df
    

        
        