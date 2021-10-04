import numpy as np
import os
import pandas as pd
from itertools import combinations

from scipy.spatial.distance import cosine

def find_semantics_cosine_similarity_pairwise(someGlovePath):
    dim = someGlovePath.split("_")[-2]
    language = someGlovePath.split("/")[-1].split("_")[0]
    
    print(dim, language)
    sem_df = pd.read_csv(someGlovePath).set_index("video")
    
    sign_pairs = list(combinations(sem_df.index, r=2))
    print(len(sign_pairs))
    #!#!#!#!
    """CREATING A DATAFRAME FOR SEM COSINE SIMILARITY"""
    index = pd.MultiIndex.from_tuples(sign_pairs, names=["s1", "s2"])
    cosSim_col = "sem_cosineSim"
    sem_output_df = pd.DataFrame(columns=[cosSim_col] , index=index)

    masterPath = "data/output/SemSim/"+language+"/"+dim+"/"
    if not os.path.exists(masterPath):
        os.makedirs(masterPath)

    k = 0
    for i, (a, b) in enumerate(sign_pairs): 
        sem_output_df.loc(axis=0)[a,b][cosSim_col] = round(1 - cosine(sem_df.loc[a], sem_df.loc[b]), 4)

        if ( i%10000 == 0 ) or ( i == ( len(sign_pairs)-1 ) ):
            sem_output_df = sem_output_df.dropna()
            sem_output_df = sem_output_df.reset_index()
            
            path = masterPath+language+"_"+dim+"_SemSim_part"+str(k).zfill(3)+".csv.gz"
            sem_output_df.to_csv(path, compression="gzip", index=False)

            #Progress
            print("Progress...", round(i/len(sign_pairs),3))
            
            k+=1
            sem_output_df = pd.DataFrame(columns = [cosSim_col], index = index)

        if ( i == ( len(sign_pairs)-1 ) ):
            del sem_output_df

    del sem_df
    
