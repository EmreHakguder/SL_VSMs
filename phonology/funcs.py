import numpy as np
import os
import pandas as pd
import json

from itertools import combinations
from scipy.spatial.distance import cosine
from sklearn.decomposition import TruncatedSVD, randomized_svd

def vectorize_phonology(somePhonDF_Path, phonColumnsDict="data/raw/phonologyData/phonCols.json"):
    #Reading in the csv file
    phon_df = pd.read_csv(somePhonDF_Path).set_index("video")
    
    #Identifying the language
    language = somePhonDF_Path.split("/")[-1].split("_")[0]

    """Creating separate DFs per phonological parameter (ENTIRE, HS_only, LOC_only, MOV_only)"""
    with open(phonColumnsDict, 'r') as f:
        phonCols = json.load(f)

    for phonType in phonCols:
        print("Now working on", language, phonType)
        cols = phonCols[phonType]
        phon_df_col = phon_df[cols]
        
        #Transforming to one-hot encoding
        phon_df_col_oneHot = pd.get_dummies(phon_df_col, drop_first=True)

        """DIMENSIONALITY REDUCTION FOR PHONOLOGY"""
        A = phon_df_col_oneHot
        
        if phonType == "MOV":
            N_components = int(len(A.columns)-1)
        else:
            N_components = int(len(A.columns)/3)
            
        svd =  TruncatedSVD(n_components = N_components)
        A_transf = svd.fit_transform(A)
        A_transf.shape
        
        print(str(len(phon_df_col_oneHot)) + " columns reduced to "+str(N_components)+" columns.")
        print("explained_variance_ratio_:", sum(svd.explained_variance_ratio_), "\n\n")
        

        phon_df_col_oneHot_svd = pd.DataFrame(A_transf, index=A.index)
        phon_df_col_oneHot_svd = phon_df_col_oneHot_svd.reset_index()
        
        masterPath = "data/output/vectorizedPhonDFs/"
        if not os.path.exists(masterPath):
            os.makedirs(masterPath)
            
        phon_df_col_oneHot_svd.to_csv(masterPath+language+"_"+phonType+"_vectorizedDF.csv.gz", index=False, compression="gzip")
    
def find_phonology_cosine_similarity_perPhonType(someVectorizdPhonDFPath):
    language = someVectorizdPhonDFPath.split("_")[0].split("/")[-1]
    phonType = someVectorizdPhonDFPath.split("_")[1]
    
    print("Now working on ", language, phonType)
    phon_df_col_oneHot_svd = pd.read_csv(someVectorizdPhonDFPath)
    phon_df_col_oneHot_svd = phon_df_col_oneHot_svd.set_index("video")
    
    #Creating unique pairs of signs
    sign_pairs = list(combinations(phon_df_col_oneHot_svd.index, r=2))
    
    """CREATING A DATAFRAME FOR PHON COSINE SIMILARITY"""
    index = pd.MultiIndex.from_tuples(sign_pairs, names=["s1", "s2"])
    cosSim_col = phonType+"_cosineSim"
    phon_output_df = pd.DataFrame(columns=[cosSim_col] , index=index)

    masterPath = "data/output/vectorized_PhonSim/"+language+"/"+phonType+"/"
    if not os.path.exists(masterPath):
        os.makedirs(masterPath)

    k = 0
    for i, (a, b) in enumerate(sign_pairs): 
        phon_output_df.loc(axis=0)[a,b][cosSim_col] = round(1 - cosine(phon_df_col_oneHot_svd.loc[a], phon_df_col_oneHot_svd.loc[b]), 4)

        if ( i%10000 == 0 ) or ( i == ( len(sign_pairs)-1 ) ):
            phon_output_df = phon_output_df.dropna()
            phon_output_df = phon_output_df.reset_index()
            
            path = masterPath+language+"_"+phonType+"_vectorizedPhonSim_part"+str(k).zfill(3)+".csv.gz"
            
            phon_output_df.to_csv(path, compression="gzip", index=False)

            #Progress
            print("Progress...", round(i/len(sign_pairs),3))
            k+=1
            phon_output_df = pd.DataFrame(columns = [cosSim_col], index = index)

        if ( i == ( len(sign_pairs)-1 ) ):
            del phon_output_df

    del phon_df_col_oneHot_svd
        

def stitch_parts(masterPath):
    csvs = [masterPath+p for p in os.listdir(masterPath) if not p.startswith(".")]
    df = pd.concat(map(pd.read_csv, csvs), ignore_index=True)
    return df