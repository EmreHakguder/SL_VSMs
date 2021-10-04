import csv
import pandas as pd
import os

def clear_phonology_df(somePhonDF_Path):
    intersect = intersect_phon_sem(somePhonDF_Path)
    
    #Identifying the language
    language = somePhonDF_Path.split("/")[-1].split("_")[0]
    phon_df = pd.read_csv(somePhonDF_Path).set_index("video")
    
    phon_df = phon_df[phon_df.index.isin(intersect)]
    phon_df = phon_df.sort_index()
    phon_df = phon_df.reset_index()
    
    masterPath =  "data/output/phonologyData/"
    if not os.path.exists(masterPath):
        os.makedirs(masterPath)
        
    phon_df.to_csv(masterPath+language+"_Phonology_clean.csv.gz", index=False)
    del phon_df
    
def clear_semantics_df(somePhonDF_Path, some_glove_path):
    #Identifying the language
    language = somePhonDF_Path.split("/")[-1].split("_")[0]
    
    intersect = intersect_phon_sem(somePhonDF_Path)
    dim = some_glove_path.split(".")[-2]

    sem_df = pd.read_table(some_glove_path, sep=" ", header=None, quoting=csv.QUOTE_NONE)
    sem_df.columns = ["video"]+[i for i in range(len(sem_df.columns)-1)]
    sem_df["video"] = sem_df["video"].str.lower()

    sem_df = sem_df.set_index("video")
    sem_df = sem_df[sem_df.index.isin(intersect)]
    
    #Splitting df into unique rows and non-unique rows // keeping only the first occurrence of duplicate rows
    index = sem_df.index
    is_duplicate = index.duplicated(keep="first")
    not_duplicate = ~is_duplicate
    
    #Keeping only the first of duplicate signs
    sem_df = sem_df[not_duplicate].copy()
    sem_df = sem_df.sort_index()
    sem_df = sem_df.reset_index()
    
    masterPath =  "data/output/semanticsData/"
    if not os.path.exists(masterPath):
        os.makedirs(masterPath)
    
    sem_df.to_csv(masterPath+language+"_Semantics_"+str(dim)+"_clean.csv.gz", index=False)
    del sem_df
    
def intersect_phon_sem(somePhonDF_Path, glovePath = "../../../Downloads/glove/glove.6B.50d.txt"):
    #Identifying the language
    language = somePhonDF_Path.split("/")[-1].split("_")[0]
    
    """PHONOLOGY DATA"""
    phon_df = pd.read_csv(somePhonDF_Path).set_index("video")
    phon_signs = list(phon_df.index)
    
    """SEMANTICS DATA"""
    sem_df = pd.read_table(glovePath, sep=" ", header=None, quoting=csv.QUOTE_NONE)
    sem_df[0] = sem_df[0].str.lower()
    sem_df = sem_df.set_index(0)
    sem_signs = list(sem_df.index)
    
    """OVERLAPPING PHONOLOGY AND SEMANICS DATA"""
    intersect = list_intersect(phon_signs, sem_signs)
     
    return intersect

def list_intersect(list1, list2):
    return list(set(list1) & set(list2))

def remove_phonology_duplicate_videos(somePhonDF_Path):
    #Identifying the language
    language = somePhonDF_Path.split("/")[-1].split("_")[0]
    
    #Reading in the phonology excel file 
    phon_df = pd.read_excel(somePhonDF_Path)
    
    #making sure all cells are of data type 'string'
    phon_df = phon_df.astype(str) 
    
    #Lowering case of video names
    phon_df.video = phon_df.video.str.lower()
    
    #Removing numbers and parentheses that identify duplicate signs
    phon_df.video = phon_df.video.str.rstrip("()0123456789")
    
    #Setting the 'video' column as dataframe index
    phon_df = phon_df.set_index("video")
    
    #Splitting df into unique rows and non-unique rows // keeping only the first occurrence of duplicate rows
    index = phon_df.index
    is_duplicate = index.duplicated(keep="first")
    not_duplicate = ~is_duplicate
    
    #Keeping only the first of duplicate signs
    phon_df = phon_df[not_duplicate].copy()
    phon_df = phon_df.reset_index()
    
    #Lower-casing certain column values
    phon_df = lower_column_values(phon_df)
    
    phon_df.to_csv("data/transforming/phonologyData/unique_signs/"+language+"_unique.csv.gz", index=False, compression='gzip')
    
    del phon_df


def lower_column_values(somePhonDF):
    #Lower-casing values of phonology columns if lower-upper case distinction in the transcription does not make a difference
    #This is necessary because there are some irregularities in transcriptions
    lowerCols = [col for col in somePhonDF.columns if ("HandShape" not in col) and ("Thumb" not in col) and ("JointConfiguration" not in col) and ("SelectedFingers" not in col)]

    for lowerCol in lowerCols:
        somePhonDF[lowerCol] = somePhonDF[lowerCol].str.lower()
    
    return somePhonDF
