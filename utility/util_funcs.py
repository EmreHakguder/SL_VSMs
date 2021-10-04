import pandas as pd

def pandas_pair_signs_alphabetically(x):
    return " + ".join(sorted([x["s1"], x["s2"]]))