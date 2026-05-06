
#%%
import pandas as pd 
import matplotlib.pyplot as plt

import numpy as np

#%%

MAX_TRIGS = 15
PRIOR_NAME = "gear1"
OUTPUT_PATH = f"results/output/max_trigs_{MAX_TRIGS}/{PRIOR_NAME}_benchmark_results.csv"

df = pd.read_csv(OUTPUT_PATH)