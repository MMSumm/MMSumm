# Create valid.csv

import os
import pandas as pd
os.chdir('/scratch/anshul/valid_data')
arr1 = os.listdir('article')
arr2 = os.listdir('img')
from tqdm import tqdm
out = []
for fn1 in tqdm(arr1):
    temp = []
    for fn2 in arr2:
        if fn1[:-4] in fn2:
            temp.append(fn2)
    out.append(temp)
data = {'article':arr1,'img':out}
df = pd.DataFrame(data)
df.to_csv('valid.csv')
