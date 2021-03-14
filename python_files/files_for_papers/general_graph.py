import os

import pandas as pd


array_pd = []
for file in os.listdir():
    if file == 'general_graph':
        continue
    array_pd.append(pd.read_csv(file))


result = pd.concat(array_pd, ignore_index=True)
result.to_csv('test.csv')