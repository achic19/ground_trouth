import random
from random import randint

import pandas as pd
import math

# generate some integers
# for i in range(10):
#     values.append([i, random.choice(['A', 'B', 'C']),
#                    randint(1, 3), randint(0, 20), randint(10, 30)])
#
# df = pd.DataFrame(values, columns=['via_to', 'link2', 'time', 'speed', 'mac'])
# print(df)
values2 = list()
for i in range(10):
    values2.append([i - 3, random.choice(['A', 'B', 'C']),
                    randint(1, 3), randint(0, 20), randint(10, 30)])

df2 = pd.DataFrame(values2, columns=['via_to', 'link2', 'time', 'speed', 'mac2'])
gg = df2['via_to']
print(df2)

size = df2.shape[0]
mean_1 = df2['via_to'].mean()
diff = df2['via_to'] - mean_1
print(mean_1)

ind_1 = df2.iloc[(diff).abs().argsort()[:size]].index[size - 1]
candidates = df2.at[ind_1, 'via_to']
df2 = df2.drop([ind_1])
print(candidates)
print(df2)
std = df2['via_to'].std()
print(std)
diff = abs(candidates - df2['via_to'].mean())
print(diff)
if std > diff:
    print(mean_1)
else:
    print(df2['via_to'].mean())
