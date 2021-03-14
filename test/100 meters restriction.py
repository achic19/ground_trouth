import pandas as pd
import random
from random import randint

import pandas as pd

# generate some integers
values = list()
for i in range(10):
    values.append([i, random.choice(['A', 'B', 'C']),
                   randint(1, 3), randint(0, 20), randint(10, 30)])

df = pd.DataFrame(values, columns=['via_to', 'link2', 'time', 'speed', 'mac'])

print(df)
df = df.drop(df[abs(df['via_to'] - df['time']) > 3].index)
print(df)
