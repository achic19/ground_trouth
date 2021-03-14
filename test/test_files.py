import pandas as pd
file =pd.read_csv('_car.csv')
file["std_speed"].fillna(0, inplace = True)
print(file.std_speed)