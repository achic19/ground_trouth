# class Student():
#   name = "Albert"
#   age = "20"
#   def greet(self,message):
#     print (message +" "+ self.name)
#
# student = Student()
# print (getattr(student , "name"))
# print (getattr(student , "age"))
# getattr(student, "greet")("Hello")

import random
from random import randint

import pandas as pd

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
print(df2)
# df2 = df2.loc[(df2['speed'] > 10) | (df2['time'] > 2 )]
df2 = df2.loc[(df2['speed'] > 10) | ((df2['speed'] < 10) & (df2['time'] > 2) & (df2['mac2'] > 15))]
print(df2)
# fds = df2.groupby('link2')
# fds2 = fds['time'].median()
# print(fds2)
# rows_list = []
# for i , group_name in enumerate(df2.link2.unique()):
#     group = str(fds.get_group(group_name)['time'].values.tolist())
#     rows_list.append([group_name,group])
# print(rows_list)
# newDF = pd.DataFrame(rows_list)
# print(newDF)
# gk = pd.concat([fds2, newDF], axis=1)
# print(gk)

# df2.at[0,'link2' ] = update
# print(df2)
# df2 = df2.drop(df2[(df2['time'] < 1.5) & (df2['speed'] < 10)].index)
# print(df2.groups['speed'])

# size = df2.shape[0]
# mean_1 = df2['via_to'].mean()
# diff = df2['via_to'] - mean_1
#
# ind_1 = df2.iloc[(diff).abs().argsort()[:size]].index[size - 1]
# candidates = df2.at[ind_1, 'via_to']
# df2 = df2.drop([ind_1])
# print(candidates)
# print(df2)
# std = df2['via_to'].std()
# diff = abs(candidates - df2['via_to'].mean())
# if std > diff:
#     print()
# df2= df2.rename(columns={'mac2':'ttt'})
# print(df2)
# print(pd.merge(df,df2[['via_to','ttt']],on=['via_to'], how='inner'))
#
# # # df = df.append(df2,ignore_index=True)
# # # print(df)
# #
# # # df['avarage'] = ''
# # # df['std'] = ''
# # # print(df)
# # # gk = df.groupby(['link1', 'link2'], as_index=False).groups
# # gk0 = df.groupby(['link1'])['speed'].mean()
# # print (gk0)
# # gk1 = df.groupby(['link1'])['speed'].std()
# # print (gk1)
# # gk =pd.concat([gk0 , gk1], axis=1)
# # gk.to_csv('mean_std.csv',header= ['mean','std'])
# # # print (gk)
# #
# # # # final = pd.DataFrame(gk.keys(), columns=['from','to'])
# # # # final.to_csv('final')
# # # for group_name in df.link.unique():
# #     group = gk.get_group(group_name)
# #     print(group)
# #     mac_records = group.loc[(group.mac.isin(['A', 'B']))]
# #     for i, record in enumerate(mac_records.time):
# #         result = group.loc[(record - group['time'] < 5) & (record - group['time'] >= 0)]
# #         index = group.index[i]
# #         print(result)
# #         df['avarage'].iloc[index] = result.speed.mean()
# #         df['std'].iloc[index] = result.speed.std()
# # print(df)
#
#
# # for group_name in df.link.unique():
# #     group = gk.get_group(group_name)
# #     print(group_name)
# #
# #     for i, record in enumerate(group.time):
# #         result = group.loc[(record - group['time'] < 5) & (record - group['time'] >= 0)]
# #         index = group.index[i]
# #         print(result)
# #         df['avarage'].iloc[index] = result.speed.mean()
# #         df['std'].iloc[index] = result.speed.std()
# # print(df)
