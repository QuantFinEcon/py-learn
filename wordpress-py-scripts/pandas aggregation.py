import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame([[1, 'cat1'], [1, 'cat1'], [4, 'cat2'], [3, 'cat1'], [5, 'cat1'],[1, 'cat2']], columns=['A', 'B'])
df = df[['A','B']]

df['count'] = df.groupby(['A','B'])['A'].transform('count')

df = df.drop_duplicates(['A','B'])

df = df.groupby(['A','B']).sum()

lot = df.unstack().plot(kind='bar',subplots=True, sharex=True, sharey=True, layout = (3,3), legend=False)
plt.show(block=True)





df = pd.DataFrame([[1, 'cat1'], [1, 'cat1'], [4, 'cat2'], [3, 'cat1'], [5, 'cat1'],[1, 'cat2']], columns=['A', 'B'])
df = df[['A','B']]

df_1 = df.groupby(['A', 'B'])['A'].agg({'counts': 'count'}).reset_index()
df_2 = df.groupby('B')['A'].agg({'average': 'mean'}).reset_index()

df = df_1.merge(df_2, on='B').drop_duplicates(['A', 'B'])
df.drop('average', axis=1, inplace=True)
df = df.groupby(['A','B']).sum()

df_2['A'] = df_2['average']
df_2 = df_2.groupby(['A','B']).sum()


fig, ax = plt.subplots(2, 2, figsize=(8, 8))

target1 = [ax[0][0], ax[0][1]]
target2 = [ax[1][0], ax[1][1]]

df.unstack().plot(kind='bar', subplots=True, rot=0, xlim=(0,5), ax=target1,
                            ylim=(0,3), layout=(2,2), legend=False)

df_2.unstack().plot(kind='bar', width=0.005, subplots=True, rot=0, xlim=(0,5), ax=target2,
                    ylim=(0,3), layout=(2,2), legend=False, color='k')

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


