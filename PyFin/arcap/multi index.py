import pandas as pd

arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
          ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]

tuples = list(zip(arrays))
tuples
tuples = list(zip(*arrays))
tuples


index = pd.MultiIndex.from_tuples(tuples, names=['first','second'])
index

s = pd.Series(np.random.randn(8), index=index)


iterables = [['bar', 'baz', 'foo', 'qux'], ['one', 'two']]

index = pd.MultiIndex.from_product(iterables, names=['first','second'])
index

s = pd.Series(np.random.randn(8), index=index)


arrays = [np.array(['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux']),
          np.array(['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'])]

df = pd.DataFrame(np.random.randn(8, 4), index=arrays)
df

df = pd.DataFrame(np.random.randn(3, 8), index=['A', 'B', 'C'], columns=index)
df

df.reindex(index)
df.reindex(index[:5])

df = df.T

df.loc['bar']
df.iloc[0:3]

# multi level indexing
df.loc[('baz', 'two'):('qux', 'one')]
df.loc[('baz', 'two'):'qux']
df.ix[[('bar', 'two'), ('qux', 'one')]]


def mklbl(prefix,n):
    return ["%s%s" % (prefix,i)  for i in range(n)]

mklbl('A',4)


miindex = pd.MultiIndex.from_product([mklbl('A',4),
                                      mklbl('B',2),
                                      mklbl('C',4),
                                      mklbl('D',2)])

micolumns = pd.MultiIndex.from_tuples([('a','foo'),('a','bar'),
                                       ('b','foo'),('b','bah')],
                                    names=['lvl0', 'lvl1'])

dfmi = pd.DataFrame(np.arange(len(miindex)*len(micolumns)).reshape((len(miindex),len(micolumns))),
                    index=miindex,
                    columns=micolumns).sort_index().sort_index(axis=1)
dfmi 

dfmi.loc[(slice('A1','A3'),slice(None), ['C1','C3']),:'a']
dfmi.loc[(slice('A1','A3'),slice('B0'), ['C1','C3']),:]
dfmi.loc[(slice('A1','A3'),slice('B1'), ['C1','C3']),:] #slice till B1
dfmi.loc[(slice('A1','A3'),['B1'], ['C1','C3']),:]

idx = pd.IndexSlice
idx[:,:,['C1','C3']]
dfmi.loc[idx[:,:,['C1','C3']],idx[:,'foo']]

# boolean indexer
mask = dfmi[('a','foo')]>200
dfmi.loc[idx[mask,:,['C1','C3']],idx[:,'foo']]
dfmi.loc(axis=1)[idx[:,'foo']] # set passed one tuple(slice(..),slice(..),...)

# set values
df2 = dfmi.copy()
df2.loc(axis=0)[:,:,['C1','C3']] = -10
df2

df2.loc[idx[:,:,['C1','C3']],:] = df2*1000
df2

df
# xs like slice
df.xs('one', level='second', axis=0)
df.loc[(slice(None),'one'),:]

df.xs(('one', 'bar'), level=('second', 'first'), axis=0)

df.xs('one', level='second', axis=0, drop_level=False)
df.xs('one', level='second', axis=0, drop_level=True)

#switch the order of two levels:
df[:5]
df[:5].swaplevel(0, 1, axis=0)

#reorder_levels function generalizes the swaplevel
df[:5].reorder_levels([0,1], axis=0)
df[:5].reorder_levels([1,0], axis=0)

# sort index
import random; random.shuffle(tuples)
s = pd.Series(np.random.randn(8), index=pd.MultiIndex.from_tuples(tuples))
s.sort_index()
s.sort_index(level=0)
s.sort_index(level=1)

# replace names
s.index.set_names(['L1', 'L2'], inplace=True)
s.sort_index(level='L1')
s.sort_index(level='L2')

# indexing
frm = pd.DataFrame(np.random.randn(5, 3))
frm.iloc[:,[0,2]]
frm.iloc[[0,2],:]
frm.take([0,2],axis=1)
frm.take([0,2],axis=0)


# MORE .xs indexing
#idx = pd.IndexSlice
#X = x.loc(axis=1)[idx[:,['PE_RATIO']]]
#X.columns = X.columns.droplevel(1)

#Y = x.swaplevel(0,1,axis=1)
#Y.xs('PE_RATIO', level=0, axis=1,drop_level=True)

X = x.xs('PE_RATIO', level=1, axis=1,drop_level=True)


# level to div/mul/add/sub

P = df.T
P.div(P['foo','one'],level=0)
P.div(P['foo'],level=1)

P
for i in P.columns: print i
for c in P.columns.levels[0]: print c
for c in P.columns.levels[1]: print c

for c in P.columns.levels[0]: 
    print P[c]

for c in P.columns.levels[0]: 
    print P[c,'two']

PP = P.copy()

for c in PP.columns.levels[0]: 
    PP[c,'new1'] = PP[c,'one']/PP[c,'two']

PP
PP.T.sort_index(level=0).T



P.groupby(level=[0]).transform(lambda x: x[])
P.groupby(level=[-1]).transform(lambda x: x)

P.groupby(level=[0]).transform(lambda x: x/x.sum())













