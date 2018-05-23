####################################
# compare tables
####################################


import os
import pandas as pd
import numpy as np
import difflib
import itertools
import time


#==============================================================================
# MANUAL TESTING
#==============================================================================
#
#os.getcwd()
#os.chdir('C:\\Users\\1580873\\Desktop\\Completed Developments\\Utility\\Source')
#
##sep=
##encoding=
#
#new_filepath = 'sample_new.csv'
#old_filepath = 'sample_old.csv'
#
#new = pd.read_csv(new_filepath , sep=',', encoding='utf-8')
#old = pd.read_csv(old_filepath , sep=',', encoding='utf-8')
#
#print(str(os.stat(new_filepath).st_size/1000000) +"MB")#mb
#print(str(os.stat(old_filepath).st_size/1000000) +"MB")#mb
#
#comparator = compare_table(new=new, old=old)
#
#comparator.shape_diff()
#comparator.find_diffcol()
#comparator.get_common_cols()
#
#comparator.find_dupPK(table=new, PK='Main.Profile.Code')
#out=comparator.find_colmembership()
#comparator.dict_to_xls(out, filename = "colmember.xlsx")
#
##mutators
#comparator.set_common_cols()
#comparator.remove_duplicates()
#
#comparator.values_change()
#comparator.values_change_with_matching()
## static: for any other tables 
#comparator.from_to(left=new, right=old) 
#
#comparator.compare_innerjoin_exclusions()





#==============================================================================
# HELPERS
#==============================================================================

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print( '%r  %2.2f ms' %  (method.__name__, (te - ts) * 1000))
        return result
    return timed

class compare_table(object):

    @timeit
    def __init__(self, new = None, old = None):
        self.t1 = new # NEW #pandas.DataFrame # LEFT 
        self.t2 = old # OLD # RIGHT
        # pre comparison cleaning
    
    @timeit
    def standardise_dtype(self, on=None, new_type=None):
        self.t1[on] = self.t1[on].astype(new_type)
        self.t2[on] = self.t2[on].astype(new_type)
        
    @timeit
    def get_common_cols(self, t1=None, t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        s1=list(t1.columns.values)
        s2=list(t2.columns.values)
        return [x for x in list(t1.columns.values) if x in list(set(s1) & set(s2))]
    
    @timeit        
    def set_common_cols(self):
        s1=list(self.t1.columns.values)
        s2=list(self.t2.columns.values)
        self.t1 = self.t1[[x for x in list(self.t1.columns.values) if x in list(set(s1) & set(s2))]]
        self.t2 = self.t2[[x for x in list(self.t2.columns.values) if x in list(set(s1) & set(s2))]]
        pass
    
    # troubleshoot why PK is not unique?
    @staticmethod
    def find_dupPK(table, PK=None):        
        df = pd.DataFrame(columns=table.columns.values)
        for d in table[PK][ table[PK].duplicated() ].unique():
            df = pd.concat([df, table.loc[table[PK]==d, :]],axis=0)
        return df
        
    @timeit            
    def remove_duplicates(self):        
        old_rows = self.t1.shape[0]
        self.t1 = self.t1.drop_duplicates(keep='first')
        print("Removed " + str(old_rows - self.t1.shape[0]) + " rows from table1 (New)")
        
        old_rows = self.t2.shape[0]
        self.t2 = self.t2.drop_duplicates(keep='first')
        print("Removed " + str(old_rows - self.t2.shape[0]) + " rows from table1 (Old)")
        pass
        
    # versioning control: test change of values for an EXACT similar table    
    @staticmethod 
    def from_to(left, right):
        out = {}
        left = left.reset_index()
        right = right.reset_index()
        boo = left.ne(right)
        idx = np.where(boo) # (row, col)
        
        try:
            # MUST BE SAME SHAPE
            chg_from = right.values[idx] # RIGHT is OLD. so from
            chg_to = left.values[idx]
        except IndexError:
            print('Ensure 2 tables in comparison are of the same shapes')
            return
        
        changed = boo.stack()[boo.stack()]
        changed.index.names = ['index','columns']
        # False positive found e.g. of 100 columns, 1 col is different --> 0.99>0.95 similarity but its a valid difference
        # how to filter out false positive (error)?
        
        # pass in "similarity" ratio to output for subjectivity evaluation
        # output the rows with mismatches too
        out['from_to_idx'] = pd.DataFrame({'from': chg_from, 'to': chg_to}, index=changed.index).dropna(axis=0, how='all')
        out['suspected_from'] = right.iloc[list(set([x[0] for x in out['from_to_idx'].index.values])),:]
        out['suspected_to'] = left.iloc[list(set([x[0] for x in out['from_to_idx'].index.values])),:]
        return out
        
    # but order or rows and columns must be the same    
    @timeit        
    def values_change_with_matching(self):
        '''
        sort rows by key(s)
        sort columns (only for common columns)
        '''
        idx = self.isolate_similar_rows(self, left=self.t1, right=self.t2, threshold=0.95)
        cc = self.get_common_cols()
        return self.from_to(left=self.t1[cc].iloc[idx['left'],:], \
                       right=self.t2[cc].iloc[idx['right'],:])
            
    @timeit        
    def values_change(self):        
        return self.from_to(self.t1, self.t2)
    
    # alternative use: from_to between 2 tables from left and right not from inner join    
    @timeit        
    def compare_innerjoin_exclusions(self, on=None):
        
        excl = self.innerjoin_exclusions(on)
        left = excl['left_only'] #ONLY IN NEW
        right = excl['right_only'] #ONLY IN OLD
        idx = self.isolate_similar_rows(self, left,right,threshold=0.95)
        # t1.iloc[[0,0,0],:] allows one-many
        return self.from_to(left=left.iloc[idx['left'],:], right=right.iloc[idx['right'],:])
    
    #which columns values is a miss? What is a 'inexact 'match?
    @staticmethod
    def similarity(row1, row2):
        '''
        useful if there are many redundant columns values
        0 to 1.00
        threshold: 0.95
        '''
        a="".join(str(x) for x in row1)
        b="".join(str(x) for x in row2)
        return difflib.SequenceMatcher(None, a, b).ratio()
    
    # DEPENDS on no. of columns and how much FP alpha tolerated
    @staticmethod
    def isolate_similar_rows(self,left,right,threshold=0.95):
        # NOTE!! Only use for subset of data.. otherwise, permutation will have Inf possibilties
        print('Please Wait. Script is running to match similar rows...')
        print('Definition of "Similar" Threshold is ' + str(threshold) + '/1.00')
        idx = pd.DataFrame([ [ x[0],x[1] ] for x in itertools.product(list(range(left.shape[0])), \
                            list(range(right.shape[0]))) ], \
                            columns = ['left','right'])
        idx['similarity'] = None
        
        for i in range(idx.shape[0]):
            li = idx.loc[i,'left']; ri = idx.loc[i,'right']
            idx.loc[i,'similarity'] = self.similarity(left.iloc[li,:],right.iloc[ri,:])
        
        return idx.loc[idx['similarity']>=threshold]
        

    #test row counts
    @timeit
    def shape_diff(self):
        r1=self.t1.shape[0]; c1=self.t1.shape[1]
        r2=self.t2.shape[0]; c2=self.t2.shape[1]
        print('t1 shape is ' + str(self.t1.shape) )
        print('t2 shape is ' + str(self.t2.shape) )
        print('t1 has ' + str(r1-r2) + ' more rows than t2')
        print('t1 has ' + str(c1-c2) + ' more columns than t2')
        pass
    
    #find columns not in both tables. Whats in A not in B? Whats in B not in A?
    @timeit
    def find_diffcol(self):
        c1=set(self.t1.columns.values)
        c2=set(self.t2.columns.values)
        
        notin1 = c1.difference(c2)
        notin2 = c2.difference(c1)
        
        if notin1 != set(): 
            print(notin1 + " not in table1 columns but in table2 columns")
        if notin2 != set(): 
            print(notin2 + " not in table2 columns but in table1 columns")
        if notin1 == set() and notin2 == set():
            print('No Differences in Columns!')
        pass
    
    #by column, test matchness between 1-1 row for those not not inner join... 
    @timeit
    def find_colmembership(self):
        '''
        returns a dictonary
        k1 common members per column
        k2 only in new, not in old
        k3 only in old, not in new
        '''
        cc = self.get_common_cols()
        
        print(self.t1[cc].dtypes)
        # by float / int not always True.. so go by length of unique values
        #str(t1['Exposure_for_RWA'].dtype).startswith('float')
        # how to handle float accuracy ???
        
        dic = {}
        for c in cc:
            u = list(self.t1[c].unique())
            dic[c] = u        
        # pretify into df
        u_t1_df = pd.DataFrame( dict([ (k,pd.Series(v)) for k,v in dic.items() ]) ).replace(np.nan,'',regex=True)
        
        dic = {}
        for c in cc:
            u = list(self.t2[c].unique())
            dic[c] = u        
        # pretify into df
        u_t2_df = pd.DataFrame( dict([ (k,pd.Series(v)) for k,v in dic.items() ]) ).replace(np.nan,'',regex=True)

        out = {}
        # k1
        dic = {}
        for c in cc:
            u=list(set(u_t1_df[c].unique()) & set(u_t2_df[c].unique()))
            if '' in u: u.remove('')
            dic[c] = u
        out['common'] = pd.DataFrame( dict([ (k,pd.Series(v)) for k,v in dic.items() ]) ).replace(np.nan,'',regex=True)

        # k2
        dic = {}
        for c in cc:
            u=list(set(u_t1_df[c].unique()) - set(u_t2_df[c].unique()))
            if '' in u: u.remove('')
            dic[c] = u
        out['newonly'] = pd.DataFrame( dict([ (k,pd.Series(v)) for k,v in dic.items() ]) ).replace(np.nan,'',regex=True)
        
        # k3
        dic = {}
        for c in cc:
            u=list(set(u_t2_df[c].unique()) - set(u_t1_df[c].unique()))
            if '' in u: u.remove('')
            dic[c] = u
        out['oldonly'] = pd.DataFrame( dict([ (k,pd.Series(v)) for k,v in dic.items() ]) ).replace(np.nan,'',regex=True)

        return out
    
    @timeit
    def innerjoin_exclusions(self, on=None):
        return self.compare_exclusions(t1=self.t1, t2=self.t2, on=on)

    #find rows not in inner join. Whats in A not in B? Whats in B not in A?
    def compare_exclusions(self,t1, t2, on=None):
        '''
        returns a dictonary
        k1 in new, not in old - LEFT JOIN exclude INNER JOIN
        k2 in old, not in new - RIGHT JOIN exclude INNER JOIN
        '''
        out = {}
#        XandY_diff = True # duplicated column names ending with _x LEFT and _y RIGHT
        if on==None: 
            on = self.get_common_cols(t1=t1,t2=t2)
            print("common cols = " + str(on) )
#            XandY_diff = False

        #check is ON is a unique KEY
        if any(t1.drop_duplicates()[on].duplicated()) or any(t2.drop_duplicates()[on].duplicated()):
            print("primary (or composite) Key is NOT UNIQUE! Try another set of \
                  columns as Key or use all (non numerical) columns.")
            return
        # self.remove_duplicates()
        t1 = t1.drop_duplicates(keep='first')
        t2 = t2.drop_duplicates(keep='first')
        
        # _merge both left_only right_only
        # in new not in old
        merge_results = pd.merge(left=t1, right=t2, on=on, how='outer', indicator=True)
        out['left_only'] = merge_results.loc[merge_results['_merge']=='left_only']
        out['both'] = merge_results.loc[merge_results['_merge']=='both']
        out['right_only'] = merge_results.loc[merge_results['_merge']=='right_only']

        return out
    
    
    @staticmethod
    def dict_to_xls(d, filename=None):

        if filename == None: filename = 'exclusions.xlsx'
        writer = pd.ExcelWriter(path=os.getcwd()+'\\'+filename)
        
        for k,v in d.items(): 
            #if 'Index' in v.columns.values: v = v.drop('Index',axis=1)
            v.to_excel(writer, sheet_name = k, 
                           index = True, header = True, na_rep='', inf_rep='',
                           merge_cells = False, float_format = '%.2f',
                           encoding = 'iso-8859-1')
        print('Written into ' + filename)
        writer.close()
        pass
    
    
    __doc__ = \
    """
    ============================
    EXAMPLES
    ============================
    comparator = compare_table(new=new, old=old)
    
    comparator.shape_diff()
    comparator.find_diffcol()
    comparator.get_common_cols()
    
    comparator.find_dupPK(table=new, PK='Main.Profile.Code')
    out=comparator.find_colmembership()
    comparator.dict_to_xls(out, filename = "colmember.xlsx")
    
    #mutators
    comparator.set_common_cols()
    comparator.remove_duplicates()
    
    comparator.values_change()
    comparator.values_change_with_matching()
    # static: for any other tables 
    comparator.from_to(left=new, right=old) 
    
    comparator.compare_innerjoin_exclusions()

    
    ============================
    THOUGHTS ON RECONCILIATION
    ============================
    pre comparison cleaning:
        - find commmon columns
        - order columns
        - trim ends, cleanup column names
        - does columns data types match? standardise dtypes
        - drop indexes like S/N 1,2,3..., no purpose other than count

    comparison exceptions handling:
        - treatment of nan, nan == nan is False pd.fillna('')
        - trim all spaces at ends
        - exactness of numbers decimals? tolerance at what dp?
        - columns present but missing values at other table...
        - can comparison be done by ignoring columns with missing values? 
        - need one column match AT LEAST!
        
    """

####################################
# test.bat
####################################

@echo off

echo %Configures setup paths of Tableau User Reconciliation weekly batch task
echo 

set PATH=%PATH%;C:\Users\1580873\Desktop\RiskView\UserReconcilation
set "python=C:\ProgramData\Anaconda3\python.exe"
set "pyscript=tableau_access_reconciliation.py"

set "tableau_filepath=User_List_data.csv" REM Filepath of Tableau Server Users 
set "um_filepath=UserManagement.xlsx" REM Filepath of UserManagement
set "library_dir=C:\Users\1580873\Desktop\RiskView\UserReconcilation" REM Working Directory of packages and files

echo %python%
echo %tableau_filepath% 
echo %um_filepath% 
echo %library_dir% 

%python% -u %pyscript% %1 %tableau_filepath% %um_filepath% %library_dir% 

echo %ERRORLEVEL% :: handling error 1=error present 0=no error
echo %End of File
pause rem call cmd
