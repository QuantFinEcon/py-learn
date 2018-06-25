
"""
Problem:    Given a N rows x M columns array, each being True or False,
            toggle to maximise the count of True
"""
from copy import deepcopy


class BoolList(list):
    """ Extends builtin list to lazy compute count composition
        @*args:bool
    """
    def __init__(self, *args):
        super().__init__(args)

    @classmethod
    def fromList(cls, lst):
        return cls(*lst)
    
    @property
    def count_true(self):
        return sum(1 for x in self if x is True)
        
    @property
    def count_false(self):
        return sum(1 for x in self if x is False)

    def __str__(self):
        return "\t".join(list(map(str,self)))
        
class BoolMatrix(list):
    """ Matrix is made up of N rows of BoolList, 
        each of M columns. Ragged arrays not allowed. 
        @*args:BoolList
    """
    def __init__(self, *args):
        self.has_next = True
        for row in args:
            assert isinstance(row, BoolList)
            self().append(row)

    def __call__(self):
        return self
    
    @property
    def TrueCount(self):
        """ returns total count of True in boolmatrix """
        return sum([x.count_true for x in self])
    
    @property
    def MostFalseRow(self):
        """ returns first row index with the most False 
            output:tuple (<index>, <max count>)
        """
        counts = [x.count_false for x in self]
        return (counts.index(max(counts)),max(counts))

    @property
    def MostFalseColumn(self):
        """ returns first column index with the most True 
            output:tuple (<index>, <max count>)
        """
        counts = [0 for _ in range(len(self[0]))]
        for _boolList in self:
            for i in range(len(_boolList)):
                if not _boolList[i]:
                    counts[i] += 1
        return (counts.index(max(counts)),max(counts))
        
    def swap(self):
        """ inverse boolean row if there is a greater increase in True counts
            @row:int index of row in BoolMatrix
            @column:int index of column in BoolMatrix
        """
        _MostFalseRow = self.MostFalseRow
        _MostFalseColumn = self.MostFalseColumn
        if _MostFalseRow[1] >= _MostFalseColumn[1]:
            # inherit list.__setitem__(self,index,value)
            row = _MostFalseRow[0]
            self[row] = BoolList.fromList([False if x else True 
                for x in self[row]])
        else:
            column = _MostFalseColumn[0]
            for row in self:
                if row[column]:
                    row[column] = False
                else:
                    row[column] = True
        
        #determine if a next swap would increase count of True
        self.has_next = self.MostFalseRow[1] > _MostFalseRow[1] \
                        or self.MostFalseColumn[1] > _MostFalseColumn[1]

    def __str__(self):
        return "\n".join(list(map(str,self)))


if __name__ == "__main__":
    
    test_cases = [
    
    BoolMatrix(BoolList(False,True,True,True),
               BoolList(True,False,False,False),
               BoolList(False,True,True,True)) , 
    
    BoolMatrix(BoolList(True,False,False,True),
               BoolList(True,False,False,True),
               BoolList(False,True,True,True),
               BoolList(False,True,True,True)) , 

    BoolMatrix(BoolList(True,False,False,True),
               BoolList(True,False,False,True),
               BoolList(False,True,True,False),
               BoolList(False,True,True,False)) , 
               
    BoolMatrix(BoolList(False,True,True,True,True),
               BoolList(True,True,True,False,True),
               BoolList(False,False,False,True,False),
               BoolList(False,True,True,True,True),
               BoolList(False,True,False,True,True)) , 

    ]

    for x in test_cases:
        
        _TrueCount = 0
        while(_TrueCount <= x.TrueCount and x.has_next):
            print("prev={0} new={1}".format(_TrueCount, x.TrueCount))
            _TrueCount = x.TrueCount
            _MostFalseRow = x.MostFalseRow
            _MostFalseColumn = x.MostFalseColumn
            # save copy of x before overriding
            _x = deepcopy(x)
            # greedily favor row-wise first if on par with column
            x.swap()
            
        print(x)
        print("\n")

