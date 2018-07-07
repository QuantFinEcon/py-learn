
def extendList(val, list=[]):
    list.append(val)
    return list

list1 = extendList(10)
list2 = extendList(123, [])
list3 = extendList('a', list1)

print("List1 = %s" % list1)
print("List2 = %s" % list2)
print("List3 = %s" % list3)


1. function pass in a python standard library inbuilt "list" object as an argument to cache calculation results. It uses append method that appends to the end of the list by default. Return type is a list. In my opinion, there are 2 other alternatives to cache intermediary results in a list. 1) One can define a callable class object by initalising a list object as attribute (wrapping) and replace __call__ so that class becomes a method. 2) One can simply inherit a list and proceed with setup stage __init__ to append. Here, note that list is mutable unlike tuple, hence we can simply use __init__ rather than __new__ which is implictly run by Python interpreter before object creation. 

2. list in Python is a data structure that can contain multiple data types compared to Java's ArrayList<Object Type> that task JIT compiler to cast all generic Object type which is the AraryList<Object> default reference type. list also has a __str__ by default so print in Python 2 will implicitly print its representation. Only reason it could break is when argument "val" type is more than numbers or strings (e.g. multiple arguments tuple *args, list), or a Python 3 interpreter is used to parse and run script. In this case, list should extend rather than append. 

3. 

# alternative 1
class extendList:

    def __init__(self,lst=None):
        if lst:
            if isinstance(lst,list):
                self.lst = lst
            elif isinstance(lst,extendList):
                self.lst = lst.lst
        else:
            self.lst = list()

    def __call__(self,*args):
        self.lst.extend(args)
        return self.lst

    def __repr__(self):
        return str(self.lst)

if __name__ == "__main__":
    list1 = extendList()
    list1(10)
    print(list1)
    
    list2 = extendList([])
    list2(123)
    print(list2)
    
    list3 = extendList(list1)
    list3('a')
    list3('a')
    list3('b','b')
    list3(['c','c'])
    print(list3)
    print(list1)
    
    
# alternative 2
class extendList(list):
    def __init__(self,*args):
        super().__init__(args)
   
    @classmethod
    def From(cls, lst): 
        assert isinstance(lst,list)
        return cls(*lst)
        
    def __call__(self,*args):
        self.extend(args)
        return self

if __name__ == "__main__":
    print( extendList(1,2,3) )
    print( extendList.From([1,2,3]) )
    myList = extendList(1,2,3)
    print(myList)
    myList(4,5,6)
    print(myList)
    myList([7,8,9])
    print(myList)



"""
Find shortest path given list of list as matrix
- only move right or down 
- sum values in matrix to get total distance
- select path with min total distance
"""

class Path:
    """ stateful key for cache """
    def __init__(self, path=""):
        self._path = path # String RRRDDD
        self.right_steps = path.count("R")
        self.down_steps = path.count("D")
#        self.length = len(self._path)

    @property
    def walked(self):
        return self._path

    @walked.setter
    def walked(self, step): # step 'R' or 'D'
        if step == 'R': # eagerly computed
            self.right_steps += 1
        elif step == 'D':
            self.down_steps += 1
        else:
            raise TypeError("Invalid steps given to path! \
                            Single step R or D only")
        self._path = self._path + step

    def __len__(self):
        return len(self._path)

    def __repr__(self):
        return self._path # use as immutable dict key

    def __eq__(self, other):
        # == --> hash(obj)  vs  is --> id(obj)
        return self._path == other

    def __hash__(self):
        return hash(self._path)
        

#a=Path("RRDDDD")
#a.walked = "RRR"
#a.walked = "R"
#a.walked = "D"
#a.walked
#
#b=Path("DDD")
#
#c = {a:1, b:2}
#c[b]
#c["DDD"]
#c[a.walked]
#d = Path(b.walked + "R")
#c[d]=5



# callable class as 'stateful' function
class FindShortestPath:
    """ memonise path walked with 
        "Path" object as key, sum of values
        as value. Path contains either 'R' or 'D'
        E.g. RRRDDD, RDRDRD
    """    
    def __init__(self, path):
        """
        Inputs:
        @path: list of lists of equal row width
        [
            ['start',3,3,1],
            [2,4,2,3],
            [5,2,3,4],
            [1,3,2,'end'],
        ]
        """
        self.path = path
        self.height = len(path)
        self.width = len(path[0])

        self.cache = dict(_=0) # start as _
        self.__call__() # call as inner helper function
        
        self._ends = {key:value 
                      for key,value in self.cache.items() 
                      if len(key) == (self.height + self.width - 2)}
    
        self.shortest_paths = [k for k,v in self._ends.items()
                               if v == min(self._ends.values())]
    
    
    def __str__(self):
        return "Shortest Paths are = " + str(self.shortest_paths)
    
    def __call__(self, _path=Path("_") ):

        # hasNext: move right
        if _path.right_steps < self.width-1:
            newPath = Path(_path.walked + "R")
            if self.path[newPath.down_steps][newPath.right_steps] == 'end':
                return
            if not newPath in self.cache:
                self.cache[newPath] = self.cache[_path] + \
                    self.path[newPath.down_steps][newPath.right_steps]
                print("Moving From {oldPath} to {newPath}".format(
                        oldPath=_path, newPath=newPath))
                self.__call__(newPath)

        # hasNext: move down
        if _path.down_steps < self.height-1:
            newPath = Path(_path.walked + "D")
            if self.path[newPath.down_steps][newPath.right_steps] == 'end':
                return
            if not newPath in self.cache:
                self.cache[newPath] = self.cache[_path] + \
                    self.path[newPath.down_steps][newPath.right_steps]
                print("Moving From {oldPath} to {newPath}".format(
                      oldPath=_path, newPath=newPath))
                self.__call__(newPath)

                
if __name__ == "__main__":
    
    paths = [
    
        [['start',1,2,3,'end']],
        [['start'],[1],[2],[3],['end']],   
        [
            ['start',3,3,1],
            [2,4,2,3],
            [5,2,3,4],
            [1,3,2,'end'],
        ]
    
    ]
    
    for path in paths:
        x = FindShortestPath(path)
#        x.width
#        x.height
#        x.cache
#        x._ends
        print(x)
        print("\n")




def IntradayRange(px:list):
    output = dict()
    _inner_range = range(0,len(px)-1,1) #buy
    _outer_range = range(1,len(px),1) #sell

    for buy_time in _inner_range:
        for sell_time in _outer_range:
            if buy_time < sell_time:
                output[(buy_time,sell_time)] = \
                    px[sell_time] - px[buy_time]

    output = sorted(output.items(), key=lambda x: x[1])
    most_profitable = output.pop()
    print("To max profit, buy time = {0}, sell time = {1}, profit = {2}"\
          .format(most_profitable[0][0],
                  most_profitable[0][1],
                  most_profitable[1]))

if __name__ == "__main__":
    px = [26,5,10,12,5,23,8,2,5,20,21,1]
    a=IntradayRange(px)



import numpy as np
px = [26,5,10,12,5,23,8,2,5,20,21,1]
np.mean(px)


x=[3,2,1,4,7]
[i*2 for i in x if i%2==0]

https://codeburst.io/building-an-api-with-django-rest-framework-and-class-based-views-75b369b30396 



