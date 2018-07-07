# =============================================================================
# 
# =============================================================================
x  =[1,2,5]+ [3,4]
x.sort()

# =============================================================================
# 
# =============================================================================
a = [[1,2],
     [3,4]]


findMatrix(a)


def findMatrix(a):
    n_rows = len(a)
    n_cols = len(a[0])
    for row in range(n_rows):
        for col in range(n_cols):
            if row == 0 and col == 0:
                pass
            elif row == 0 and col!= 0:
                a[row][col] += a[row][col-1]
            elif row != 0 and col == 0:
                a[row][col] += a[row-1][col]
            elif row != 0 and col != 0:
                a[row][col] += a[row-1][col] + a[row][col-1] - a[row-1][col-1]
            else:
                pass
    return a


# =============================================================================
# 
# =============================================================================
nodes = "(B,D) (D,E) (A,B) (C,F) (E,G) (A,C)"
#( A(B(D(E(G)))) (C(F)) )

nodes = "(B,D) (D,E) (A,C) (C,F) (E,G) (A,C)"


def SExpression(nodes):
    x = nodes.split(" ")
    x = [k.lstrip("(").rstrip(")") for k in x]
    x = [(k[0],k[2]) if k[0]<k[2] else (k[2],k[0]) for k in x] # sort order
    x.sort()
    
    # > 2 children
    counter = dict()
    for parent, child in x:
        if parent in counter:
            counter[parent] += 1
        else:
            counter[parent] = 1
    if any([k>2 for k in list(counter.values())]):
        return "E1"
    
    # duplicate edges
    if len(set(x)) < len(x):
        return "E2"

    root = x[0][0]

    if x[0][0] == x[1][0]: 
        root_binary = True
    else:
        root_binary = False

    if root_binary:
        _x1 = [x[0]]
        x.remove(_x1[0])
        _x2 = [x[0]]
        x.remove(_x2[0])
        
        for parent, child in x:
            if parent == _x1[len(_x1)-1][1]:
                _x1.append((parent,child))
            elif parent == _x2[len(_x2)-1][1]:
                _x2.append((parent,child))
            else:
                pass
        
        branch1 = root + "(" + "(".join([k[1] for k in _x1]) + ")"*len(_x1)
        branch2 = "(" + "(".join([k[1] for k in _x2]) + ")"*len(_x2)    
        
        
        children1= [k[1] for k in _x1]
        children2= [k[1] for k in _x2]
                
        # cycle detected
        if set(children1) in set(children2):
            return "E3"
        
        return "(" + branch1 + branch2 + ")"
        
    else:
        _x1 = [x[0]]
        x.remove(_x1[0])
        for parent, child in x:
            if parent == _x1[len(_x1)-1][1]:
                _x1.append((parent,child))
            else:
                pass
        branch1 = root + "(" + "(".join([k[1] for k in _x1]) + ")"*len(_x1)
        return "(" + branch1 + ")"
        

