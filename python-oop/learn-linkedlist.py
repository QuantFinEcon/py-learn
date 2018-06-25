# -*- coding: utf-8 -*-
"""
linked-list
"""

class Node:
    def __init__( self, data, prevNode, nextNode ):
        self.data = data
        self.prevNode = prevNode
        self.nextNode = nextNode

class DoublyLinkedList:
              
    def __init__( self, data=None ):
        self.first = None
        self.last = None
        self.count = 0
    
    def addFirst( self, data ):
        if self.count == 0:
            self.first = Node( data, None, None )
            self.last = self.first
        elif self.count > 0:
            # create a new node pointing to self.first
            node = Node( data, None, self.first )
            # have self.first point back to the new node
            self.first.prevNode = node
            # finally point to the new node as the self.first
            self.first = node
        self.current = self.first
        self.count += 1
    
    def popFirst( self ):
        if self.count == 0:
            raise RuntimeError("Cannot pop from an empty linked list")
        result = self.first.data
        if self.count == 1:
            self.first = None
            self.last = None
        else:
            self.first = self.first.nextNode
            self.first.prevNode = None
        self.current = self.first
        self.count -= 1
        return result
    
    def popLast( self ):
        if self.count == 0:
            raise RuntimeError("Cannot pop from an empty linked list")
        result = self.last.data
        if self.count == 1:
            self.first = None
            self.last = None
        else:
            self.last = self.last.prevNode
            self.last.nextNode = None
        self.count -= 1
        return result
    
    def addLast( self, data ):
        if self.count == 0:
            self.addFirst(0)
        else:
            self.last.nextNode = Node( data, self.last, None )
            self.last = self.last.nextNode
            self.count += 1
    
    def __repr__( self ):
        result = ""
        if self.count == 0:
            return "..."
        cursor = self.first
        for i in range( self.count ):
            result += "{}".format(cursor.data)
            cursor = cursor.nextNode
            if cursor is not None:
                result += " --- "
        return result
    
    def __iter__( self ):
        # lets Python know this class is iterable
        return self
    
    def next( self ):
        # provides things iterating over class with next element
        if self.current is None:
            # allows us to re-iterate
            self.current = self.first
            raise StopIteration
        else:
            result = self.current.data
            self.current = self.current.nextNode
            return result
    
    def __len__( self ):
        return self.count
     

dll = DoublyLinkedList()
dll.addFirst("days")
dll.addFirst("dog")
dll.addLast("of summer")
 
assert list( dll ) == ["dog", "days", "of summer" ]
assert dll.popFirst() == "dog"
assert list( dll ) == ["days", "of summer"]
assert dll.popLast() == "of summer"
assert list( dll ) == ["days"]

# =============================================================================
# 
# =============================================================================

class Node():
    def __init__(self, next_node=None, previous_node=None, data=None):
        self.next_node = next_node
        self.previous_node = previous_node
        self.data = data



class LinkedList():
    def __init__(self, node):
        assert isinstance(node, Node)
        self.first_node = node
        self.last_node = node

    def push(self, node):
        '''Pushes the node <node> at the "front" of the ll
        '''
        node.next_node = self.first_node
        node.previous_node = None
        self.first_node.previous_node = node
        self.first_node = node

    def pop(self):
        '''Pops the last node out of the list'''
        old_last_node = self.last_node
        to_be_last = self.last_node.previous_node
        to_be_last.next_node = None
        old_last_node.previous_node = None

        # Set the last node to the "to_be_last"
        self.previous_node = to_be_last

        return old_last_node

    def remove(self, node):
        '''Removes and returns node, and connects the previous and next
        nicely
        '''
        next_node = node.next_node
        previous_node = node.previous_node

        previous_node.next_node = next_node
        next_node.previous_node = previous_node

        # Make it "free"
        node.next_node = node.previous_node = None

        return node

    def __str__(self):
        next_node = self.first_node
        s = ""
        while next_node:
            s += "--({:0>2d})--\n".format(next_node.data)
            next_node = next_node.next_node

        return s

        return 


node1 = Node(data=1)

linked_list = LinkedList(node1)


for i in xrange(10):
    if i == 5:
        node5 = Node(data=5)
        linked_list.push(node5)
    else:
        linked_list.push(Node(data=i))

print linked_list

print "popping"
print linked_list.pop().data

print "\n\n"
print linked_list


print "\n\n"
linked_list.push(Node(data=10))

print "\n\n"
print linked_list


linked_list.remove(node5)

print "\n\n"
print linked_list

