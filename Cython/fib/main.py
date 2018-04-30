import os
os.getcwd()
# os.chdir('C:\\Users\\yeoshuiming\\Dropbox\\GitHub\\py-learn\\Cython')
import datetime


start = datetime.datetime.now()
import fib
fib.fib(10000000)
print("Time Taken: "+ str(datetime.datetime.now() - start))

def fib2(n):
    """Print the Fibonacci series up to n."""
    a, b = 0, 1
    while b < n:
        print(b, end=' ')
        a, b = b, a + b

    print()

start = datetime.datetime.now()
fib2(10000000)
print("Time Taken: "+ str(datetime.datetime.now() - start))
