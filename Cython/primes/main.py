import os
print(os.getcwd())
import primes
import datetime

start = datetime.datetime.now()
primes.primes(1000)
print("Time Taken: "+ str(datetime.datetime.now() - start))

start = datetime.datetime.now()
primes.primes_python(1000)
print("Time Taken: "+ str(datetime.datetime.now() - start))


