
from glob import glob
import multiprocessing
import itertools
import time
import os

## Let's clear cache between each one
## cd libexec/python --> ipython
#
#from sutils import get_cache
#cache = get_cache('docker')
#
## Function to clean cache
#def clean_cache(cache=None):
#    if cache is None:
#        cache = get_cache('docker')
#    files = glob("%s/*" %cache)
#    [os.remove(filey) for filey in files]

# Here is the Singularity Python API client
from docker.api import DockerApiConnection
client = DockerApiConnection(image="ubuntu:latest")
images = client.get_images()

# We will give the multiprocessor a function to download a layer
def download_layer(client,image_id,cache_base,prefix):
    targz = client.get_layer(image_id=image_id,
                             download_folder=cache_base,
                             prefix=prefix)
    client.update_token()
    return targz


class SingularityMulti(object):

    def __init__(self, workers=None):
        '''initialize a worker to run a function with some arguments 
        '''
        if workers is None:
            workers = 4
        self.workers = workers

    def _wrapper(self,func_args):
        function, args = func_args
        return function(*args)

    def _package(self,func, args):
        return zip(itertools.repeat(func), args)

    def run(self,func,tasks):
        start_time = time.time()
        with multiprocessing.Pool(processes=self.workers) as pool:
            results = pool.map(self._wrapper,self._package(func,tasks))
        end_time = time.time()
        self.runtime = end_time - start_time 
        pool.close()

if __name__ == '__main__':
    # Add the calls to the task queue
    tasks = []
    for ii in range(len(images)):
        image_id = images[ii]
        targz = "%s/%s.tar.gz" %(cache,image_id)
        prefix = "[%s/%s] Download" %((ii+1),len(images))
        tasks.append((client,image_id,cache,prefix))



# =============================================================================
# parallel lookup
# =============================================================================
#https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

import multiprocessing
from multiprocessing import Process, Pool
import urllib
import timeit
import random

def negative(i): 
    return -1*i
list_of_ints = [random.randint(0,100) for _ in range(50)]

multiprocessing.cpu_count()
pool = Pool(processes=4)
results = pool.map(negative, list_of_ints)

pool.map?


# =============================================================================
# 
# =============================================================================

def http_get(url):  
    print("opening {url}".format(url=url))
    result = {"url": url, 
              "data": urllib.request.urlopen(url, timeout=5).read()}
    return result

urllib.request.urlopen('http://www.google.com/',timeout=5).read()

urls = ['http://www.google.com/','http://www.google.com/','http://www.google.com/',
        'http://www.google.com/','http://www.google.com/','http://www.google.com/',
        'http://www.google.com/','http://www.google.com/','http://www.google.com/']


pool = Pool(processes=4)

timeit.timeit("",
              setup="""
              
              """)

results = pool.map(http_get, urls)







