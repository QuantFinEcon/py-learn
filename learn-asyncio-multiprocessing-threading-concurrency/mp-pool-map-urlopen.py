
from multiprocessing import Pool
from urllib import request
from timeit import default_timer as timer


def http_get(url):
#    print("opening {url}".format(url=url))
    result = {"url": url, 
              "data": request.urlopen(url, timeout=5).read()}
    return result

#urllib.request.urlopen('http://www.google.com/',timeout=5).read()

if __name__ == "__main__":

    urls = ['http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/',
            'http://www.google.com/','http://www.google.com/','http://www.google.com/']
    
    # =============================================================================
    # timeit
    # =============================================================================
    start = timer()
    
    results = [http_get(url) for url in urls]
#    print(results)
    
    time_taken = timer() - start
    print("sequential time: {0:.5f}ms".format(time_taken))


    with Pool(processes=8) as pool:

        start = timer()
        
        results = pool.map(http_get, urls, chunksize=5)
        pool.close()
        pool.join()
    #    print(results)
        
        time_taken = timer() - start
        print("parallel time: {0:.5f}ms".format(time_taken))
    
