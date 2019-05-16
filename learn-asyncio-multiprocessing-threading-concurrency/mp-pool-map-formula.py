from multiprocessing import Pool
import random
from timeit import default_timer as timer

def formula(i): 
    return (i**5)/7.8*12.7

if __name__ == "__main__":
    
    list_of_ints = [random.randint(0,10000) for _ in range(10000)]
    
    # =============================================================================
    # timeit
    # =============================================================================
    start = timer()
    
    results = list(map(formula, list_of_ints))
#    print(results)
    
    time_taken = timer() - start
    print("sequential time: {0:.5f}ms".format(time_taken))

    with Pool(processes=8) as pool:

        start = timer()
        
        results = pool.map(formula, list_of_ints, chunksize=512)
        pool.close()
        pool.join()
    #    print(results)
        
        time_taken = timer() - start
        print("parallel time: {0:.5f}ms".format(time_taken))
