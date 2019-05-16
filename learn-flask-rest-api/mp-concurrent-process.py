"""
Process.start() works outside IDLE
"""

import multiprocessing

class Summarize( multiprocessing.Process ):
    """ fetch items from queue and count them """
    def __init__( self, queue ):
        self.queue= queue
        super().__init__()
    def run( self ):
        """Waits for a termination"""
        print( self.__class__.__name__, "start" )
        count= 0
        item= self.queue.get()
        while item != (None, None, None):
            print( item )
            count += 1
            item= self.queue.get()
        print( self.__class__.__name__, "finish", count )


if __name__ == "__main__":

    result_queue = multiprocessing.SimpleQueue()

    for i in range(100):
        result_queue.put( (i,i+1,i+2) )

    """ start 4 concurrent processes """
    # for orderly termination
    for i in range(4):
        result_queue.put( (None, None, None) )
    
    multi_processes = []
    for i in range(4):
        one_process = Summarize( result_queue )
        multi_processes.append(one_process)

    for process in multi_processes:
        process.start()
    
    # wait for the processes to finish execution and join back into the parent process
    for process in multi_processes:
        process.join()







