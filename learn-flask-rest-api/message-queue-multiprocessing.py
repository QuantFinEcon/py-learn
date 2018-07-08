
"""
multiprocessing use pickling to serialise objects for transmission. 
Pickling is not the fastest serialisation technique.

We can use queues and pipes to serialize objects that are then transmitted to other
processes. pickle that object so that it is transmitted via the queue to another process

one important design consideration when using multiprocessing: it's
generally best to avoid having multiple processes (or multiple threads) attempting to
update shared objects. synchronization and locking issues

Using process-level synchronization via RESTful web services or multiprocessing
can prevent synchronization issues because there are no shared objects. 

design principle is to look at the processing as a pipeline of discrete steps.
Each processing step will have an input queue and an output queue; 
the step will fetch an object, perform some processing, and write the object.

The multiprocessing philosophy matches the POSIX concept of a shell pipeline
written as process1 | process2 | process3. This kind of shell pipeline involves
three concurrent processes interconnected with pipes. The important difference is
that we don't need to use STDIN, STDOUT, and explicit serialization of the objects.
We can trust the multiprocessing module to handle the OS-level infrastructure.

The POSIX shell pipelines are limited, in that each pipe has a single producer and a
single consumer. The Python multiprocessing module allows us to create message
queues that include multiple consumers. This allows us to have a pipeline that
fans out from one source process to multiple sink processes.

To maximize throughput on a given computer system, we need to have enough
work pending so that no processor or core is ever left with nothing useful to do.
When any given OS process is waiting for a resource, at least one other process
should be ready to run. 

gets a request from a queue, processes that request, and places the 
results into another queue. decomposes the larger problem into a number of 
stages that form a pipeline. stages will run concurrently. independent queues. 
no issue with complex locking or shared resources. 


For the simulation of our casino game, we can break the simulation down into a
three-step pipeline:
    1. An overall driver puts simulation requests into a processing queue.
    2. A "pool" of simulators will get a request from the processing queue, perform
        the simulation, and put the statistics into a results queue.
    3. A summarizer will get the results from the results queue and create a final
        tabulation of the results.

Using a process pool allows us to have as many simulations running concurrently
as our CPU can handle

"""

import multiprocessing
multiprocessing.Process?

#dir(multiprocessing.Process)
multiprocessing.Process.start?
multiprocessing.Process.run?
# wait for all child processes to finish before joining back to parent
multiprocessing.Process.join?
multiprocessing.Process.terminate?
multiprocessing.Process.pid?
multiprocessing.Process.is_alive?
multiprocessing.Process.sentinel?

# 'authkey',
# 'daemon',
# 'exitcode',
# 'ident',
# 'name',

import threading
threading.Thread?

class Simulation( multiprocessing.Process ):
    """ defining processes as subclasses of 
        multiprocessing.Process """

    def __init__( self, setup_queue, result_queue ):
        """ each instance of Process has a 
            setup_queue --> result_queue """
        self.setup_queue = setup_queue # tuples of (table, player)
        self.result_queue = result_queue
        super().__init__()

    def run( self ):
        """
        Must Override to interact with states
        Process.start() --> Proces.run() --> wait for termination
            from sentinel object (None, None)
        GET from setup_queue. 
        POST to result_queue 
        """
        print( self.__class__.__name__, "start" )
        
        simulation_counts = 0
        item = self.setup_queue.get()
        while item != (None,None):
            
            table, player = item            
            self.sim = Simulate( table, player, samples=1 )
            results = list( self.sim )
            
            self.result_queue.put( (table, player, results[0]) )
            item = self.setup_queue.get()
            simulation_counts += 1

        print( self.__class__.__name__, "finish", simulation_counts )

class Simulate:
    def __init__( self, table, player, samples ):
            pass
    def __iter__( self ): 
        yields summaries

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

"""
build queues that define the processing pipeline to transfer data
FIFO - sentinel object (None,None,..) last to put
"""
multiprocessing.Queue?
#setup_queue = multiprocessing.SimpleQueue()
result_queue = multiprocessing.SimpleQueue()

result_queue.get?
result_queue.put?
result_queue.empty?

""" start single process """
for i in range(100):
    result_queue.put( (i,i+1,i+2) )
result_queue.put( (None, None, None) )

result = Summarize( result_queue )
result.start() # WILL NOT WORK IN IDLE INTERPRETER
result.run()
result.terminate()

""" start 4 concurrent processes """
for i in range(100):
    result_queue.put( (i,i+1,i+2) )

multi_processes = []
for i in range(4):
    one_process = Summarize( result_queue )
    one_process.start()
    multi_processes.append(one_process)

# for orderly termination
for process in multi_processes:
    result_queue.put( (None, None, None) )

# wait for the processes to finish execution and join back into the parent process
for process in multi_processes:
    process.join()

#multi_processes[0].run()



