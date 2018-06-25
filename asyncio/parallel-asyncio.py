# =============================================================================
# https://hackernoon.com/asyncio-for-the-working-python-developer-5c468e6e2e8e
# =============================================================================
import asyncio
import random
import time
from time import sleep

#import gc
#gc.collect()
#gc.garbage


"""
context switch in asyncio represents the event loop yielding the 
flow of control from one coroutine to the next
"""
async def foo():
    print('Running in foo')
    await asyncio.sleep(2) #await gives back control to event loop
    print('Explicit context switch to foo')

async def bar():
    print('Explicit context to bar')
    await asyncio.sleep(2) #working
    print('Implicit context switch to bar')


event_loop = asyncio.get_event_loop()

tasks = [event_loop.create_task( foo() ), 
         event_loop.create_task( bar() )]
wait_tasks = asyncio.wait(tasks)
event_loop.run_until_complete(wait_tasks)
event_loop.close()

[x for x in dir(event_loop) if not x.startswith("_")]
if event_loop.is_closed():
    asyncio.set_event_loop(asyncio.get_event_loop())

asyncio.wait?
"""
Signature: asyncio.wait(fs, *, loop=None, timeout=None, 
                        return_when='ALL_COMPLETED')
Docstring:
Wait for the Futures and coroutines given by fs to complete.

The sequence futures must not be empty.

Coroutines will be wrapped in Tasks.

Returns two sets of Future: (done, pending).

Usage:

    done, pending = yield from asyncio.wait(fs)

Note: This does not raise TimeoutError! Futures that aren't done
when the timeout occurs are returned in the second set.
"""

# =============================================================================
# order of execution - heavy job first?
# =============================================================================
def task(pid):
    """Synchronous non-deterministic task.
    """
    sleep(random.randint(0, 2) * 0.001)
    print('Task %s done' % pid)

async def task_coro(pid):
    """Asynchronous Coroutine non-deterministic task
    """
    await asyncio.sleep(random.randint(0, 2) * 0.001)
    print('Task %s done' % pid)

def synchronous():
    """ linear order of execution """
    for i in range(1, 10):
        task(i)

async def asynchronous():
    tasks = [asyncio.ensure_future(task_coro(i)) for i in range(1, 10)]
    await asyncio.wait(tasks)

synchronous()

""" Random order of execution """
for _ in range(10):
    print("Trial "+ str(_))
    if event_loop.is_closed():
        event_loop = asyncio.new_event_loop()
    event_loop.run_until_complete(asynchronous())
    event_loop.close()
    print("\n")













