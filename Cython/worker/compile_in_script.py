import pyximport
pyximport.install()

import worker
print(worker) #worker.pyd not worker.py
worker.HardWorker
worker.HardWorker(worker.add_simple_stuff).work_hard()

