import numpy as np
import matplotlib.pyplot as plt

"""
The other parameters you can configure are, with their defaults

left = 0.125
the left side of the subplots of the figure
right = 0.9
the right side of the subplots of the figure
bottom = 0.1
the bottom of the subplots of the figure
top = 0.9
the top of the subplots of the figure
wspace = 0.2
the amount of width reserved for blank space between subplots, expressed as a fraction of the average axis width
hspace = 0.2
the amount of height reserved for white space between subplots, expressed as a fraction of the average axis height
"""

x = np.random.rand(20)
y = np.random.rand(20)

fig, axes = plt.subplots(nrows=4, ncols=3)

fig = plt.figure(figsize=(6.5,12))
fig.subplots_adjust(wspace=1.5,hspace=1.5)

    
iplot = 420
for i in range(7):
    iplot += 1
    ax = fig.add_subplot(iplot)

    ax.plot(x,y,'ko')
    ax.set_xlabel("x")
    ax.set_ylabel("y")

plt.savefig("subplots_example.pdf",bbox_inches='tight')
    

    
    
    
    