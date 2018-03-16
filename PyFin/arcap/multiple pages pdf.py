"""
This is a demo of creating a pdf file with several pages,
as well as adding metadata and annotations to pdf files.
"""

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('multipage_pdf.pdf') as pdf:
    plt.figure(figsize=(3, 3))
    plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
    plt.title('Page One')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    plt.rc('text', usetex=True)
    plt.figure(figsize=(8, 6))
    x = np.arange(0, 5, 0.1)
    plt.plot(x, np.sin(x), 'b-')
    plt.title('Page Two')
    pdf.attach_note("plot of sin(x)")  # you can add a pdf note to
                                       # attach metadata to a page
    pdf.savefig()
    plt.close()

    plt.rc('text', usetex=False)
    fig = plt.figure(figsize=(4, 5))
    plt.plot(x, x*x, 'ko')
    plt.title('Page Three')
    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
    plt.close()

    # We can also set the file's metadata via the PdfPages object:
    d = pdf.infodict()
    d['Title'] = 'Multipage PDF Example'
    d['Author'] = u'Jouni K. Sepp\xe4nen'
    d['Subject'] = 'How to create a multipage pdf file and set its metadata'
    d['Keywords'] = 'PdfPages multipage keywords author title subject'
    d['CreationDate'] = datetime.datetime(2009, 11, 13)
    d['ModDate'] = datetime.datetime.today()
    
#==============================================================================
#     
#==============================================================================

from matplotlib import pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages
 
# The PDF document
pdf_pages = PdfPages('my-fancy-document.pdf')
 
for i in xrange(3):
  # Create a figure instance (ie. a new page)
  fig = plot.figure(figsize=(8.27, 11.69), dpi=100)
 
  x.plot()
 
  # Done with the page
  pdf_pages.savefig(fig)
 
# Write the PDF document to the disk
pdf_pages.close()
#==============================================================================
# 
#==============================================================================

import numpy
from matplotlib import pyplot as plt
 
# Prepare the data
t = numpy.linspace(-numpy.pi, numpy.pi, 1024)
s = numpy.random.randn(2, 256)
 
#
# Do the plot
#
grid_size = (5, 2)
 
# Plot 1
plot.subplot2grid(grid_size, (0, 0), rowspan=2, colspan=2)
plt.plot(t, numpy.sinc(t), c= '#000000')
 
# Plot 2
plot.subplot2grid(grid_size, (2, 0), rowspan=3, colspan=1)
plot.scatter(s[0], s[1], c= '#000000')
 
# Plot 2
plot.subplot2grid(grid_size, (2, 1), rowspan=3, colspan=1)
plot.plot(numpy.sin(2 * t), numpy.cos(0.5 * t), c= '#000000')
 
# Automagically fits things together
plot.tight_layout()
 
# Done !
plot.show()

#==============================================================================
# 
#==============================================================================

import numpy
 
from matplotlib import pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages
 
# Generate the data
data = numpy.random.randn(20, 1024)
data.shape 

# The PDF document
pdf_pages = PdfPages('histograms.pdf')
 
# Generate the pages
nb_plots = data.shape[0]
nrows=5
ncols=2
grid_size = (nrows, ncols) #( #rows, #cols)
nb_plots_per_page = nrows*ncols
nb_pages = int(numpy.ceil(nb_plots / float(nb_plots_per_page)))

 
for i, samples in enumerate(data):
    print i
    print samples
  # Create a figure instance (ie. a new page) if needed
    if i % nb_plots_per_page == 0:
        fig = plot.figure(figsize=(8.27, 5), dpi=100)
 
  # Plot stuffs !
    plot.subplot2grid(grid_size, loc = (i % nb_plots_per_page, i / nrows), colspan=1, rowspan=1)
    plot.hist(samples, 32, normed=1, facecolor='#808080', alpha=0.75)
 
  # Close the page if needed
    if (i + 1) % nb_plots_per_page == 0 or (i + 1) == nb_plots: # terminate pdf-page or pdf
        plot.tight_layout()
        pdf_pages.savefig(fig)
 
# Write the PDF document to the disk
pdf_pages.close()

#==============================================================================
# subplot2grid vss. gridspec
#==============================================================================
ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
ax3 = plt.subplot2grid((3,3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3,3), (2, 0))
ax5 = plt.subplot2grid((3,3), (2, 1))
plt.tight_layout()

import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1,:-1]) # last columns -1
ax3 = plt.subplot(gs[1:, -1])
ax4 = plt.subplot(gs[-1,0]) # last row -1
ax5 = plt.subplot(gs[-1,-2])
plt.tight_layout()





exec("a=1")

def draw_menu(options):
    for counter, option in enumerate(options): # index=key, value
            print(" %s %s" %(counter, option))
draw_menu(data, 2)
data
#==============================================================================
# 
#==============================================================================

import numpy
 
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
 
# Generate the data
data = np.random.randn(20, 20)
data.shape 

nrow = 5
ncol = 2
total = 20
pages = total/(nrow*ncol)
    
# The PDF document
pdf_pages = PdfPages('gridspec_test.pdf')
base = 0
#create one pdf page
for i in xrange(pages):
    plots_per_pg = nrow*ncol
    # Create a figure instance (ie. a new page)
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(nrow, ncol)
    # plot 
    for i in range(base+0,base+nrow*ncol-1,1):
        #        ax = plt.Subplot(fig, inner_grid[j]) OR # a1=plt.subplot(gs[r,c])
        #        ax.plot(*squiggle_xy(a, b, c, d))
        #        ax.set_xticks([])
        #        ax.set_yticks([])
        #        fig.add_subplot(ax)
        exec("a"+str(i)+"="+"data[:,"+str(i)+"]") # extract data
        exec("ax"+str(i)+"="+"plt.subplot("+"gs["+str(i % plots_per_pg)+"]"+")") # choose loc in grid
        exec("ax"+str(i)+"."+"plot("+"a"+str(i)+")") # plot in grid
    base += plots_per_pg
    
    plt.tight_layout()
    # Done with the page
    pdf_pages.savefig(fig)
 
# Write the PDF document to the disk
pdf_pages.close()
    
for i in xrange(10): print i
    
11 % 10
    
    

