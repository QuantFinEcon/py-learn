#!/usr/bin/env python -

# Examples on Granger-causality, written by Harold W. Gutch, June 2009
#
# Used in HG's talk for the time series seminar at the MPI DS in June
# 2009, organized (among others) by J. Nagler, A. Witt and L. Anand
#
# 
# I am sure a lot of the stuff here is redundant, can be written more
# nicely etc.  I might do so some day.  Really, all of this is very simple
# stuff, so this task shouldn't take more than at most an hour or two
#

import numpy as n

fullsize = 10000
windowsize = 1000

# 2 uniform sources
X = n.random.randn(fullsize)
X = X - sum(X)/fullsize
Y = n.random.randn(fullsize)
Y = Y - sum(Y)/fullsize

Z = Y.copy();
Z[1:] = Z[1:] + X[0:-1]

# so: X influences Z with timelag 1


covarXY = n.zeros(fullsize-windowsize)

for i in range(0,fullsize-windowsize):
    EX = X[i:i+windowsize]
    EX2 = sum(EX*EX)/(windowsize+1)
    EX = sum(EX)/(windowsize+1)
    EZ = Z[i:i+windowsize]
    EZ2 = sum(EZ*EZ)/(windowsize+1)
    EZ = sum(EZ)/(windowsize+1)

    covarXY[i] = (sum(X[i:i+windowsize]*Z[i:i+windowsize])/(windowsize+1) - EX*EZ)/(np.sqrt((EX2-EX*EX)*(EZ2-EZ*EZ)))


figure()
plot(X[0:100], 'g')

plot(Z[0:100], 'r')
plot(covarXY[0:100], 'b')
maxval = max(max(X[0:100]),max(Z[0:100]))
minval = min(min(X[0:100]),min(Z[0:100]))

matplotlib.pyplot.text(-10,(maxval-minval)*0.9+minval,'X',color='g')
matplotlib.pyplot.text(-10,(maxval-minval)*0.8+minval,'Y',color='r')
matplotlib.pyplot.text(-15,(maxval-minval)*0.7+minval,'Cov(X,Y)',color='g')
xlabel('time')
A = sum(Z**2)/fullsize
B = sum(Z[0:-1]*Z[1:])/(fullsize-1);
C = sum(Z[0:-2]*Z[2:])/(fullsize-2);

a2 = (2*A*C-B)/(4*A**2-B**2);
a1 = (1-a2)*B/(2*A);

windowpredictionerror = n.zeros(fullsize-2)
windowpredictionerrorcause = covarXY = n.zeros(fullsize-windowsize-2)

windowpredictionerror = Z[2:] - a1*Z[1:-1] - a2*Z[:-2]
windowpredictionerrorcause = windowpredictionerror - X[1:-1]

wpeplot = n.zeros(fullsize-2-windowsize+1)
wpecplot = n.zeros(fullsize-2-windowsize+1)

for i in range(0,fullsize-2-windowsize):
    wpemean = sum((windowpredictionerror[i:i+windowsize]))/(windowsize+1)
    wpeplot[i] = sum((windowpredictionerror[i:i+windowsize] - wpemean)**2)/(windowsize+1)

    wpemean = sum((windowpredictionerrorcause[i:i+windowsize]))/(windowsize+1)
    wpecplot[i] = sum((windowpredictionerrorcause[i:i+windowsize] - wpemean)**2)/(windowsize+1)


figure()
plot(wpeplot[0:100], 'r')
plot(wpecplot[0:100], 'g')
maxval = max(max(wpeplot[0:100]), max(wpecplot[0:100]))
minval = min(min(wpeplot[0:100]), min(wpecplot[0:100]))
matplotlib.pyplot.text(-10,(maxval-minval)*1.1+minval,'prediction without X',color='r')
matplotlib.pyplot.text(-10,(maxval-minval)*(-0.1)+minval,'prediction with X',color='g')

predictionerror = n.zeros(fullsize-2)
predictionerror = Z[2:] - a1*Z[1:-1] - a2*Z[:-2]
variance = sum(predictionerror**2)/(fullsize-2) - (sum(predictionerror)/(fullsize-2))**2
print variance

predictionerror = predictionerror - X[1:-1]
variance = sum(predictionerror**2)/(fullsize-2) - (sum(predictionerror)/(fullsize-2))**2
print variance

