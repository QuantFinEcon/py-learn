#
# Monte Carlo valuation of European call options with NumPy
# mcs_vector_numpy.py
#
import math
import numpy as np
from time import time

np.random.seed(20000)
t0 = time()

# Parameters
S0 = 100.; K = 105.; T = 1.0; r = 0.05; sigma = 0.2
M = 50; dt = T / M; I = 250000

#==============================================================================
# # Simulating I paths with M time steps
# S = np.zeros((M + 1, I))
# S[0] = S0
# for t in range(1, M + 1):
#    z = np.random.standard_normal(I) # pseudorandom numbers
#    S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z)
# 
#==============================================================================

# Simulating I paths with M time steps (LOG VERSION)
z = np.random.standard_normal((M + 1, I))
S = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z, axis=0))
# sum instead of cumsum would also do if only the final values are of interest
S[0] = S0

# vectorized operation per time step over all paths
# Calculating the Monte Carlo estimator
C0 = math.exp(-r * T) * np.sum(np.maximum(S[-1] - K, 0)) / I

# Results output
tnp1 = time() - t0
print "European Option Value %7.3f" % C0
print "Duration in Seconds %7.3f" % tnp1

import matplotlib.pyplot as plt
plt.plot(S[:, :10])
plt.grid(True)
plt.xlabel('time step')
plt.ylabel('index level')

plt.hist(S[-1], bins=50) # end of period S values
plt.grid(True)
plt.xlabel('index level')
plt.ylabel('frequency')

plt.hist(np.maximum(S[-1] - K, 0), bins=50) # end of period F values
plt.grid(True)
plt.xlabel('option inner value')
plt.ylabel('frequency')
plt.ylim(0, 50000)

winprob = float(np.count_nonzero(np.maximum(S[-1] - K, 0)))/np.shape(S)[1]
print r"probability of option ending in-the-money is %3.3f%%" % (winprob*100)
