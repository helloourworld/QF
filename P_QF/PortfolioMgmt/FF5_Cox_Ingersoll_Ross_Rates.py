# https://quant-trading.co/montecarlo-simulation-for-cox-ingersoll-ross-cir-process/?ct=t(EMAIL_CAMPAIGN_7_12_2024_8_0_COPY_01)&mc_cid=147ff74aa2&mc_eid=d966a09ccf

# a montecarlo simulation for a Cox,Ingersoll, Ross – CIR process in python. Montecarlo simulation is a powerful technique
# that allows you visualize different paths a financial asset could take in the future. You can also use this technique
# for derivatives pricing. In this notebook we are showing how you can run a montecarlo simulation for a Cox, Ingersoll,
# Ross – CIR process in python. 
# 
# Remember that a CIR process is one of the most used stochastic processes
# in finance to model the behavior of interest rates. If you would like to know more about the mathematic
# of this process please look here MONTECARLO.

from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import scipy.stats
from numpy.random import rand
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# a) The number of simulations that we are going to run, b) The number of steps to use in each path the asset will follow, c) The initial interest rate, d) The time step in the simulation, e) The mean reversion speed, f) The long run mean of interest rates, g) The volatility of interest rates, h) The degrees of freedom.

# Number of paths - We are using 1000 paths for illustrative purposes - The more paths, the more accurate.
M = 1000
N = 250  # Number of steps - We are using daily steps. In one year there are apox 250 working days
r0 = 0.055  # Initial r - interest rate
T = 1  # Time to maturity - We are using one year
delta_t = T/N  # Time step for the simulation.

kappa = 2  # Mean reversion speed

theta = 0.03  # Long run mean
sigma = 0.2  # Volatility

d = 4*theta*kappa/sigma**2  # Degrees of freedom

# Generate uniformly distributed random variables

x = rand(M, N)
x

# Initialize the inverse Chi Square function and generate random variables


# We need to transform the previously generated uniform random variables into Chi Squared variables. We can do that using the inverse Chi Square function as following.

# Non Central Chi Square inverse function
ncx2inv = scipy.stats.distributions.ncx2.ppf

# Create a vector with the initial interest rate


# We need to create a column vector that contains the initial interest rate. The number of rows will be equal to the number of simulations that we are going to run

initial_r = r0*np.ones((M, 1))  # Initial r vector

initial_r[0:5]

# Create a matrix


# We need to store the results from our montecarlo simulation, so we create a matrix with M rows and d columns.
# Then we append the previously created interest rate vector to it

r = np.zeros((M, N))  # Declaration of the interest rate matrix (r)
r = np.append(initial_r, r, axis=1)

r

# Create a matrix for w and lamda


# We need to create matrixes for w and lamda, which are auxiliary variables for our calculations

w = np.zeros((M, N))
lamda = np.zeros((M, N))

# Generate the simulation paths


# Using the discretized equation of the geometric brownian motion and the matrix of normal random variables we can create the paths for our simulation as follows

# Calculate the paths

for i in range(0, N):
    coeff1 = 4*kappa
    coeff2 = (sigma**2)*(1-np.exp(-kappa*delta_t))
    coeff3 = np.exp(-kappa*delta_t)

    # Non central parameter according to the notes in Glasserman
    lamda[:, i] = r[:, i]*((coeff1*coeff3)/coeff2)

    # We generate a vector of non central chi squares
    w[:, i] = ncx2inv(x[:, i], d, lamda[:, i])
    # ncx2.ppf(prb, df, nc)-> df:degrees of freedom, nc:Non centrality parameter
    r[:, i+1] = (coeff2/coeff1)*w[:, i]

r

# Plot the paths

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax.plot(r[0:100, :].transpose()*100, c='silver')
ax.plot(r[99:100, :].transpose()*100, c='darkblue')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f%%'))
ax.set_facecolor('white')

plt.yticks(fontname="Tahoma", fontsize=15)
plt.xticks(fontname="Tahoma", fontsize=15)
plt.ylabel("Interest Rate", fontsize=15)
plt.xlabel("Business Days", fontsize=15)


plt.title("Cox, Ingersoll, Ross - Montecarlo Simulation",
          size=25, family='Tahoma')
plt.box(on=None)


plt.subplots_adjust(bottom=0.1, right=1.8, top=1.0)
plt.show()
