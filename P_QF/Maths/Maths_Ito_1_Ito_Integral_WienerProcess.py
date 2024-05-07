# -*- coding: utf-8 -*-
"""
@author: lyu

From Wiki ItoIntegralWienerProcess

"""
# A simulation of Ito Integral of a Wiener process with time step dt = .0001
import matplotlib.pyplot as pl
import numpy as np

t0 = 0.0
dt = 0.0001
t_final = 3.9
T = np.arange(t0, t_final, dt)
ax = pl.figure().add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')

x = 0.0
sigma = 100.0

# Ito Integral
np.random.seed(2)
for t in T:
    new_x = x + sigma * np.random.normal(0, dt)
    ax.plot([t, t+dt], [(x*x-t)/2, (new_x*new_x-t-dt)/2], 'b-', linewidth=0.5)
    x = new_x
    # if t>=1.0010:
    #     break
x = 0.0
sigma = 100.0
np.random.seed(2)
new_x2 = [x + sigma * np.random.normal(0, dt) for t in T]
new_xx = np.cumsum(new_x2)
Y = [(new_x*new_x-t)/2 + .1 for t, new_x in zip(T, new_xx)]

ax.plot(T, Y, 'k-', linewidth=1, label="Ito Integral")

# sigma * dW process
x = 0.0
np.random.seed(2)
for t in T:
    new_x = x + sigma * np.random.normal(0, dt)
    ax.plot([t, t+dt], [x, new_x], 'r-', linewidth=.5)
    x = new_x
ax.plot(T, new_xx-.2, 'g--', linewidth=0.5, label="sigma*dW process")
pl.show()
