# -*- coding: utf-8 -*-
"""
@author: lyu

https://mathworld.wolfram.com/BrentsMethod.html

Find a root of a function in a bracketing interval using Brent’s method.

Uses the classic Brent’s method to find a root of the function f on the sign changing interval [a , b]. Generally considered the best of the rootfinding routines here.

f must be continuous. f(a) and f(b) must have opposite signs.

Brent's method uses a Lagrange interpolating polynomial of degree 2. Brent (1973) claims that this method will always converge as long as the values of the function are computable within a given region containing a root. 

"""
def f(x):
    return (x**2 -1)

from scipy import optimize

root = optimize.brentq(f, -2, 0)
print(root)

root = optimize.brentq(f, 0, 2)
print(root)