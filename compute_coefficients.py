import numpy as np
from scipy.special import legendre
import matplotlib.pyplot as plt
from scipy.integrate import quad as integrate

def f(theta, l):
    return legendre(l)(np.cos(theta))*np.sin(theta)

def compute_coefficient(l):
    coefficient = (1/2)*(integrate(f,0,np.pi/2,args=(l),epsabs=1e-3)[0]
                        -integrate(f,np.pi/2,np.pi,args=(l),epsabs=1e-3)[0])
    return coefficient

for i in range(7):
    print('a{} ='.format(i), np.round(compute_coefficient(i),decimals=15))
