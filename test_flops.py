#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 07:14:19 2022

@author: adelprete
"""

def computeFlops(N, n, m):
    flops = 0
    for i in range(0,N-1): # N-1 iters
#        flops += (1./3)*m**3        
        for j in range(i, N): # N-i iters
            for k in range(i+1, j+2): # j+1-i iters
                if(k<=j):
                    flops += m**3
                    flops += m**3
                if(j<N-1):
                    flops += n*m*m
                    flops += m*m*n
    return flops
                    
def computeFlopsTheory(N, n, m):
    flops = 0
    for i in range(0,N-1): # N-1 iters
#        flops += (1./3)*m**3        
#        for j in range(0, N-i): # N-i iters
#            flops += 4*j
        flops += 4*(N-1-i)*(N-1-i)/2
    return flops
#    return (2*N*N - 5*N + 4)*(m**3 + n*m*m/2) + (N*m**3)/3

import numpy as np
import matplotlib.pyplot as plt
import orc.utils.plot_utils as plut
    
n = 1
m = 1
N = np.array(range(1,100))
flops = np.zeros(len(N))
flops_theory = np.zeros(len(N))

for (i,Ni) in enumerate(N):
    flops[i] = computeFlops(Ni, n, m)
    flops_theory[i] = computeFlopsTheory(Ni, n, m)
    
(f, ax) = plut.create_empty_figure(1)
ax.plot(N, flops, label='Flops', alpha=0.7)
ax.plot(N, flops_theory, label='Flops theory', alpha=0.7)
ax.plot(N, [50*(i**2) for i in N], label='N^2', alpha=0.7)
ax.plot(N, [0.9*(i**3) for i in N], label='N^3', alpha=0.7)
ax.set_xlabel('Horizon length N')
ax.set_ylabel('N Flops')
leg = ax.legend()
leg.get_frame().set_alpha(0.5)