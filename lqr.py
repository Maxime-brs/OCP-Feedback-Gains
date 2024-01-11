# -*- coding: utf-8 -*-
"""
Generic LQR solver class.

@author: adelpret
"""

import numpy as np
from numpy.linalg import inv

def a2s(a, format_string ='{0:.2f} '):
    ''' array to string '''
    if(len(a.shape)==0):
        return format_string.format(a);

    if(len(a.shape)==1):
        res = '[';
        for i in range(a.shape[0]):
            res += format_string.format(a[i]);
        return res+']';
        
    res = '[[';
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            res += format_string.format(a[i,j]);
        res = res[:-1]+'] [';
    return res[:-2]+']'; 
            
            
class LqrProblem:
    def __init__(self, x0, A, B, Q, R, Qf, N):
        # the task is defined by a quadratic cost: 
        # sum_{i=0}^N-1 0.5 x' Q x +  0.5 u' R u + 0.5 x' Qf x
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.N = N
        
        self.n = n = A.shape[0]
        self.m = m = B.shape[1]
        
        
    def f(self, x, u, i=0):
        return self.A@x + self.B@u


class RiccatiSolver:
    def __init__(self, name, problem, DEBUG=False):
        self.name = name
        self.problem = problem
        self.DEBUG = DEBUG
        
    ''' Simulate system forward with computed control law '''
    def simulate_system(self, x0, U_bar, KK):
        n = x0.shape[0]
        m = U_bar.shape[1]
        N = U_bar.shape[0]
        X = np.zeros((N+1, n))
        U = np.zeros((N, m))
        X[0,:] = x0
        for i in range(N):
            U[i,:] = U_bar[i,:] + KK[i,:,:] @ X[i,:]
            X[i+1,:] = self.problem.f(X[i,:], U[i,:], i)
        return (X,U)
        
        
    def solve(self, x0):
        problem = self.problem
        n = problem.n
        m = problem.m
        N = problem.N
        rx = list(range(0,n))
        ru = list(range(0,m))
        
        self.kk  = np.zeros((N,m))      # feedforward control inputs
        self.KK  = np.zeros((N,m,n))    # feedback gains

        # derivatives of the cost function w.r.t. x and u
        self.l_x = np.zeros((N+1, n))
        self.l_xx = np.zeros((N+1, n, n))
        self.l_u = np.zeros((N, m))
        self.l_uu = np.zeros((N, m, m))
        self.l_xu = np.zeros((N, n, m))
        
        # the cost-to-go is defined by a quadratic function: 0.5 x' Q_{xx,i} x + Q_{x,i} x + ...
        self.Q_xx = np.zeros((N, n, n))
        self.Q_x  = np.zeros((N, n))
        self.Q_uu = np.zeros((N, m, m))
        self.Q_u  = np.zeros((N, m))
        self.Q_xu = np.zeros((N, n, m))
        
        # the Value function is defined by a quadratic function: 0.5 x' V_{xx,i} x + V_{x,i} x
        V_xx = np.zeros((N+1, n, n))
        V_x  = np.zeros((N+1, n))
        
        # dynamics derivatives w.r.t. x and u
        A = np.zeros((N, n, n))
        B = np.zeros((N, n, m))
        
        # initialize value function
        self.l_x[-1,:]  = 0.0
        self.l_xx[-1,:,:] = problem.Qf
        V_xx[N,:,:] = self.l_xx[N,:,:]
        V_x[N,:]    = self.l_x[N,:]
        
        for i in range(N-1, -1, -1):
            if(self.DEBUG):
                print("\n *** Time step %d ***" % i)
                
            # compute dynamics Jacobians
            A[i,:,:] = problem.A
            B[i,:,:] = problem.B
                
            # compute the gradient of the cost function
#            self.l_x[i,:]    = self.cost_running_x(i, X_bar[i,:], U_bar[i,:])
            self.l_xx[i,:,:] = problem.Q
#            self.l_u[i,:]    = self.cost_running_u(i, X_bar[i,:], U_bar[i,:])
            self.l_uu[i,:,:] = problem.R
#            self.l_xu[i,:,:] = self.cost_running_xu(i, X_bar[i,:], U_bar[i,:])
            
            # compute regularized cost-to-go
            self.Q_x[i,:]     = self.l_x[i,:] + A[i,:,:].T @ V_x[i+1,:]
            self.Q_u[i,:]     = self.l_u[i,:] + B[i,:,:].T @ V_x[i+1,:]
            self.Q_xx[i,:,:]  = self.l_xx[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ A[i,:,:]
            self.Q_uu[i,:,:]  = self.l_uu[i,:,:] + B[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
            self.Q_xu[i,:,:]  = self.l_xu[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
            
            if(self.DEBUG):
                print("Q_x, Q_u, Q_xx, Q_uu, Q_xu", a2s(self.Q_x[i,rx]), a2s(self.Q_u[i,ru]), 
                        a2s(self.Q_xx[i,rx,:]), a2s(self.Q_uu[i,ru,:]), a2s(self.Q_xu[i,rx,0]))
                
            Qbar_uu       = self.Q_uu[i,:,:] #+ mu*np.identity(m)
            Qbar_uu_pinv  = np.linalg.inv(Qbar_uu)
            self.kk[i,:]       = - Qbar_uu_pinv @ self.Q_u[i,:]
            self.KK[i,:,:]     = - Qbar_uu_pinv @ self.Q_xu[i,:,:].T
            if(self.DEBUG):
                print("Qbar_uu, Qbar_uu_pinv",a2s(Qbar_uu), a2s(Qbar_uu_pinv));
                print("kk, KK", a2s(self.kk[i,ru]), a2s(self.KK[i,ru,rx]));
                
            # update Value function
            V_x[i,:]    = (self.Q_x[i,:] + 
                self.KK[i,:,:].T @ self.Q_u[i,:] +
                self.KK[i,:,:].T @ self.Q_uu[i,:,:] @ self.kk[i,:] +
                self.Q_xu[i,:,:] @ self.kk[i,:])
            V_xx[i,:]   = (self.Q_xx[i,:,:] +
                self.KK[i,:,:].T @ self.Q_uu[i,:,:] @ self.KK[i,:,:] +
                self.Q_xu[i,:,:] @ self.KK[i,:,:] + 
                self.KK[i,:,:].T @ self.Q_xu[i,:,:].T)
                
            # V_x[i,:]    = self.Q_x[i,:]  - self.Q_xu[i,:,:] @ Qbar_uu_pinv @ self.Q_u[i,:]
            # V_xx[i,:]   = self.Q_xx[i,:] - self.Q_xu[i,:,:] @ Qbar_uu_pinv @ self.Q_xu[i,:,:].T
                    
        (X, U) = self.simulate_system(x0, self.kk, self.KK)
        return (X, U, self.KK)

    