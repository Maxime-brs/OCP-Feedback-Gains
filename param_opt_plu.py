# -*- coding: utf-8 -*-
"""
Solver for an LQR problem based on KKT system and parametric optimization for computing
the Riccati gains.

@author: adelpret
"""

import numpy as np
from numpy.linalg import inv
from scipy.linalg import lu, ldl, solve_triangular
from lqr import a2s, LqrProblem, RiccatiSolver

class KktSolverPLU:
    
    def __init__(self, name, problem, DEBUG=False):
        self.name = name
        self.problem = problem
        self.DEBUG = DEBUG
        self.flops = 0
        
    def solve(self, x0):
        problem = self.problem
        n, m, N = problem.n, problem.m, problem.N
        nx, nu = N*n, N*m
        
        # System dynamics can be written either as:
        #   X = G*U + X0
        # or equivalenttly as:
        #   Dx*X + Du*U = D0*x0
        # We use the latter formulation because it is more efficient 
        # as it does not require any inversion.
        self.Dx = np.zeros((nx, nx))
        self.Du = np.zeros((nx, nu))
        self.D0 = np.zeros((nx, n))
        self.H = np.zeros((nx+nu, nx+nu))
        I = np.identity(n)
        
        self.D0[:n,:] = problem.A
        for i in range(N):
            ixm1, ix, ix1 = (i-1)*n, i*n, (i+1)*n
            iu, iu1 = i*m, (i+1)*m
            self.Dx[ix:ix1, ix:ix1] = I
            self.Du[ix:ix1, iu:iu1] = -problem.B
            if(i>0):
                self.Dx[ix:ix1, ixm1:ix] = -problem.A
            
            if(i<N-1):
                self.H[ix:ix1, ix:ix1] = problem.Q
            else:
                self.H[ix:ix1, ix:ix1] = problem.Qf
            self.H[nx+iu:nx+iu1, nx+iu:nx+iu1] = problem.R
        
        # just for debug
#        self.G = np.zeros((nx, nu)) 
#        self.X0 = np.zeros((nx, n))
#        Dx_inv = inv(self.Dx)
#        self.G = -Dx_inv @ self.Du
#        self.X0 = Dx_inv @ self.D0 @ x0
        
        # COMPUTE KKT SYSTEM
        self.D = np.hstack((self.Dx, self.Du))
        self.M = np.zeros((nx+nu+nx, nx+nu+nx))
        self.M[0:nx+nu, 0:nx+nu] = self.H
        self.M[nx+nu:,  0:nx+nu] = self.D
        self.M[0:nx+nu, nx+nu: ] = self.D.T
        self.kkt_vec = np.zeros(nx+nu+nx)
        self.kkt_vec[nx+nu:] = self.D0 @ x0
        
        # SOLVE KKT SYSTEM
#        self.Minv = inv(self.M)
#        self.kkt_sol = self.Minv @ self.kkt_vec
        (self.P, self.L, self.U) = lu(self.M)
        PT_vec = self.P.T @ self.kkt_vec
        Linv_PT_vec = solve_triangular(self.L, PT_vec, lower=True)
        self.kkt_sol = solve_triangular(self.U, Linv_PT_vec, lower=False)
        
        # EXTRACT OPTIMAL STATE AND CONTROL TRAJECTORIES
        X = np.zeros((N+1, n))
        X[0,:] = x0
        X[1:,:] = self.kkt_sol[:nx].reshape((N, n))
        U = self.kkt_sol[nx:nx+nu].reshape((N, m))
        
        return X, U
    
    
    def computeGainsWithPLUupdates(self):
        ''' COMPUTE RICCATI GAINS BY UPDATING THE LU DECOMPOSITION
            Complexity N^3
        '''
        problem = self.problem
        n, m, N = problem.n, problem.m, problem.N
        nx, nu = N*n, N*m
        
        k = 2*nx+nu
        Im = np.identity(m)
        self.K_plu = np.zeros((N,m,n))
        
        Linv_P_A = solve_triangular(self.L, self.P.T[:,nx+nu:nx+nu+n] @ problem.A, lower=True)
        Minv_A = solve_triangular(self.U, Linv_P_A, lower=False)
        self.K_plu[0,:,:] = Minv_A[nx:nx+m, :]
#        self.K_plu[0,:,:] = self.Minv[nx:nx+m, nx+nu:nx+nu+n] @ problem.A
        
#        (self.Pi[0], self.Li[0], self.Ui[0]) = lu(self.M)
        (Pim1, Lim1, Uim1) = (self.P, self.L, self.U)
        
        for i in range(1,N):
            Pi = np.zeros((k+i*m, k+i*m))
            Li = np.zeros((k+i*m, k+i*m))
            Ui = np.zeros((k+i*m, k+i*m))
            
            F = np.zeros((k+(i-1)*m, m))
            F[nx+(i-1)*m:nx+i*m, :] = Im

#            U0 = inv(Li-1) @ inv(Pi-1) @ F
            Pinv_F = Pim1[nx+(i-1)*m:nx+i*m,:].T
            U0 = solve_triangular(Lim1, Pinv_F, lower=True)

#            L0 = inv(Ui[i-1].T) @ F
            # TODO: optimize this operation exploiting sparsity of F
            L0 = solve_triangular(Uim1.T, F, lower=True)

#            L1 = -F.T @ inv(Ui[i-1]) @ U0
            Uinv_U0 = solve_triangular(Uim1, U0, lower=False)
            L1 = -Uinv_U0[nx+(i-1)*m:nx+i*m, :]
            
            Pi[:-m,:-m] = Pim1
            Ui[:-m,:-m] = Uim1
            Li[:-m,:-m] = Lim1
            
            Pi[-m:,-m:] = Im
            Li[-m:,:-m] = L0.T
            Li[-m:,-m:] = L1
            Ui[:-m,-m:] = U0
            Ui[-m:,-m:] = Im
            
            # DEBUG
#            if(not np.allclose(self.Mi[i] - Pi[i] @ Li[i] @ Ui[i], np.zeros_like(self.Mi[i]))):
#                print("PLU update failed!", i)
#                print("M\n", self.Mi[i])
#                print("Pi[i] @ Li[i] @ Ui[i]\n", Pi[i] @ Li[i] @ Ui[i])
            
            # K = Fu * W * Fx
            # K = Fu * Uinv * Linv * PT * Fx
#            Mi_inv = inv(Ui[i]) @ inv(Li[i]) @ Pi[i].T
            PT_Fx = Pi.T[:, nx+nu+(i-1)*n:nx+nu+i*n]
#            Linv_PT = solve_triangular(Li[i], Pi[i].T, lower=True)
            Linv_PT = solve_triangular(Li, PT_Fx, lower=True)
            Mi_inv = solve_triangular(Ui, Linv_PT, lower=False)
#            self.K_plu[i,:,:] = Mi_inv[nx+i*m:nx+(i+1)*m, nx+nu+(i-1)*n:nx+nu+i*n]
            self.K_plu[i,:,:] = Mi_inv[nx+i*m:nx+(i+1)*m, :]
            
            Pim1, Lim1, Uim1 = Pi, Li, Ui
#        print("PLU")
#        for i in range(N):
#            print("K"+str(i)+"\t", self.K_plu[i,:,:])
        return self.K_plu
