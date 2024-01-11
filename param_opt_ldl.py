# -*- coding: utf-8 -*-
"""
Solver for an LQR problem based on KKT system and parametric optimization for computing
the Riccati gains by updating the LDL decomposition of the KKT matrix.

@author: adelpret
"""

import numpy as np
from numpy.linalg import inv
from scipy.linalg import ldl, solve_triangular, solve_banded
from lqr import a2s, LqrProblem, RiccatiSolver

class KktSolverLDL:
    
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
        
        # COMPUTE KKT SYSTEM
        self.D = np.hstack((self.Dx, self.Du))
        self.M = np.zeros((nx+nu+nx, nx+nu+nx))
        self.M[0:nx+nu, 0:nx+nu] = self.H
        self.M[nx+nu:,  0:nx+nu] = self.D
        self.M[0:nx+nu, nx+nu: ] = self.D.T
        self.kkt_vec = np.zeros(nx+nu+nx)
        self.kkt_vec[nx+nu:] = self.D0 @ x0
        
        # SOLVE KKT SYSTEM WITH LDL DECOMPOSITION
#        self.kkt_sol = inv(self.M) @ self.kkt_vec
        (self.L, self.D, self.perm) = ldl(self.M)
        nk = self.M.shape[0]
        self.D_diag_ord = np.zeros((3,nk))
        self.perm_inv = np.zeros(nk, np.int)
        p = self.perm
        for i in range(nk):
            self.perm_inv[p[i]] = i
            for j in range(max(0,i-1), min(nk,i+2)):
                self.D_diag_ord[1 + i - j, j] = self.D[i,j]
        self.kkt_sol = self.solveKKT(self.kkt_vec)
        
        # EXTRACT OPTIMAL STATE AND CONTROL TRAJECTORIES
        X = np.zeros((N+1, n))
        X[0,:] = x0
        X[1:,:] = self.kkt_sol[:nx].reshape((N, n))
        U = self.kkt_sol[nx:nx+nu].reshape((N, m))
        
        return X, U
    
    def solveKKT(self, rhs):
        ''' Solve the KKT system for the specified right hand side, which can be either a vector or a matrix '''
        p = self.perm
#        L[p,:]@D@L[p,:].T = M[p,:][:,p]
        
#        I = np.identity(self.M.shape[0])
#        P = I[p,:]
#        sol = P.T @ inv(self.L.T@P.T) @ inv(self.D) @ inv(P@self.L) @ P @ self.kkt_vec
#        sol = P.T @ inv(self.L.T@P.T) @ inv(self.D) @ inv(self.L[p,:]) @ P @ self.kkt_vec
#        sol = P.T @ inv(self.L[p,:].T) @ inv(self.D) @ inv(self.L[p,:]) @ P @ self.kkt_vec
#        sol = P.T @ inv(self.L[p,:].T) @ inv(self.D) @ inv(self.L[p,:]) @ self.kkt_vec[p]
        
        if(len(rhs.shape)==1):
            Linv_k = solve_triangular(self.L[p,:], rhs[p], lower=True, unit_diagonal=True)
        else:
            Linv_k = solve_triangular(self.L[p,:], rhs[p,:], lower=True, unit_diagonal=True)
#        sol = P.T @ inv(self.L[p,:].T) @ inv(self.D) @ Linv_k
#        Dinv_Linv_k = inv(self.D) @ Linv_k
        Dinv_Linv_k = solve_banded((1,1), self.D_diag_ord, Linv_k)
#        sol = P.T @ inv(self.L[p,:].T) @ Dinv_Linv_k
        LTinv_Dinv_Linv_k = solve_triangular(self.L[p,:], Dinv_Linv_k, lower=True, trans=1, unit_diagonal=True)
#        sol = P.T @ LTinv_Dinv_Linv_k 
        sol = LTinv_Dinv_Linv_k[self.perm_inv]
        return sol
    
    def computeGainsWithLDLupdates(self):
        ''' COMPUTE RICCATI GAINS BY UPDATING THE LDL DECOMPOSITION
            Complexity N^3
            
            It works but it must be optimized and it seems very numerically unstable
        '''
        problem = self.problem
        n, m, N = problem.n, problem.m, problem.N
        nx, nu = N*n, N*m
        
        k = 2*nx+nu
        Im = np.identity(m)
        self.K_ldl = np.zeros((N,m,n))
        
#        Linv_P_A = solve_triangular(self.L, self.P.T[:,nx+nu:nx+nu+n] @ problem.A, lower=True)
#        Minv_A = solve_triangular(self.U, Linv_P_A, lower=False)
        rhs = np.zeros((k,n))
        rhs[nx+nu:nx+nu+n,:] = problem.A
        Minv_A = self.solveKKT(rhs)
        self.K_ldl[0,:,:] = Minv_A[nx:nx+m, :]
#        self.K_ldl[0,:,:] = self.Minv[nx:nx+m, nx+nu:nx+nu+n] @ problem.A
        
        (pim1, pim1_inv, Lim1, Dim1, Dim1_diag_ord) = (
                self.perm, self.perm_inv, self.L, self.D, self.D_diag_ord)
#        Mim1 = self.M
        for i in range(1,N):
            pi = np.zeros(k+i*m, np.int)
            pi_inv = np.zeros(k+i*m, np.int)
            Li = np.zeros((k+i*m, k+i*m))
            Di = np.zeros((k+i*m, k+i*m))
            Di_diag_ord = np.zeros((3, k+i*m))
            
            F = np.zeros((k+(i-1)*m, m))
            F[nx+(i-1)*m:nx+i*m, :] = Im

#            L0 = inv(Dim1) @ inv(Lim1) @ F 
            Linv_F = solve_triangular(Lim1[pim1,:], F[pim1,:], lower=True, unit_diagonal=True)
#            L0 = inv(Dim1) @ Linv_F
            L0 = solve_banded((1,1), Dim1_diag_ord, Linv_F)
            d = -L0.T @ Dim1 @ L0
            
            pi[:-m] = pim1
            pi_inv[:-m] = pim1_inv
            Di[:-m,:-m] = Dim1
            Li[:-m,:-m] = Lim1
            Di_diag_ord[:,:-m] = Dim1_diag_ord
            
            pi[-m:]     = [int(i) for i in range(k+(i-1)*m, k+i*m)]
            pi_inv[-m:] = [int(i) for i in range(k+(i-1)*m, k+i*m)]
            Li[-m:,:-m] = L0.T
            Li[-m:,-m:] = Im
            Di[-m:,-m:] = d
            for ii in range(k+(i-1)*m, k+i*m):
                for jj in range(max(0,ii-1), min(k+i*m,ii+2)):
                    Di_diag_ord[1 + ii - jj, jj] = Di[ii,jj]
            
            # DEBUG
#            Mi = np.zeros((k+i*m, k+i*m))
#            Mi[:k+(i-1)*m,:k+(i-1)*m] = Mim1
#            Mi[-m:, nx+(i-1)*m:nx+i*m] = Im
#            Mi[nx+(i-1)*m:nx+i*m, -m:] = Im
#            Mim1 = Mi
#            if(not np.allclose(Mi - Li @ Di @ Li.T, np.zeros_like(Mi))):
#                print("LDL update failed!", i)
#                print("M\n", Mi)
#                print("Li[i] @ Di[i] @ Li[i].T\n", Li @ Di @ Li.T)
            # END DEBUG
            
            # K = Fu * W * Fx
            # K = Fu * Uinv * Linv * PT * Fx
##            Mi_inv = inv(Ui[i]) @ inv(Li[i]) @ Pi[i].T
#            PT_Fx = Pi.T[:, nx+nu+(i-1)*n:nx+nu+i*n]
##            Linv_PT = solve_triangular(Li[i], Pi[i].T, lower=True)
#            Linv_PT = solve_triangular(Li, PT_Fx, lower=True)
#            Mi_inv = solve_triangular(Ui, Linv_PT, lower=False)
##            self.K_plu[i,:,:] = Mi_inv[nx+i*m:nx+(i+1)*m, nx+nu+(i-1)*n:nx+nu+i*n]
                  
            Fx = np.zeros((k+i*m, n))
            Fx[nx+nu+(i-1)*n:nx+nu+i*n, :] = np.identity(n)
            # TODO: try to exploit the structure of Fx in this solve_triangular
            Linv_k = solve_triangular(Li[pi,:], Fx[pi,:], lower=True, unit_diagonal=True)
            # TODO: try to exploit the structure of D, i.e. block diagonal
            Dinv_Linv_k = solve_banded((1,1), Di_diag_ord, Linv_k)
            LTinv_Dinv_Linv_k = solve_triangular(Li[pi,:], Dinv_Linv_k, lower=True, trans=1, unit_diagonal=True)
            Mi_inv = LTinv_Dinv_Linv_k[pi_inv]
            
            self.K_ldl[i,:,:] = Mi_inv[nx+i*m:nx+(i+1)*m, :]
            
            pim1, pim1_inv, Lim1, Dim1, Dim1_diag_ord = pi, pi_inv, Li, Di, Di_diag_ord
#        print("LDL")
#        for i in range(N):
#            print("K"+str(i)+"\t", self.K_ldl[i,:,:])
        return self.K_ldl
    