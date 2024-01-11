# -*- coding: utf-8 -*-
"""
Solver for an LQR problem based on a recursive formula for the inverse of the 
augmented KKT system for computing the Riccati gains.

@author: adelpret
"""

import numpy as np
from numpy.linalg import inv
from scipy.linalg import lu, ldl, solve_triangular
from lqr import a2s, LqrProblem, RiccatiSolver

class KktSolverAugRec:
    
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
    
    
    def computeGainsWithAugKkt(self):
        problem = self.problem
        n, m, N = problem.n, problem.m, problem.N
        nx, nu = N*n, N*m
        
        # COMPUTE RICCATI GAINS INVERTING AUGMENTED KKT MATRIX
        # This works but is inefficient, complexity N^4
        k = 2*nx+nu
        Im = np.identity(m)
#        self.Mi = N*[None,] 
#        self.Mi[0] = self.M
        Mim1 = self.M
        self.K_augkkt = np.zeros((N,m,n))
                
        Linv_P_A = solve_triangular(self.L, self.P.T[:,nx+nu:nx+nu+n] @ problem.A, lower=True)
        Minv_A = solve_triangular(self.U, Linv_P_A, lower=False)
        self.K_augkkt[0,:,:] = Minv_A[nx:nx+m, :]        
#        self.K_augkkt[0,:,:] = Minv_A[nx:nx+m, nx+nu:nx+nu+n] @ problem.A
        for i in range(1,N):
            Mi = np.zeros((k+i*m, k+i*m))
            Mi[:k+(i-1)*m,:k+(i-1)*m] = Mim1
            Mi[-m:, nx+(i-1)*m:nx+i*m] = Im
            Mi[nx+(i-1)*m:nx+i*m, -m:] = Im
#            Mi_inv = inv(self.Mi[i])
#            self.K_augkkt[i,:,:] = Mi_inv[nx+i*m:nx+(i+1)*m, nx+nu+(i-1)*n:nx+nu+i*n]
            
            (L, D, perm) = ldl(Mi)
            rhs = np.zeros((k+i*m,n))
            rhs[nx+nu+(i-1)*n:nx+nu+i*n,:] = np.identity(n)
            Mi_inv_lambdai = self.solveLDL(L, D, perm, rhs)
            self.K_augkkt[i,:,:] = Mi_inv_lambdai[nx+i*m:nx+(i+1)*m,:]
            
            Mim1 = Mi
#        print("augmented KKT")
#        for i in range(N):
#            print("K"+str(i)+"\t", self.K_augkkt[i,:,:])
        return self.K_augkkt
    
    
    def solveLDL(self, L, D, perm, rhs):
        ''' Solve the specified linear system for the specified right hand side, 
            which can be either a vector or a matrix 
        '''
        from scipy.linalg import solve_banded
        p = perm
        nk = L.shape[0]
        D_diag_ord = np.zeros((3,nk))
        perm_inv = np.zeros(nk, np.int)
        for i in range(nk):
            perm_inv[p[i]] = i
            for j in range(max(0,i-1), min(nk,i+2)):
                D_diag_ord[1 + i - j, j] = D[i,j]
        
        if(len(rhs.shape)==1):
            Linv_k = solve_triangular(L[p,:], rhs[p], lower=True, unit_diagonal=True)
        else:
            Linv_k = solve_triangular(L[p,:], rhs[p,:], lower=True, unit_diagonal=True)
#        sol = P.T @ inv(self.L[p,:].T) @ inv(self.D) @ Linv_k
#        Dinv_Linv_k = inv(self.D) @ Linv_k
        Dinv_Linv_k = solve_banded((1,1), D_diag_ord, Linv_k)
#        sol = P.T @ inv(self.L[p,:].T) @ Dinv_Linv_k
        LTinv_Dinv_Linv_k = solve_triangular(L[p,:], Dinv_Linv_k, lower=True, trans=1, unit_diagonal=True)
#        sol = P.T @ LTinv_Dinv_Linv_k 
        sol = LTinv_Dinv_Linv_k[perm_inv]
        return sol
    
        
#        Fbar = np.zeros((m,k))
#        Fbar[:,nx:nx+m] = Im
#        A = self.Minv @ Fbar.T
#        B = inv(Fbar @ A)
#        M1_inv = self.Minv - A@B@A.T
#        K1 = M1_inv[nx+m:nx+2*m, nx+nu:nx+nu+n]
#        print("Almost Efficient KKT K1\t", K1) # this works, but still not efficient
        
#        Au1 = self.Minv[nx+m:nx+2*m, nx:nx+m]
#        B0  = inv(self.Minv[nx  :nx+m,   nx:nx+m])
#        Al1 = self.Minv[nx+nu:nx+nu+n, nx:nx+m]
#        K1 = self.Minv[nx+m:nx+2*m, nx+nu:nx+nu+n] - Au1 @ B0 @ Al1.T
#        print("Efficient KKT K1\t", K1) # this works, is efficient, but is limited to K1
        
    def computeGainsWithRecursive_DEBUG(self):
        problem = self.problem
        n, m, N = problem.n, problem.m, problem.N
        nx, nu = N*n, N*m
        # COMPUTE RICCATI GAINS USING EFFICIENT RECURSIVE UPDATE OF AUGMENTED KKT MATRIX
        # Complexity N^3 if done naively, N^2 is exploiting structure
        nl = nx+nu
        self.Mi_inv = N*[None,] 
        W = self.Mi_inv # shortcut
        M_read = N*[None,]
        M_write = N*[None,]
        
        W[0] = self.Minv
        M_read[0] = 0*W[0]
        M_write[0] = np.ones_like(W[0])
        
        self.K_augkkteff = np.zeros((N,m,n))
        Linv_P_A = solve_triangular(self.L, self.P[:,nx+nu:nx+nu+n] @ problem.A, lower=True)
        Minv_A = solve_triangular(self.U, Linv_P_A, lower=False)
        self.K_augkkteff[0,:,:] = Minv_A[nx:nx+m, :]
#        self.K_augkkteff[0,:,:] = self.Minv[nx:nx+m, nx+nu:nx+nu+n] @ problem.A
        for i in range(0,N-1): # N-1 iters
            mi, mip1, mip2 = nx+i*m, nx+(i+1)*m, nx+(i+2)*m
            li, lip1 = nl+i*n, nl+(i+1)*n
            Bi = inv(W[i][mi:mip1, mi:mip1])
            M_read[i][mi:mip1, mi:mip1] = 1.0

#            Ai = W[i][:,mi:mip1]            
#            W[i+1] = W[i] - Ai @ Bi @ Ai.T
            
            W[i+1] = np.copy(W[i])
            M_read[i+1] = 0*W[i]
            M_write[i+1] = 0*W[i]
                        
            def updateW(r0, r1, c0, c1):
                W[i+1][r0:r1, c0:c1] -= W[i][r0:r1,mi:mip1] @ Bi @ W[i][c0:c1,mi:mip1].T
                M_write[i+1][r0:r1, c0:c1] = 1.0
                M_read[i][r0:r1,mi:mip1] = 1.0
                M_read[i][c0:c1,mi:mip1] = 1.0
                M_read[i][r0:r1,c0:c1] = 1.0
                
            for j in range(i, N): # N-i iters
                mj, mjp1 = nx+j*m, nx+(j+1)*m
                lj, ljp1 = nl+j*n, nl+(j+1)*n
                for k in range(i+1, j+2): # j+1-i iters
                    mk, mkp1 = nx+k*m, nx+(k+1)*m
                    if(k<=j):
                        updateW(mj, mjp1, mk, mkp1)
                    if(j<N-1):
                        updateW(lj, ljp1, mk, mkp1)
                
            self.K_augkkteff[i+1,:,:] = W[i+1][li:lip1, mip1:mip2].T
            M_read[i+1][li:lip1, mip1:mip2] = 1
            
        for i in range(0,N):
            print("M read", i, "\n", M_read[i][nx:,nx:nl])
            print("M write", i, "\n", M_write[i][nx:,nx:nl])
            print("M write-read", i, "\n", M_write[i][nx:,nx:nl]-M_read[i][nx:,nx:nl])
            
        print("Generalized Efficient KKT")
        for i in range(N):
            print("K"+str(i)+"\t", self.K_augkkteff[i,:,:])


        return self.K_augkkteff
        
    def computeGainsWithRecursive(self):
        problem = self.problem
        n, m, N = problem.n, problem.m, problem.N
        nx, nu = N*n, N*m
        # COMPUTE RICCATI GAINS USING EFFICIENT RECURSIVE UPDATE OF AUGMENTED KKT MATRIX
        # Complexity N^3
        nl = nx+nu
        
        # compute a part  of M inverse [nx:, nx:nl]
        Wi = np.zeros_like(self.M)
        Linv_P = solve_triangular(self.L, self.P.T[:,nx:nl], lower=True)
        Wi[:,nx:nl] = solve_triangular(self.U, Linv_P, lower=False)
#        W[0] = inv(self.M)
        
        # compute K0 from M inverse
        self.K_augkkteff = np.zeros((N,m,n))
#        self.K_augkkteff[0,:,:] = W[0][nx:nx+m, nx+nu:nx+nu+n] @ problem.A
        Linv_P_A = solve_triangular(self.L, self.P.T[:,nx+nu:nx+nu+n] @ problem.A, lower=True)
        Minv_A = solve_triangular(self.U, Linv_P_A, lower=False)
        self.K_augkkteff[0,:,:] = Minv_A[nx:nx+m, :]
        
        for i in range(0,N-1): # N-1 iters
            mi, mip1, mip2 = nx+i*m, nx+(i+1)*m, nx+(i+2)*m
            li, lip1 = nl+i*n, nl+(i+1)*n
            Bi = inv(Wi[mi:mip1, mi:mip1])

            # inefficient way of updating W
#            Ai = W[i][:,mi:mip1]            
#            W[i+1] = W[i] - Ai @ Bi @ Ai.T
            
            Wip1 = np.copy(Wi)
                        
            def updateW(r0, r1, c0, c1):
                Wip1[r0:r1, c0:c1] -= Wi[r0:r1,mi:mip1] @ Bi @ Wi[c0:c1,mi:mip1].T
                
            for j in range(i, N): # N-i iters
                mj, mjp1 = nx+j*m, nx+(j+1)*m
                lj, ljp1 = nl+j*n, nl+(j+1)*n
                for k in range(i+1, j+2): # j+1-i iters
                    mk, mkp1 = nx+k*m, nx+(k+1)*m
                    if(k<=j):
                        updateW(mj, mjp1, mk, mkp1)
                    if(j<N-1):
                        updateW(lj, ljp1, mk, mkp1)
                
            self.K_augkkteff[i+1,:,:] = Wip1[li:lip1, mip1:mip2].T
            
            Wi = Wip1
            
#        print("Recursive KKT inverse")
#        for i in range(N):
#            print("K"+str(i)+"\t", self.K_augkkteff[i,:,:])

        return self.K_augkkteff
    
#    def computeGainsWithRecursive(self):
#        problem = self.problem
#        n, m, N = problem.n, problem.m, problem.N
#        nx, nu = N*n, N*m
#        # COMPUTE RICCATI GAINS USING EFFICIENT RECURSIVE UPDATE OF AUGMENTED KKT MATRIX
#        # Complexity N^3
#        nl = nx+nu
#        self.Mi_inv = N*[None,] 
#        W = self.Mi_inv # shortcut     
#        
#        # compute a part  of M inverse [nx:, nx:nl]
#        W[0] = np.zeros_like(self.M)
#        Linv_P = solve_triangular(self.L, self.P.T[:,nx:nl], lower=True)
#        W[0][:,nx:nl] = solve_triangular(self.U, Linv_P, lower=False)
##        W[0] = inv(self.M)
#        
#        # compute K0 from M inverse
#        self.K_augkkteff = np.zeros((N,m,n))
##        self.K_augkkteff[0,:,:] = W[0][nx:nx+m, nx+nu:nx+nu+n] @ problem.A
#        Linv_P_A = solve_triangular(self.L, self.P.T[:,nx+nu:nx+nu+n] @ problem.A, lower=True)
#        Minv_A = solve_triangular(self.U, Linv_P_A, lower=False)
#        self.K_augkkteff[0,:,:] = Minv_A[nx:nx+m, :]
#        
#        for i in range(0,N-1): # N-1 iters
#            mi, mip1, mip2 = nx+i*m, nx+(i+1)*m, nx+(i+2)*m
#            li, lip1 = nl+i*n, nl+(i+1)*n
#            Bi = inv(W[i][mi:mip1, mi:mip1])
#
#            # inefficient way of updating W
##            Ai = W[i][:,mi:mip1]            
##            W[i+1] = W[i] - Ai @ Bi @ Ai.T
#            
#            W[i+1] = np.copy(W[i])
#                        
#            def updateW(r0, r1, c0, c1):
#                W[i+1][r0:r1, c0:c1] -= W[i][r0:r1,mi:mip1] @ Bi @ W[i][c0:c1,mi:mip1].T
#                
#            for j in range(i, N): # N-i iters
#                mj, mjp1 = nx+j*m, nx+(j+1)*m
#                lj, ljp1 = nl+j*n, nl+(j+1)*n
#                for k in range(i+1, j+2): # j+1-i iters
#                    mk, mkp1 = nx+k*m, nx+(k+1)*m
#                    if(k<=j):
#                        updateW(mj, mjp1, mk, mkp1)
#                    if(j<N-1):
#                        updateW(lj, ljp1, mk, mkp1)
#                
#            self.K_augkkteff[i+1,:,:] = W[i+1][li:lip1, mip1:mip2].T
#            
##        print("Recursive KKT inverse")
##        for i in range(N):
##            print("K"+str(i)+"\t", self.K_augkkteff[i,:,:])
#
#        return self.K_augkkteff


    def computeGainsWithParamOpt(self):
        ''' EXTRACT FEEDBACK GAINS WITH PARAMETRIC OPTIMIZATION
            This does not provide the Riccati gains!
        '''
        problem = self.problem
        n, m, N = problem.n, problem.m, problem.N
        nx, nu = N*n, N*m
        
        
        Linv_P = solve_triangular(self.L, self.P, lower=True)
        Minv = solve_triangular(self.U, Linv_P, lower=False)
        self.du_dp = Minv[nx:nx+nu, nx+nu:]
        self.K_param = np.zeros((N,m,n))
        self.K_param[0,:,:] = self.du_dp[:m,:] @ self.D0 # this works!
        for i in range(1,N):
            # these gains are not equal to the Riccati gains!
            self.K_param[i,:,:] = self.du_dp[i*m:(i+1)*m, (i-1)*n:i*n]
#        print("du_dp\n", self.du_dp)
#        print("KKT param-optim")
#        for i in range(N):
#            print("K"+str(i)+"\t", self.K_param[i])
        return self.K_param