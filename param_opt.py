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

class KktSolver:
    
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
        self.D = np.zeros((nx, nx+nu))
        self.D0 = np.zeros((nx, n))
        self.H = np.zeros((nx+nu, nx+nu))
        I = np.identity(n)
        
        # variables order is: u0, x1, u1, x2, u2, ..., xN
        self.D0[:n,:] = problem.A
        for i in range(N):
            ixm1, ix, ix1 = (i-1)*(m+n)+m, i*(m+n)+m, (i+1)*(m+n)
            iu, iu1 = i*(m+n), i*(m+n)+m
            self.D[i*n:i*n+n, ix:ix1] = I
            self.D[i*n:i*n+n, iu:iu1] = -problem.B
            if(i>0):
                self.D[i*n:i*n+n, ixm1:ixm1+n] = -problem.A
            
            if(i<N-1):
                self.H[ix:ix1, ix:ix1] = problem.Q
            else:
                self.H[ix:ix1, ix:ix1] = problem.Qf
            self.H[iu:iu1, iu:iu1] = problem.R
        
        # COMPUTE KKT SYSTEM
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
        U = np.zeros((N, m))
        X[0,:] = x0
        for i in range(N):
            iz = i*(m+n)
            X[i+1,:] = self.kkt_sol[iz+m:iz+m+n]
            U[i,:] = self.kkt_sol[iz:iz+m]
        
        return X, U
    
    
    def computeGainsWithRedKkt(self):
        ''' COMPUTE RICCATI GAINS INVERTING REDUCED KKT MATRIX
            At each iteration we remove u_i and x_{i+1}, the constraint
            x_{i+1} = A x_i + B u_i, and (maybe not needed) modify the kkt vector d.
            This works but is inefficient, complexity N^4
        '''
        
        problem = self.problem
        n, m, N = problem.n, problem.m, problem.N
        nx, nu, nz, z = N*n, N*m, N*(m+n), n+m
        
        self.Mi = N*[None,] 
        self.Mi[0] = self.M
        self.K_redkkt = np.zeros((N,m,n))
        
        # COMPUTE K0
        Linv_P_A = solve_triangular(self.L, self.P.T[:,nx+nu:nx+nu+n] @ problem.A, lower=True)
        Minv_A = solve_triangular(self.U, Linv_P_A, lower=False)
        self.K_redkkt[0,:,:] = Minv_A[:m, :]
#        self.K_redkkt[0,:,:] = inv(self.M)[:m, nx+nu:nx+nu+n] @ problem.A
#        print(0, "Mi\n", self.Mi[0])
        
        for i in range(1,N):
            Ni = N-i
            nxi, nui, nzi = Ni*n, Ni*m, Ni*(m+n)
            ki = 2*nxi+nui             # size of this KKT system
            self.Mi[i] = np.zeros((ki, ki))
            
            # copy the top left corner, but skipping the first two row/cols
            self.Mi[i][:nzi,:nzi] = self.Mi[i-1][z:nzi+z, z:nzi+z]
            
            # copy the rows corresponding to the dynamics
            self.Mi[i][-nxi:, :nzi] = self.Mi[i-1][-nxi:, z:nzi+z]

            # make matrix symmetric                        
            self.Mi[i][:-nxi, -nxi:] = self.Mi[i][-nxi:, :-nxi].T
#            print(i, "Mi\n", self.Mi[i])

#            Mi_inv = inv(self.Mi[i])
            # select rows corresponding to u_i
#            self.K_redkkt[i,:,:] = Mi_inv[:m, nzi:nzi+n] @ problem.A
            
            (L, D, perm) = ldl(self.Mi[i])
            rhs = np.zeros((ki,n))
            rhs[nzi:nzi+n,:] = problem.A
            Mi_inv_A = self.solveLDL(L, D, perm, rhs)
            self.K_redkkt[i,:,:] = Mi_inv_A[:m,:]
        
#        print("reduced KKT")
#        for i in range(N):
#            print("K"+str(i)+"\t", self.K_redkkt[i,:,:])
        return self.K_redkkt
    
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
    
    
if __name__=='__main__':
    import matplotlib.pyplot as plt
    import orc.utils.plot_utils as plut
    from numpy.linalg import norm
    from param_opt_plu import KktSolverPLU
    from param_opt_ldl import KktSolverLDL
    from param_opt_aug_rec import KktSolverAugRec

    from numpy.random import randint
    seed = randint(0, 100000000)
    print("Seed", seed)
#    np.random.seed(79217176)
    np.random.seed(seed)

    np.set_printoptions(linewidth=100, precision=3, suppress=True);
    
    ''' Test LQR with a simple double integrator
    '''
    PLOT_STATE = 1
    USE_AUG_KKT = 1
    USE_RED_KKT = 1
    USE_RECURSIVE = 1
    USE_PLU_UPDATES = 0
    USE_LDL_UPDATES = 0
    SYSTEM_ID = 2
    N = 15                # horizon size
    dt = 0.1               # control time step
    DEBUG = False
        
    if(SYSTEM_ID==1):
        A = np.array([[1.0, dt], [0.0, 1.0]])
        B = np.array([[0.5*dt*dt], [dt]])
        n = A.shape[0]                          # state size
        m = B.shape[1]                          # control size
        x0 = np.array([10.0, 0.0]);     # initial state
    elif(SYSTEM_ID==2):
        from numpy.random import random
        n = 2
        m = 1
        A = random((n,n))
        B = random((n,m))
        x0 = random(n)
    
    Q = 1e-2 * np.identity(n)
    R = 1e-3 * np.identity(m)
    Qf = 1 * np.identity(n)
    problem = LqrProblem(x0, A, B, Q, R, Qf, N)
    print("Eigenvalues of A", np.linalg.eigvals(A))
        
    riccati_solver = RiccatiSolver("Riccati", problem, DEBUG)
    kkt_solver = KktSolver("KKT", problem, DEBUG)
    kkt_aug_rec = KktSolverAugRec("KKT-AUG-REC", problem, DEBUG)
    kkt_ldl = KktSolverLDL("KKT-LDL", problem, DEBUG)
    kkt_plu = KktSolverPLU("KKT-PLU", problem, DEBUG)
    
    (X_riccati, U_riccati, K_riccati) = riccati_solver.solve(x0)
    (X_kkt, U_kkt) = kkt_solver.solve(x0)
    (X_aug, U_aug) = kkt_aug_rec.solve(x0)
    (X_ldl, U_ldl) = kkt_ldl.solve(x0)
    (X_plu, U_plu) = kkt_plu.solve(x0)
    
    if(PLOT_STATE):
        nplot = int(x0.shape[0]/2)
        (f, ax) = plut.create_empty_figure(nplot,2)
        ax = ax.reshape(nplot*2)
        t = range(0, N+1)
        for i in range(len(ax)):
            ax[i].plot(t, X_riccati[:,i], label='Riccati x '+str(i), alpha=0.5)
            ax[i].plot(t, X_kkt[:,i], '--', label='KKT x '+str(i), alpha=0.5)
            ax[i].plot(t, X_ldl[:,i], ':', label='KKT-LDL x '+str(i), alpha=0.5)
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel(r'$x_'+str(i)+'$')
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)
    
    if(USE_AUG_KKT):
        K_augkkt = kkt_aug_rec.computeGainsWithAugKkt()
        print("\nRiccati vs AUG-KKT gains")
#        for i in range(N): print("K"+str(i)+"\t", K_riccati[i,:,:], K_augkkt[i,:,:])
        for i in range(N):
            print("||K"+str(i)+" err|| = %3f\t"%norm(K_riccati[i,:,:]-K_augkkt[i,:,:]))
            
    if(USE_RED_KKT):
        K_redkkt = kkt_solver.computeGainsWithRedKkt()
        print("\nRiccati vs RED-KKT gains")
#        for i in range(N): print("K"+str(i)+"\t", K_riccati[i,:,:], K_redkkt[i,:,:])
        for i in range(N):
            print("||K"+str(i)+" err|| = %3f\t"%norm(K_riccati[i,:,:]-K_redkkt[i,:,:]))
        
    if(USE_RECURSIVE):
        K_recur = kkt_aug_rec.computeGainsWithRecursive()
        print("\nRiccati vs RECURSIVE gains")
#        for i in range(N): print("K"+str(i)+"\t", K_riccati[i,:,:], K_recur[i,:,:])
        for i in range(N):
            print("||K"+str(i)+" err|| = %3f\t"%norm(K_riccati[i,:,:]-K_recur[i,:,:]))
            
    if(USE_PLU_UPDATES):
        K_plu = kkt_plu.computeGainsWithPLUupdates()
        print("\nRiccati vs PLU gains")
#        for i in range(N): print("K"+str(i)+"\t", K_riccati[i,:,:], K_plu[i,:,:])
        for i in range(N):
            print("||K"+str(i)+" err|| = %3f\t"%norm(K_riccati[i,:,:]-K_plu[i,:,:]))
            
    if(USE_LDL_UPDATES):
        K_ldl = kkt_ldl.computeGainsWithLDLupdates()
        print("\nRiccati vs LDL gains")
#        for i in range(N): print("K"+str(i)+"\t", K_riccati[i,:,:], K_ldl[i,:,:])
        for i in range(N):
            print("||K"+str(i)+" err|| = %3f\t"%norm(K_riccati[i,:,:]-K_ldl[i,:,:]))
            
            
        
    print("Eigenvalues of A", np.linalg.eigvals(A))
    