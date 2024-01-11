# -*- coding: utf-8 -*-
"""
Profile different approaches for computing Riccati gains.

@author: adelpret
"""

import numpy as np
from numpy.linalg import inv
from lqr import a2s, LqrProblem, RiccatiSolver
from param_opt import KktSolver
from param_opt_ldl import KktSolverLDL
from param_opt_plu import KktSolverPLU
from param_opt_aug_rec import KktSolverAugRec
from time import time
    
if __name__=='__main__':
    import matplotlib.pyplot as plt
    import orc.utils.plot_utils as plut
    from numpy.linalg import norm
    np.set_printoptions(linewidth=100, precision=3, suppress=True);
    
    USE_AUG_KKT = 1
    USE_RED_KKT = 1
    USE_RECURSIVE = 1
    USE_PLU_UPDATES = 1
    USE_LDL_UPDATES = 1
    USE_DDP = 1
    
    PLOT_FLOPS = 0
    SYSTEM_ID = 2
    N = [16, 32, 64, 128]    # horizon size
    DEBUG = False
        
    if(SYSTEM_ID==1):
        dt = 0.1                    # control time step
        A = np.array([[1.0, dt], [0.0, 1.0]])
        B = np.array([[0.5*dt*dt], [dt]])
        n = A.shape[0]                          # state size
        m = B.shape[1]                          # control size
        x0 = np.array([10.0, 0.0]);     # initial state
    elif(SYSTEM_ID==2):
        from numpy.random import random
        n, m = 6, 3
#        n, m = 2, 1
        A = random((n,n))
        B = random((n,m))
        x0 = random(n)
    
    Q = 1e-2 * np.identity(n)
    R = 1e-3 * np.identity(m)
    Qf = 1 * np.identity(n)
    problem = LqrProblem(x0, A, B, Q, R, Qf, N[0])
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
    

    time_augkkt = np.zeros(len(N))
    time_redkkt = np.zeros(len(N))
    time_recur = np.zeros(len(N))
    time_plu = np.zeros(len(N))
    time_ldl = np.zeros(len(N))
    time_ddp = np.zeros(len(N))
    flops_augkkt = np.zeros(len(N))
    flops_recur = np.zeros(len(N))
    
    for (i,Ni) in enumerate(N):
        print("************************ N %d ********************************"%Ni)
        problem.N = Ni
        
        if(USE_DDP):
            start = time()
            riccati_solver.solve(x0)
            time_ddp[i] = 1e3*(time()-start)
            print("DDP", time_ddp[i])
            
        if(USE_RECURSIVE):
            kkt_aug_rec.solve(x0)
            start = time()
            K_recur = kkt_aug_rec.computeGainsWithRecursive()
            time_recur[i] = 1e3*(time()-start)
            flops_recur[i] = kkt_solver.flops
            print("RECURSIVE", time_recur[i])
            
        if(USE_LDL_UPDATES):
            kkt_ldl.solve(x0)
            start = time()
#            import cProfile
#            cProfile.run('kkt_ldl.computeGainsWithLDLupdates()')
            K_ldl = kkt_ldl.computeGainsWithLDLupdates()
            time_ldl[i] = 1e3*(time()-start)
            print("LDL", time_ldl[i])
            
        if(USE_PLU_UPDATES):
            kkt_plu.solve(x0)
            start = time()
            K_plu = kkt_plu.computeGainsWithPLUupdates()
            time_plu[i] = 1e3*(time()-start)
            print("PLU", time_plu[i])
            
        if(USE_RED_KKT):
            kkt_solver.solve(x0)
            start = time()
            K_redkkt = kkt_solver.computeGainsWithRedKkt()
            time_redkkt[i] = 1e3*(time()-start)
            print("RED-KKT", time_redkkt[i])
            
        if(USE_AUG_KKT):
            try:
                kkt_aug_rec.solve(x0)
                start = time()
                K_augkkt = kkt_aug_rec.computeGainsWithAugKkt()
                time_augkkt[i] = 1e3*(time()-start)
                print("AUG-KKT", time_augkkt[i])
            except:
                print("AUG-KKT failed")
                time_augkkt[i] = np.nan
        
    (f, ax) = plut.create_empty_figure(1)
    if(USE_AUG_KKT):     ax.plot(N, time_augkkt, label='Aug KKT', alpha=0.7)
    if(USE_RED_KKT):     ax.plot(N, time_redkkt, label='Red KKT', alpha=0.7)
    if(USE_RECURSIVE):   ax.plot(N, time_recur, label='Recursive', alpha=0.7)
    if(USE_PLU_UPDATES): ax.plot(N, time_plu, label='PLU', alpha=0.7)
    if(USE_LDL_UPDATES): ax.plot(N, time_ldl, label='LDL', alpha=0.7)
    if(USE_DDP):         ax.plot(N, time_ddp, label='DDP', alpha=0.7)
#    ax.plot(N, [0.2*i**2 for i in N], label='N^2', alpha=0.7)
#    ax.plot(N, [0.002*i**3 for i in N], label='N^3', alpha=0.7)
    ax.set_xlabel('Horizon length N')
    ax.set_ylabel('Computation time [ms]')
    ax.set_yscale('log')
    leg = ax.legend()
    leg.get_frame().set_alpha(0.5)
    
    if(PLOT_FLOPS):
        (f, ax) = plut.create_empty_figure(1)
    #    ax.plot(N, flops_augkkt, label='Aug KKT', alpha=0.7)
        ax.plot(N, flops_recur, label='Recursive', alpha=0.7)
        ax.plot(N, [50*(i**2) for i in N], label='N^2', alpha=0.7)
        ax.plot(N, [0.9*(i**3) for i in N], label='N^3', alpha=0.7)
        ax.set_xlabel('Horizon length N')
        ax.set_ylabel('N Flops')
        leg = ax.legend()
        leg.get_frame().set_alpha(0.5)
    