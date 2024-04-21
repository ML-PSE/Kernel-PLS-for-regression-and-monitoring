"""
NIPALS implementation for kernel partial least squares.

@author: Ankur Kumar @ MLforPSE.com
"""

import numpy as np
import random

def KerNIPALS(K, Y, nlatents):
    """
    NIPALS implementation for kernel partial least squares.

    Args:
        K    : kernel (Gram) matrix (number of samples  x number of samples) 
        Y    : training outputs (number of samples  x dimY) 
        nlatents  : number of score vectors to extract 

    Returns:
        Bf     : matrix of dual-form regression coefficients (number of samples x dimY)     
        T,U    : matrix of latent vectors (number of samples x nlatents) 
        
    """
      
    max_iterations = 50
    crit = 1e-8

    N = K.shape[0]
    m = Y.shape[1]
    
    T = np.empty((N, nlatents))
    U = np.empty((N, nlatents))

    Kres = np.copy(K)
    Yres = np.copy(Y)
    
    for num_lv in range(nlatents):
        print('finding latent #: {}'.format(num_lv+1))
        
        #initialization
        u = Yres[:, random.randint(0, m-1)][:, None]
        iteration_count = 0
        convergence_metric = crit * 10.0
        
        # inner iterations
        while iteration_count < max_iterations and convergence_metric > crit:
            u_old = np.copy(u)
            
            w = np.dot(Kres, u)
            t = w/np.linalg.norm(w)
            c = np.dot(Yres.T, t)
            u = np.dot(Yres, c)
            u = u/np.linalg.norm(u)
                        
            convergence_metric = np.linalg.norm(u-u_old)/np.linalg.norm(u)
            iteration_count += 1

        if iteration_count >= max_iterations:
            raise Exception('KPLS failed to converge for component: {}'.format(num_lv+1))
        
        # store component
        T[:, num_lv] = t[:,0]
        U[:, num_lv] = u[:,0]
        
        # deflate
        Ktt = np.dot(np.dot(Kres, t), t.T)
        Kres = Kres - Ktt.T - Ktt + np.dot(t, np.dot(t.T, Ktt))
        Yres = Yres - np.dot(t, np.dot(t.T, Yres))
        
    # matrix for regression
    temp = np.linalg.inv(np.dot(T.T, np.dot(K, U)))
    Bf =  np.dot(np.dot(np.dot(U, temp), T.T), Y)

    return Bf, T, U
