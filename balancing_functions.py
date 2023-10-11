import numpy as np 
import scipy

import user_defined_functions as funs


# --------------------------------------------------------------------------
# ----- Compute the Gramians using the frequency-domain lifted -------------
# ----- Lyapunov equations (these are obtained by harmonic     -------------
# ----- balancing the differntial Lyapunov equations)          -------------
# --------------------------------------------------------------------------

def compute_gramian_coefficients(freqs,A,M):
    
    nf = len(freqs)
    n = A.shape[0]//nf
    X = scipy.linalg.solve_continuous_lyapunov(A,-M@(M.conj().T))
    X = X[n*(nf//2):n*(nf//2 + 1),:]
            
    return X


def reconstruct_gramian(freqs,X,t):
    
    n = X.shape[0]
    nf = X.shape[-1]//n
    
    P = np.zeros((n,n))
    for k in range (nf//2+1):
        k0 = k*n
        k1 = (k+1)*n
        
        if k < nf//2:   P += 2*(X[:,k0:k1]*np.exp(-1j*freqs[k]*t)).real
        else:           P += X[:,k0:k1].real
    
    return P



# --------------------------------------------------------------------------
# ----- Execute Part I of Alg. 1 in Padovan & Rowley, JCP, 2023 ------------
# --------------------------------------------------------------------------


def compute_matrix_X(gammas,freqs,A,M):
    
    nf = len(freqs)
    m = M.shape[-1]//nf
    
    Id = np.diag(np.ones(A.shape[0]))
    X = np.zeros((A.shape[0],len(gammas)*m),dtype=np.complex128)
    
    for i in range (len(gammas)):
        
        i0 = i*m
        i1 = (i+1)*m
        
        X[:,i0:i1] = scipy.linalg.solve(1j*gammas[i]*Id - A,M[:,m*(nf//2)]).reshape(-1,m)
        
    return X


def evaluate_gramian(gammas,freqs,X,t):
    
    nf = len(freqs)
    n = X.shape[0]//nf
    m = X.shape[-1]//len(gammas)
    dg = gammas[1] - gammas[0]
    
    Z = np.zeros((n,X.shape[-1]),dtype=np.complex128)
    
    for i in range (X.shape[-1]):
        
        if gammas[i//m] == 0:   cxi = np.sqrt(0.5*dg/np.pi)
        else:                   cxi = np.sqrt(dg/np.pi)
            
        Xi = X[:,i].reshape((n,nf),order='F')
        Z[:,i] = cxi*np.sum(Xi*np.exp(1j*freqs*t),axis=-1)
        
    Z = np.concatenate((Z.real,Z.imag),axis=-1)
    P = Z@Z.T
    
    return P

