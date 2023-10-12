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
# ----- Compute the Gramians using their time-domain definition ------------
# --------------------------------------------------------------------------

def evaluate_linearized_rhs(t,q,A,H,fQ,Tp): 
    tt = np.mod(t,Tp)
    return (A + np.einsum('ijk,j',H,fQ(tt)) + np.einsum('ijk,k',H,fQ(tt)))@q

def compute_gramian_time_domain(tensors,tbflow,Q,t,taus):
    
    A, H, B = tensors[0], tensors[1], tensors[2]
    dtau = taus[1] - taus[0]
    X = np.zeros((A.shape[0],len(taus)*B.shape[-1]))
    fQ = scipy.interpolate.interp1d(tbflow,Q,kind='linear',fill_value='extrapolate')
    Tp = tbflow[-1] + tbflow[1] - tbflow[0]
    
    count = 0
    for j in range (B.shape[-1]):
        q0 = B[:,j]
        
        for i in range (len(taus)):
            print("Impulse %d/%d"%(i+1,len(taus)))
            sol = scipy.integrate.solve_ivp(evaluate_linearized_rhs,[taus[i],t],\
                                            q0,'RK45',t_eval=[t],args=(A,H,fQ,Tp)) 
            X[:,count] = sol.y[:,-1]
            count += 1
    
    
    return np.sqrt(dtau)*X


# --------------------------------------------------------------------------
# ----- Compute the Gramians using the frequential factors -----------------
# --------------------------------------------------------------------------


def compute_matrix_X(gammas,freqs,A,M):
    
    nf = len(freqs)
    m = M.shape[-1]//nf
    
    Id = np.diag(np.ones(A.shape[0]))
    X = np.zeros((A.shape[0],len(gammas)*m),dtype=np.complex128)
    
    for i in range (len(gammas)):
        
        i0 = i*m
        i1 = (i+1)*m
        
        X[:,i0:i1] = scipy.linalg.solve(1j*gammas[i]*Id - A,M[:,m*(nf//2):m*(nf//2+1)]).reshape(-1,m)
        
    return X

# This function implements Algorithm 1 in the JCP paper
def compute_matrix_X_efficient(gammas,m,freqs,A,M):
    
    
    nf = len(freqs)
    n = A.shape[0]//nf
    p = M.shape[-1]//nf
    ncols_X = ((len(gammas)-2)*(2*m + 1) + 2*(m+1))*p
    
    omega = freqs[nf//2+1]
    Id = np.diag(np.ones(A.shape[0]))
    X = np.zeros((A.shape[0],ncols_X),dtype=np.complex128)
    
    gvec = np.zeros(ncols_X//p)
    count = 0
    for i in range (len(gammas)):
        
        # Compute factorization (or preconditioner)
        lu, piv = scipy.linalg.lu_factor(1j*gammas[i]*Id - A)
        
        if i == 0 or i == len(gammas)-1:    Range = np.arange(0,m+1,1)
        else:                               Range = np.arange(-m,m+1,1) 
        
        for j in range (len(Range)):
            idx0 = count*p
            idx1 = (count+1)*p
            
            Mj = M[:,p*(nf//2):p*(nf//2+1)].copy()
            Xj = np.zeros_like(Mj,dtype=np.complex128)
            for k in range (p): 
                Mjk = Mj[:,k].reshape((n,nf),order='F')
                Mj[:,k] = np.roll(Mjk,Range[j]).reshape(-1,order='F')
                sol = scipy.linalg.lu_solve((lu,piv),Mj[:,k]).reshape((n,nf),order='F')
                Xj[:,k] = np.roll(sol,-Range[j]).reshape(-1,order='F')
                
            X[:,idx0:idx1] = Xj
            
            gvec[count] = np.abs(gammas[i] + Range[j]*omega)
            count += 1
            
    return X, np.sort(np.abs(gvec))


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














