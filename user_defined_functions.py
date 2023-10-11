import numpy as np 
import scipy


# ----------------------------------------------------
# ----- Define the dynamics of the system ------------
# ----------------------------------------------------

def compute_third_order_tensor(params,n):

    _, alpha, beta = params[0], params[1], params[2]
    Q1 = np.random.randn(n,n**2)
    Q2 = np.random.randn(n,n**2)
    L = Q1.copy()
    R = np.zeros((n**2,n**2))
    for k in range (n**2):
        x1, y1, _ = Q1[0,k], Q1[1,k], Q1[2,k] 
        x2, y2, z2 = Q2[0,k], Q2[1,k], Q2[2,k] 
        R[:,k] = np.reshape(np.einsum('i,j',Q1[:,k],Q2[:,k]),-1)
        L[:,k] = np.asarray([-alpha*x1*z2-beta*x1*y2,-alpha*y1*z2+beta*x1*x2,alpha*(x1*x2+y1*y2)])

    H = (L@R.T@scipy.linalg.inv(R@R.T)).reshape((n,n,n))
    H[np.abs(H) < 1e-12] = 0.0
    validate_third_order_tensor(params,H,n)

    return H

def validate_third_order_tensor(params,H,n):
    _, alpha, beta = params[0], params[1], params[2]
    for k in range (10):
        Q1 = np.random.randn(n)
        Q2 = np.random.randn(n)

        x1, y1, _ = Q1[0], Q1[1], Q1[2] 
        x2, y2, z2 = Q2[0], Q2[1], Q2[2] 
        v1 = np.asarray([-alpha*x1*z2-beta*x1*y2,-alpha*y1*z2+beta*x1*x2,alpha*(x1*x2+y1*y2)])
        v2 = np.einsum('ijk,j,k',H,Q1,Q2)

        error = np.linalg.norm(v1 - v2)
        if error > 1e-10:
            raise ValueError ("Third-order tensor was not computed correctly. Error = %1.12e"%error)



def evaluate_rhs(t,q,A,H,B,wf): 
    return A@q + np.einsum('ijk,j,k',H,q,q) + B@[np.sin(wf*t)]


# ----------------------------------------------------
# ----- Define forward and inverse fft ---------------
# ----------------------------------------------------

def fft(t,Q,n):

    T = t[-1] + t[1] - t[0]
    freqs = (2*np.pi/T)*np.arange(-n,n+1,1)
    QHat = (1/len(t))*scipy.fft.rfft(Q,axis=-1)[:,:(n+1)]
    QHat = np.concatenate((np.fliplr(QHat).conj(),QHat[:,1:]),axis=-1)

    return freqs, QHat

def ifft(freqs,QHat,t,dt):
    
    Q = np.zeros((QHat.shape[0],len(t)),dtype=np.complex128)
    for k in range (len(freqs)):
        Q += QHat[:,k].reshape(-1,1)*np.exp(1j*freqs[k]*(t + dt))
    
    return Q.real


# ----------------------------------------------------
# ----- Compute frequency domain matrices ------------
# ----------------------------------------------------

def compute_lifted_frequency_domain_matrices(tensors,freqs,QHat):

    A, H, B, C = tensors[0], tensors[1], tensors[2], tensors[3]

    n = A.shape[0]          # Size of the system
    m = B.shape[-1]         # Size of the input
    p = C.shape[0]          # Size of the output
    nf = len(freqs)         # Total number of frequencies (including negative freqs.)
    

    T = np.zeros((n*nf,n*nf),dtype=np.complex128)
    B_ = np.zeros((n*nf,m*nf),dtype=np.complex128)
    C_ = np.zeros((p*nf,n*nf),dtype=np.complex128)

    for i in range (-(nf//2),nf//2+1):

        i0 = (i+nf//2)*n 
        i1 = i0 + n
        T[i0:i1,i0:i1] = -1j*freqs[i+nf//2]*np.diag(np.ones(n)) + A \
            + np.einsum('ijk,j',H,QHat[:,nf//2]) + np.einsum('ijk,k',H,QHat[:,nf//2])
        
        j0 = (i+nf//2)*m
        j1 = j0 + m
        B_[i0:i1,j0:j1] = B 
        
        j0 = (i+nf//2)*p
        j1 = j0 + p
        C_[j0:j1,i0:i1] = C

        for j in range (-(nf//2),nf//2+1):

            j0 = (j+nf//2)*n 
            j1 = j0 + n
            k = i - j

            if np.abs(k) <= nf//2 and k != 0:

                T[i0:i1,j0:j1] = np.einsum('ijk,j',H,QHat[:,k+nf//2]) + np.einsum('ijk,k',H,QHat[:,k+nf//2])

                # Since we are considering time-invariant matrices B and C, we do not need
                # to populate the off-diagonal entries of the matrices B_ and C_

    return T, B_, C_


# ----------------------------------------------------
# ----- Compute residual for Newton's method ---------
# ----------------------------------------------------

def compute_frequency_domain_residual(t,tensors,freqs,QHat,wf):
    
    n, nf = QHat.shape
    A, H, B = tensors[0], tensors[1], tensors[2]
    
    Q = ifft(freqs,QHat,t,0)
    res = np.zeros(Q.shape)
    for k in range (len(t)):
        res[:,k]  = evaluate_rhs(t[k],Q[:,k],A,H,B,wf)
        
    _, reshat = fft(t,res,nf//2)
    reshat -= QHat*(1j*freqs) 
    
    return reshat.reshape(-1,order='F')
    

# ----------------------------------------------------
# ----- Compute base flow using Newton's method ------
# ----------------------------------------------------

def newton_harmonic_balance(t,tensors,freqs,QHat,wf,tol):

    n, nf = QHat.shape
    rhat = compute_frequency_domain_residual(t,tensors,freqs,QHat,wf)
    error = np.linalg.norm(rhat)

    iter = 0
    print("Computing the base flow using Newton's method...")
    print("Iteration %d,\t Residual norm = %1.12e"%(iter,error))
    
    while error > tol and iter < 20: 
        
        
        T, _, _ = compute_lifted_frequency_domain_matrices(tensors,freqs,QHat)
        dQHat = scipy.linalg.solve(-T,rhat).reshape((n,nf),order='F')
        QHat = QHat + dQHat 

        rhat = compute_frequency_domain_residual(t,tensors,freqs,QHat,wf)
        error = np.linalg.norm(rhat)

        iter += 1
        print("Iteration %d,\t Residual norm = %1.12e"%(iter,error))

    print("Done.")
    
    return QHat





