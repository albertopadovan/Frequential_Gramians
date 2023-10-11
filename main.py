import numpy as np 
import scipy
import matplotlib.pyplot as plt 

import user_defined_functions as funs
import balancing_functions as bal


# -------------------------------------------------------------------
# --------- Define system -------------------------------------------
# -------------------------------------------------------------------

n = 3                                       # number of degrees of freedom
mu = 1/5; alpha = 1/5; beta = 1/5
params = [mu,alpha,beta]

omega = np.sqrt(1 - beta**2*mu/alpha)       # natural frequency of the unforced limit cycle
A = np.asarray([[mu,-1,0],[1,mu,0],[0,0,-alpha]])
H = funs.compute_third_order_tensor(params,n)
B = 1e-1*np.asarray([1,1,0]).reshape(-1,1)
C = np.ones(n).reshape(1,-1)
tensors = [A,H,B,C]


# -------------------------------------------------------------------
# --------- Compute base flow ---------------------------------------
# -------------------------------------------------------------------

wf = 2*omega            # Forcing frequency (twice the natural frequency)
Tw = 2*np.pi/omega
dt = Tw/100
time = np.arange(0,100*Tw,dt)
sol = scipy.integrate.solve_ivp(funs.evaluate_rhs,[0,time[-1]],\
                                np.zeros(n),'RK45',t_eval=time,args=(A,H,B,wf))
Q = sol.y

# Generate initial condition for Newton's method
idx0 = np.argmin(np.abs(time - 98*Tw))
idxf = np.argmin(np.abs(time - 99*Tw))
time = time[idx0:idxf] - time[idx0]
Q = Q[:,idx0:idxf]
U = np.sin(2*omega*time).reshape(1,-1)
freqs, QHat = funs.fft(time,Q,10)

QHat = funs.newton_harmonic_balance(time,tensors,freqs,QHat,wf,1e-9)
Q = funs.ifft(freqs,QHat,time,0)

# -------------------------------------------------------------------
# --------- Compute the Gramians ------------------------------------
# -------------------------------------------------------------------

dg = omega/2/30
gammas = np.arange(0,10*omega/2+dg,dg)


T, B_, C_ = funs.compute_lifted_frequency_domain_matrices(tensors,freqs,QHat)

t = 0.3

X = bal.compute_matrix_X(gammas,freqs,T,B_)
P = bal.evaluate_gramian(gammas,freqs,X,t)

Y = bal.compute_matrix_X(-gammas,freqs,T.conj().T,C_.conj().T)
Q = bal.evaluate_gramian(gammas,freqs,Y,t)

X = bal.compute_gramian_coefficients(freqs,T,B_)
P_ = bal.reconstruct_gramian(freqs,X,t)

Y = bal.compute_gramian_coefficients(freqs,T.conj().T,C_.conj().T)
Q_ = bal.reconstruct_gramian(freqs,Y,t)

