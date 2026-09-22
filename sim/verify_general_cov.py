"""General-covariance version of Theorem 1.

Claim (a): the scalar result  beta* = A/(A+D),  A = tr(Pperp C_n Pperp),
           D = tr(Pperp C_d Pperp),  holds for ARBITRARY C_n and C_d, needing
           only (i) drift in span(B) and (ii) E[n d^H] = 0.
Claim (b): the MMSE member of the wider class  h = P_B h0 + Psi Pperp h0  is the
           Wiener filter Psi* = C_d,perp (C_d,perp + C_n,perp)^{-1}; the scalar
           family is Psi = (1-beta) I, and is optimal iff the two restricted
           covariances are proportional.
"""
import numpy as np, sys
sys.path.insert(0,'sim')
from isac_core import Scenario, steering, draw_diffuse_clustered
rng=np.random.default_rng(12); sc=Scenario(); N,r=sc.N,sc.r

# --- an ANISOTROPIC calibration noise covariance: correlated across subcarriers ---
kk=np.arange(sc.K)
Rk=np.exp(-np.abs(kk[:,None]-kk[None,:])/12.0)          # exponential correlation
Ra=np.eye(sc.Nr)+0.4*(np.ones((sc.Nr,sc.Nr))-np.eye(sc.Nr))
C_n=np.kron(Rk,Ra)*3e-4                                  # (N,N), k-major
Ln=np.linalg.cholesky(C_n+1e-12*np.eye(N))

# --- C_d for the clustered DMC model, by ensemble average of b b^H ---
def clustered_atoms(M):
    L1=len(sc.theta_bg); per=int(np.ceil(M/L1)); ph=[];xi=[]
    for i in range(L1):
        ph.append(rng.vonmises(sc.theta_bg[i],40.0,per))
        xi.append(sc.tau_bg[i]+rng.exponential(0.30e-6,per))
    return (np.clip(np.concatenate(ph)[:M],-np.pi/2.2,np.pi/2.2),
            np.clip(np.concatenate(xi)[:M],0.02e-6,2.8e-6))
def atom(p,x):
    return np.outer(np.exp(-1j*2*np.pi*kk*sc.df*x), steering(p,sc.Nr)).ravel()
ph,xi=clustered_atoms(6000)
C_d=np.zeros((N,N),complex)
for p,x in zip(ph,xi):
    b=atom(p,x); C_d+=np.outer(b,b.conj())
C_d/= len(ph)
C_d*= (10**(-30/10)*sc.E_spec)/np.trace(C_d).real          # scale to -30 dB
Ld=np.linalg.cholesky(C_d+1e-12*np.eye(N))

A=np.trace(sc.P_perp@C_n@sc.P_perp).real
D=np.trace(sc.P_perp@C_d@sc.P_perp).real
b_star=A/(A+D)
print("="*74)
print("CLAIM (a): scalar theorem under arbitrary covariances")
print("="*74)
print(f"  A = tr(Pperp C_n Pperp) = {A:.6f}")
print(f"  D = tr(Pperp C_d Pperp) = {D:.6f}")
print(f"  beta* = A/(A+D)         = {b_star:.5f}")
T=4000
Dl=(rng.standard_normal((T,N))+1j*rng.standard_normal((T,N)))/np.sqrt(2)@Ld.T
Nz=(rng.standard_normal((T,N))+1j*rng.standard_normal((T,N)))/np.sqrt(2)@Ln.T
sd2=2e-5
Del=(rng.standard_normal((T,r))+1j*rng.standard_normal((T,r)))*np.sqrt(sd2/2)
Ht=sc.h_spec[None,:]+Dl
Hc=(sc.B@(sc.g[None,:]-Del).T).T+Dl+Nz
PHc=Hc@sc.PB.T
def mse(bt):
    E=(1-bt)*Hc+bt*PHc-Ht
    return float(np.mean(np.einsum('ti,ti->t',E.conj(),E).real))
const=sd2*np.trace(sc.B.conj().T@sc.B).real+np.trace(sc.PB@C_n@sc.PB).real
print(f"\n  {'beta':>8} {'MSE emp':>12} {'MSE theory':>12}")
for bt in [0.0,0.2,b_star,0.6,1.0]:
    th=const+(1-bt)**2*A+bt**2*D
    tag="  <- beta*" if abs(bt-b_star)<1e-9 else ""
    print(f"  {bt:8.4f} {mse(bt):12.6f} {th:12.6f}{tag}")
g=np.linspace(0,1,401); print(f"  empirical argmin beta = {g[int(np.argmin([mse(x) for x in g]))]:.4f}")

print("\n"+"="*74)
print("CLAIM (b): scalar vs matrix Wiener filter on the complement")
print("="*74)
w,V=np.linalg.eigh(sc.P_perp); U=V[:,w>0.5]                 # orthonormal basis, (N,N-r)
Cd_p=U.conj().T@C_d@U; Cn_p=U.conj().T@C_n@U
Psi=Cd_p@np.linalg.inv(Cd_p+Cn_p)
risk_mat=np.trace(Psi@Cn_p).real
risk_sca=(1-b_star)**2*np.trace(Cn_p).real+b_star**2*np.trace(Cd_p).real
print(f"  matrix Wiener risk        = {risk_mat:.6e}")
print(f"  best scalar (beta*) risk  = {risk_sca:.6e}")
print(f"  penalty for using scalar  = {10*np.log10(risk_sca/risk_mat):.2f} dB")
# Cn^-1 Cd is not Hermitian even when both factors are, so a Hermitian solver
# must not be applied to the product.  Use the generalized Hermitian problem
# Cd v = lambda Cn v, equivalently the spectrum of Cn^-1/2 Cd Cn^-1/2, and state
# the rank threshold: Cd is numerically singular here, so quote the spread over
# the numerically supported eigenvalues.
import scipy.linalg as _sla
ev=np.sort(_sla.eigh(Cd_p, Cn_p, eigvals_only=True).real)
tol=ev.max()*Cd_p.shape[0]*np.finfo(float).eps
kept=ev[ev>tol]
print(f"  generalized eigenvalues of (Cd, Cn): {len(kept)} of {len(ev)} above "
      f"tol={tol:.2e}; spread over those = {kept.max()/kept.min():.3g}")
# A spread computed across a numerically singular Cd is dominated by the
# floating-point floor.  Report instead a threshold-free measure of how far the
# two restricted covariances are from proportional: the effective rank
# tr(C)^2/tr(C^2), which is p for a white covariance.
er=lambda C: (np.trace(C).real**2)/np.trace(C@C).real
print(f"  effective rank of Cd_perp = {er(Cd_p):.2f} of {Cd_p.shape[0]};"
      f"  of Cn_perp = {er(Cn_p):.2f}")
print(f"  (the scalar weight is exact only when the two are proportional)")
print(f"  raw risk (beta=0)         = {np.trace(Cn_p).real:.6e}")
print(f"  full-projection (beta=1)  = {np.trace(Cd_p).real:.6e}")
print("="*74)
