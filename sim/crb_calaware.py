"""Calibration-aware nuisance-eliminated CRB for the target parameters.

Proposition 5 eliminates the background gains using the payload alone.  The
receiver of Section VI also holds the calibration snapshot, whose information
2 Re{[I, jI]^H C_cal^{-1} [I, jI]} adds to the nuisance block BEFORE the Schur
complement.  Writing T = W/sigma_w^2, the elimination still has a closed form,

    J_eff = 2 Re{ D_eta^H [ T - T B (B^H T B + C_cal^{-1})^{-1} B^H T ] D_eta },

which returns Proposition 5 as C_cal^{-1} -> 0 and the clairvoyant information as
C_cal^{-1} -> infinity.  In the collinear limit it returns Theorem 2(ii).

For W = s I and C_cal = sigma_perp^2 (B^H B)^{-1} the bracket collapses to
T (I - w P_B) with w = T sigma_perp^2 / (1 + T sigma_perp^2), so the three bounds
differ only through w: clairvoyant w = 0, calibration-aware w as above, and
payload-only w = 1.

NUMERICS.  The Fisher matrix mixes amplitude, seconds and radians, so its
singular values span many decades and np.linalg.pinv() silently truncates the
weak direction of a near-collinear draw.  Every inverse below is taken after
nondimensionalizing by `SCALES`, and `unit_invariance()` asserts that expressing
the delay in microseconds leaves the converted bound unchanged.

STATISTICS.  sqrt(mean over draws of the per-draw CRB) is the right reference for
an ensemble RMSE, and it is stable for the clairvoyant and calibration-aware
bounds.  It is NOT stable for the payload-only bound: that bound diverges as the
target atom approaches span(B), one draw in 2000 contributes about a quarter of
the angle-variance sum, split halves disagree by 50%, and the value drifts
monotonically with the number of draws.  For the payload-only bound this script
therefore reports the MEDIAN per-draw inflation over the clairvoyant bound, which
is stable, and prints the mean only with its instability diagnostics.
"""
import numpy as np, sys
sys.path.insert(0,'sim')
from isac_core import Scenario, steering

sc=Scenario(); Nr,K,df,N = sc.Nr,sc.K,sc.df,sc.N
kk=np.arange(K); B=sc.B; G=B.conj().T@B; Gi=np.linalg.inv(G)
S_SYM, SW2 = 13, 0.1                              # (M-1) unit-energy payload symbols
T = S_SYM/SW2                                     # W = s I, T = W/sigma_w^2
AM = np.sqrt(1e-3)                                # |alpha_r|
SCALES = np.array([AM, AM, 1e-6, 1.])             # amplitude, amplitude, us, rad

def atom(th,ta):   return np.outer(np.exp(-1j*2*np.pi*kk*df*ta), steering(th,Nr)).ravel()
def d_dtau(th,ta): return np.outer((-1j*2*np.pi*kk*df)*np.exp(-1j*2*np.pi*kk*df*ta), steering(th,Nr)).ravel()
def d_dth(th,ta):
    n=np.arange(Nr); a=steering(th,Nr)*(1j*np.pi*n*np.cos(th))
    return np.outer(np.exp(-1j*2*np.pi*kk*df*ta), a).ravel()

def weight(sig2):
    """w in M = T (I - w P_B):  0 clairvoyant, 1 payload-only, else calibration-aware."""
    if sig2 is None: return 1.0
    if sig2 == 0:    return 0.0
    return T*sig2/(1.0+T*sig2)

def fim(th,ta,alpha,sig2):
    D=np.column_stack([atom(th,ta), 1j*atom(th,ta), alpha*d_dtau(th,ta), alpha*d_dth(th,ta)])
    BD=B.conj().T@D
    return 2*T*np.real(D.conj().T@D - weight(sig2)*BD.conj().T@Gi@BD)

def crb(th,ta,alpha,sig2):
    """Per-draw CRB, inverted in nondimensionalized coordinates."""
    J=fim(th,ta,alpha,sig2)
    Jn=J*SCALES[:,None]*SCALES[None,:]
    C=np.linalg.inv(Jn)*SCALES[:,None]*SCALES[None,:]
    return C[2,2], C[3,3]

def unit_invariance(th,ta,alpha,sig2,tol=1e-8):
    """Converted bounds must not depend on the unit chosen for the delay."""
    a=np.array(crb(th,ta,alpha,sig2))
    J=fim(th,ta,alpha,sig2); u=np.array([1.,1.,1e-6,1.])
    Ju=J*u[:,None]*u[None,:]; sc2=SCALES/u
    Cu=np.linalg.inv(Ju*sc2[:,None]*sc2[None,:])*sc2[:,None]*sc2[None,:]
    b=np.array([Cu[2,2]*1e-12, Cu[3,3]])
    return np.max(np.abs(a-b)/a) < tol

if __name__=="__main__":
    th0,ta0,al0 = np.deg2rad(20.), .7e-6, .3
    # closed form against the explicit augmented real Fisher matrix
    D=np.column_stack([atom(th0,ta0),1j*atom(th0,ta0),al0*d_dtau(th0,ta0),al0*d_dth(th0,ta0)])
    Z=np.column_stack([B,1j*B]); Ir=np.hstack([np.eye(B.shape[1]),1j*np.eye(B.shape[1])])
    for sig2 in (1e-4,):
        Cci=G/sig2                                  # C_cal^{-1}, sigma_delta^2 = 0
        A=np.column_stack([D,Z]); J=2*T*np.real(A.conj().T@A)
        J[4:,4:] += 2*np.real(Ir.conj().T@Cci@Ir)
        Sch=J[:4,:4]-J[:4,4:]@np.linalg.solve(J[4:,4:],J[4:,:4])
        print(f"augmented-FIM Schur complement vs closed form: max rel discrepancy "
              f"{np.abs(Sch-fim(th0,ta0,al0,sig2)).max()/np.abs(Sch).max():.2e}")
    print("unit invariance (payload-only, worst-conditioned case):",
          all(unit_invariance(th0,ta0,al0,s) for s in (0,1e-4,1e-1,None)))

    rng=np.random.default_rng(7); ND=2000
    TAU=rng.uniform(.1e-6,3e-6,ND); TH=np.deg2rad(rng.uniform(-50,50,ND))
    AL=AM*np.exp(1j*rng.uniform(0,2*np.pi,ND))
    per={}
    for lab,sig in (('clairvoyant',0),('1e-5',1e-5),('1e-4',1e-4),('1e-3',1e-3),
                    ('1e-2',1e-2),('1e-1',1e-1),('payload-only',None)):
        per[lab]=np.array([crb(TH[j],TAU[j],AL[j],sig) for j in range(ND)])
    cl=per['clairvoyant']
    print(f"\n{'bound':13s} {'sqrt(mean) ns':>14} {'deg':>8} | "
          f"{'median inflation dB: delay':>28} {'angle':>7}")
    for lab,d in per.items():
        md=10*np.log10(np.median(d[:,0]/cl[:,0])); ma=10*np.log10(np.median(d[:,1]/cl[:,1]))
        print(f"{lab:13s} {np.sqrt(d[:,0].mean())*1e9:14.3f} "
              f"{np.rad2deg(np.sqrt(d[:,1].mean())):8.4f} | {md:28.3f} {ma:7.3f}")
    p=per['payload-only']; h=ND//2
    print("\npayload-only mean is NOT a converged statistic:")
    print(f"  top draw = {p[:,1].max()/p[:,1].sum()*100:.1f}% of the angle-variance sum")
    print(f"  split halves: {np.rad2deg(np.sqrt(p[:h,1].mean())):.2f} vs "
          f"{np.rad2deg(np.sqrt(p[h:,1].mean())):.2f} deg")
    print("  sqrt(mean) angle vs number of draws: " +
          ", ".join(f"n={n}:{np.rad2deg(np.sqrt(p[:n,1].mean())):.1f}" for n in (250,500,1000,2000)))
    print("  -> quote the median inflation, not the mean.")
