"""CRB for the target parameters with the BACKGROUND GAINS as nuisance.

Section V currently plots a genie bound computed with g known.  Following the
nuisance-parameter treatment of [Rivetti et al.], augment the parameter vector
with g and take the Schur complement.  Because the nuisance derivatives
d(mu)/d(Re g_i), d(mu)/d(Im g_i) span exactly the complex subspace span(B), the
elimination reduces to replacing the Gram matrix of the target derivatives by
its projection onto span(B)^perp:

    J_eff = (2/sigma^2) Re{ D_eta^H P_perp D_eta }.

Verified below against the explicit Schur complement.  It reproduces Theorem 2:
as b(tau_r,theta_r) -> span(B), P_perp b -> 0 and J_eff becomes singular."""
import numpy as np, sys
sys.path.insert(0,'sim')
from isac_core import Scenario, steering

sc = Scenario(); Nr, K, df = sc.Nr, sc.K, sc.df
kk = np.arange(K)
def atom(th,ta):  return np.outer(np.exp(-1j*2*np.pi*kk*df*ta), steering(th,Nr)).ravel()
def d_dtau(th,ta):return np.outer((-1j*2*np.pi*kk*df)*np.exp(-1j*2*np.pi*kk*df*ta), steering(th,Nr)).ravel()
def d_dth(th,ta):
    n=np.arange(Nr); a=steering(th,Nr)*(1j*np.pi*n*np.cos(th))
    return np.outer(np.exp(-1j*2*np.pi*kk*df*ta), a).ravel()

def fim(th,ta,alpha,B,sigma2=1.0):
    b,bt,bth = atom(th,ta), d_dtau(th,ta), d_dth(th,ta)
    D = np.column_stack([b, 1j*b, alpha*bt, alpha*bth])
    Z = np.column_stack([B, 1j*B])
    g = lambda U,V: (2/sigma2)*np.real(U.conj().T@V)
    Jee,Jez,Jzz = g(D,D), g(D,Z), g(Z,Z)
    Js = Jee - Jez@np.linalg.pinv(Jzz)@Jez.T
    P  = np.eye(B.shape[0]) - B@np.linalg.solve(B.conj().T@B, B.conj().T)
    Jp = g(D, P@D)
    return Jee, Js, Jp, P

if __name__=="__main__":
    B=sc.B; alpha=0.3
    th0,ta0 = sc.theta_bg[0], sc.tau_bg[0]
    Jg,Js,Jp,_ = fim(np.deg2rad(20.),0.7e-6,alpha,B)
    print("Schur elimination vs projection onto span(B)^perp:")
    print(f"  max|J_schur - J_proj| = {np.abs(Js-Jp).max():.3e}  (matrix scale {np.abs(Js).max():.3e})")
    print("\nCRB inflation from treating the background gains as unknown.")
    print("Target swept in angle toward background path 1 (10 deg), delay offset +0.30 us.")
    print(f"{'d_theta[deg]':>12} {'|P_perp b|^2/|b|^2':>19} {'tau infl[dB]':>13} {'theta infl[dB]':>15}")
    for dth in [40.,20.,10.,5.,2.,1.,0.5,0.2,0.0]:
        th=th0+np.deg2rad(dth); ta=ta0+0.30e-6
        Jg,Js,Jp,P = fim(th,ta,alpha,B)
        b=atom(th,ta); frac=np.vdot(b,P@b).real/np.vdot(b,b).real
        cg,cs=np.linalg.pinv(Jg),np.linalg.pinv(Js)
        print(f"{dth:12.2f} {frac:19.5f} {10*np.log10(cs[2,2]/cg[2,2]):13.2f} {10*np.log10(cs[3,3]/cg[3,3]):15.2f}")
