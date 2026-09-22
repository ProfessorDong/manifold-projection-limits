"""Table II: in-span fraction mu-bar of an unmodeled field (Proposition 2).

Evaluates the mean-squared-coherence identity

    mu_bar = E_pi[ ||Q^H b(phi,xi)||^2 ] / E_pi[ ||b(phi,xi)||^2 ],   Q = orth(B),

deterministically, so the table carries no Monte Carlo error.  Since
||b(phi,xi)||^2 = N for every atom, mu_bar = tr(P_B E_pi[b b^H])/N, and with
angle and delay drawn independently E_pi[b b^H] = C_tau (x) C_a.  The delay
factor is integrated ANALYTICALLY through the characteristic function of the
delay law; Gauss-Laguerre quadrature on the exponential does not converge at
large K*df, where the integrand oscillates over many cycles across the nodes.

Two choices of pi:
  manifold-spread : phi ~ U[-60,60] deg,  xi ~ U[0.05, 1.70] us
  clustered       : phi ~ N(theta_i, 9 deg), xi ~ t_i + Exp(0.30 us),
                    i uniform over the L+1 background paths  (Quitin et al.)
"""
import numpy as np, sys
sys.path.insert(0,'sim')
from isac_core import build_B

TH = np.deg2rad([10.,-25.,35.,60.]); TA = np.array([.20,.55,.90,1.25])*1e-6
DF = 15e3
SG = np.deg2rad(9.0); TAU_D = 0.30e-6
XI_LO, XI_HI = 0.05e-6, 1.70e-6

def _angular_cov(Nr, centers, weights, angles):
    """C_a = E[a(phi) a(phi)^H] by quadrature over the angular law."""
    A = np.exp(1j*np.pi*np.outer(np.arange(Nr), np.sin(angles)))
    return (A*weights) @ A.conj().T

def mubar(Nr, K, kind, df=None, nang=200):
    df = DF if df is None else df
    B = build_B(TH, TA, Nr, K, df); G = B.conj().T @ B
    kk = np.arange(K)
    F = np.exp(-2j*np.pi*df*np.outer(kk, TA))          # delay factors of the atoms
    V = np.exp(1j*np.pi*np.outer(np.arange(Nr), np.sin(TH)))
    om = 2*np.pi*df*(kk[:,None] - kk[None,:])          # 2 pi df (k - l)
    H = np.zeros((len(TH), len(TH)), complex)
    if kind == "manifold":
        z, w = np.polynomial.legendre.leggauss(nang)
        Ca = _angular_cov(Nr, None, w/2, z*np.pi/3)
        # E[exp(-j om U)] for U ~ U[XI_LO, XI_HI]
        Ct = np.exp(-1j*om*(XI_LO+XI_HI)/2)*np.sinc(om*(XI_HI-XI_LO)/(2*np.pi))
        H = (F.conj().T @ Ct @ F) * (V.conj().T @ Ca @ V)
    else:
        hx, hw = np.polynomial.hermite_e.hermegauss(nang); hw = hw/hw.sum()
        for j in range(len(TH)):
            Ca = _angular_cov(Nr, None, hw, TH[j] + SG*hx)
            # E[exp(-j om (t_j + Exp(tau_d)))] exactly
            Ct = np.exp(-1j*om*TA[j])/(1 + 1j*om*TAU_D)
            H += ((F.conj().T @ Ct @ F) * (V.conj().T @ Ca @ V))/len(TH)
    return np.trace(np.linalg.solve(G, H)).real/(Nr*K)

if __name__ == "__main__":
    print(f"{'(Nr,K)':>9} {'N':>6} {'kappa':>9} {'manifold':>10} {'clustered':>10} {'clus/kap':>9} {'man/kap':>8}")
    for Nr,K in ((4,64),(8,256),(8,512)):
        N = Nr*K; kap = 4/N
        m = mubar(Nr,K,"manifold"); c = mubar(Nr,K,"clustered")
        print(f"{str((Nr,K)):>9} {N:6d} {kap:9.5f} {m:10.4f} {c:10.4f} {c/kap:8.1f}x {m/kap:7.1f}x")
    print("\nSubcarrier-spacing sweep at fixed N=4096 and fixed physical delay support:")
    for df in (15e3, 60e3):
        c = mubar(8,512,"clustered",df=df); m = mubar(8,512,"manifold",df=df)
        print(f"  df={df/1e3:5.0f} kHz: support={(XI_HI-XI_LO)*df*100:5.2f}% of 1/df, "
              f"clustered mu={c:.5f} ({c/(4/4096):.1f}x kappa), manifold mu={m:.5f} ({m/(4/4096):.1f}x)")
