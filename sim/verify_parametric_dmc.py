"""Akram's question: would a parametric dense-multipath model plus a matrix Wiener
filter beat the scalar plug-in weight of Section V?

The scalar weight is the MMSE filter within the class Psi = (1-beta)I.  The
unrestricted MMSE filter on the complement is the Wiener filter
Psi* = Cd_perp (Cd_perp + Cn_perp)^{-1}, and the gap between them is set by how far
the two restricted covariances are from proportional.  Two questions follow.

(1) How large is the gap under the WHITE calibration noise the manuscript's model
    assumes?  Section V-B illustrates it with a correlated C_n instead, which is
    much kinder to the scalar weight: holding the complement noise energy A fixed
    holds the raw risk, the full-projection risk and the best scalar risk fixed, so
    the noise SHAPE alone moves the matrix filter.  See main() part 1.

(2) Can a parametric model realize the gap from the single calibration snapshot?
    Richter's DMC has three parameters (power, delay decay, base delay) and assumes
    a spatially white field, so it is misspecified for both diffuse fields used
    here.  It is fitted by ML on the complement and evaluated on the same snapshot,
    exactly as a receiver would have to.  See main() part 2.

Filters compared, all on the same draws:
  raw       Psi = I                                            (no projection)
  proj      Psi = 0                                            (full projection)
  scalar    w = A / ||Pperp h_cal||^2, the manuscript's plug-in    (1 parameter)
  param     ML fit of (alpha, tau_d, tau_r), then Psi_hat          (3 parameters)
  oracle    Psi from the exact ensemble C_d                 (252x252, unknowable)
"""
import numpy as np, sys
sys.path.insert(0, 'sim')
from isac_core import Scenario, steering, draw_diffuse_clustered

sc = Scenario()
N, r, Nr, K, df = sc.N, sc.r, sc.Nr, sc.K, sc.df
M = N - r
SN2 = 1e-4
kk = np.arange(K)

# ---- orthonormal basis of the complement; index = k*Nr + n --------------------
Q = np.linalg.qr(sc.B, mode='complete')[0]
U = Q[:, r:]
assert U.shape == (N, M) and np.allclose(U.conj().T @ U, np.eye(M), atol=1e-9)
assert np.abs(sc.B.conj().T @ U).max() < 1e-9

def toeplitz_from(gen):
    i = np.arange(len(gen)); dk = i[:, None] - i[None, :]
    return np.where(dk >= 0, gen[np.abs(dk)], np.conj(gen[np.abs(dk)]))

def effrank(C):
    return (np.trace(C).real**2)/np.trace(C @ C).real

# ---- exact ensemble covariance of Scenario.draw_diffuse ----------------------
# d = sum_j c_j b(phi_j, xi_j) s, c iid CN(0,1), phi ~ U[-pi/3, pi/3],
# xi ~ U[0.05, 1.70] us  =>  C_d = (P_d/N) kron(C_tau, C_theta), unit diagonal.
_g = 200001
_xi = np.linspace(0.05e-6, 1.70e-6, _g)
_phi = np.linspace(-np.pi/3, np.pi/3, _g)
C_SPREAD = np.kron(
    toeplitz_from(np.exp(-1j*2*np.pi*np.outer(kk, _xi)*df).mean(axis=1)),
    toeplitz_from(np.exp(1j*np.pi*np.outer(np.arange(Nr), np.sin(_phi))).mean(axis=1)))
assert np.allclose(np.diag(C_SPREAD), 1.0, atol=1e-6)

# ---- ensemble covariance of the clustered field, by atom averaging -----------
def _clustered_cov(n_atoms=6000, seed=12):
    rng = np.random.default_rng(seed)
    L1 = len(sc.theta_bg); per = int(np.ceil(n_atoms/L1)); ph = []; xi = []
    for i in range(L1):
        ph.append(rng.vonmises(sc.theta_bg[i], 40.0, per))
        xi.append(sc.tau_bg[i] + rng.exponential(0.30e-6, per))
    ph = np.clip(np.concatenate(ph)[:n_atoms], -np.pi/2.2, np.pi/2.2)
    xi = np.clip(np.concatenate(xi)[:n_atoms], 0.02e-6, 2.8e-6)
    C = np.zeros((N, N), complex)
    for p, x in zip(ph, xi):
        b = np.outer(np.exp(-1j*2*np.pi*kk*df*x), steering(p, Nr)).ravel()
        C += np.outer(b, b.conj())
    return C*(N/np.trace(C).real)                    # unit diagonal on average
C_CLUST = _clustered_cov()

FIELDS = {
    'spread':  (C_SPREAD, lambda rng, dB: sc.draw_diffuse(rng, dB)),
    'clustered': (C_CLUST, lambda rng, dB: draw_diffuse_clustered(sc, rng, dB)),
}
def C_true(field, dB):
    return (10**(dB/10.)*sc.E_spec/N)*FIELDS[field][0]

# ---- three-parameter DMC family (Richter): exponential PDP, spatially white ---
# C_tau[k,l] = tau_r exp(-j w tau_d)/(1 + j w tau_r),  w = 2 pi df (k-l)
def dmc_unit(tau_d, tau_r):
    w = 2*np.pi*df*np.arange(K)
    return np.kron(toeplitz_from(np.exp(-1j*w*tau_d)/(1 + 1j*w*tau_r)), np.eye(Nr))

TAU_D = np.linspace(0.02e-6, 1.20e-6, 13)
TAU_R = np.geomspace(0.06e-6, 3.0e-6, 13)
GRID = []
def build_grid():
    if GRID: return
    print(f"  precomputing {len(TAU_D)*len(TAU_R)} eigendecompositions "
          f"({M}x{M}) ...", flush=True)
    for td in TAU_D:
        for tr in TAU_R:
            S = U.conj().T @ dmc_unit(td, tr) @ U
            lam, V = np.linalg.eigh((S + S.conj().T)/2)
            GRID.append((np.maximum(lam, 0.0), V))

def fit_dmc(y, sn2, n_alpha=60):
    """ML over (alpha, tau_d, tau_r) for y ~ CN(0, alpha S(td,tr) + sn2 I)."""
    best = None
    a_hi = max(np.vdot(y, y).real/M, 1e-18)*40
    alphas = np.geomspace(a_hi*1e-5, a_hi, n_alpha)
    for lam, V in GRID:
        p = np.abs(V.conj().T @ y)**2
        den = alphas[:, None]*lam[None, :] + sn2
        nll = np.log(den).sum(axis=1) + (p[None, :]/den).sum(axis=1)
        j = int(np.argmin(nll))
        if best is None or nll[j] < best[0]:
            best = (nll[j], alphas[j], lam, V)
    _, a, lam, V = best
    return V @ ((a*lam/(a*lam + sn2))[:, None]*V.conj().T)

def run(field, dB, sn2=SN2, T=160, seed=11):
    rng = np.random.default_rng(seed)
    S = U.conj().T @ C_true(field, dB) @ U
    S = (S + S.conj().T)/2
    Psi_o = S @ np.linalg.inv(S + sn2*np.eye(M))
    A = sn2*M
    draw = FIELDS[field][1]
    err = {k: 0.0 for k in ('raw', 'proj', 'scalar', 'param', 'oracle')}
    for _ in range(T):
        d = draw(rng, dB)
        n = (rng.standard_normal(N) + 1j*rng.standard_normal(N))*np.sqrt(sn2/2)
        s = U.conj().T @ d
        y = U.conj().T @ (d + n)
        w = float(np.clip(A/max(np.vdot(y, y).real, 1e-300), 0, 1))
        Psi_p = fit_dmc(y, sn2)
        for k, e in (('raw', y - s), ('proj', -s), ('scalar', (1-w)*y - s),
                     ('param', Psi_p @ y - s), ('oracle', Psi_o @ y - s)):
            err[k] += np.vdot(e, e).real
    return {k: v/T for k, v in err.items()}, effrank(S)

def main():
    print("="*78)
    print("PART 1  the noise SHAPE, not the diffuse field, sets the gap in Sec. V-B")
    print("="*78)
    Rk = np.exp(-np.abs(kk[:, None] - kk[None, :])/12.0)
    Ra = np.eye(Nr) + 0.4*(np.ones((Nr, Nr)) - np.eye(Nr))
    Cn_corr = np.kron(Rk, Ra)*3e-4
    Cd_p = U.conj().T @ C_true('clustered', -30) @ U
    A = np.trace(U.conj().T @ Cn_corr @ U).real
    D = np.trace(Cd_p).real
    bst = A/(A + D)
    risk_sca = (1 - bst)**2*A + bst**2*D
    print(f"  clustered C_d at -30 dB: effective rank {effrank(Cd_p):.2f} of {M}, D = {D:.6f}")
    print(f"  complement noise energy A = {A:.6f} HELD FIXED, so beta* = {bst:.4f},")
    print(f"  raw = {A:.4e}, full projection = {D:.4e}, best scalar = {risk_sca:.4e}")
    print(f"  are the same in both rows below; only the noise shape moves.\n")
    print(f"  {'noise shape':>26} {'effrank(Cn)':>12} {'matrix Wiener':>14} {'scalar penalty':>15}")
    for name, Cn in (("correlated (as in Sec. V-B)", Cn_corr),
                     ("white (the paper's model)", np.eye(N)*(A/M))):
        Cn_p = U.conj().T @ Cn @ U
        Cn_p = (Cn_p + Cn_p.conj().T)/2
        rm = np.trace(Cd_p @ np.linalg.inv(Cd_p + Cn_p) @ Cn_p).real
        print(f"  {name:>26} {effrank(Cn_p):12.2f} {rm:14.4e} "
              f"{10*np.log10(risk_sca/rm):12.2f} dB")

    print("\n" + "="*78)
    print("PART 2  can three parameters realize it from one snapshot?")
    print("="*78)
    build_grid()
    print("  complement risk E||e_perp||^2, same draws, white noise sn2 = 1e-4\n")
    hdr = (f"  {'field':>10} {'dB':>5} {'effrank':>8} {'raw':>9} {'proj':>9} "
           f"{'scalar':>9} {'param':>9} {'oracle':>9} {'sc/or':>7} {'pa/or':>7}")
    print(hdr); print("  " + "-"*(len(hdr)-2))
    for field in ('spread', 'clustered'):
        for dB in (-50, -40, -35, -30, -20):
            e, eff = run(field, dB)
            print(f"  {field:>10} {dB:5d} {eff:8.2f} {e['raw']:9.5f} {e['proj']:9.5f} "
                  f"{e['scalar']:9.5f} {e['param']:9.5f} {e['oracle']:9.5f} "
                  f"{e['scalar']/e['oracle']:7.2f} {e['param']/e['oracle']:7.2f}")
    print(f"\n  effrank = (tr S)^2 / tr S^2 out of {M}.  The DMC family is misspecified")
    print("  for both fields (neither has an exponential PDP, neither is spatially")
    print("  white), yet it recovers all but about 1 dB of the oracle's advantage.")

def part3():
    """Is the concentration an artifact of the 0.96 MHz scenario simulated here?
    Both risks are closed form in the eigenvalues of S, so no draws are needed."""
    from isac_core import build_B
    print("\n" + "="*78)
    print("PART 3  does the concentration survive more bandwidth?")
    print("="*78)
    th = sc.theta_bg; ta = sc.tau_bg; gg = sc.g
    spread = 1.70e-6 - 0.05e-6
    Cth = toeplitz_from(np.exp(1j*np.pi*np.outer(np.arange(Nr), np.sin(_phi))).mean(axis=1))
    print(f"  {'K':>5} {'BW[MHz]':>8} {'spread*BW':>10} {'N-L-1':>7} {'effrank':>8} "
          f"{'effrank/p':>10} {'scalar':>10} {'oracle':>10} {'ratio':>7}")
    for Kx in (64, 128, 256, 512, 1024):
        Bx = build_B(th, ta, Nr, Kx, df); Nx = Nr*Kx; rx = Bx.shape[1]; Mx = Nx - rx
        Es = np.abs(np.vdot(Bx @ gg, Bx @ gg)).real
        Ux = np.linalg.qr(Bx, mode='complete')[0][:, rx:]
        Ct = toeplitz_from(np.exp(-1j*2*np.pi*np.outer(np.arange(Kx), _xi)*df).mean(axis=1))
        Cdx = (10**(-30/10.)*Es/Nx)*np.kron(Ct, Cth)
        Sx = Ux.conj().T @ Cdx @ Ux; Sx = (Sx + Sx.conj().T)/2
        lam = np.maximum(np.linalg.eigvalsh(Sx), 0)
        D = lam.sum(); A = SN2*Mx
        scl = A*D/(A + D); orc = SN2*np.sum(lam/(lam + SN2))
        ef = D**2/np.sum(lam**2)
        print(f"  {Kx:5d} {Kx*df/1e6:8.2f} {spread*Kx*df:10.2f} {Mx:7d} {ef:8.2f} "
              f"{ef/Mx:10.4f} {scl:10.5f} {orc:10.5f} {scl/orc:7.2f}")
    print("\n  Effective rank grows in proportion to the bandwidth, but so does N-L-1;")
    print("  their ratio barely moves, and the scalar penalty grows rather than shrinks.")

if __name__ == "__main__":
    main()
    part3()
