"""Numbers quoted in the manuscript that are not produced by any other script.

Each block prints the value as it appears in the paper, with the section that
quotes it.  Fast: no measured data, no long Monte Carlo.
"""
import numpy as np, sys
sys.path.insert(0, 'sim')
from isac_core import Scenario, build_B

sc = Scenario(); B = sc.B; N = sc.N; P = sc.P_perp
TH, TA, G = sc.theta_bg, sc.tau_bg, sc.g

print("=" * 72)
print("Sec. IV  per-chain drift is not confined to span(B):  'about 47%'")
# A residual per-chain calibration coefficient multiplies antenna m by (1+e_m),
# so the error is diag(e) B g, not B delta.  Index = k*Nr + m, antenna varies
# fastest, hence np.tile(e, K).
rng = np.random.default_rng(3); hs = B @ G; acc = []
for _ in range(20000):
    e = (rng.standard_normal(sc.Nr) + 1j*rng.standard_normal(sc.Nr))/np.sqrt(2)
    v = np.tile(e, sc.K) * hs
    acc.append((np.vdot(v, v).real - np.vdot(v, P @ v).real, np.vdot(v, v).real))
acc = np.array(acc)
print(f"  E||P_B diag(e)Bg||^2 / E||diag(e)Bg||^2 = {acc[:,0].sum()/acc[:,1].sum():.4f}")

print("=" * 72)
print("Sec. II  a perturbed atom does leave energy outside the nominal span")
for lab, th, ta in (("theta_0 10 -> 11 deg", np.deg2rad(11.), TA[0]),
                    ("theta_0 10 -> 10.5 deg", np.deg2rad(10.5), TA[0]),
                    ("tau_0 + 10 ns", TH[0], TA[0] + 10e-9)):
    b = build_B([th], [ta], sc.Nr, sc.K, sc.df)[:, 0]
    print(f"  {lab:24s} out-of-span fraction {np.vdot(b, P@b).real/np.vdot(b, b).real:.6f}")

print("=" * 72)
print("Sec. IV  the full-support example needs INDEPENDENT angle and delay")
n = 400_000; kk = np.arange(sc.K); ll = np.arange(sc.Nr)
def entry(s, t):                      # E[ b_{k=1,l=1} b_{k=0,l=0}^* ]
    return np.mean(np.exp(1j*(-2*np.pi*sc.df*1*t + np.pi*1*s)))
U = rng.uniform(0, 1, n)
print(f"  independent sin(theta) ~ U[-1,1], tau ~ U[0,1/df):  "
      f"{entry(rng.uniform(-1,1,n), rng.uniform(0,1/sc.df,n)):.4f}")
print(f"  both driven by one uniform variate (same marginals): "
      f"{entry(2*U-1, U/sc.df):.4f}")

print("=" * 72)
print("Table II caption  quadrature law vs simulator law, Config. A")
from table_inspan import mubar
om = 2*np.pi*sc.df*(kk[:, None] - kk[None, :])
F = np.exp(-2j*np.pi*sc.df*np.outer(kk, TA)); V = np.exp(1j*np.pi*np.outer(ll, np.sin(TH)))
H = np.zeros((4, 4), complex); ang = np.linspace(-np.pi, np.pi, 16384, endpoint=False)
for j in range(4):                    # clipped von Mises(40) x clipped exponential
    w = np.exp(40*(np.cos(ang - TH[j]) - 1)); w /= w.sum()
    Av = np.exp(1j*np.pi*np.outer(ll, np.sin(np.clip(ang, -np.pi/2.2, np.pi/2.2))))
    cap = 2.8e-6 - TA[j]; z = 1/.30e-6 + 1j*om
    Ct = np.exp(-1j*om*TA[j])*((1 - np.exp(-z*cap))/(.30e-6*z) + np.exp(-z*cap))
    H += ((F.conj().T @ Ct @ F)*(V.conj().T @ ((Av*w) @ Av.conj().T) @ V))/4
mu_sim = np.trace(np.linalg.solve(B.conj().T @ B, H)).real/N
mu_tab = mubar(4, 64, 'clustered')
print(f"  Gaussian 9 deg / unbounded exponential (table): {mu_tab:.6f}")
print(f"  clipped von Mises(40) / clipped exponential   : {mu_sim:.6f}")
print(f"  difference                                    : {mu_tab-mu_sim:.6f}")

print("=" * 72)
print("Thm. 2 proof  the floor binds unbiased estimators only")
v, a = 1.0, 0.2
for lam in (1.0, 0.5, 0.2):
    print(f"  lambda={lam:4.2f}: risk = lam^2 v + (1-lam)^2 |alpha|^2 = "
          f"{lam**2*v + (1-lam)**2*abs(a)**2:.4f}   (floor v = {v})")
z = a + (rng.normal(size=1_000_000) + 1j*rng.normal(size=1_000_000))*np.sqrt(v/2)
print(f"  Monte Carlo at lambda=0.5: {np.mean(abs(.5*z - a)**2):.4f}")
print("  sup over alpha is unbounded for lambda<1, so v is also the minimax risk.")
