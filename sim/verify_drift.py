"""
(a) beta* is invariant to in-manifold drift  (drift lives in span(B), where the
    shrinkage family applies unit weight -> cannot be reduced by any beta).
(b) Proposition 1 with drift: R_proj = (1+rho)/(rho+kappa), now WITH diffuse too.
"""
import numpy as np, sys
sys.path.insert(0,'sim')
from isac_core import Scenario, est_structured, est_shrink, beta_plugin

rng = np.random.default_rng(23)
sc = Scenario(); N, r = sc.N, sc.r
kappa = r/N
trB = np.trace(sc.B.conj().T@sc.B).real

def batch(sd2, sn2, ddB, T=500):
    """Returns arrays: Ht, Hc, PHc (=P_B Hc), each (T,N)."""
    Ht=np.empty((T,N),complex); Hc=np.empty((T,N),complex)
    for t in range(T):
        delta=(rng.standard_normal(r)+1j*rng.standard_normal(r))*np.sqrt(sd2/2)
        n=(rng.standard_normal(N)+1j*rng.standard_normal(N))*np.sqrt(sn2/2)
        d=sc.draw_diffuse(rng, ddB)
        Ht[t]=sc.B@sc.g+d; Hc[t]=sc.B@(sc.g-delta)+d+n
    PHc = Hc @ sc.PB.T
    return Ht, Hc, PHc

def mse_beta(Ht,Hc,PHc,b):
    E=(1-b)*Hc + b*PHc - Ht
    return float(np.mean(np.einsum('ti,ti->t', E.conj(), E).real))

def nrm2(A):
    return np.einsum('ti,ti->t', A.conj(), A).real

print("="*80)
print("(a) beta* invariance to drift   [sn2=1e-4, diffuse=-30 dB]")
print("="*80)
sn2, ddB = 1e-4, -30.0
A = sn2*(N-r)
print(f"  {'sd2':>10} {'rho':>9} {'argmin_beta (emp)':>18} {'beta* pred':>11} {'beta_hat':>9}")
for sd2 in [0.0, 1e-6, 1e-5, 1e-4, 1e-3]:
    Ht,Hc,PHc = batch(sd2, sn2, ddB, 400)
    Dd = float(np.mean(nrm2(Ht - Ht@sc.PB.T)))          # E||P_perp d||^2
    grid=np.linspace(0,1,401)
    mses=[mse_beta(Ht,Hc,PHc,b) for b in grid]
    bemp=grid[int(np.argmin(mses))]
    bhat=float(np.mean([beta_plugin(Hc[i],sc,sn2) for i in range(Hc.shape[0])]))
    rho = sd2*r/(sn2)
    print(f"  {sd2:10.1e} {rho:9.3f} {bemp:18.3f} {A/(A+Dd):11.3f} {bhat:9.3f}")

print("\n" + "="*80)
print("(b) Proposition 1 with drift AND diffuse:  R_proj = raw_err / structured_err")
print("="*80)
print(f"  kappa = {kappa:.6f}, 1/kappa = {1/kappa:.1f}")
print(f"  {'sd2':>9} {'rho':>8} {'diffuse':>9} {'R_proj emp':>11} {'R_proj Prop1':>13} {'R_shrink':>10}")
for sd2 in [0.0, 1e-5, 1e-4]:
    for ddB in [None, -30.0]:
        Ht,Hc,PHc = batch(sd2, sn2, ddB, 400)
        raw=float(np.mean(nrm2(Hc-Ht)))
        st =float(np.mean(nrm2(PHc-Ht)))
        bs=np.array([beta_plugin(Hc[i],sc,sn2) for i in range(Hc.shape[0])])[:,None]
        sh =float(np.mean(nrm2((1-bs)*Hc + bs*PHc - Ht)))
        rho = sd2*r/sn2
        prop1 = (1+rho)/(rho+kappa)
        lab = "none" if ddB is None else f"{ddB:.0f} dB"
        print(f"  {sd2:9.1e} {rho:8.2f} {lab:>9} {raw/st:11.2f} {prop1:13.2f} {raw/sh:10.2f}")
print("="*80)
print("Prop.1 closed form applies only when there is NO unmodeled diffuse (rows 'none').")
print("With diffuse present, R_proj collapses below 1 while the shrinkage stays >= 1.")
