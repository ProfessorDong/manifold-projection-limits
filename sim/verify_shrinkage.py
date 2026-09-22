"""
Verification of the optimal partial-projection (shrinkage) result.

Model:  h_true = B g + d           (current background, d = unmodeled diffuse)
        h_cal  = B(g - delta) + d + n     (calibration snapshot; d common, quasi-static)
        delta ~ CN(0, sd2 I),  n ~ CN(0, sn2 I)

Estimator family:  h_beta = (1-beta) h_cal + beta P_B h_cal

Claim (Theorem):
   MSE(beta) = [ sd2 tr(B^H B) + sn2 r ]  +  (1-beta)^2 A + beta^2 D
   with A = sn2 (N-r) = E||P_perp n||^2 ,  D = E||P_perp d||^2
   => beta* = A/(A+D),  MSE_perp(beta*) = A D/(A+D)  <  min(A, D)
"""
import numpy as np, sys
sys.path.insert(0, 'sim')
from isac_core import Scenario, est_raw, est_structured, est_shrink, beta_oracle, beta_plugin

rng = np.random.default_rng(11)
sc = Scenario()
N, r = sc.N, sc.r

def trial(sd2, sn2, diffuse_dB, rng):
    g = sc.g
    delta = (rng.standard_normal(r) + 1j*rng.standard_normal(r)) * np.sqrt(sd2/2)
    n = (rng.standard_normal(N) + 1j*rng.standard_normal(N)) * np.sqrt(sn2/2)
    d = sc.draw_diffuse(rng, diffuse_dB)
    h_true = sc.B @ g + d
    h_cal = sc.B @ (g - delta) + d + n
    return h_true, h_cal, d, n

def mse(vals): return float(np.mean(vals))

print("="*78)
print("CHECK 1: MSE(beta) decomposition and beta* optimality")
print("="*78)
sd2, sn2, ddB, T = 0.0, 1e-4, -20.0, 600
Hs = [trial(sd2, sn2, ddB, rng) for _ in range(T)]
D_emp = mse([np.vdot(sc.P_perp@d, sc.P_perp@d).real for _,_,d,_ in Hs])
A_th = sn2*(N-r)
print(f"  A (theory) = sn2(N-r) = {A_th:.5f}")
print(f"  D (empirical E||P_perp d||^2) = {D_emp:.5f}")
b_star = A_th/(A_th+D_emp)
print(f"  beta* = A/(A+D) = {b_star:.5f}")
print(f"\n  {'beta':>8} {'MSE emp':>12} {'MSE theory':>12}")
const = sd2*np.trace(sc.B.conj().T@sc.B).real + sn2*r
for b in [0.0, 0.25, b_star, 0.75, 1.0]:
    e = mse([np.vdot(est_shrink(hc,sc,b)-ht, est_shrink(hc,sc,b)-ht).real for ht,hc,_,_ in Hs])
    th = const + (1-b)**2*A_th + b**2*D_emp
    tag = "  <- beta*" if abs(b-b_star)<1e-9 else ""
    print(f"  {b:8.4f} {e:12.5f} {th:12.5f}{tag}")
print(f"\n  predicted MSE_perp(beta*) = A*D/(A+D) = {A_th*D_emp/(A_th+D_emp):.6f}")
print(f"  vs raw (A) = {A_th:.6f}   vs structured (D) = {D_emp:.6f}")
print(f"  -> gain over raw = {A_th/(A_th*D_emp/(A_th+D_emp)):.2f}x ,"
      f"  over structured = {D_emp/(A_th*D_emp/(A_th+D_emp)):.2f}x")

print("\n" + "="*78)
print("CHECK 2: plug-in beta (no oracle knowledge) vs oracle beta")
print("="*78)
print(f"  {'diffuse dB':>10} {'beta*':>8} {'beta_hat':>9} {'NMSE raw':>10} {'NMSE str':>10} "
      f"{'NMSE b*':>10} {'NMSE bhat':>10}")
for ddB in [None, -45, -41.5, -35, -30, -25, -20, -15, -10]:
    T = 400
    er=[];es=[];eb=[];ep=[];bh=[]
    Ds=[]
    for _ in range(T):
        ht, hc, d, n = trial(sd2, sn2, ddB, rng)
        Ds.append(np.vdot(sc.P_perp@d, sc.P_perp@d).real)
        er.append(np.vdot(hc-ht,hc-ht).real)
        hs = est_structured(hc,sc); es.append(np.vdot(hs-ht,hs-ht).real)
        bhat = beta_plugin(hc, sc, sn2); bh.append(bhat)
        hp = est_shrink(hc,sc,bhat); ep.append(np.vdot(hp-ht,hp-ht).real)
    Dm = np.mean(Ds); bstar = A_th/(A_th+Dm)
    for _ in range(T):
        pass
    # oracle-beta arm on a fresh but identical-statistics batch
    eb=[]
    for _ in range(T):
        ht, hc, d, n = trial(sd2, sn2, ddB, rng)
        hb = est_shrink(hc,sc,bstar); eb.append(np.vdot(hb-ht,hb-ht).real)
    lab = "none" if ddB is None else f"{ddB}"
    print(f"  {lab:>10} {bstar:8.4f} {np.mean(bh):9.4f} "
          f"{np.mean(er)/sc.E_spec:10.2e} {np.mean(es)/sc.E_spec:10.2e} "
          f"{np.mean(eb)/sc.E_spec:10.2e} {np.mean(ep)/sc.E_spec:10.2e}")
print("="*78)
