"""Two claims from Akram's review, checked numerically.

(A) 3 dB safety bound.  If the persistent-case plug-in beta = A/(A+D) is used
    when the true cross-correlation is C01 < D, the excess risk relative to the
    optimal weight is (D-C01)^2/(A+D), so the ratio is
        1 + (D-C01)^2 / [ (A+D)D - C01^2 ],
    maximised at C01 = 0 where it equals 1 + D/(A+D) < 2, i.e. < 3.01 dB.
    Requires D0 = D1 = D (stationary field); it fails if D1 < D0.

(B) Combined drift + unmodeled field.  With a persistent d, the raw error is
    -B*delta + n and the projected error is -B*delta + P_B n - P_perp d, so
        R_proj = (1+rho)/(rho + kappa + nu),   nu = f_perp*D_tot/(N sigma^2),
    which contains Prop.1 (nu=0) and the crossover (rho=0) as special cases."""
import numpy as np, sys
sys.path.insert(0,'sim')
from isac_core import Scenario

print("="*72); print("(A) 3 dB safety bound, D0 = D1 = D")
print(f"{'A/D':>8} {'C01/D':>7} {'ratio':>9} {'dB':>7} {'bound 1+D/(A+D)':>17}")
worst=0.0
for AD in (0.01,0.1,1.0,10.0):
    for c in (0.0,0.25,0.5,0.75,1.0):
        D=1.0; A=AD*D; C=c*D
        R_opt =D-C**2/(A+D)
        R_plug=(D/(A+D))**2*(A+D)+D-2*(D/(A+D))*C
        r=R_plug/R_opt; worst=max(worst,r)
        if c in (0.0,0.5): print(f"{AD:8.2f} {c:7.2f} {r:9.4f} {10*np.log10(r):7.3f} {1+D/(A+D):17.4f}")
print(f"  worst ratio over the grid: {worst:.4f} = {10*np.log10(worst):.3f} dB   (bound: 2 = 3.010 dB)")
print("  counterexample if D1 < D0:  D0=1, D1=0.1, A->0, C01=0 ->",
      f"ratio = {1+1.0**2/((1e-9+1.0)*0.1):.1f}  (bound needs D1 >= D0)")

print("="*72); print("(B) combined drift + unmodeled field:  R=(1+rho)/(rho+kappa+nu)")
sc=Scenario(); N,r=sc.N,sc.r; kap=r/N; g=np.random.default_rng(5); T=3000
print(f"{'rho':>8} {'diffuse dB':>11} {'nu':>9} {'R pred':>9} {'R emp':>9} {'ratio':>7}")
for rho in (0.0,0.05,0.5):
    for dB in (-60,-40,-30):
        sn2=1e-4; sd2=rho*sn2/r
        # nu from the measured out-of-span diffuse energy
        Ds=[];
        for _ in range(400):
            d=sc.draw_diffuse(g,dB); p=sc.P_perp@d; Ds.append(np.vdot(p,p).real)
        nu=np.mean(Ds)/(N*sn2)
        pred=(1+rho)/(rho+kap+nu)
        raw=proj=0.0
        for _ in range(T):
            dl=(g.standard_normal(r)+1j*g.standard_normal(r))*np.sqrt(sd2/2)
            n=(g.standard_normal(N)+1j*g.standard_normal(N))*np.sqrt(sn2/2)
            d=sc.draw_diffuse(g,dB)
            h=sc.h_spec+d; h0=sc.B@(sc.g-dl)+d+n
            e=h0-h; raw+=np.vdot(e,e).real
            e=sc.PB@h0-h; proj+=np.vdot(e,e).real
        emp=raw/proj
        print(f"{rho:8.3f} {dB:11d} {nu:9.4f} {pred:9.3f} {emp:9.3f} {emp/pred:7.3f}")
print(f"\nE_spec = ||Bg||^2 = {sc.E_spec:.2f}   (Akram's item 13: she estimated ~370)")
print("path powers |g_i|^2 =", np.round(np.abs(sc.g)**2,4))
