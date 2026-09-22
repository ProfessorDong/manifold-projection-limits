"""Corollary (uniform dominance): the plug-in partial projection has strictly
smaller MSE than the raw snapshot for EVERY deterministic unmodeled field d,
given Gaussian complement noise and p = N-L-1 > 4.

Unlike verify_shrinkage.py, d is held FIXED across trials (deterministic), which
is what a dominance claim requires.  Degenerate and adversarial d are included.
Run with OMP_NUM_THREADS=4; unconstrained BLAS oversubscribes on this box."""
import numpy as np, sys
sys.path.insert(0, 'sim')
from isac_core import Scenario

sc = Scenario(); N, r = sc.N, sc.r; p = N - r
sn2 = 1e-4; A = sn2 * p
T = 20000
print(f"N={N}  L+1={r}  p=N-L-1={p}  (>4 required)   A={A:.4e}   trials={T}")

def risk(d, seed):
    """MC risk of raw vs plug-in for a FIXED d, vectorized over trials."""
    g = np.random.default_rng(seed)
    ht = sc.h_spec + d
    n = (g.standard_normal((T, N)) + 1j*g.standard_normal((T, N))) * np.sqrt(sn2/2)
    hc = ht[None, :] + n
    Pp = hc @ sc.P_perp.T                       # P_perp is Hermitian
    nrm = np.einsum('ti,ti->t', Pp.conj(), Pp).real
    b = np.clip(A / np.maximum(nrm, 1e-300), 0.0, 1.0)
    e_raw = hc - ht
    e_shr = e_raw - b[:, None] * Pp             # (1-b)hc + b*PB*hc - ht
    f = lambda e: np.einsum('ti,ti->t', e.conj(), e).real.mean()
    return f(e_raw), f(e_shr), b.mean()

rng = np.random.default_rng(90210)
u = sc.P_perp @ (rng.standard_normal(N) + 1j*rng.standard_normal(N))
u /= np.sqrt(np.vdot(u, u).real)                # unit vector in the complement
cases = [("d = 0 (dictionary exact)",            np.zeros(N, complex)),
         ("d entirely IN span(B)",               sc.PB @ sc.h_spec * 0.1),
         ("d rank-1 in complement, |d|^2=A/1e4", u*np.sqrt(A/1e4)),
         ("d rank-1 in complement, |d|^2=A/100", u*np.sqrt(A/100)),
         ("d rank-1 in complement, |d|^2=A",     u*np.sqrt(A)),
         ("d rank-1 in complement, |d|^2=100A",  u*np.sqrt(100*A)),
         ("d rank-1 in complement, |d|^2=1e4A",  u*np.sqrt(1e4*A))]
for x in (-50, -40, -30, -20, -10):
    cases.append((f"physical diffuse {x} dB",
                  sc.draw_diffuse(np.random.default_rng(7), x)))   # ONE fixed draw

print(f"\n{'deterministic field d':<38} {'raw':>11} {'plug-in':>11} {'ratio':>7} {'E[beta]':>8}  dom?")
allok = True
for i, (lab, d) in enumerate(cases):
    rr, rs, bm = risk(d, 1000 + i)
    ok = rs < rr; allok &= ok
    print(f"{lab:<38} {rr:11.4e} {rs:11.4e} {rr/rs:7.3f} {bm:8.4f}  {'yes' if ok else 'NO'}")
print("\nALL DOMINATE" if allok else "\nDOMINANCE VIOLATED")
