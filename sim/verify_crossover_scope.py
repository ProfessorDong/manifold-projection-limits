"""The crossover of Corollary 2 requires a persistent unmodeled field.

Corollary 2 is derived under the model of Section III, in which the SAME d is
present at calibration and at use.  That matters: the raw error is then
h0_hat - h = n alone, because d cancels, which is what makes a large d harmful to
the raw estimate's competitor rather than to the raw estimate itself.

From Theorem 1, with the raw snapshot at w = 1 and full projection at w = 0,

    R_perp(1) = A + D0 + D1 - 2 C01,     R_perp(0) = D1,

so full projection beats the raw snapshot exactly when 2 C01 < A + D0.  Writing
D0 = D1 = D and C01 = rho_c D for a cross-epoch correlation coefficient rho_c,

    projection is HARMFUL  <=>  D (2 rho_c - 1) > A,

which has no solution for rho_c <= 1/2.  A crossover therefore exists only if the
field is more than half correlated between the two epochs; below that, full
projection never hurts however much diffuse energy there is.  Setting rho_c = 1
recovers Corollary 2's condition D > A.
"""
import numpy as np

def predicted(A, D0, D1, C01):
    return A + D0 + D1 - 2*C01, D1                      # raw (w=1), projection (w=0)

def measured(A, D, rho_c, p=252, T=40000, seed=3):
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal((T, p)) + 1j*rng.standard_normal((T, p)))/np.sqrt(2)
    u = (rng.standard_normal((T, p)) + 1j*rng.standard_normal((T, p)))/np.sqrt(2)
    n = (rng.standard_normal((T, p)) + 1j*rng.standard_normal((T, p)))*np.sqrt(A/(2*p))
    x0 = z*np.sqrt(D/p)
    x1 = (rho_c*z + np.sqrt(max(1-rho_c**2, 0))*u)*np.sqrt(D/p)
    return (np.mean(np.sum(np.abs(x0 + n - x1)**2, axis=1)),
            np.mean(np.sum(np.abs(x1)**2, axis=1)))

if __name__ == "__main__":
    A = 1.0
    print("harmful <=> D (2 rho_c - 1) > A;  A = 1")
    print(f"{'rho_c':>6} {'D*':>8}   " + "  ".join(f"D={D:<5g}" for D in (0.1, 1, 10, 100)))
    for rho_c in (1.0, 0.75, 0.6, 0.5, 0.25, 0.0):
        star = A/(2*rho_c - 1) if rho_c > 0.5 else np.inf
        row = []
        for D in (0.1, 1.0, 10.0, 100.0):
            raw, prj = predicted(A, D, D, rho_c*D)
            row.append(f"{'harmful' if prj > raw else 'helps  '}")
        print(f"{rho_c:6.2f} {star:8.3g}   " + "  ".join(f"{r:<7}" for r in row))
    print("\nMonte Carlo check against the closed form (p = 252):")
    for rho_c in (1.0, 0.5, 0.0):
        for D in (1.0, 10.0):
            mr, mp = measured(A, D, rho_c)
            pr, pp = predicted(A, D, D, rho_c*D)
            print(f"  rho_c={rho_c:3.1f} D={D:4.1f}: raw {mr:7.3f} (pred {pr:7.3f}), "
                  f"projection {mp:7.3f} (pred {pp:7.3f})")
    print("\nSo Corollary 2's condition is the rho_c = 1 face of 2 C01 > A + D0.")
