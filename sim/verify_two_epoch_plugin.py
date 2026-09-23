"""Does Corollary 3's deterministic factor-two bound transfer to the plug-in?

Corollary 3 compares two DETERMINISTIC weights on the complement: the persistence
weight w = D/S and the optimum w* = C01/S, and caps their risk ratio at 1 + D/S < 2.
The plug-in of Proposition 4 replaces D/S by a function of the same noise whose
risk is being evaluated, so the cap does not transfer automatically.  This script
measures the gap in the two regimes the manuscript quotes.

Model (two epochs, complement only, dimension p):
    y = x0 + n,   estimand x1,   risk E||w y - x1||^2
    x0, x1 independent fields of energies D0, D1 (so C01 = 0 and w* = 0)
    n white of energy A
    w_hat = (1 - A/||y||^2)_+
Three field laws, and it matters which hypothesis each one breaks:

    iso     x ~ CN(0, (D/p) I).  Gaussian, covariance effective rank p.  Both of
            Proposition 4's hypotheses hold.

    spike   x = sqrt(D) z v, with v a FRESH uniform random unit direction every
            trial and z ~ CN(0,1).  Marginally E[x x^H] = (D/p) I, so the
            effective-rank condition (23) HOLDS with room to spare -- the
            covariance is not rank one.  What fails is GAUSSIANITY: the energy
            ||x||^2 = D|z|^2 has relative variance 1 for every p, so it never
            concentrates.  This is exactly the counterexample the manuscript gives
            after Proposition 4, and it is the law that produces the 6.6% excess
            over Corollary 3's deterministic bound.

    rank1   the same field with v held FIXED across trials, which is the genuinely
            rank-one conditional model: Sigma = D v v^H + (A/p) I has effective
            rank about 3.95 at p = 252, A = D = 1, so condition (23) fails here.

Reported for both, so the two failure modes are not conflated.
"""
import numpy as np

def risks(p, A, D0, D1, law, T=400000, seed=5, block=4000):
    rng = np.random.default_rng(seed)
    S = A + D0; w_pers = D0/S                    # Corollary 3's deterministic weight
    tot_plug = tot_pers = tot_opt = 0.0
    vfix = rng.standard_normal(p); vfix /= np.linalg.norm(vfix)
    def draw(D):
        if law == 'iso':
            return (rng.standard_normal((block, p)) + 1j*rng.standard_normal((block, p))) \
                   * np.sqrt(D/(2*p))
        if law == 'rank1':                      # one fixed direction: covariance IS rank one
            v = np.tile(vfix, (block, 1))
        else:                                   # 'spike': fresh direction each trial
            v = rng.standard_normal((block, p)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        z = (rng.standard_normal(block) + 1j*rng.standard_normal(block))/np.sqrt(2)
        return v*(z*np.sqrt(D))[:, None]
    done = 0
    while done < T:
        x0, x1 = draw(D0), draw(D1)
        n = (rng.standard_normal((block, p)) + 1j*rng.standard_normal((block, p)))*np.sqrt(A/(2*p))
        y = x0 + n
        w = np.maximum(1 - A/np.sum(np.abs(y)**2, axis=1), 0)
        tot_plug += np.sum(np.abs(w[:, None]*y - x1)**2)
        tot_pers += np.sum(np.abs(w_pers*y - x1)**2)
        tot_opt  += np.sum(np.abs(x1)**2)          # w* = 0 is optimal when C01 = 0
        done += block
    return tot_plug/done, tot_pers/done, tot_opt/done, w_pers

if __name__ == "__main__":
    p = 252                                        # N - L - 1 for Table I, Config. A
    print(f"complement dimension p = {p}, C01 = 0 so the optimal weight is w* = 0")
    print(f"{'law':6s} {'A':>6} {'D':>8} {'R_opt':>10} {'R_pers':>10} {'ratio':>7}"
          f" {'R_plugin':>11} {'ratio':>8}   bound 1+D/S")
    for law in ('iso', 'spike', 'rank1'):
        for A, D in ((1.0, 1.0), (1.0, 0.05), (1.0, 1e-6)):
            pl, pe, op, wp = risks(p, A, D, D, law)
            print(f"{law:6s} {A:6.1f} {D:8.0e} {op:10.3e} {pe:10.3e} {pe/op:7.4f}"
                  f" {pl:11.3e} {pl/op:8.4f}   {1+D/(A+D):.4f}")
    print("\nReadings quoted in Section V:")
    pl, pe, op, _ = risks(p, 1.0, 1e-6, 1e-6, 'iso')
    print(f"  A=1, D0=D1=1e-6, isotropic: plug-in complement risk {pl:.3e} against an"
          f" optimum of {op:.3e}")
    for law in ('iso', 'spike', 'rank1'):
        pl, pe, op, _ = risks(p, 1.0, 1.0, 1.0, law)
        print(f"  A=D0=D1=1, {law:5s}: plug-in/optimal = {pl/op:.4f} against the"
              f" deterministic bound 1.5  ({100*(pl/op-1.5)/1.5:+.1f}%)")

    print("\nWhich hypothesis each law breaks, at p = 252, A = D = 1:")
    rng2 = np.random.default_rng(11)
    for law in ('iso', 'spike', 'rank1'):
        if law == 'rank1':
            vf = rng2.standard_normal(p); vf /= np.linalg.norm(vf)
            Sig = 1.0*np.outer(vf, vf) + (1.0/p)*np.eye(p)
            relvar = None
        else:
            if law == 'iso':
                x = (rng2.standard_normal((40000, p)) + 1j*rng2.standard_normal((40000, p))) \
                    / np.sqrt(2*p)
            else:
                v = rng2.standard_normal((40000, p)); v /= np.linalg.norm(v, axis=1, keepdims=True)
                z = (rng2.standard_normal(40000) + 1j*rng2.standard_normal(40000))/np.sqrt(2)
                x = v*z[:, None]
            Sig = (x.conj().T @ x).real/len(x) + (1.0/p)*np.eye(p)
            e = np.sum(np.abs(x)**2, axis=1); relvar = e.var()/e.mean()**2
        er = np.trace(Sig)**2/np.trace(Sig @ Sig)
        note = "" if relvar is None else f", energy relative variance {relvar:.4f}"
        print(f"  {law:5s}: covariance effective rank {er:7.2f} of {p}{note}")
    print("  condition (23) needs the effective rank to diverge; concentration also needs")
    print("  a fourth-moment hypothesis, which Gaussianity supplies and 'spike' does not.")
