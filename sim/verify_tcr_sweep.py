"""Akram's question: does the sensing advantage of partial projection survive a
sweep of the target-to-clutter ratio, or is it an artifact of the single TCR of
-30 dB used in Fig. 4?

Fig. 4 fixes TCR and sweeps the two nuisance axes.  This sweeps TCR itself at the
operating point of Sec. VII, with the diffuse level held at the value where full
projection has already collapsed (-30 dB) so that the comparison is informative.
Same estimator chain and same matched-filter grid as fig_B_sensing.py.
"""
import numpy as np, sys
sys.path.insert(0, 'sim')
from isac_core import Scenario, steering, beta_plugin

def run(sc, snr_db, tcr_db, ddB, sn2, T, seed, ng=121):
    rng = np.random.default_rng(seed); Nr, K, df = sc.Nr, sc.K, sc.df; Mp = 13
    sw2 = 10**(-snr_db/10); am = 10**(tcr_db/20.); kk = np.arange(K)
    tg = np.linspace(.1e-6, 3e-6, ng); hg = np.deg2rad(np.linspace(-60, 60, ng))
    E = np.exp(1j*2*np.pi*np.outer(tg, kk)*df)
    Am = np.exp(1j*np.pi*np.outer(np.sin(hg), np.arange(Nr)))
    dt = tg[1]-tg[0]; dh = hg[1]-hg[0]
    acc = {k: [] for k in ('raw', 'str', 'shr')}
    for _ in range(T):
        n = (rng.standard_normal(sc.N)+1j*rng.standard_normal(sc.N))*np.sqrt(sn2/2)
        d = sc.draw_diffuse(rng, ddB); ht = sc.h_spec+d; hc = ht+n
        tau = rng.uniform(tg[0]+dt, tg[-1]-dt); th = np.deg2rad(rng.uniform(-50, 50))
        al = am*np.exp(1j*rng.uniform(0, 2*np.pi))
        hr = al*np.outer(np.exp(-1j*2*np.pi*kk*df*tau), steering(th, Nr)).ravel()
        H = (ht+hr).reshape(K, Nr)
        X = np.exp(1j*np.pi/4*(2*rng.integers(0, 4, size=(K, Mp))+1))
        W = np.sqrt(sw2/2)*(rng.standard_normal((K, Mp, Nr))+1j*rng.standard_normal((K, Mp, Nr)))
        Y = H[:, None, :]*X[:, :, None]+W; b = beta_plugin(hc, sc, sn2)
        for key, hb in (('raw', hc), ('str', sc.PB@hc), ('shr', (1-b)*hc+b*(sc.PB@hc))):
            Hb = hb.reshape(K, Nr)
            Z = np.conj(X)[:, :, None]*Y-(np.abs(X)**2)[:, :, None]*Hb[:, None, :]
            P = np.abs(E@((Z.sum(axis=1))@Am.conj().T))**2
            it, ih = np.unravel_index(np.argmax(P), P.shape)
            def par(ym, y0, yp, c, st):
                den = ym-2*y0+yp
                return c+(.5*(ym-yp)/den)*st if abs(den) > 0 else c
            tr = par(P[it-1, ih], P[it, ih], P[it+1, ih], tg[it], dt) if 0 < it < ng-1 else tg[it]
            hh = par(P[it, ih-1], P[it, ih], P[it, ih+1], hg[ih], dh) if 0 < ih < ng-1 else hg[ih]
            acc[key].append(((tr-tau)**2, (np.rad2deg(hh-th))**2))
    out = {}
    for k, v in acc.items():
        v = np.array(v)
        out[k] = (np.sqrt(v[:, 0].mean()), np.sqrt(v[:, 1].mean()))
    return out

if __name__ == "__main__":
    sc = Scenario(); T = 500
    TCR = [-40, -35, -30, -25, -20, -15, -10, -5, 0]
    print("diffuse -30 dB, calibration sn2 = 1e-4, SNR 10 dB, T = 500 draws/point")
    print(f"\n{'TCR':>5} | {'delay RMSE [ns]':>30} | {'angle RMSE [deg]':>30} | "
          f"{'shr/str':>8} {'shr/raw':>8}")
    print(f"{'dB':>5} | {'raw':>9} {'full proj':>9} {'partial':>9} | "
          f"{'raw':>9} {'full proj':>9} {'partial':>9} | {'(delay)':>8} {'(delay)':>8}")
    print("-"*96)
    for t in TCR:
        o = run(sc, 10., float(t), -30, 1e-4, T, seed=7000+t)
        print(f"{t:5d} | {o['raw'][0]*1e9:9.2f} {o['str'][0]*1e9:9.2f} {o['shr'][0]*1e9:9.2f} | "
              f"{o['raw'][1]:9.3f} {o['str'][1]:9.3f} {o['shr'][1]:9.3f} | "
              f"{o['shr'][0]/o['str'][0]:8.3f} {o['shr'][0]/o['raw'][0]:8.3f}")
