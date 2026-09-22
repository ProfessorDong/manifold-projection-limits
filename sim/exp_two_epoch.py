"""TWO-EPOCH TEST of the paper's central modelling assumption.

The manuscript writes  h = Bg + d  and  h0_hat = B(g-delta) + d + n,  i.e. the
SAME realization d at the calibration epoch and at use.  That is what makes
beta* = A/(A+D).  With distinct fields d0, d1 the out-of-span risk is

    R(w) = w^2 (A + D0) + D1 - 2 w C01,    w = 1-beta,
    w* = clip[ C01 / (A + D0) ],   C01 = Re E[x1^H x0],  xj = P_perp dj.

Same-d is the special case C01 = D0 = D1.  Here we MEASURE C01 on real channels
by fitting B at one snapshot and evaluating the residual at later snapshots.

DICHASUS cf06 moves the transmitter along a trajectory, so lag maps to physical
displacement: this measures how fast the unmodelled field decorrelates in space.
A perfectly static user is the zero-displacement limit."""
import numpy as np, json, sys
sys.path.insert(0,'sim')
from tfrecord_reader import iter_records, parse_tensorproto
from exp_real_dichasus import steer, omp, LAM, ASSIGN, CPO, STO

C = 299792458.0
spec = json.load(open('data/spec.json')); BW = spec['bandwidth']

def load_seq(path, nrec):
    """Consecutive snapshots WITH positions (no stride)."""
    H, P = [], []
    for i, rec in enumerate(iter_records(path, max_records=nrec)):
        t = parse_tensorproto(rec['csi']); h = t[...,0] + 1j*t[...,1]
        K = h.shape[1]; k = np.arange(K)
        h = h*np.exp(1j*(STO[:,None]*2*np.pi*k[None,:]/K - CPO[:,None]))
        H.append(h[ASSIGN.ravel()])
        P.append(np.array(parse_tensorproto(rec['pos-tachy'])).ravel())
    return np.array(H), np.array(P)

if __name__ == "__main__":
    az = np.deg2rad(np.linspace(-80,80,161)); el = np.deg2rad(np.array([-10.,0.,10.]))
    A = np.array([[steer(a,e) for a in az] for e in el])
    NREC = 220
    H, POS = load_seq('data/cf06_head.bin', NREC)
    S, Nr, K = H.shape; N = Nr*K
    print(f"loaded {S} consecutive snapshots  N={N}  lambda={LAM:.4f} m")
    step = np.linalg.norm(np.diff(POS[:,:3],axis=0),axis=1)
    print(f"median inter-snapshot displacement: {np.median(step)/LAM:.3f} lambda "
          f"({np.median(step)*100:.2f} cm)\n")

    LAGS = [0,1,2,4,8,16,32,64,128]
    ANCH = list(range(0, S-max(LAGS), 20))
    Lp = 8
    print(f"Dictionary fitted at each anchor with L+1={Lp}; residual x = P_perp h")
    print(f"{'lag':>4} {'displ [lam]':>12} {'corr C01/sqrt(D0 D1)':>21} {'D1/D0':>8}")
    rows=[]
    for lag in LAGS:
        cs, ds, dd = [], [], []
        for a in ANCH:
            B,_,x0,_ = omp(H[a], Lp, A, az, el)
            Q,_ = np.linalg.qr(B)
            h1 = H[a+lag].ravel()
            x1 = h1 - Q@(Q.conj().T@h1)
            D0 = np.vdot(x0,x0).real; D1 = np.vdot(x1,x1).real
            cs.append(np.vdot(x1,x0).real/np.sqrt(D0*D1)); ds.append(D1/D0)
            dd.append(np.linalg.norm(POS[a+lag,:3]-POS[a,:3])/LAM)
        rows.append((lag,np.mean(dd),np.mean(cs),np.mean(ds)))
        print(f"{lag:4d} {np.mean(dd):12.3f} {np.mean(cs):21.4f} {np.mean(ds):8.3f}")
    np.save('data/two_epoch_corr.npy', np.array(rows))

    print("\nConsequence for the optimal weight (A = sigma_perp^2 (N-L-1)):")
    print(f"{'lag':>4} {'corr':>7} {'w* (measured)':>15} {'w* (same-d)':>13} {'beta* meas':>11} {'beta* same-d':>13}")
    lag_c = {r[0]: r[2] for r in rows}
    for lag in LAGS:
        c = lag_c[lag]
        for Aratio in [1.0]:            # A = D0
            D0 = 1.0; D1 = 1.0; A = Aratio*D0
            w_meas = np.clip(c*np.sqrt(D0*D1)/(A+D0), 0, 1)
            w_same = D0/(A+D0)
            print(f"{lag:4d} {c:7.4f} {w_meas:15.4f} {w_same:13.4f} "
                  f"{1-w_meas:11.4f} {1-w_same:13.4f}")
