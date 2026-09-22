"""Measured-data provenance and the choice of offset-compensation convention.

Two things the manuscript states about Section VII-B that no other script checks:

(1) Which reference-transmitter compensation is applied, and what the choice costs.
    The published per-antenna offsets leave the subcarrier indexing and the sign of
    the exponent open to interpretation, so we take per dataset the combination
    minimising the order-4 specular residual.  This prints all combinations so the
    choice is auditable, and shows that it is the timing offset alone for 0152.

(2) What the 32 selected snapshots actually cover: timestamps, chronological order
    and spatial extent.  The hall set moves; the lab set is close to one geometry
    repeated, which the manuscript now says.
"""
import numpy as np, sys, json
sys.path.insert(0, 'sim')
from tfrecord_reader import iter_records, parse_tensorproto
import exp_real_dichasus as E1
import exp_real_env2 as E2

AZ = np.deg2rad(np.linspace(-80, 80, 161))
EL = np.deg2rad(np.linspace(-80, 80, 17))
SETS = (('cf06', E1, 'data/cf06_head.bin', 'data/offsets_cf06.json', E1.ASSIGN.ravel()),
        ('0152', E2, 'data/d0152_head.bin', 'data/offsets_015x.json', E2.SUB.ravel()))

def load(path, sub, CPO, STO, mode, stride=8, maxrec=400):
    out = []
    for i, rec in enumerate(iter_records(path, max_records=maxrec)):
        if i % stride: continue
        t = parse_tensorproto(rec['csi']); h = t[..., 0] + 1j*t[..., 1]
        K = h.shape[1]; k = np.arange(K)
        s = STO[:, None]*2*np.pi*k[None, :]/K; c = CPO[:, None]
        if   mode == 'sto-':     h = h*np.exp(-1j*s)
        elif mode == 'sto+':     h = h*np.exp(+1j*s)
        elif mode == 'sto- c-':  h = h*np.exp(-1j*(s + c))
        elif mode == 'sto+ c-':  h = h*np.exp(+1j*(s - c))
        out.append(h[sub])
    return np.array(out)

if __name__ == "__main__":
    print("(1) order-4 residual share over all 32 selected snapshots, by convention")
    for lab, E, path, offf, sub in SETS:
        A = np.array([[E.steer(a, e) for a in AZ] for e in EL])
        off = json.load(open(offf))
        CPO, STO = np.array(off['cpo']), np.array(off['sto'])
        print(f"  {lab}:")
        for mode in ('none', 'sto-', 'sto+', 'sto- c-', 'sto+ c-'):
            H = load(path, sub, CPO, STO, mode); fr = []
            for h in H:
                Bm = E.omp(h, 4, A, AZ, EL)[0]; v = h.ravel()
                q, _ = np.linalg.qr(Bm); d = v - q @ (q.conj().T @ v)
                fr.append(np.vdot(d, d).real/np.vdot(v, v).real)
            print(f"    {mode:8s} {np.mean(fr):.5f}")

    print("\n(2) provenance of the 32 selected snapshots (every eighth record)")
    for lab, _, path, _, _ in SETS:
        ts, pos, idx = [], [], []
        for i, rec in enumerate(iter_records(path, max_records=400)):
            if i % 8: continue
            idx.append(i)
            if 'time' in rec: ts.append(float(np.ravel(rec['time'])[0]))
            for k in ('pos-tachy', 'pos-lidar'):
                if k in rec: pos.append(np.ravel(parse_tensorproto(rec[k]))[:3]); break
        ts = np.array(ts); pos = np.array(pos)
        d = np.diff(ts)
        ext = pos.max(0) - pos.min(0)
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        print(f"  {lab}: records {idx[0]}..{idx[-1]}, span {ts.max()-ts.min():.2f} s, "
              f"{int((d<0).sum())} of {len(d)} timestamp differences negative")
        print(f"        extent per axis [m] {np.round(ext,4)}, widest {ext.max():.4f}, "
              f"median step {np.median(step):.4f}, path {step.sum():.4f}")
