"""REAL-DATA EXPERIMENT (DICHASUS dichasus-cf06, Arena2036 industrial hall).
Array A only: 2x4 URA, 0.118 m spacing = 0.500 lambda at 1.272 GHz. Nr=8, K=1024.
Measurable on real data: the specular-fit residual d = P_perp h for each assumed
model order, hence the true diffuse-to-specular ratio and R_proj once controlled
synthetic calibration noise is added to the REAL channel."""
import numpy as np, json, sys
sys.path.insert(0, 'sim')
from tfrecord_reader import iter_records, parse_tensorproto

C = 299792458.0
spec = json.load(open('data/spec.json'))
BW = spec['bandwidth']; FC = 1.272e9; LAM = C/FC
ARR = spec['antennas'][0]
ASSIGN = np.array(ARR['assignments']); NROW, NCOL = ASSIGN.shape
DX = ARR['spacingX']; DY = ARR['spacingY']
off = json.load(open('data/offsets_cf06.json'))
CPO = np.array(off['cpo']); STO = np.array(off['sto'])

def load(path, stride=8, maxrec=400, compensate=True):
    out = []
    for i, rec in enumerate(iter_records(path, max_records=maxrec)):
        if i % stride: continue
        t = parse_tensorproto(rec['csi']); h = t[...,0] + 1j*t[...,1]
        if compensate:
            K = h.shape[1]; k = np.arange(K)
            # convention determined empirically (minimises specular residual):
            #   h_comp = h * exp(-j CPO) * exp(+j 2 pi k STO / K)
            h = h*np.exp(1j*(STO[:,None]*2*np.pi*k[None,:]/K - CPO[:,None]))
        out.append(h[ASSIGN.ravel()])
    return np.array(out)

def steer(az, el):
    n = np.arange(NCOL)[None,:]; m = np.arange(NROW)[:,None]
    return np.exp(1j*2*np.pi/LAM*(n*DX*np.cos(el)*np.sin(az) + m*DY*np.sin(el))).ravel()

def omp(h, n_paths, A, az_grid, el_grid, nfft=4096):
    Nr, K = h.shape; df = BW/K; kk = np.arange(K)
    r = h.copy(); atoms = []; params = []
    for _ in range(n_paths):
        c = np.einsum('eai,ik->eak', A.conj(), r)
        # b^H r = sum_k exp(+j 2pi k df tau) c[k]  ->  inverse DFT convention
        P = np.abs(np.fft.ifft(c, n=nfft, axis=-1))**2
        ie, ia, it = np.unravel_index(np.argmax(P), P.shape)
        tau = it/(nfft*df)
        params.append((np.rad2deg(az_grid[ia]), np.rad2deg(el_grid[ie]), tau*1e9))
        atoms.append(np.outer(A[ie,ia], np.exp(-1j*2*np.pi*kk*df*tau)).ravel())
        B = np.array(atoms).T
        g, *_ = np.linalg.lstsq(B, h.ravel(), rcond=None)
        r = (h.ravel() - B@g).reshape(Nr, K)
    return B, g, r.ravel(), params

if __name__ == "__main__":
# Elevation grid: -80:10:80.  The 3-point grid used earlier ({-10,0,10})
# under-covers the elevation aperture and leaves real path energy in the
# residual; broadening it lowers the order-4 hall share from 0.51 to 0.42,
# whereas on i.i.d. Gaussian data of the same size the same broadening moves
# the residual by 0.0005, so the reduction is coverage and not overfitting.
# Refining further to -80:5:80 moves the shares by less than 0.002.
    az_grid = np.deg2rad(np.linspace(-80,80,161))
    el_grid = np.deg2rad(np.linspace(-80,80,17))
    A = np.array([[steer(a,e) for a in az_grid] for e in el_grid])
    H = load('data/cf06_head.bin'); S, Nr, K = H.shape; N = Nr*K
    print(f"loaded {S} snapshots  Nr={Nr} K={K} N={N}")
    print(f"lambda={LAM:.4f} m  spacing={DX/LAM:.3f} lambda  BW={BW/1e6:.2f} MHz\n")

    # sanity: does offset compensation actually improve the specular fit?
    Hu = load('data/cf06_head.bin', compensate=False)
    for lab, HH in (("compensated", H), ("uncompensated", Hu)):
        f = []
        for s in range(min(6, S)):
            _,_,d,_ = omp(HH[s], 4, A, az_grid, el_grid)
            f.append(np.vdot(d,d).real/np.vdot(HH[s].ravel(),HH[s].ravel()).real)
        print(f"  residual share with 4 paths, {lab:14s}: {np.mean(f):.4f}")
    print()

    print("Diffuse (residual) share of channel energy vs assumed model order")
    print(f"{'L+1':>5} {'DMC share':>11} {'DMC/spec [dB]':>14} {'isotropic (L+1)/N':>18}")
    Ls = [1,2,3,4,6,8,12,16,24,32]; res = {}
    for p in Ls:
        fr = [np.vdot(d,d).real/np.vdot(H[s].ravel(),H[s].ravel()).real
              for s in range(S) for d in [omp(H[s], p, A, az_grid, el_grid)[2]]]
        f = float(np.mean(fr)); res[p] = f
        print(f"{p:5d} {f:11.4f} {10*np.log10(f/(1-f)):14.2f} {p/N:18.2e}")
    np.save('data/dmc_share.npy', np.array([[p,res[p]] for p in Ls]))
    print("\nexample strongest paths (snapshot 0, L+1=4): az[deg], el[deg], tau[ns]")
    for q in omp(H[0], 4, A, az_grid, el_grid)[3]: print("   ", np.round(q,2))
