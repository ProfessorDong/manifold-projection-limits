"""
Does the background-estimator choice matter for SENSING?
Pipeline: residual = Y - hhat_bg * X  (genie symbols, to isolate clutter suppression)
          -> whitened delay-angle map -> parabolic peak refinement -> RMSE
Compares raw / structured / shrinkage(plug-in beta) background suppression,
vs target-to-clutter ratio and vs unmodeled diffuse level. CRB overlaid.
"""
import numpy as np, sys
sys.path.insert(0,'sim')
from isac_core import Scenario, steering, beta_plugin

def run(sc, snr_db, tcr_db, ddB, sd2, sn2, T, seed=0, ngrid=121):
    rng = np.random.default_rng(seed)
    Nr,K,df = sc.Nr, sc.K, sc.df
    M, Mp = 14, 13
    sw2 = 10**(-snr_db/10)
    # |alpha_r| set by target-to-clutter ratio relative to the DIRECT path gain |g0|=1
    a_mag = 10**(tcr_db/20.0)
    tau_grid = np.linspace(0.1e-6,3.0e-6,ngrid); th_grid = np.deg2rad(np.linspace(-60,60,ngrid))
    kk = np.arange(K)
    E = np.exp(1j*2*np.pi*np.outer(tau_grid,kk)*df)              # (G,K)
    Amat = np.exp(1j*np.pi*np.outer(np.sin(th_grid), np.arange(Nr)))  # (G,Nr)
    errs = {k:[] for k in ['raw','str','shr']}
    for t in range(T):
        # --- truth ---
        delta=(rng.standard_normal(sc.r)+1j*rng.standard_normal(sc.r))*np.sqrt(sd2/2)
        n=(rng.standard_normal(sc.N)+1j*rng.standard_normal(sc.N))*np.sqrt(sn2/2)
        d=sc.draw_diffuse(rng, ddB)
        h_true = sc.B@sc.g + d                       # (N,) k-major
        h_cal  = sc.B@(sc.g-delta) + d + n
        dt=(tau_grid[1]-tau_grid[0]); dth=(th_grid[1]-th_grid[0])
        tau_r = rng.uniform(tau_grid[0]+dt, tau_grid[-1]-dt)
        th_r  = np.deg2rad(rng.uniform(-50,50))
        al_r  = a_mag*np.exp(1j*rng.uniform(0,2*np.pi))
        hr = al_r*np.outer(np.exp(-1j*2*np.pi*kk*df*tau_r), steering(th_r,Nr)).ravel()
        H = (h_true+hr).reshape(K,Nr)
        X = (rng.integers(0,4,size=(K,Mp))*0+1)*np.exp(1j*np.pi/4*(2*rng.integers(0,4,size=(K,Mp))+1))
        W = np.sqrt(sw2/2)*(rng.standard_normal((K,Mp,Nr))+1j*rng.standard_normal((K,Mp,Nr)))
        Y = H[:,None,:]*X[:,:,None] + W                       # (K,Mp,Nr)
        bhat = beta_plugin(h_cal, sc, sn2)
        ests = {'raw':h_cal, 'str':sc.PB@h_cal, 'shr':(1-bhat)*h_cal + bhat*(sc.PB@h_cal)}
        for key,hb in ests.items():
            Hb = hb.reshape(K,Nr)
            Z = np.conj(X)[:,:,None]*Y - (np.abs(X)**2)[:,:,None]*Hb[:,None,:]   # (K,Mp,Nr)
            zk = Z.sum(axis=1)                                  # (K,Nr) coherent over m
            Bk = zk @ Amat.conj().T                             # (K,G_theta)
            P  = np.abs(E @ Bk)**2                              # (G_tau,G_theta)
            it,ih = np.unravel_index(np.argmax(P), P.shape)
            # separable 3-point parabolic refinement
            def par(ym,y0,yp,c,step):
                den=ym-2*y0+yp
                return c + (0.5*(ym-yp)/den)*step if abs(den)>0 else c
            tr = par(P[it-1,ih],P[it,ih],P[it+1,ih],tau_grid[it],dt) if 0<it<ngrid-1 else tau_grid[it]
            hr_= par(P[it,ih-1],P[it,ih],P[it,ih+1],th_grid[ih],dth) if 0<ih<ngrid-1 else th_grid[ih]
            errs[key].append(((tr-tau_r)**2, (np.rad2deg(hr_-th_r))**2))
    out={}
    for k,v in errs.items():
        v=np.array(v); out[k]=(np.sqrt(v[:,0].mean()), np.sqrt(v[:,1].mean()))
    # genie CRB (known symbols, known background)
    SNRint = Nr*K*Mp*a_mag**2/sw2
    crb_t = np.sqrt(1/(2*SNRint*(2*np.pi*df)**2*(K**2-1)/12))
    crb_a = np.rad2deg(np.sqrt(1/(2*SNRint*np.pi**2*np.cos(np.deg2rad(20))**2*(Nr**2-1)/12)))
    out['crb']=(crb_t,crb_a)
    return out

sc=Scenario()
sn2=1e-4; sd2=0.0; T=300
print("="*94)
print("Sensing accuracy vs TARGET-TO-CLUTTER RATIO   (SNR=10 dB, diffuse=-30 dB, 300 trials)")
print("="*94)
print(f"{'TCR dB':>7} | {'delay RMSE [ns]':^34} | {'angle RMSE [deg]':^34}")
print(f"{'':>7} | {'raw':>9}{'struct':>9}{'shrink':>9}{'CRB':>7} | {'raw':>9}{'struct':>9}{'shrink':>9}{'CRB':>7}")
for tcr in [-9,-20,-30,-40,-45]:
    o=run(sc,10.0,tcr,-30.0,sd2,sn2,T,seed=100+abs(tcr))
    print(f"{tcr:7d} | {o['raw'][0]*1e9:9.2f}{o['str'][0]*1e9:9.2f}{o['shr'][0]*1e9:9.2f}{o['crb'][0]*1e9:7.2f}"
          f" | {o['raw'][1]:9.3f}{o['str'][1]:9.3f}{o['shr'][1]:9.3f}{o['crb'][1]:7.3f}")
print("\n"+"="*94)
print("Sensing accuracy vs UNMODELED DIFFUSE LEVEL   (SNR=10 dB, TCR=-30 dB, 300 trials)")
print("="*94)
print(f"{'diff dB':>7} | {'raw':>9}{'struct':>9}{'shrink':>9}{'CRB':>7} | {'raw':>9}{'struct':>9}{'shrink':>9}{'CRB':>7}")
for dd in [None,-45,-35,-30,-25,-20]:
    o=run(sc,10.0,-30.0,dd,sd2,sn2,T,seed=700+(0 if dd is None else abs(int(dd))))
    lab="none" if dd is None else f"{dd:.0f}"
    print(f"{lab:>7} | {o['raw'][0]*1e9:9.2f}{o['str'][0]*1e9:9.2f}{o['shr'][0]*1e9:9.2f}{o['crb'][0]*1e9:7.2f}"
          f" | {o['raw'][1]:9.3f}{o['str'][1]:9.3f}{o['shr'][1]:9.3f}{o['crb'][1]:7.3f}")
print("="*94)
