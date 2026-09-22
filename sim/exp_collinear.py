"""
Near-collinear regime: target placed at angular separation dtheta from the DIRECT
path, at the SAME delay. Here background-suppression error aliases directly onto
the target atom, so the background estimator choice should matter most.
Reports |alpha_hat - alpha|^2 (reflection-coefficient MSE) and the Theorem-1 floor.
"""
import numpy as np, sys
sys.path.insert(0,'sim')
from isac_core import Scenario, steering, beta_plugin

def run(sc, dtheta_deg, snr_db, tcr_db, ddB, sd2, sn2, T, seed=0):
    rng=np.random.default_rng(seed); Nr,K,df=sc.Nr,sc.K,sc.df; Mp=13
    sw2=10**(-snr_db/10); a_mag=10**(tcr_db/20.0)
    kk=np.arange(K)
    th_r=sc.theta_bg[0]+np.deg2rad(dtheta_deg); tau_r=sc.tau_bg[0]   # same delay as direct path
    b=np.outer(np.exp(-1j*2*np.pi*kk*df*tau_r), steering(th_r,Nr)).ravel()   # target atom (N,)
    res={k:[] for k in ['raw','str','shr']}
    for t in range(T):
        delta=(rng.standard_normal(sc.r)+1j*rng.standard_normal(sc.r))*np.sqrt(sd2/2)
        n=(rng.standard_normal(sc.N)+1j*rng.standard_normal(sc.N))*np.sqrt(sn2/2)
        d=sc.draw_diffuse(rng,ddB)
        h_true=sc.B@sc.g+d; h_cal=sc.B@(sc.g-delta)+d+n
        al=a_mag*np.exp(1j*rng.uniform(0,2*np.pi))
        H=(h_true+al*b).reshape(K,Nr)
        X=np.exp(1j*np.pi/4*(2*rng.integers(0,4,size=(K,Mp))+1))
        W=np.sqrt(sw2/2)*(rng.standard_normal((K,Mp,Nr))+1j*rng.standard_normal((K,Mp,Nr)))
        Y=H[:,None,:]*X[:,:,None]+W
        bh=beta_plugin(h_cal,sc,sn2)
        for key,hb in {'raw':h_cal,'str':sc.PB@h_cal,'shr':(1-bh)*h_cal+bh*(sc.PB@h_cal)}.items():
            Hb=hb.reshape(K,Nr)
            Z=np.conj(X)[:,:,None]*Y-(np.abs(X)**2)[:,:,None]*Hb[:,None,:]
            zk=Z.sum(axis=1).ravel()                       # (N,) k-major
            al_hat=np.vdot(b,zk)/(Mp*np.vdot(b,b).real)    # LS on the known target atom
            res[key].append(np.abs(al_hat-al)**2)
    return {k:float(np.mean(v)) for k,v in res.items()}

sc=Scenario(); sn2=1e-4; T=400
Cd=np.linalg.inv(sn2*np.linalg.inv(sc.B.conj().T@sc.B))   # C_cal^-1 with sd2=0
floor=1.0/np.real(np.linalg.inv(np.linalg.inv(Cd))[0,0]) if False else None
Ccal=sn2*np.linalg.inv(sc.B.conj().T@sc.B)
thm1=1.0/np.real(np.linalg.inv(Ccal)[0,0])
print("="*88)
print(f"Near-collinear regime (target at direct-path delay).  SNR=10 dB, TCR=-30 dB, {T} trials")
print(f"Theorem 1 floor 1/[C_cal^-1]_00 (sd2=0, no diffuse) = {thm1:.3e}   |alpha_r|^2 = {10**(-30/10):.1e}")
print("="*88)
for ddB in [None, -30.0]:
    lab = "no diffuse" if ddB is None else f"diffuse {ddB:.0f} dB"
    print(f"\n--- {lab} ---")
    print(f"{'dtheta':>8} {'MSE raw':>12} {'MSE struct':>12} {'MSE shrink':>12}   winner")
    for dt in [0.5,1,2,5,10,20]:
        o=run(sc,dt,10.0,-30.0,ddB,0.0,sn2,T,seed=int(1000+10*dt+(0 if ddB is None else 1)))
        w=min(o,key=o.get)
        print(f"{dt:8.1f} {o['raw']:12.3e} {o['str']:12.3e} {o['shr']:12.3e}   {w}")
print("="*88)
