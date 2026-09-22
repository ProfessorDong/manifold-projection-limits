"""Table III and the model-order sensitivity of Section VII-B, from one place.

For each measured snapshot, OMP with least-squares refitting selects L+1 specular
atoms from an angle-delay dictionary; the reported share is ||P_perp h||^2/||h||^2
averaged over snapshots.  The share is a property of THIS dictionary and fit.  It
is not a decomposition of the channel into specular and diffuse parts, and it is
not a bound on physical diffuse power: broadening the elevation grid lowers it
(see the sensitivity block below), while an independently calibrated dictionary,
not fitted to the channel being denoised, would raise it.

Also reports, at a fixed raw calibration NMSE, the error reduction of the full
projection and of the plug-in partial projection across assumed model orders.
"""
import numpy as np, sys, json
sys.path.insert(0,'sim')
import exp_real_dichasus as E1
import exp_real_env2 as E2

AZ  = np.deg2rad(np.linspace(-80,80,161))
EL  = np.deg2rad(np.linspace(-80,80,17))     # see module docstrings
EL3 = np.deg2rad(np.array([-10.,0.,10.]))    # the earlier, under-covering grid
ORDERS = [1,2,4,8,12,16,32]
NMSE_REF = 1e-2

def shares(E, path, el, orders, omax=None):
    A=np.array([[E.steer(a,e) for a in AZ] for e in el])
    H=E.load(path); en=np.sum(np.abs(H)**2,axis=(1,2))
    omax=omax or max(orders)
    out=np.zeros((len(H),len(orders)))
    for s,h in enumerate(H):
        B=E.omp(h,omax,A,AZ,el)[0]; v=h.ravel()
        for j,r in enumerate(orders):
            q,_=np.linalg.qr(B[:,:r]); d=v-q@(q.conj().T@v)
            out[s,j]=np.vdot(d,d).real/en[s]
    return out, en

if __name__=="__main__":
    N=8192
    nv={lab: float(np.load(f'data/noisevar_{lab}.npy')[2]) for lab in ('cf06','0152')}
    res={}
    for lab,E,path in (('cf06',E1,'data/cf06_head.bin'),('0152',E2,'data/d0152_head.bin')):
        f,en = shares(E,path,EL,ORDERS); res[lab]=f
        m=f.mean(0); w=np.average(f,0,weights=en)
        print(f"\n=== {lab} ===  {f.shape[0]} snapshots, records 0,8,...,{8*(f.shape[0]-1)}")
        print("L+1      :"+"".join(f"{r:>9d}" for r in ORDERS))
        print("share    :"+"".join(f"{x:>9.3f}" for x in m))
        print("dB       :"+"".join(f"{10*np.log10(x/(1-x)):>+9.1f}" for x in m))
        print("enr-wtd  :"+"".join(f"{x:>9.3f}" for x in w))
        print(f"noise-corrected (nu={nv[lab]:.4f}):"
              +"".join(f"{x-nv[lab]:>9.3f}" for x in m))
        f3,_=shares(E,path,EL3,ORDERS)
        print("el={-10,0,10} grid:"+"".join(f"{x:>9.3f}" for x in f3.mean(0)))

    print("\n=== deep fit: does the residual plateau? (cf06) ===")
    fd,_=shares(E1,'data/cf06_head.bin',EL,[32,64,128],omax=128)
    print("L+1      :"+"".join(f"{r:>9d}" for r in [32,64,128]))
    print("share    :"+"".join(f"{x:>9.4f}" for x in fd.mean(0)))

    print(f"\n=== error reduction at raw NMSE = {NMSE_REF:g}, across assumed order ===")
    rng=np.random.default_rng(0)
    for lab,E,path in (('cf06',E1,'data/cf06_head.bin'),('0152',E2,'data/d0152_head.bin')):
        A=np.array([[E.steer(a,e) for a in AZ] for e in EL]); H=E.load(path)
        print(f"  {lab}:  L+1    full proj.   partial")
        rp=[];rs=[]
        for r in ORDERS:
            num=[];ds=[];dh=[]
            for h in H:
                v=h.ravel(); hn=np.vdot(v,v).real; s2=NMSE_REF*hn/N
                B=E.omp(h,r,A,AZ,EL)[0]; q,_=np.linalg.qr(B)
                n=(rng.standard_normal(N)+1j*rng.standard_normal(N))*np.sqrt(s2/2)
                hc=v+n; Ph=q@(q.conj().T@hc)
                num.append(np.vdot(n,n).real)
                e=Ph-v; ds.append(np.vdot(e,e).real)
                Pp=hc-Ph
                b=float(np.clip(s2*(N-r)/max(np.vdot(Pp,Pp).real,1e-300),0,1))
                e=(1-b)*hc+b*Ph-v; dh.append(np.vdot(e,e).real)
            rp.append(np.mean(num)/np.mean(ds)); rs.append(np.mean(num)/np.mean(dh))
            print(f"       {r:4d}   {rp[-1]:9.4f}   {rs[-1]:8.4f}")
        rp=np.array(rp); rs=np.array(rs)
        print(f"    -> full projection inflates the error by {1/rp.max():.1f} to {1/rp.min():.1f}x;"
              f" risk varies by {rp.max()/rp.min():.1f}x across orders")
        print(f"    -> partial projection in [{rs.min():.4f},{rs.max():.4f}],"
              f" varying by {rs.max()/rs.min():.2f}x")
