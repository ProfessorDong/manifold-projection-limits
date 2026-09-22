"""Second measured environment: DICHASUS dichasus-0152 (indoor LoS lab room,
co-located 4x8 URA). Processing identical to dichasus-cf06 so the two
environments are directly comparable: one 2x4 sub-block, Nr=8, K=1024, N=8192."""
import numpy as np, json, sys
sys.path.insert(0,'sim')
from tfrecord_reader import iter_records, parse_tensorproto

C=299792458.0
spec=json.load(open('data/spec015.json'))
BW=spec['bandwidth']; FC=1.272e9; LAM=C/FC
ARR=spec['antennas'][0]
FULL=np.array(ARR['assignments'])                 # (4,8)
SUB=FULL[:2,:4]                                   # 2x4 sub-block, as in cf06
NROW,NCOL=SUB.shape
DX=ARR['spacingX']; DY=ARR['spacingY']
off=json.load(open('data/offsets_015x.json'))
CPO=np.array(off['cpo']); STO=np.array(off['sto'])

def load(path,stride=8,maxrec=400,compensate=True):
    out=[]
    for i,rec in enumerate(iter_records(path,max_records=maxrec)):
        if i%stride: continue
        t=parse_tensorproto(rec['csi']); h=t[...,0]+1j*t[...,1]
        if compensate:
            K=h.shape[1]; k=np.arange(K)
            # Convention chosen per dataset as the one MINIMISING the specular
            # residual.  That is the most favourable available treatment of the
            # offsets for the structural model; it is NOT independent validation
            # of the correction, and the resulting share is a property of this
            # dictionary and fit, not a bound on physical diffuse power.
            h=h*np.exp(-1j*(STO[:,None]*2*np.pi*k[None,:]/K))
        out.append(h[SUB.ravel()])
    return np.array(out)

def steer(az,el):
    n=np.arange(NCOL)[None,:]; m=np.arange(NROW)[:,None]
    return np.exp(1j*2*np.pi/LAM*(n*DX*np.cos(el)*np.sin(az)+m*DY*np.sin(el))).ravel()

def omp(h,n_paths,A,az,el,nfft=4096):
    Nr,K=h.shape; df=BW/K; kk=np.arange(K)
    r=h.copy(); atoms=[]
    for _ in range(n_paths):
        c=np.einsum('eai,ik->eak',A.conj(),r)
        P=np.abs(np.fft.ifft(c,n=nfft,axis=-1))**2
        ie,ia,it=np.unravel_index(np.argmax(P),P.shape)
        tau=it/(nfft*df)
        atoms.append(np.outer(A[ie,ia],np.exp(-1j*2*np.pi*kk*df*tau)).ravel())
        B=np.array(atoms).T
        g,*_=np.linalg.lstsq(B,h.ravel(),rcond=None)
        r=(h.ravel()-B@g).reshape(Nr,K)
    return B,g,r.ravel()

if __name__=="__main__":
# Elevation grid: -80:10:80.  The 3-point grid used earlier ({-10,0,10})
# under-covers the elevation aperture and leaves real path energy in the
# residual; broadening it lowers the order-4 hall share from 0.51 to 0.42,
# whereas on i.i.d. Gaussian data of the same size the same broadening moves
# the residual by 0.0005, so the reduction is coverage and not overfitting.
# Refining further to -80:5:80 moves the shares by less than 0.002.
    az=np.deg2rad(np.linspace(-80,80,161)); el=np.deg2rad(np.linspace(-80,80,17))
    A=np.array([[steer(a,e) for a in az] for e in el])
    H=load('data/d0152_head.bin'); S,Nr,K=H.shape; N=Nr*K
    print(f"dichasus-0152 (indoor LoS lab room): {S} snapshots  Nr={Nr} K={K} N={N}")
    Hu=load('data/d0152_head.bin',compensate=False)
    for lab,HH in (("compensated",H),("uncompensated",Hu)):
        f=[np.vdot(d,d).real/np.vdot(HH[s].ravel(),HH[s].ravel()).real
           for s in range(min(6,S)) for d in [omp(HH[s],6,A,az,el)[2]]]
        print(f"  residual share, 6 paths, {lab:14s}: {np.mean(f):.4f}")
    print(f"\n{'L+1':>5} {'DMC share':>11} {'DMC/spec [dB]':>14} {'crossover NMSE_raw':>19}")
    rows=[]
    for p in [1,2,4,6,8,12,16,32]:
        fr=[np.vdot(d,d).real/np.vdot(H[s].ravel(),H[s].ravel()).real
            for s in range(S) for d in [omp(H[s],p,A,az,el)[2]]]
        f=float(np.mean(fr)); kap=p/N
        rows.append((p,f))
        print(f"{p:5d} {f:11.4f} {10*np.log10(f/(1-f)):14.2f} {f/(1-kap):19.4f}")
    np.save('data/dmc_share_env2.npy',np.array(rows))
