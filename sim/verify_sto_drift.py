"""Is the cross-snapshot decorrelation physical, or residual reference-transmitter drift?

The modulus |<h2,h1>| is already invariant to a per-snapshot GLOBAL phase, so a
global phase cannot explain the collapse.  A per-snapshot residual sampling-time
offset can: it multiplies subcarrier k by exp(j 2 pi k dTau/K), which destroys the
coherent sum over k even when the physical channel is unchanged.

Maximise the correlation jointly over a delay shift and a global phase:
    c(D) = sum_k <h2[k],h1[k]> exp(j 2 pi k D / K)   -> an inverse DFT over k,
    coherence = max_D |c(D)| / (||h1|| ||h2||).
If this is large while the raw correlation is ~0, the collapse is a timing
artifact and NOT physical decorrelation."""
import numpy as np, sys
sys.path.insert(0,'sim')
from exp_real_dichasus import steer, omp, LAM
from exp_two_epoch import load_seq

az=np.deg2rad(np.linspace(-80,80,161)); el=np.deg2rad(np.linspace(-80,80,17))
A=np.array([[steer(a,e) for a in az] for e in el])
H,POS=load_seq('data/cf06_head.bin',220); S,Nr,K=H.shape
ANCH=list(range(0,S-140,20)); Lp=8; NFFT=1<<14

def coh(u,v):
    """u,v shape (Nr,K).  raw modulus, and modulus maximised over a delay shift."""
    n=np.sqrt(np.vdot(u,u).real*np.vdot(v,v).real)
    per_k=np.einsum('ak,ak->k', v.conj(), u)      # inner product per subcarrier
    raw=abs(per_k.sum())/n
    best=np.abs(np.fft.ifft(per_k, n=NFFT)).max()*NFFT/n
    return raw, best

print(f"{'k':>3} {'displ[lam]':>10} | {'FULL raw':>9} {'FULL +dTau':>11} | {'RESID raw':>10} {'RESID +dTau':>12}")
for k in [1,2,4,8,16,32]:
    fr=[];fb=[];rr=[];rb=[]
    for a in ANCH:
        B,_,_,_=omp(H[a],Lp,A,az,el); Q,_=np.linalg.qr(B)
        h1=H[a+1]; h2=H[a+1+k]
        x,y=coh(h1,h2); fr.append(x); fb.append(y)
        f=lambda h:(h.ravel()-Q@(Q.conj().T@h.ravel())).reshape(Nr,K)
        x,y=coh(f(h1),f(h2)); rr.append(x); rb.append(y)
    d=np.mean([np.linalg.norm(POS[a+1+k,:3]-POS[a+1,:3])/LAM for a in ANCH])
    print(f"{k:3d} {d:10.3f} | {np.mean(fr):9.4f} {np.mean(fb):11.4f} |"
          f" {np.mean(rr):10.4f} {np.mean(rb):12.4f}")
