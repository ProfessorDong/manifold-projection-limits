"""DECISIVE CHECK on the 0.074-lambda decorrelation result.

Two objections must be excluded before the result can stand:
 (1) C01 = Re E[x1^H x0] can vanish while the field is perfectly predictable
     (x1 = j x0 gives Re=0 but |correlation|=1).  So report Re, Im AND modulus.
 (2) If the dataset lacks cross-snapshot phase coherence, a random global phase
     per snapshot destroys ANY correlation, residual or not.  Diagnostic: the
     SPECULAR part must stay correlated over 2 cm.  If the full channel
     decorrelates as fast as the residual, the effect is a processing artifact."""
import numpy as np, sys
sys.path.insert(0,'sim')
from exp_real_dichasus import steer, omp, LAM
from exp_two_epoch import load_seq

az=np.deg2rad(np.linspace(-80,80,161)); el=np.deg2rad(np.linspace(-80,80,17))
A=np.array([[steer(a,e) for a in az] for e in el])
H,POS=load_seq('data/cf06_head.bin',220); S=H.shape[0]
ANCH=list(range(0,S-140,20)); Lp=8

def cc(u,v):
    return np.vdot(v,u)/np.sqrt(np.vdot(u,u).real*np.vdot(v,v).real)

print(f"{'k':>3} {'displ[lam]':>10} | {'FULL channel':>28} | {'RESIDUAL P_perp h':>28}")
print(f"{'':>3} {'':>10} | {'Re':>8} {'Im':>8} {'|.|':>9} | {'Re':>8} {'Im':>8} {'|.|':>9}")
for k in [0,1,2,4,8,16,32]:
    fr=[];fi=[];fm=[];rr=[];ri=[];rm=[]
    for a in ANCH:
        B,_,_,_=omp(H[a],Lp,A,az,el); Q,_=np.linalg.qr(B)
        h1=H[a+1].ravel(); h2=H[a+1+k].ravel()
        c=cc(h1,h2); fr.append(c.real); fi.append(c.imag); fm.append(abs(c))
        x1=h1-Q@(Q.conj().T@h1); x2=h2-Q@(Q.conj().T@h2)
        c=cc(x1,x2); rr.append(c.real); ri.append(c.imag); rm.append(abs(c))
    d=np.mean([np.linalg.norm(POS[a+1+k,:3]-POS[a+1,:3])/LAM for a in ANCH])
    print(f"{k:3d} {d:10.3f} | {np.mean(fr):8.4f} {np.mean(fi):8.4f} {np.mean(fm):9.4f} |"
          f" {np.mean(rr):8.4f} {np.mean(ri):8.4f} {np.mean(rm):9.4f}")
print("\nIf FULL |.| collapses as fast as RESIDUAL |.|, the dataset lacks")
print("cross-snapshot phase coherence and the result is an artifact.")
