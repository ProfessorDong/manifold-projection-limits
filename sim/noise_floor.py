"""Measure the DICHASUS receiver noise variance directly from the delay-domain
noise floor, so that the residual shares of Table III can be noise-corrected.

sigma_w^2 (per frequency sample) = K * median_bin |ifft(h)|^2 over delay bins
that carry no channel energy.  The median is used rather than the mean so that
residual channel tails do not bias the floor upward."""
import numpy as np, json, sys
sys.path.insert(0,'sim')
from tfrecord_reader import iter_records, parse_tensorproto

# Each dataset has its own array geometry and sub-array; using the cf06
# assignment for the 0152 file would estimate the floor on the wrong antennas.
SPEC={'cf06': ('data/spec.json',   lambda a: np.array(a).ravel()),
      '0152': ('data/spec015.json',lambda a: np.array(a)[:2,:4].ravel())}

def load(path, off_json, sign, stride=8, maxrec=400):
    specf,pick=SPEC[sign]
    ASSIGN=pick(json.load(open(specf))['antennas'][0]['assignments'])
    off=json.load(open(off_json)); CPO=np.array(off['cpo']); STO=np.array(off['sto'])
    H=[]
    for i,rec in enumerate(iter_records(path, max_records=maxrec)):
        if i%stride: continue                       # same records the analysis uses
        t=parse_tensorproto(rec['csi']); h=(t[...,0]+1j*t[...,1])[ASSIGN]
        K=h.shape[1]; k=np.arange(K)
        c=CPO[ASSIGN]; s=STO[ASSIGN]
        if sign=='cf06':   h=h*np.exp(1j*(s[:,None]*2*np.pi*k[None,:]/K - c[:,None]))
        else:              h=h*np.exp(-1j*(s[:,None]*2*np.pi*k[None,:]/K))
        H.append(h)
    return np.array(H)

def noise_var(H, guard=64):
    """Delay bins outside the channel support, taken relative to the peak bin."""
    S,Nr,K=H.shape; out=[]
    for s in range(S):
        g=np.fft.ifft(H[s],axis=1)
        p=np.mean(np.abs(g)**2,axis=0)
        pk=int(np.argmax(p))
        idx=np.arange(K); dist=np.minimum((idx-pk)%K,(pk-idx)%K)
        floor=np.median(p[dist>guard])          # median over noise-only bins
        out.append(floor*K)
    return float(np.mean(out))

if __name__=="__main__":
    for lab,path,oj,sg in (("cf06",'data/cf06_head.bin','data/offsets_cf06.json','cf06'),
                           ("0152",'data/d0152_head.bin','data/offsets_015x.json','0152')):
        H=load(path,oj,sg)
        s2=noise_var(H)
        tot=float(np.mean(np.abs(H)**2))
        nu=s2/tot
        print(f"{lab}: sigma_w^2={s2:.4e}  total/sample={tot:.4e}")
        print(f"      noise share nu = {nu:.4f}   implied SNR = {10*np.log10(1/nu-1):.2f} dB")
        np.save(f'data/noisevar_{lab}.npy', np.array([s2,tot,nu]))
