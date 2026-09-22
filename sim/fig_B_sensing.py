"""Figure B (2x2): sensing accuracy for the three background estimators.
Left column  : vs unmodeled diffuse power   (good calibration)  -> projection collapses
Right column : vs calibration noise         (no diffuse)        -> raw collapses
Only the shrinkage estimator is best-or-tied in BOTH regimes."""
import numpy as np, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,'sim')
from isac_core import Scenario, steering, beta_plugin
from crb_calaware import crb as crb_perdraw

C1,C2,C3="#2a78d6","#eb6834","#1baf7a"; MUT="#52514e"; INK="#0b0b0b"

def run(sc,snr_db,tcr_db,ddB,sn2,T,seed,ng=121):
    rng=np.random.default_rng(seed); Nr,K,df=sc.Nr,sc.K,sc.df; Mp=13
    sw2=10**(-snr_db/10); am=10**(tcr_db/20.); kk=np.arange(K)
    tg=np.linspace(.1e-6,3e-6,ng); hg=np.deg2rad(np.linspace(-60,60,ng))
    E=np.exp(1j*2*np.pi*np.outer(tg,kk)*df); Am=np.exp(1j*np.pi*np.outer(np.sin(hg),np.arange(Nr)))
    dt=tg[1]-tg[0]; dh=hg[1]-hg[0]; acc={k:[] for k in ('raw','str','shr')}
    bnd=[]                      # per-draw calibration-aware CRB, same draws as the errors
    for _ in range(T):
        n=(rng.standard_normal(sc.N)+1j*rng.standard_normal(sc.N))*np.sqrt(sn2/2)
        d=sc.draw_diffuse(rng,ddB); ht=sc.h_spec+d; hc=ht+n
        tau=rng.uniform(tg[0]+dt,tg[-1]-dt); th=np.deg2rad(rng.uniform(-50,50))
        al=am*np.exp(1j*rng.uniform(0,2*np.pi))
        bnd.append(crb_perdraw(th,tau,al,sn2))
        hr=al*np.outer(np.exp(-1j*2*np.pi*kk*df*tau),steering(th,Nr)).ravel()
        H=(ht+hr).reshape(K,Nr)
        X=np.exp(1j*np.pi/4*(2*rng.integers(0,4,size=(K,Mp))+1))
        W=np.sqrt(sw2/2)*(rng.standard_normal((K,Mp,Nr))+1j*rng.standard_normal((K,Mp,Nr)))
        Y=H[:,None,:]*X[:,:,None]+W; b=beta_plugin(hc,sc,sn2)
        for key,hb in (('raw',hc),('str',sc.PB@hc),('shr',(1-b)*hc+b*(sc.PB@hc))):
            Hb=hb.reshape(K,Nr)
            Z=np.conj(X)[:,:,None]*Y-(np.abs(X)**2)[:,:,None]*Hb[:,None,:]
            P=np.abs(E@((Z.sum(axis=1))@Am.conj().T))**2
            it,ih=np.unravel_index(np.argmax(P),P.shape)
            def par(ym,y0,yp,c,st):
                den=ym-2*y0+yp; return c+(.5*(ym-yp)/den)*st if abs(den)>0 else c
            tr=par(P[it-1,ih],P[it,ih],P[it+1,ih],tg[it],dt) if 0<it<ng-1 else tg[it]
            hh=par(P[it,ih-1],P[it,ih],P[it,ih+1],hg[ih],dh) if 0<ih<ng-1 else hg[ih]
            acc[key].append(((tr-tau)**2,(np.rad2deg(hh-th))**2))
    out={}
    for k,v in acc.items():
        v=np.array(v)
        bs=np.array([[np.sqrt(v[i,0].mean()),np.sqrt(v[i,1].mean())]
                     for i in [np.random.default_rng(s).integers(0,len(v),len(v)) for s in range(200)]])
        out[k]=(np.sqrt(v[:,0].mean()),np.sqrt(v[:,1].mean()),
                np.percentile(bs[:,0],[2.5,97.5]),np.percentile(bs[:,1],[2.5,97.5]))
    # The reference is the calibration-aware bound of Prop. 5 evaluated on the SAME
    # draws as the errors above and aggregated the same way, sqrt(mean over draws of
    # the per-draw CRB).  It conditions on the diffuse field being known or absent,
    # which is the qualification stated in Sec. VI.
    bnd=np.array(bnd)
    out['crb']=(np.sqrt(bnd[:,0].mean()), np.rad2deg(np.sqrt(bnd[:,1].mean())))
    return out

sc=Scenario(); T=500
dd=[None,-50,-45,-40,-35,-30,-25,-20]; xd=[-55 if x is None else x for x in dd]
res_d=[run(sc,10.,-30.,x,1e-4,T,seed=2000+i) for i,x in enumerate(dd)]
sn_list=[1e-5,1e-4,1e-3,3e-3,1e-2,3e-2,1e-1]
res_n=[run(sc,10.,-30.,None,s,T,seed=3000+i) for i,s in enumerate(sn_list)]


def dump(fn,xs,res,idx,scale):
    import numpy as _np
    rows=[]
    for x,r in zip(xs,res):
        row=[x]
        for k in ('raw','str','shr'):
            row += [r[k][idx]*scale, r[k][2+idx][0]*scale, r[k][2+idx][1]*scale]
        row.append(r['crb'][idx]*scale)
        rows.append(row)
    _np.savetxt(fn,_np.array(rows),
        header='x raw rlo rhi str slo shi shr hlo hhi crb',comments='')
dump('figs/data/figB_diffuse_delay.dat',xd,res_d,0,1e9)
dump('figs/data/figB_diffuse_angle.dat',xd,res_d,1,1.0)
dump('figs/data/figB_noise_delay.dat',  sn_list,res_n,0,1e9)
dump('figs/data/figB_noise_angle.dat',  sn_list,res_n,1,1.0)
print('figB data exported')

plt.rcParams.update({"font.size":8,"axes.linewidth":.6,"xtick.major.width":.6,
                     "ytick.major.width":.6,"mathtext.fontset":"cm"})
fig,ax=plt.subplots(2,2,figsize=(7.1,5.0))
series=[('raw',C1,'o','raw calibration snapshot'),
        ('str',C2,'s','full manifold projection'),
        ('shr',C3,'^','optimal shrinkage (plug-in)')]

def panel(a,xs,res,idx,scale,ylab,ttl,xlab,xticklab=None,logx=False):
    for key,col,mk,lab in series:
        y=[r[key][idx]*scale for r in res]
        lo=[r[key][2+idx][0]*scale for r in res]; hi=[r[key][2+idx][1]*scale for r in res]
        a.plot(xs,y,color=col,marker=mk,ms=4,lw=1.5,mfc="white",mew=1.1,label=lab,zorder=3)
        a.fill_between(xs,lo,hi,color=col,alpha=.18,lw=0,zorder=2)
    a.plot(xs,[r['crb'][idx]*scale for r in res],color="black",lw=1.1,ls=":",label="CRB",zorder=4)
    a.set_yscale("log")
    if logx: a.set_xscale("log")
    a.set_xlabel(xlab); a.set_ylabel(ylab); a.set_title(ttl,fontsize=8.5,pad=4)
    if xticklab is not None: a.set_xticks(xs); a.set_xticklabels(xticklab)
    a.grid(True,which="major",color="#d8d8d4",lw=.5,zorder=0)
    a.grid(True,which="minor",color="#efefed",lw=.3,zorder=0)
    a.set_axisbelow(True); a.tick_params(colors=MUT,labelsize=7)
    for s in ("top","right"): a.spines[s].set_visible(False)

lbl=["none"]+[str(int(v)) for v in xd[1:]]
panel(ax[0,0],xd,res_d,0,1e9,"delay RMSE [ns]",
      r"(a) vs. diffuse   ($\sigma_\perp^2=10^{-4}$)","unmodeled diffuse [dB rel. specular]",lbl)
panel(ax[1,0],xd,res_d,1,1.,"angle RMSE [deg]","(c)","unmodeled diffuse [dB rel. specular]",lbl)
panel(ax[0,1],sn_list,res_n,0,1e9,"delay RMSE [ns]",
      r"(b) vs. calibration noise   (no diffuse)",r"calibration noise $\sigma_\perp^2$",logx=True)
panel(ax[1,1],sn_list,res_n,1,1.,"angle RMSE [deg]","(d)",r"calibration noise $\sigma_\perp^2$",logx=True)
ax[0,0].legend(fontsize=6.3,frameon=False,loc="upper left",handlelength=2.1,
               labelcolor=INK,borderpad=.2,labelspacing=.26)
ax[0,0].annotate("projection\ncollapses",xy=(-22,600),fontsize=6.5,color=C2,ha="center")
ax[0,1].annotate("raw\ncollapses",xy=(4e-2,600),fontsize=6.5,color=C1,ha="center")
fig.tight_layout(pad=.4,w_pad=1.8,h_pad=1.2)
fig.savefig("figs/figB_sensing.pdf",bbox_inches="tight")
fig.savefig("figs/figB_sensing.png",dpi=200,bbox_inches="tight")
print("OK")
