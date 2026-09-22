"""Does manifold projection help on REAL measured channels?

For a real channel h (DICHASUS cf06, array A) and an OMP dictionary B fitted to
that same channel (i.e. the BEST case for the structural prior):
    raw error        = ||n||^2                     with  n ~ CN(0, s2 I) synthetic
    structured error = ||P_B(h+n) - h||^2 = ||P_perp h||^2 + ||P_B n||^2
    shrinkage        = plug-in beta_hat
Parameterise the calibration quality by the raw NMSE  = s2 N / ||h||^2, so the
result is scale-free.  R_proj = NMSE_raw / (f_d + NMSE_raw * kappa).
"""
import numpy as np, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,'sim')
import exp_real_dichasus as E

C1,C2,C3="#2a78d6","#eb6834","#1baf7a"; MUT="#52514e"; INK="#0b0b0b"
rng=np.random.default_rng(0)

az=np.deg2rad(np.linspace(-80,80,161)); el=np.deg2rad(np.linspace(-80,80,17))
A=np.array([[E.steer(a,e) for a in az] for e in el])
H=E.load('data/cf06_head.bin'); S,Nr,K=H.shape; N=Nr*K
Ls=[4,16]; LMAX=max(Ls)
print(f"{S} real snapshots, N={N}")

# --- OMP once per snapshot to LMAX; prefixes give the lower orders ---
store=[]
for s in range(S):
    h=H[s].ravel()
    B,g,d,_=E.omp(H[s],LMAX,A,az,el)
    per={}
    for p in Ls:
        Bp=B[:,:p]; gp,*_=np.linalg.lstsq(Bp,h,rcond=None)
        dp=h-Bp@gp
        Q,_=np.linalg.qr(Bp)                       # orthonormal basis of span(Bp)
        per[p]=(Q,np.vdot(dp,dp).real)
    store.append((h,np.vdot(h,h).real,per))

nm=np.logspace(-3,0.3,22)                          # raw calibration NMSE
print(f"\n{'L+1':>4} {'DMC share':>10} {'crossover NMSE_raw':>19}")
curves={}
for p in Ls:
    fd=np.mean([per[p][1]/hn for _,hn,per in store])
    kap=p/N
    print(f"{p:4d} {fd:10.4f} {fd/(1-kap):19.4f}")
    rp=[];rs=[]
    for nmse in nm:
        num=[];den_s=[];den_h=[]
        for h,hn,per in store:
            Q,dn=per[p]
            s2=nmse*hn/N
            n=(rng.standard_normal(N)+1j*rng.standard_normal(N))*np.sqrt(s2/2)
            hc=h+n
            num.append(np.vdot(n,n).real)
            Pn=Q@(Q.conj().T@n); Ph=Q@(Q.conj().T@hc)
            e_s=Ph-h; den_s.append(np.vdot(e_s,e_s).real)
            Pp=hc-Ph                                   # P_perp hc
            b=float(np.clip(s2*(N-p)/max(np.vdot(Pp,Pp).real,1e-300),0,1))
            e_h=(1-b)*hc+b*Ph-h; den_h.append(np.vdot(e_h,e_h).real)
        rp.append(np.mean(num)/np.mean(den_s)); rs.append(np.mean(num)/np.mean(den_h))
    curves[p]=(np.array(rp),np.array(rs),fd/(1-kap))


import numpy as _np
_cols=[nm]; _hdr='nmse'
for _p in Ls:
    _cols += [curves[_p][0], curves[_p][1]]; _hdr += f' proj{_p} shr{_p}'
_np.savetxt('figs/data/figD_real.dat',_np.array(_cols).T,header=_hdr,comments='')
with open('figs/data/figD_marks.dat','w') as _f:
    _f.write('L1 crossover\n')
    for _p in Ls: _f.write(f'{_p} {curves[_p][2]}\n')
print('figD data exported')

plt.rcParams.update({"font.size":8,"axes.linewidth":.6,"xtick.major.width":.6,
                     "ytick.major.width":.6,"mathtext.fontset":"cm"})
fig,ax=plt.subplots(figsize=(5.2,3.4))
sty={4:"-",16:"--"}
for p in Ls:
    rp,rs,_=curves[p]
    ax.plot(nm,rp,color=C2,ls=sty[p],lw=1.5,label=fr"full projection, $L{{+}}1={p}$")
    ax.plot(nm,rs,color=C3,ls=sty[p],lw=1.5,label=fr"optimal shrinkage, $L{{+}}1={p}$")
ax.axhline(1,color=MUT,lw=.7,ls="--")
ax.fill_between(nm,8e-3,1,color="#f6e0e0",alpha=.75,zorder=0)
ax.annotate("projection HARMFUL",xy=(1.2e-3,1.3e-2),ha="left",fontsize=7.5,
            color="#8c2f2f",weight="bold")
for p_ in Ls:
    xc=curves[p_][2]
    ax.plot([xc],[1.0],marker="o",ms=5,mfc="white",mec=INK,mew=1.2,zorder=6)
    ax.annotate(f"{xc:.2f}",xy=(xc,1.35),ha="center",fontsize=6.5,color=INK)
ax.annotate("shrinkage never below 1",xy=(1.15e-3,1.28),fontsize=6.8,color=C3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"raw calibration NMSE  $\sigma_\perp^2 N/\|\mathbf{h}\|^2$")
ax.set_ylabel("error reduction vs. raw snapshot")
ax.set_title("Measured channels (DICHASUS, industrial hall)",fontsize=8.5,pad=4)
ax.set_xlim(1e-3,2); ax.set_ylim(8e-3,4)
ax.grid(True,which="major",color="#d8d8d4",lw=.5); ax.grid(True,which="minor",color="#efefed",lw=.3)
ax.set_axisbelow(True); ax.tick_params(colors=MUT,labelsize=7)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.legend(fontsize=6.4,frameon=False,loc="lower right",handlelength=2.4,labelcolor=INK,
          labelspacing=.28,borderpad=.2)
fig.tight_layout(pad=.4)
fig.savefig("figs/figD_real.pdf",bbox_inches="tight"); fig.savefig("figs/figD_real.png",dpi=200,bbox_inches="tight")
print("\nwritten figs/figD_real.pdf")
