"""Figure A: (a) Prop.1 projection gain vs drift fraction; (b) collapse under
unmodeled diffuse scattering, with the shrinkage estimator for reference."""
import numpy as np, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,'sim')
from isac_core import Scenario, make_config, proj_energy

C1,C2,C3 = "#2a78d6","#eb6834","#1baf7a"; MUT="#52514e"; INK="#0b0b0b"
rng = np.random.default_rng(17)

def R_meas(Nr,K,rhos,T=2000):
    B,Gi = make_config(Nr,K); N,r = B.shape
    PBn_op = (B @ Gi) @ B.conj().T          # (N,N) only for small configs
    out=[]
    for rho in rhos:
        sn2 = 1.0/(N*(1+rho)); sd2 = rho*sn2/r
        d = (rng.standard_normal((T,r))+1j*rng.standard_normal((T,r)))*np.sqrt(sd2/2)
        n = (rng.standard_normal((T,N))+1j*rng.standard_normal((T,N)))*np.sqrt(sn2/2)
        Bd = d @ B.T
        e_raw = -Bd + n
        raw = np.einsum('ti,ti->t', e_raw.conj(), e_raw).real
        Pn  = n @ PBn_op.T
        e_st = -Bd + Pn
        st  = np.einsum('ti,ti->t', e_st.conj(), e_st).real
        out.append(raw.mean()/st.mean())
    return np.array(out)

rho_f = np.logspace(-4,2,300); rho_m = np.logspace(-4,2,11)
cfgs = [(4,64,C1,"o",r"$N_r{=}4$, $K{=}64$"), (8,512,C2,"s",r"$N_r{=}8$, $K{=}512$")]

plt.rcParams.update({"font.size":8,"axes.linewidth":.6,"xtick.major.width":.6,
                     "ytick.major.width":.6,"mathtext.fontset":"cm"})
fig,ax = plt.subplots(1,2,figsize=(7.1,2.8))

for Nr,K,col,mk,lab in cfgs:
    kap=4/(Nr*K); Rp=(1+rho_f)/(rho_f+kap)
    Rm=R_meas(Nr,K,rho_m, T=20000 if Nr==4 else 2000)
    rs=kap/(1-2*kap)
    ax[0].plot(rho_f,Rp,color=col,lw=1.5,zorder=3,marker=mk,markevery=[0],ms=4,
               mfc="white",mec=col,mew=1.1,label=lab)
    ax[0].plot(rho_m,Rm,ls="none",marker=mk,ms=4,mfc="white",mec=col,mew=1.1,zorder=4)
    np.savetxt(f'figs/data/figA_pred_{Nr}_{K}.dat', np.c_[rho_f,Rp], header='rho R', comments='')
    np.savetxt(f'figs/data/figA_mc_{Nr}_{K}.dat', np.c_[rho_m,Rm], header='rho R', comments='')
    open('figs/data/figA_params.dat','a').write(f'{Nr} {K} {1/kap} {rs}\n')
    ax[0].axhline(1/kap,color=col,lw=.6,ls=(0,(1,2)),zorder=1)
    ax[0].axvline(rs,color=col,lw=.6,ls=(0,(3,2)),zorder=1,ymax=.58)
    ax[0].annotate(rf"$1/\kappa={round(1/kap)}$",xy=(1.15e-4,1/kap*1.28),color=col,fontsize=6.5)
    ax[0].annotate(r"$\rho^\star$",xy=(rs,200),ha="center",color=col,fontsize=6.8)
ax[0].axhline(1,color=MUT,lw=.6,ls="--",zorder=1)
ax[0].annotate("no benefit",xy=(9e1,1.9),ha="right",color=MUT,fontsize=6.5)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlim(1e-4,1e2); ax[0].set_ylim(.7,2600)
ax[0].set_xlabel(r"drift fraction $\rho=\sigma_\delta^2(L{+}1)/\sigma_\perp^2$")
ax[0].set_ylabel(r"projection gain $R_{\mathrm{proj}}$")
ax[0].set_title("(a) in-manifold gain drift", fontsize=8.5, pad=4)
ax[0].legend(fontsize=6.6,frameon=False,loc="lower left",handlelength=2.2,
             labelcolor=INK,borderpad=.2,labelspacing=.28,bbox_to_anchor=(-0.015,-0.02))

sc=Scenario(); sn2=1e-4; A=sn2*(sc.N-sc.r)
dd=[-60,-55,-50,-45,-42,-38.8,-35,-30,-25,-20,-15,-10]; Rp_,Rs_=[],[]
for x in dd:
    T=400; raws=[];strs=[];shrs=[]
    for _ in range(T):
        n=(rng.standard_normal(sc.N)+1j*rng.standard_normal(sc.N))*np.sqrt(sn2/2)
        d=sc.draw_diffuse(rng,x); ht=sc.h_spec+d; hc=ht+n
        er=hc-ht; raws.append(np.vdot(er,er).real)
        es=sc.PB@hc-ht; strs.append(np.vdot(es,es).real)
        Pp=sc.P_perp@hc; b=float(np.clip(A/max(np.vdot(Pp,Pp).real,1e-300),0,1))
        eh=(1-b)*hc+b*(sc.PB@hc)-ht; shrs.append(np.vdot(eh,eh).real)
    Rp_.append(np.mean(raws)/np.mean(strs)); Rs_.append(np.mean(raws)/np.mean(shrs))
ax[1].axhspan(8e-4,1,color="#f6e0e0",alpha=.7,zorder=0)
ax[1].annotate("projection harmful",xy=(-10.3,1.6e-3),ha="right",fontsize=6.5,color="#8c2f2f")
ax[1].annotate("shrinkage never below 1",xy=(-59.6,1.35),fontsize=6.4,color=C3)
ax[1].plot(dd,Rp_,color=C2,marker="s",ms=4,lw=1.5,mfc="white",mew=1.1,label="full projection",zorder=3)
ax[1].plot(dd,Rs_,color=C3,marker="^",ms=4.4,lw=1.5,mfc="white",mew=1.1,label="optimal shrinkage",zorder=4)
ax[1].axhline(1,color=MUT,lw=.6,ls="--",zorder=1)
ax[1].axhline(64,color=C2,lw=.6,ls=(0,(1,2)),zorder=1)
ax[1].annotate(r"$1/\kappa=64$",xy=(-60.2,76),fontsize=6.5,color=C2)
ax[1].axvline(-38.8,color=MUT,lw=.6,ls=(0,(3,2)),zorder=1)
ax[1].annotate("crossover\n$-38.8$ dB",xy=(-38.0,140),fontsize=6.4,color=MUT)
ax[1].set_yscale("log"); ax[1].set_ylim(8e-4,400); ax[1].set_xlim(-61,-9)
ax[1].set_xlabel("unmodeled diffuse power [dB rel. specular]")
ax[1].set_ylabel("error reduction vs. raw snapshot")
ax[1].set_title("(b) unmodeled diffuse scattering", fontsize=8.5, pad=4)
ax[1].legend(fontsize=6.6,frameon=False,loc="lower left",handlelength=2.2,
             labelcolor=INK,borderpad=.2,labelspacing=.28)

for a in ax:
    a.grid(True,which="major",color="#d8d8d4",lw=.5,zorder=0)
    a.grid(True,which="minor",color="#efefed",lw=.3,zorder=0)
    a.set_axisbelow(True); a.tick_params(colors=MUT,labelsize=7)
    for s in ("top","right"): a.spines[s].set_visible(False)
fig.tight_layout(pad=.4,w_pad=1.8)
fig.savefig("figs/figA_projection.pdf",bbox_inches="tight")
fig.savefig("figs/figA_projection.png",dpi=200,bbox_inches="tight")
np.savetxt("figs/data/figA_diffuse.dat", np.c_[dd,Rp_,Rs_], header="dB Rproj Rshrink", comments="")
print("R_proj  :", [f"{v:.3f}" for v in Rp_])
print("R_shrink:", [f"{v:.3f}" for v in Rs_])
print("OK")
