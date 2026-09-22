"""Figure C: operating-region map. Where does the structural prior help?
x: unmodeled diffuse (DMC) power relative to the specular background [dB]
y: calibration noise sigma_perp^2
Crossover locus  R_proj = 1  <=>  f_perp * P_diff = sigma_perp^2 (N - r).
Measured DMC ranges from Poutanen (TAP 2011) and Quitin (EuCAP 2010) overlaid."""
import numpy as np, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,'sim')
from isac_core import Scenario, draw_diffuse_clustered, beta_plugin
from table_inspan import mubar

C1,C2,C3="#2a78d6","#eb6834","#1baf7a"; MUT="#52514e"; INK="#0b0b0b"
rng=np.random.default_rng(101); sc=Scenario()

# Out-of-span fraction under the measurement-based clustered DMC model, taken
# from the Proposition-2 identity (exact quadrature) so that the plotted locus
# carries no Monte Carlo error; the markers below verify it by simulation.
f_perp=1.0-mubar(sc.Nr,sc.K,"clustered")
_D=[];_P=[]
for _ in range(8000):
    _d=draw_diffuse_clustered(sc,rng,-30.)
    _pp=sc.P_perp@_d; _D.append(np.vdot(_pp,_pp).real); _P.append(np.vdot(_d,_d).real)
f_mc=float(np.mean(_D)/np.mean(_P))
print(f"clustered-DMC out-of-span fraction: identity {f_perp:.4f}, simulated {f_mc:.4f}")
print(f"locus intercept 10log10((N-r)/(f_perp*E_spec)) = "
      f"{10*np.log10((sc.N-sc.r)/(f_perp*sc.E_spec)):.4f} dB")

sp2=np.logspace(-6,-0.5,200)
dmc_cross=10*np.log10(sp2*(sc.N-sc.r)/(f_perp*sc.E_spec))    # crossover locus

np.savetxt('figs/data/figC_locus.dat', np.c_[dmc_cross,sp2], header='dmc_dB sigma2', comments='')
plt.rcParams.update({"font.size":8,"axes.linewidth":.6,"xtick.major.width":.6,
                     "ytick.major.width":.6,"mathtext.fontset":"cm"})
fig,ax=plt.subplots(figsize=(5.4,3.6))

ax.fill_betweenx(sp2,dmc_cross,15,color="#f6e0e0",zorder=0)
ax.fill_betweenx(sp2,-70,dmc_cross,color="#e3f1e8",zorder=0)
ax.plot(dmc_cross,sp2,color=INK,lw=1.4,zorder=4)
ax.annotate("structural prior\nHARMFUL",xy=(-24,2e-5),ha="center",va="center",
            fontsize=8.5,color="#8c2f2f",weight="bold")
ax.annotate("structural prior\nhelps",xy=(-57,1.2e-2),ha="center",va="center",
            fontsize=8.5,color="#1f6b45",weight="bold")
ax.annotate(r"$R_{\mathrm{proj}}=1$",xy=(-44,2.8e-5),fontsize=7,color=INK,rotation=58)

# measured DMC evidence
# --- measured DMC evidence: every scenario sits deep in the harmful region ---
ax.axvspan(-8,10,color="#7a2020",alpha=.13,zorder=1)
ax.axvspan(-8,-3,color=C1,alpha=.42,zorder=2)
ax.axvspan(5,10,color=C2,alpha=.42,zorder=2)
ax.axvline(-3.7,color=C3,lw=1.6,ls="--",zorder=3)
ax.annotate("measured DMC in real indoor channels",xy=(1,4.2e-6),ha="center",
            fontsize=6.8,color="#7a2020",weight="bold")
ax.annotate("LOS\n[13]",xy=(-5.5,1.3e-6),ha="center",fontsize=6.0,color=C1)
ax.annotate("NLOS\n[13]",xy=(7.5,1.3e-6),ha="center",fontsize=6.0,color=C2)
ax.annotate("mean [14]",xy=(-3.2,3.5e-1),ha="left",fontsize=6.0,color=C3)
# realistic calibration operating point
ax.axhline(1e-4,color=MUT,lw=.8,ls=(0,(4,2)),zorder=3)
ax.annotate(r"$\sigma_\perp^2=10^{-4}$",xy=(-69,1.25e-4),fontsize=6.4,color=MUT)
ax.annotate("",xy=(-8,4.5e-4),xytext=(-36.6,4.5e-4),
            arrowprops=dict(arrowstyle="<->",color="#7a2020",lw=1.0))
ax.annotate("28--46 dB margin",xy=(-22,6.5e-4),ha="center",fontsize=6.6,color="#7a2020")

# verification markers: measured R_proj at three points on the locus
open('figs/data/figC_marks.dat','w').write('dmc_dB sigma2 Rproj\n')
for s in [1e-5,1e-3,1e-1]:
    x=10*np.log10(s*(sc.N-sc.r)/(f_perp*sc.E_spec))
    num=[];den=[];T=1500
    for _ in range(T):
        n=(rng.standard_normal(sc.N)+1j*rng.standard_normal(sc.N))*np.sqrt(s/2)
        d=draw_diffuse_clustered(sc,rng,x); ht=sc.h_spec+d; hc=ht+n
        num.append(np.vdot(n,n).real); e=sc.PB@hc-ht; den.append(np.vdot(e,e).real)
    Rm=np.mean(num)/np.mean(den)           # ratio of MEANS, per the definition
    ax.plot([x],[s],marker="o",ms=5,mfc="white",mec=INK,mew=1.2,zorder=5)
    open('figs/data/figC_marks.dat','a').write(f'{x} {s} {Rm}\n')
    print(f"  locus check: sigma_perp^2={s:.0e}, DMC={x:+.1f} dB -> measured R_proj={Rm:.3f}")

ax.set_yscale("log"); ax.set_xlim(-70,15); ax.set_ylim(8e-7,1.0)
ax.set_xlabel("unmodeled diffuse (DMC) power rel. specular [dB]")
ax.set_ylabel(r"calibration noise variance $\sigma_\perp^2$")
ax.grid(True,which="major",color="#d8d8d4",lw=.5,zorder=2)
ax.set_axisbelow(False)
ax.tick_params(colors=MUT,labelsize=7)
for s_ in ("top","right"): ax.spines[s_].set_visible(False)
fig.tight_layout(pad=.4)
fig.savefig("figs/figC_region.pdf",bbox_inches="tight")
fig.savefig("figs/figC_region.png",dpi=200,bbox_inches="tight")
print("written figs/figC_region.pdf")
