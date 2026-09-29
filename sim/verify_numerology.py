"""Akram's point: Table I quotes a Doppler bin lambda/(2 T_cpi) without giving
T_cpi, and Section III requires every propagation delay to be shorter than the
cyclic prefix without giving T_cp.  Both are checkable, and the second turns out
to constrain Config. B.

3GPP NR (TS 38.211): with T_c = 1/(480000 x 4096) s and kappa = 64,
  normal   CP:  N_cp = 144 kappa 2^-mu
  extended CP:  N_cp = 512 kappa 2^-mu   (defined only for mu = 2, 60 kHz)
and a slot is 14 symbols for normal CP, 12 for extended.
"""
import numpy as np
T_c = 1.0/(480000*4096); kap = 64; c = 299792458.0; fc = 3.5e9
lam = c/fc

MAX_BG  = 1.25e-6      # last background delay, Table I
MAX_DMC = 2.80e-6      # clustered diffuse delays are clipped here (isac_core.py)

print(f"lambda = c/f_c = {lam*1000:.2f} mm;  longest modeled delay = {MAX_DMC*1e6:.2f} us"
      f"  (background {MAX_BG*1e6:.2f} us + diffuse tail)\n")
print(f"{'cfg':>6} {'SCS':>8} {'mu':>3} {'CP type':>9} {'T_cp[us]':>9} {'covers?':>8} "
      f"{'symbols':>8} {'T_cpi[ms]':>10} {'bin[m/s]':>9}")
for name, scs, mu in (("A", 15e3, 0), ("B", 60e3, 2), ("B", 60e3, 2)):
    for cp, ncp, nsym in ((("normal", 144*kap*2**-mu, 14),) if (name, mu) != ("B", 2)
                          else (("normal", 144*kap*2**-mu, 14),
                                ("extended", 512*kap*2**-mu, 12))):
        T_cp = ncp*T_c
        T_sym = 1.0/scs
        T_cpi = nsym*(T_sym + T_cp)
        bin_v = lam/(2*T_cpi)
        ok = "yes" if T_cp > MAX_DMC else "NO"
        print(f"{name:>6} {scs/1e3:6.0f}kHz {mu:3d} {cp:>9} {T_cp*1e6:9.3f} {ok:>8} "
              f"{nsym:8d} {T_cpi*1e3:10.3f} {bin_v:9.1f}")
    if name == "A":
        continue
    break

print("\nSo Config. A fits inside the normal cyclic prefix, while Config. B does not:")
print("at 60 kHz the normal CP is 1.172 us, shorter than the 2.80 us delay support,")
print("and the extended CP (4.167 us, defined only at 60 kHz) is required.")
print("\nTable I's Doppler bins follow from T_cpi = one slot:")
for name, scs, nsym, ncp in (("A", 15e3, 14, 144*kap), ("B", 60e3, 12, 512*kap/4)):
    T_cpi = nsym*(1.0/scs + ncp*T_c)
    print(f"  Config. {name}: T_cpi = {T_cpi*1e3:.3f} ms -> "
          f"lambda/(2 T_cpi) = {lam/(2*T_cpi):.1f} m/s")
print("\nThe projection gain of Fig. 2(a) depends on the geometry only through")
print("kappa = (L+1)/N, so it is unchanged by the subcarrier spacing.")
