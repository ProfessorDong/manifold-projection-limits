"""Where does a real array sit on the projection-gain curve?

Argos [Shepard et al., MobiCom 2012, Sec. 5.4] reports that over 4 hours the
reciprocity-calibration coefficients deviate from their mean by <0.7% in
amplitude and <2.6% in angle (= 0.08 rad).  That is a measured lower bound on
the per-gain relative drift eps^2 = sigma_delta^2 / E|g|^2.

Claim to verify:  rho = sigma_delta^2 (L+1) / sigma_perp^2 = eps^2 * SNR_cal,
with SNR_cal = E||Bg||^2 / E||n||^2, exactly when the atoms are orthogonal, and
approximately otherwise.  Then evaluate R_proj = (1+rho)/(rho+kappa) there."""
import numpy as np, sys
sys.path.insert(0, 'sim')
from isac_core import Scenario

sc = Scenario(); N, r = sc.N, sc.r
kappa = r / N
rho_star = kappa / (1 - 2*kappa)
print(f"N={N}  L+1={r}  kappa={kappa:.6f}  1/kappa={1/kappa:.1f}  rho*={rho_star:.5f}")

# --- Argos drift -> relative gain perturbation -------------------------------
amp, ang = 0.007, 0.026*np.pi                      # 0.7 %, 2.6 % of pi
eps2 = abs((1+amp)*np.exp(1j*ang) - 1.0)**2
print(f"\nArgos 4 h drift: amp {amp:.3%}, angle {ang:.4f} rad")
print(f"  -> eps^2 = {eps2:.4e}  ({10*np.log10(eps2):.1f} dB relative gain error)")

# --- check rho = eps^2 * SNR_cal on the actual (non-orthogonal) dictionary ---
print(f"\n{'SNR_cal dB':>10} {'rho (defn)':>12} {'eps^2*SNR':>12} {'ratio':>7} {'R_proj':>8}")
Eg2 = np.mean(np.abs(sc.g)**2)
Espec = np.vdot(sc.h_spec, sc.h_spec).real          # ||Bg||^2, atoms NOT orthogonal
for snr_db in [10, 20, 30, 40, 50]:
    snr = 10**(snr_db/10)
    sn2 = Espec / (N * snr)                         # sigma_perp^2 from SNR_cal
    sd2 = eps2 * Eg2                                # sigma_delta^2 from eps^2
    rho = sd2 * r / sn2                             # the paper's definition
    Rp  = (1 + rho) / (rho + kappa)
    print(f"{snr_db:10d} {rho:12.4f} {eps2*snr:12.4f} {rho/(eps2*snr):7.3f} {Rp:8.2f}")

print(f"\nIdeal (no drift) projection gain 1/kappa = {1/kappa:.1f}")
print("Gram of B (off-diagonal magnitudes, normalized):")
G = sc.B.conj().T @ sc.B; d = np.sqrt(np.diag(G).real)
Gn = np.abs(G / np.outer(d, d)); np.fill_diagonal(Gn, 0)
print(f"  max |<b_i,b_j>|/(|b_i||b_j|) = {Gn.max():.4f}  (0 = orthogonal)")
