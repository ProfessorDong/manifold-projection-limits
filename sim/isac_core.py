"""
Core model + background estimators for the v3 manuscript.
Conventions: ||a(theta)||^2 = N_r (physical array gain).
"""
import numpy as np

# ---------------- geometry / dictionary ----------------
def steering(theta, Nr):
    return np.exp(1j * np.pi * np.arange(Nr) * np.sin(theta))

def build_B(theta_list, tau_list, Nr, K, df):
    """Stacked background dictionary B in C^{Nr*K x (L+1)}, column i = atom i."""
    B = np.zeros((Nr * K, len(theta_list)), dtype=complex)
    kk = np.arange(K)
    for i, (th, ta) in enumerate(zip(theta_list, tau_list)):
        ph = np.exp(-1j * 2 * np.pi * kk * df * ta)          # (K,)
        B[:, i] = np.outer(ph, steering(th, Nr)).ravel()      # k-major, then antenna
    return B

def projector(B):
    return B @ np.linalg.solve(B.conj().T @ B, B.conj().T)

# ---------------- default scenario ----------------
class Scenario:
    def __init__(self, Nr=4, K=64, df=15e3):
        self.Nr, self.K, self.df = Nr, K, df
        self.theta_bg = np.deg2rad([10., -25., 35., 60.])
        self.tau_bg = np.array([.20, .55, .90, 1.25]) * 1e-6
        self.g = np.array([1.0*np.exp(1j*.2), .45*np.exp(1j*.8),
                           .30*np.exp(1j*1.7), .20*np.exp(-1j*.6)])
        self.L = len(self.theta_bg) - 1
        self.B = build_B(self.theta_bg, self.tau_bg, Nr, K, df)
        self.PB = projector(self.B)
        self.P_perp = np.eye(Nr*K) - self.PB
        self.r = self.B.shape[1]                 # L+1
        self.N = Nr * K
        self.h_spec = self.B @ self.g
        self.E_spec = np.vdot(self.h_spec, self.h_spec).real

    def draw_diffuse(self, rng, diffuse_dB, n_atoms=50):
        """Unmodeled diffuse background at `diffuse_dB` relative to specular power."""
        if diffuse_dB is None:
            return np.zeros(self.N, dtype=complex)
        phi = rng.uniform(-np.pi/3, np.pi/3, n_atoms)
        xi = rng.uniform(0.05e-6, 1.70e-6, n_atoms)
        c = (rng.standard_normal(n_atoms) + 1j*rng.standard_normal(n_atoms))/np.sqrt(2)
        kk = np.arange(self.K)
        Ph = np.exp(-1j*2*np.pi*np.outer(kk, xi)*self.df)          # (K, n)
        Av = np.exp(1j*np.pi*np.outer(np.arange(self.Nr), np.sin(phi)))  # (Nr, n)
        # atom j contributes outer(Ph[:,j], Av[:,j]) flattened (k-major, antenna-minor)
        d = np.einsum('kj,nj,j->kn', Ph, Av, c).reshape(self.N)
        # Scale to the prescribed ENSEMBLE energy, not to a prescribed energy in
        # every realization.  ||b(phi,xi)||^2 = N for every atom and E|c|^2 = 1,
        # so E||d||^2 = n_atoms * N exactly.  Renormalizing each draw instead
        # would make the coefficients depend on the drawn geometry and break the
        # independent-coefficient hypothesis of Proposition 2.
        d *= np.sqrt(10**(diffuse_dB/10.0) * self.E_spec / (n_atoms*self.N))
        return d

# ---------------- background estimators ----------------
def est_raw(h_cal):
    return h_cal

def est_structured(h_cal, sc):
    return sc.PB @ h_cal

def est_shrink(h_cal, sc, beta):
    """(1-beta) * raw + beta * projected."""
    return (1.0 - beta) * h_cal + beta * (sc.PB @ h_cal)

def beta_oracle(sc, sigma_n2, D_perp):
    """beta* = A/(A+D),  A = sigma_n^2 (N-r) = E||P_perp n||^2,  D = E||P_perp d||^2."""
    A = sigma_n2 * (sc.N - sc.r)
    return A / (A + D_perp) if (A + D_perp) > 0 else 1.0

def beta_plugin(h_cal, sc, sigma_n2):
    """Data-driven beta:  A / ||P_perp h_cal||^2 , clipped to [0,1].
    Uses E||P_perp h_cal||^2 = D + A, so beta = A/(A+D) needs no oracle knowledge."""
    A = sigma_n2 * (sc.N - sc.r)
    Pp = sc.P_perp @ h_cal
    denom = np.vdot(Pp, Pp).real
    return float(np.clip(A / max(denom, 1e-300), 0.0, 1.0))

# ---------------- cheap projection (avoids forming the N x N projector) --------
def proj_energy(B, BhB_inv, V):
    """||P_B v||^2 for each row v of V (T,N), without forming P_B."""
    C = V @ B.conj()                       # (T,r)
    return np.einsum('tr,rs,ts->t', C.conj(), BhB_inv, C).real

def make_config(Nr, K, df=15e3):
    th = np.deg2rad([10., -25., 35., 60.]); ta = np.array([.20,.55,.90,1.25])*1e-6
    B = build_B(th, ta, Nr, K, df)
    return B, np.linalg.inv(B.conj().T @ B)

def draw_diffuse_clustered(sc, rng, diffuse_dB, n_atoms=200, kappa_vm=40.0, tau_decay=0.30e-6):
    """Measurement-based DMC model (Quitin et al., EuCAP 2010; Poutanen et al., TAP 2011):
    diffuse energy forms clusters CO-LOCATED with the specular paths -- Von Mises in
    azimuth about each specular angle, exponentially decaying delay after each
    specular delay.  Contrast with the angular-white model used in the model-error
    literature."""
    L1 = len(sc.theta_bg)
    per = int(np.ceil(n_atoms / L1))
    phi, xi = [], []
    for i in range(L1):
        phi.append(rng.vonmises(np.sin(sc.theta_bg[i]) * 0 + sc.theta_bg[i], kappa_vm, per))
        xi.append(sc.tau_bg[i] + rng.exponential(tau_decay, per))
    phi = np.clip(np.concatenate(phi)[:n_atoms], -np.pi/2.2, np.pi/2.2)
    xi  = np.clip(np.concatenate(xi)[:n_atoms], 0.02e-6, 2.8e-6)
    c = (rng.standard_normal(n_atoms) + 1j*rng.standard_normal(n_atoms))/np.sqrt(2)
    kk = np.arange(sc.K)
    Ph = np.exp(-1j*2*np.pi*np.outer(kk, xi)*sc.df)
    Av = np.exp(1j*np.pi*np.outer(np.arange(sc.Nr), np.sin(phi)))
    d = np.einsum('kj,nj,j->kn', Ph, Av, c).reshape(sc.N)
    d *= np.sqrt(10**(diffuse_dB/10.0) * sc.E_spec / (n_atoms*sc.N))   # ensemble energy
    return d
