import numpy as np

# ---------- frame estimation (origin + axis alignment) ----------
def _origin(X, Y, Z):
    """Centroid of the cloud (torus center in world coords)."""
    return np.array([np.mean(X), np.mean(Y), np.mean(Z)])

def _align_to_z(P):
    """
    Rotate points so the torus axis is ~ ẑ.
    Returns: Pa (aligned points), R (3x3), where Pa = (P - P.mean) @ R
    """
    Pc = P - P.mean(axis=0)
    # PCA via SVD on covariance
    U, S, Vt = np.linalg.svd(Pc, full_matrices=False)
    # Principal axes as columns of R; choose so that axis ~ smallest variance -> ẑ
    # After SVD on centered data, rows of Vt are principal directions.
    # We want R so that the 3rd principal axis maps to ẑ.
    R = np.eye(3)
    # Columns of R are the directions we map into x,y,z
    # Use first two principal directions as x,y, smallest-variance as z
    R[:, 0] = Vt[0]            # x̂'
    R[:, 1] = Vt[1]            # ŷ'
    R[:, 2] = Vt[2]            # ẑ'  (torus axis)
    Pa = Pc @ R                # align axis to z
    # Ensure right-handed and axis pointing +z
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
        Pa[:, 2] *= -1
    if np.mean(Pa[:,2]) < 0:
        R[:, 2] *= -1
        Pa[:, 2] *= -1
    return Pa, R

def _unalign_from_z(Pa, P_mean, R):
    """Inverse of _align_to_z: world coords from aligned coords."""
    return Pa @ R.T + P_mean

# ---------- R, r fitting on the aligned frame ----------
def _fit_R_aligned(Pa, iters=48):
    """
    Fit major radius R on aligned points Pa by minimizing dispersion of
    tube radii t(R) = sqrt( (rho - R)^2 + z^2 ) via golden-section search.
    """
    x, y, z = Pa[:,0], Pa[:,1], Pa[:,2]
    rho = np.hypot(x, y)

    # robust bracket for R (exclude extremes)
    R_lo, R_hi = np.percentile(rho, [10, 90])

    def tube_r(R):
        return np.hypot(rho - R, z)

    def objective(R):
        t = tube_r(R)
        med = np.median(t)
        return np.median(np.abs(t - med))   # MAD

    phi = (1 + np.sqrt(5)) / 2
    a, b = R_lo, R_hi
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    fc, fd = objective(c), objective(d)
    for _ in range(iters):
        if fc > fd:
            a = c; c = d; fc = fd
            d = a + (b - a) / phi
            fd = objective(d)
        else:
            b = d; d = c; fd = fc
            c = b - (b - a) / phi
            fc = objective(c)
    return 0.5 * (a + b)

def _fit_r_aligned(Pa, R_hat):
    """Minor radius r as the median tube radius given R_hat."""
    x, y, z = Pa[:,0], Pa[:,1], Pa[:,2]
    rho = np.hypot(x, y)
    t = np.hypot(rho - R_hat, z)
    return float(np.median(t))

# ---------- one-shot fitter (origin, rotation, R, r) ----------
def _fit_torus_params(X, Y, Z):
    """
    Return (origin, Rmat, R_hat, r_hat).
    - origin: 3-vector
    - Rmat: 3x3 rotation used by _align_to_z
    - R_hat: major radius
    - r_hat: minor radius
    """
    P = np.column_stack([X, Y, Z])
    origin = _origin(X, Y, Z)
    Pa, Rmat = _align_to_z(P)        # centered inside
    R_hat = _fit_R_aligned(Pa)
    r_hat = _fit_r_aligned(Pa, R_hat)
    return origin, Rmat, R_hat, r_hat

# ---------- snapping (in aligned frame) ----------
def _snap_aligned_to_torus(Pa, R_hat, r_hat, eps=1e-12):
    """
    Snap aligned points (axis=ẑ) to torus surface with (R_hat, r_hat).
    Returns Ps (aligned & snapped).
    """
    x, y, z = Pa[:,0], Pa[:,1], Pa[:,2]
    phi = np.arctan2(y, x)
    cx, cy = R_hat * np.cos(phi), R_hat * np.sin(phi)
    vx, vy, vz = x - cx, y - cy, z
    d = np.sqrt(vx*vx + vy*vy + vz*vz) + eps
    s = r_hat / d
    Xs = cx + s * vx
    Ys = cy + s * vy
    Zs =      s * vz
    return np.column_stack([Xs, Ys, Zs])

# ---------- convenience: full auto-snap ----------
def project_points_to_torus_auto(X, Y, Z, eps=1e-12):
    """
    Auto-fit (origin, axis, R, r) and snap.
    Returns (Xs, Ys, Zs, params) where params is a dict.
    """
    P = np.column_stack([X, Y, Z])
    origin, Rmat, R_hat, r_hat = _fit_torus_params(X, Y, Z)
    # align frame the same way fit was done
    Pa = (P - origin) @ Rmat
    Ps = _snap_aligned_to_torus(Pa, R_hat, r_hat, eps=eps)
    # back to world frame
    Pw = Ps @ Rmat.T + origin
    return Pw[:,0], Pw[:,1], Pw[:,2]
