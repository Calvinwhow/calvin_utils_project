import numpy as np

# ---------- fit sphere (center + radius) ----------
def _fit_sphere_center_radius(X, Y, Z):
    """
    Algebraic least-squares sphere fit, then robust radius via median distance.
    Returns (center, radius).
    """
    P = np.column_stack([X, Y, Z])
    A = np.column_stack([2*X, 2*Y, 2*Z, np.ones_like(X)])
    b = (X*X + Y*Y + Z*Z)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, cz, d = sol
    c = np.array([cx, cy, cz], dtype=float)

    # robust radius: median distance to fitted center
    dist = np.linalg.norm(P - c, axis=1)
    R = float(np.median(dist))
    return c, R

# ---------- snap to fitted sphere ----------
def project_points_to_sphere_auto(X, Y, Z, eps=1e-12):
    """
    Auto-fit sphere center c and radius R, then project each point along
    the ray from c to the point:  p' = c + R * (p - c)/||p - c||.
    Returns (Xs, Ys, Zs, params)
    """
    P = np.column_stack([X, Y, Z])
    c, R = _fit_sphere_center_radius(X, Y, Z)

    V = P - c
    nrm = np.linalg.norm(V, axis=1, keepdims=True) + eps
    Ps = c + (R * V / nrm)

    return Ps[:,0], Ps[:,1], Ps[:,2]
