import taichi as ti
from options import real, dim

@ti.func
def SvdDifferential(F, U, S, V, dF):
    dS = (U.transpose() @ dF @ V)
    eps = 1e-10
    Ut = U.transpose()
    dP = Ut @ dF @ V
    dPt = dP.transpose()
    Sij = ti.Matrix.zero(real, dim, dim)
    ti.loop_config(serialize=True)
    for i, j in ti.ndrange(3, 3):
        if (i >= j):
            continue
        if (S[i, i] - S[j, j] > eps):
            Sij[i, j] = 1. / (S[j, j] * S[j, j] - S[i, i] * S[i, i])
            Sij[j, i] = -Sij[i, j]
    domega_U = Sij * (dP @ S + S @ dPt)
    domega_V = Sij * (S @ dP + dPt @ S)
    dU = U @ domega_U
    dV = V @ domega_V
    return dU, dS, dV

@ti.func
def W(xij: ti.math.vec3, h: real) -> real:
    q = xij.norm() / h
    ret = 0.
    if q < 1:
        ret = 1 / (ti.math.pi * h ** 3) * (1 - 1.5 * q ** 2 + 0.75 * q ** 3)
    elif q >= 1 and q < 2:
        ret = 1 / (4 * ti.math.pi * h ** 3) * (2 - q) ** 3
    return ret

@ti.func
def nabla_W(xij: ti.math.vec3, h: real) -> ti.math.vec3:
    q = xij.norm() / h
    ret = ti.Vector.zero(real, dim)
    if q < 1:
        ret = 1 / (ti.math.pi * h ** 3) * (-3 * xij / h ** 2 + 0.75 * 3 * q * xij / h ** 2)
    elif q >= 1 and q < 2:
        ret = 1 / (4 * ti.math.pi * h ** 3) * -3 * (2 - q) ** 2 * xij / (q * h * h)
    return ret

@ti.func
def backward_svd(gu, gsigma, gv, u, sig, v):
    # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
    vt = v.transpose()
    ut = u.transpose()
    sigma_term = u @ gsigma @ vt

    s = ti.Vector.zero(real, dim)
    if ti.static(dim==2):
        s = ti.Vector([sig[0, 0], sig[1, 1]]) ** 2
    else:
        s = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]]) ** 2
    F = ti.Matrix.zero(real, dim, dim)
    for i, j in ti.static(ti.ndrange(dim, dim)):
        if i == j: F[i, j] = 0
        else: F[i, j] = 1./clamp(s[j] - s[i])
    u_term = u @ ((F * (ut@gu - gu.transpose()@u)) @ sig) @ vt
    v_term = u @ (sig @ ((F * (vt@gv - gv.transpose()@v)) @ vt))
    return u_term + v_term + sigma_term

@ti.func
def clamp(a):
    # remember that we don't support if return in taichi
    # stop the gradient ...
    if a>=0:
        a = max(a, 1e-6)
    else:
        a = min(a, -1e-6)
    return a