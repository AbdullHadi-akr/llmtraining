# SPDX-License-Identifier: Apache-2.0
"""Anisotropic, nondimensional heat-conduction PDE for Modulus-Sym.

Residual (nondimensional, matching the current PyTorch implementation):

    r = T_t
        - ( Fo_xx*T_xx + Fo_yy*T_yy + Fo_zz*T_zz
            + 2*(Fo_xy*T_xy + Fo_xz*T_xz + Fo_yz*T_yz) )
        - Qsrc

The Fourier numbers ``Fo_*`` and the source ``Qsrc`` are supplied per collocation
point as *input keys* (constant per point, fed from numpy). They are only ever
*multiplied* by derivatives of ``T`` — never differentiated — so no spurious
``grad(Fo)`` / ``grad(Qsrc)`` terms are generated (this reproduces the
constant-per-point coefficient assumption of the current model, i.e. no ``div(lambda)``
term).
"""

from sympy import Symbol, Function

from physicsnemo.sym.eq.pde import PDE


class AnisotropicHeatNonDim(PDE):
    """Nondimensional anisotropic heat equation with per-point coefficients."""

    name = "AnisotropicHeatNonDim"

    def __init__(self) -> None:
        # coordinates (network inputs)
        x, y, z, t = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("t")
        coords = {"x": x, "y": y, "z": z, "t": t}

        # temperature field (network output)
        T = Function("T")(*coords.values())

        # per-point coefficients supplied as data (functions of coords so Modulus
        # registers them as input keys, but never differentiated below)
        Fo_xx = Function("Fo_xx")(*coords.values())
        Fo_yy = Function("Fo_yy")(*coords.values())
        Fo_zz = Function("Fo_zz")(*coords.values())
        Fo_xy = Function("Fo_xy")(*coords.values())
        Fo_xz = Function("Fo_xz")(*coords.values())
        Fo_yz = Function("Fo_yz")(*coords.values())
        Qsrc = Function("Qsrc")(*coords.values())

        div = (
            Fo_xx * T.diff(x, 2)
            + Fo_yy * T.diff(y, 2)
            + Fo_zz * T.diff(z, 2)
            + 2.0
            * (
                Fo_xy * T.diff(x).diff(y)
                + Fo_xz * T.diff(x).diff(z)
                + Fo_yz * T.diff(y).diff(z)
            )
        )

        self.equations = {}
        self.equations["heat_residual"] = T.diff(t) - div - Qsrc
