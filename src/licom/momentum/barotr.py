"""
File: barotr.py
Description: Pure barotropic RK core functions for LICOM.
    Contains only the mathematical kernels — no JIT, no state declarations.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-22
Updated: 2026-04-13

REVISION HISTORY:
    22/09/2025 - Initial implementation of barotropic solver
    15/03/2026 - Unified JIT and sharding compatibility
    19/03/2026 - Refactor imports to package-level paths
    04/04/2026 - Fix JAX multidimensional UnspecifiedValue sharding issue and strip isb from jit state
    07/04/2026 - Optimized JIT static bindings and moved logic details to markdown
    13/04/2026 - Stripped to pure RK cores; schedule/state control moved to Schedule
"""

# Third-party imports
import jax

from .barotr_pure import _step_rk_logic

# =====================================================================
# Pure RK Barotropic Cores
# =====================================================================
#
# State layout (from schedule.STATE_KEYS):
#   [0:16]  core fields:  h0, h0p, ub, vb, ubp, vbp, celerity_x/y, ub_ct/vb_ct, ub_cx/ub_cy, vb_cx/vb_cy, advx, advy
#   [16:23] coupling:     pax, pxb, whx, pay, pyb, why, wgp  (read-only in barotropic)
#
# Consts layout (from schedule.CONST_SOURCES):
#   [0:5]   Dg fields:    dzph_x, dzph_y, rdx, rdy, a_f
#
# _step_rk_logic expects consts = (dzph_x, dzph_y, rdx, rdy, a_f, pax, pxb, whx, pay, pyb, why, wgp)
# We compose this from Dg consts + coupling state fields.
# =====================================================================


def _barotr_rk2_core(state, consts, nbb, dtb):
    # Extract coupling fields (constant within nbb barotropic sub-steps)
    pax, pxb, whx, pay, pyb, why, wgp = state[16:23]
    # Compose full consts for _step_rk_logic: Dg fields + coupling fields
    full_consts = consts + (pax, pxb, whx, pay, pyb, why, wgp)

    def body_fun(i, val):
        h0, h0p, ub, vb, ubp, vbp, *_ = val
        # RK2 Step 1
        h0_1, ub_1, vb_1, *_ = _step_rk_logic(
            h0, h0p, ub, vb, ubp, vbp, full_consts, dtb / 2.0, 0.0, False
        )
        # RK2 Step 2
        res2 = _step_rk_logic(h0_1, h0p, ub_1, vb_1, ubp, vbp, full_consts, dtb, 0.0, True)
        h0_2, ub_2, vb_2 = res2[0], res2[1], res2[2]
        # Core fields updated; coupling fields [16:23] unchanged (pass-through)
        return (h0_2, h0_2, ub_2, vb_2, ub_2, vb_2, *res2[3:], pax, pxb, whx, pay, pyb, why, wgp)

    return jax.lax.fori_loop(0, nbb, body_fun, state)


def _barotr_rk3_core(state, consts, nbb, dtb):
    # Extract coupling fields (constant within nbb barotropic sub-steps)
    pax, pxb, whx, pay, pyb, why, wgp = state[16:23]
    # Compose full consts for _step_rk_logic: Dg fields + coupling fields
    full_consts = consts + (pax, pxb, whx, pay, pyb, why, wgp)

    def body_fun(i, val):
        h0, h0p, ub, vb, ubp, vbp, *_ = val
        # RK3 Step 1
        res1 = _step_rk_logic(h0, h0p, ub, vb, ubp, vbp, full_consts, dtb / 3.0, 1.0, False)
        # RK3 Step 2
        res2 = _step_rk_logic(
            res1[0], h0p, res1[1], res1[2], ubp, vbp, full_consts, dtb / 2.0, 1.0, False
        )
        # RK3 Step 3
        res3 = _step_rk_logic(
            res2[0], h0p, res2[1], res2[2], ubp, vbp, full_consts, dtb, 1.0, True
        )
        # Core fields updated; coupling fields [16:23] unchanged (pass-through)
        return (res3[0], res3[0], res3[1], res3[2], res3[1], res3[2], *res3[3:], pax, pxb, whx, pay, pyb, why, wgp)

    return jax.lax.fori_loop(0, nbb, body_fun, state)
