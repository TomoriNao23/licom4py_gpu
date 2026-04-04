"""
File: barotr.py
Description: High-performance barotropic time stepping methods for Momentum class.
    Consolidated with JIT compatibility for sharded JAX execution.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-22
Updated: 2026-03-19

REVISION HISTORY:
    22/09/2025 - Initial implementation of barotropic solver
    15/03/2026 - Unified JIT and sharding compatibility
    19/03/2026 - Refactor imports to package-level paths
"""

import jax
import jax.numpy as jnp
from jax import jit
from jax.sharding import PartitionSpec as P

# Local application imports
from licom.mesh import Communication, GPU_Mesh, Cube
from licom.duogrid import Dg
from licom.operators import AGrid, Poly, Remap

# =====================================================================
# 1. Physics Operators (Pure Functions)
# =====================================================================

def get_celerity(h, dzph_x, dzph_y):
    hx, hy = Poly.scalar_xy(h)
    celerity_x = jnp.sqrt(9.8 * (hx + dzph_x))
    celerity_y = jnp.sqrt(9.8 * (hy + dzph_y))
    return jnp.where(celerity_x < 10.0, 10.0, celerity_x), \
           jnp.where(celerity_y < 10.0, 10.0, celerity_y)

def get_vel_vis_2d(celerity_x, celerity_y, h, u, v):
    he, hw , hn, hs = Poly.vector(h)
    vel_vis_x = jnp.zeros_like(u).at[:, 3:-2, :].set(
        4.9 / celerity_x[:, 3:-2, :] * (he[:, 2:-3, :] - hw[:, 3:-2, :]))
    vel_vis_y = jnp.zeros_like(v).at[:, :, 3:-2].set(
        4.9 / celerity_y[:, :, 3:-2] * (hn[:, :, 2:-3] - hs[:, :, 3:-2]))
    return u.at[:, 3:-2, :].set(u[:, 3:-2, :] + vel_vis_x[:, 3:-2, :]), \
           v.at[:, :, 3:-2].set(v[:, :, 3:-2] + vel_vis_y[:, :, 3:-2])

def flux_calculation(h0, ub_cx, vb_cy, dzph_x, dzph_y):
    he, hw, hn, hs = Poly.vector(h0)
    h_upwind_x = jnp.zeros_like(ub_cx).at[:, 1:, :].set(
        jnp.where(ub_cx[:, 1:, :] > 0.0, he[:, :-1, :], hw[:, 1:, :]))
    h_upwind_y = jnp.zeros_like(vb_cy).at[:, :, 1:].set(
        jnp.where(vb_cy[:, :, 1:] > 0.0, hn[:, :, :-1], hs[:, :, 1:]))
    return (h_upwind_x + dzph_x) * ub_cx, (h_upwind_y + dzph_y) * vb_cy

def fb_scheme(h0, h0_tem, beta_d):
    return (1.0 - beta_d) * h0 + beta_d * h0_tem

def calculate_pgf(h0):
    gradx, grady = AGrid.grad(h0)
    return -9.80 * gradx, -9.80 * grady

def get_pgf_vis_2d(celerity_x, celerity_y, rdx, rdy, u, v, pgf_u, pgf_v):
    ue, uw = Poly.vector_ew(u)
    vn, vs = Poly.vector_ns(v)
    
    vel_vis_x = jnp.zeros_like(u).at[:, 3:-2, :].set(0.5 * celerity_x[:, 3:-2, :] * (ue[:, 2:-3, :] - uw[:, 3:-2, :]))
    vel_vis_y = jnp.zeros_like(v).at[:, :, 3:-2].set(0.5 * celerity_y[:, :, 3:-2] * (vn[:, :, 2:-3] - vs[:, :, 3:-2]))
    
    vel_vis_x = jnp.zeros_like(u).at[:, 3:-3, 3:-3].set(
        rdx[:, 3:-3, 3:-3] * (vel_vis_x[:, 4:-2, 3:-3] - vel_vis_x[:, 3:-3, 3:-3]))
    vel_vis_y = jnp.zeros_like(v).at[:, 3:-3, 3:-3].set(
        rdy[:, 3:-3, 3:-3] * (vel_vis_y[:, 3:-3, 4:-2] - vel_vis_y[:, 3:-3, 3:-3]))
    
    return pgf_u.at[:, 3:-3, 3:-3].set(pgf_u[:, 3:-3, 3:-3] - vel_vis_x[:, 3:-3, 3:-3]), \
           pgf_v.at[:, 3:-3, 3:-3].set(pgf_v[:, 3:-3, 3:-3] - vel_vis_y[:, 3:-3, 3:-3])

def calculate_advection(vb_cx, ub_cy, ub_cx, vb_cy, vb_ct, ub_ct, rdx, rdy):
    vort = AGrid.vorticity(vb_cx, ub_cy)
    kin_u = 0.5 * (ub_cx**2 + vb_cx**2)
    kin_v = 0.5 * (vb_cy**2 + ub_cy**2)
    advx = jnp.zeros_like(vb_cx).at[:, :-1, :].set(
        vort[:, :-1, :] * vb_ct[:, :-1, :] - (kin_u[:, 1:, :] - kin_u[:, :-1, :]) * rdx[:, :-1, :])
    advy = jnp.zeros_like(vb_cy).at[:, :, :-1].set(
        -vort[:, :, :-1] * ub_ct[:, :, :-1] - (kin_v[:, :, 1:] - kin_v[:, :, :-1]) * rdy[:, :, :-1])
    return advx, advy

# =====================================================================
# 2. Step Logic (Pure Functions, JIT-ready)
# =====================================================================

def _step_rk_logic(h0, h0p, ub, vb, ubp, vbp, consts, dt, beta_d, is_laststep):
    dzph_x, dzph_y, pax, pxb, whx, pay, pyb, why, wgp, rdx, rdy, a_f = consts

    celerity_x, celerity_y = get_celerity(h0, dzph_x, dzph_y)
    ub_ct, vb_ct, ub_cx, ub_cy, vb_cx, vb_cy = Remap.vector_trans_2d(ub, vb)
    ub_cx, vb_cy = get_vel_vis_2d(celerity_x, celerity_y, h0, ub_cx, vb_cy)
    flux_hu, flux_hv = flux_calculation(h0, ub_cx, vb_cy, dzph_x, dzph_y)
    
    if is_laststep:
        flux_hu, flux_hv = Communication.boundary_communication(flux_hu, flux_hv)
        
    div_out = AGrid.div(flux_hu, flux_hv)
    new_h0 = h0p - div_out * dt
    new_h0 = Cube.ext_scalar(new_h0)
    
    h0_tem = fb_scheme(new_h0, h0, beta_d)
    pgf_u, pgf_v = calculate_pgf(h0_tem)
    pgf_u, pgf_v = get_pgf_vis_2d(celerity_x, celerity_y, rdx, rdy, ub_ct, vb_ct, pgf_u, pgf_v)
    
    advx, advy = calculate_advection(vb_cx, ub_cy, ub_cx, vb_cy, vb_ct, ub_ct, rdx, rdy)
    rhs_u = pgf_u + advx + a_f * vb_ct
    rhs_v = pgf_v + advy - a_f * ub_ct
    
    new_ub = ubp + rhs_u * dt
    new_vb = vbp + rhs_v * dt
    new_ub, new_vb = Cube.ext_vector(new_ub, new_vb)
    
    return new_h0, new_ub, new_vb, celerity_x, celerity_y, \
           ub_ct, vb_ct, ub_cx, ub_cy, vb_cx, vb_cy, advx, advy

# =====================================================================
# 3. Main Solver JIT (Handles all tiles)
# =====================================================================

def _barotr_rk2_core(state, consts, nbb, dtb):
    def body_fun(i, val):
        h0, h0p, ub, vb, ubp, vbp, *others = val
        # RK2 Step 1
        h0_1, ub_1, vb_1, *_ = _step_rk_logic(h0, h0p, ub, vb, ubp, vbp, consts, dtb/2.0, 0.0, False)
        # RK2 Step 2
        res2 = _step_rk_logic(h0_1, h0p, ub_1, vb_1, ubp, vbp, consts, dtb, 0.0, True)
        h0_2, ub_2, vb_2 = res2[0], res2[1], res2[2]
        return (h0_2, h0_2, ub_2, vb_2, ub_2, vb_2, jnp.int32(i + 1), *res2[3:])

    return jax.lax.fori_loop(0, nbb, body_fun, state)

def _barotr_rk3_core(state, consts, nbb, dtb):
    def body_fun(i, val):
        h0, h0p, ub, vb, ubp, vbp, *others = val
        res1 = _step_rk_logic(h0, h0p, ub, vb, ubp, vbp, consts, dtb/3.0, 1.0, False)
        res2 = _step_rk_logic(res1[0], h0p, res1[1], res1[2], ubp, vbp, consts, dtb/2.0, 1.0, False)
        res3 = _step_rk_logic(res2[0], h0p, res2[1], res2[2], ubp, vbp, consts, dtb, 1.0, True)
        return (res3[0], res3[0], res3[1], res3[2], res3[1], res3[2], jnp.int32(i + 1), *res3[3:])

    return jax.lax.fori_loop(0, nbb, body_fun, state)

# Lazy-initialized jit wrappers.
# Cannot be created at module load time because the mesh (and hence sharding)
# is not yet configured. They are built on first call using the actual
# .sharding of the input arrays, which eliminates UnspecifiedValue errors
# that arise when large arrays (C1536) can no longer be inlined as XLA
# constants and must be treated as real sharded inputs.
_barotr_rk2_jit = None
_barotr_rk3_jit = None


def _make_barotr_jit(core_fn, state, consts):
    """Build a jit with explicit in_shardings/out_shardings from actual arrays.

    Every entry must have a concrete Sharding (not None/UnspecifiedValue).
    When in_shardings contains None, JAX records UnspecifiedValue for that slot.
    If the corresponding input array then carries a real NamedSharding, JAX
    calls  dst_sharding.addressable_devices_indices_map()  at runtime and
    crashes because UnspecifiedValue does not implement the Sharding interface.

    Therefore we require ALL inputs to carry concrete shardings before
    _make_barotr_jit is called.  The caller (_pack) is responsible for
    placing 0-d scalars like isb onto a replicated mesh sharding so their
    sharding is NamedSharding(mesh, P()) rather than SingleDeviceSharding.
    """
    def _sh(x):
        if isinstance(x, jax.Array) and hasattr(x, 'sharding'):
            return x.sharding
        return None   # non-JAX objects only (should not occur in practice)

    state_sh  = tuple(_sh(x) for x in state)
    consts_sh = tuple(_sh(x) for x in consts)

    return jit(
        core_fn,
        static_argnums=(2, 3),
        in_shardings=(state_sh, consts_sh),
        out_shardings=state_sh,
    )

# =====================================================================
# 4. Class Methods
# =====================================================================

def add_barotropic_methods(cls):
    def _pack(self):
        # isb (barotropic step counter) is a 0-d int32 scalar.  By default it
        # lives on a single device (SingleDeviceSharding(GPU:0)), which is
        # incompatible with the multi-device NamedSharding of the 2-D fields.
        # Placing it on a fully-replicated mesh sharding makes it visible on
        # all devices and gives it a concrete NamedSharding(mesh, P()) that
        # _make_barotr_jit can record in in_shardings / out_shardings.
        # Without this, the jit compiles with UnspecifiedValue for that slot
        # and crashes at runtime with:
        #   AttributeError: 'UnspecifiedValue' object has no attribute
        #   'addressable_devices_indices_map'
        from jax.sharding import NamedSharding
        isb_replicated = jax.device_put(
            self.isb, NamedSharding(GPU_Mesh.mesh, P())
        )
        state = (self.h0, self.h0p, self.ub, self.vb, self.ubp, self.vbp, isb_replicated,
                 self.celerity_x, self.celerity_y, self.ub_ct, self.vb_ct, self.ub_cx,
                 self.ub_cy, self.vb_cx, self.vb_cy, self.advx, self.advy)
        consts = (Dg.dzph_x, Dg.dzph_y, self.pax, self.pxb, self.whx,
                  self.pay, self.pyb, self.why, self.wgp, Dg.rdx, Dg.rdy, Dg.a_f)
        return state, consts

    def _unpack(self, state):
        (self.h0, self.h0p, self.ub, self.vb, self.ubp, self.vbp, self.isb,
         self.celerity_x, self.celerity_y, self.ub_ct, self.vb_ct, self.ub_cx, 
         self.ub_cy, self.vb_cx, self.vb_cy, self.advx, self.advy) = state

    def barotr_rk2(self):
        global _barotr_rk2_jit
        state, consts = self._pack()
        with GPU_Mesh.mesh:
            if _barotr_rk2_jit is None:
                # Build the jit INSIDE the mesh context so that XLA traces
                # with the active mesh. Closure-captured Dg.* arrays in
                # agrid.py / remap.py (Dg.rda, Dg.rdx, Dg.a_gct, ...) are
                # then visible as sharded arrays rather than unsharded
                # constants, preventing the UnspecifiedValue sharding error.
                _barotr_rk2_jit = _make_barotr_jit(_barotr_rk2_core, state, consts)
            self._unpack(_barotr_rk2_jit(state, consts, self.nbb, self.dtb))

    def barotr_rk3(self):
        global _barotr_rk3_jit
        state, consts = self._pack()
        with GPU_Mesh.mesh:
            if _barotr_rk3_jit is None:
                _barotr_rk3_jit = _make_barotr_jit(_barotr_rk3_core, state, consts)
            self._unpack(_barotr_rk3_jit(state, consts, self.nbb, self.dtb))

    cls.barotr_rk2 = barotr_rk2
    cls.barotr_rk3 = barotr_rk3
    cls._pack = _pack
    cls._unpack = _unpack
    return cls