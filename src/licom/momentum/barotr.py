"""
File: barotropic.py
Description: High-performance barotropic time stepping methods for Momentum class.
             Using single-graph JIT compilation for the ENTIRE RK2/RK3 loops.
"""

import jax
import jax.numpy as jnp
from backend.cube_grid.use_mpp import FMS_chtholly
from duogrid import Dg
from operators.agrid import agrid_div, agrid_grad, agrid_vorticity
from operators.remap import vector_trans_2d
from operators.poly import (
    scalar_interpolation_xy, vector_interpolation,
    vector_interpolation_ew, vector_interpolation_ns
)

# =====================================================================
# 1. 纯函数物理算子 (无 JIT，由上层循环调用)
# =====================================================================

def get_celerity(h, dzph_x, dzph_y):
    celerity_x, celerity_y = jnp.zeros_like(h), jnp.zeros_like(h)
    hx, hy = scalar_interpolation_xy(h)
    celerity_x = celerity_x.at[:,:].set(jnp.sqrt(9.8 * (hx + dzph_x)))
    celerity_y = celerity_y.at[:,:].set(jnp.sqrt(9.8 * (hy + dzph_y)))
    return jnp.where(celerity_x < 10.0, 10.0, celerity_x), jnp.where(celerity_y < 10.0, 10.0, celerity_y)

def get_vel_vis_2d(celerity_x, celerity_y, h, u, v):
    vel_vis_x, vel_vis_y = jnp.zeros_like(u), jnp.zeros_like(v)
    he, hw , hn, hs = vector_interpolation(h)
    vel_vis_x = vel_vis_x.at[3:-2,:].set(4.9 / celerity_x[3:-2,:] * (he[2:-3,:] - hw[3:-2,:]))
    vel_vis_y = vel_vis_y.at[:,3:-2].set(4.9 / celerity_y[:,3:-2] * (hn[:,2:-3] - hs[:,3:-2]))
    return u.at[3:-2,:].set(u[3:-2,:] + vel_vis_x[3:-2,:]), v.at[:,3:-2].set(v[:,3:-2] + vel_vis_y[:,3:-2])

def flux_calculation(h0, ub_cx, vb_cy, dzph_x, dzph_y):
    he, hw, hn, hs = vector_interpolation(h0)
    h_upwind_x = jnp.zeros_like(ub_cx).at[1:, :].set(jnp.where(ub_cx[1:, :] > 0.0, he[:-1, :], hw[1:, :]))
    h_upwind_y = jnp.zeros_like(vb_cy).at[:, 1:].set(jnp.where(vb_cy[:, 1:] > 0.0, hn[:, :-1], hs[:, 1:]))
    return (h_upwind_x + dzph_x) * ub_cx, (h_upwind_y + dzph_y) * vb_cy

def fb_scheme(h0, h0_tem, beta_d):
    return (1.0 - beta_d) * h0 + beta_d * h0_tem

def calculate_pgf(h0):
    gradx, grady = agrid_grad(h0)
    return -9.80 * gradx, -9.80 * grady

def get_pgf_vis_2d(celerity_x, celerity_y, rdx, rdy, u, v, pgf_u, pgf_v):
    vel_vis_x, vel_vis_y = jnp.zeros_like(u), jnp.zeros_like(v)
    ue, uw = vector_interpolation_ew(u)
    vn, vs = vector_interpolation_ns(v)
    vel_vis_x = vel_vis_x.at[3:-2,:].set(0.5 * celerity_x[3:-2,:] * (ue[2:-3,:] - uw[3:-2,:]))
    vel_vis_y = vel_vis_y.at[:,3:-2].set(0.5 * celerity_y[:,3:-2] * (vn[:,2:-3] - vs[:,3:-2]))
    vel_vis_x = vel_vis_x.at[3:-3,3:-3].set(rdx[3:-3,3:-3] * (vel_vis_x[4:-2,3:-3] - vel_vis_x[3:-3,3:-3]))
    vel_vis_y = vel_vis_y.at[3:-3,3:-3].set(rdy[3:-3,3:-3] * (vel_vis_y[3:-3,4:-2] - vel_vis_y[3:-3,3:-3]))
    return pgf_u.at[3:-3,3:-3].set(pgf_u[3:-3,3:-3] - vel_vis_x[3:-3,3:-3]), pgf_v.at[3:-3,3:-3].set(pgf_v[3:-3,3:-3] - vel_vis_y[3:-3,3:-3])

def calculate_advection(vb_cx, ub_cy, ub_cx, vb_cy, vb_ct, ub_ct, rdx, rdy):
    vort = agrid_vorticity(vb_cx, ub_cy)
    kin_u = 0.5 * (ub_cx**2 + vb_cx**2)
    kin_v = 0.5 * (vb_cy**2 + ub_cy**2)
    advx = jnp.zeros_like(vb_cx).at[:-1, :].set(vort[:-1, :] * vb_ct[:-1, :] - (kin_u[1:, :] - kin_u[:-1, :]) * rdx[:-1, :])
    advy = jnp.zeros_like(vb_cy).at[:, :-1].set(-vort[:, :-1] * ub_ct[:, :-1] - (kin_v[:, 1:] - kin_v[:, :-1]) * rdy[:, :-1])
    return advx, advy

# =====================================================================
# 2. 单步逻辑 (被循环调用，内部带有 pure_callback 阻断点)
# =====================================================================

def _step_rk_logic(h0, h0p, ub, vb, ubp, vbp, consts, dt, beta_d, is_laststep):
    """单步执行逻辑，完全是纯函数，consts 包含网格常量"""
    dzph_x, dzph_y, pax, pxb, whx, pay, pyb, why, wgp, rdx, rdy, a_f = consts

    celerity_x, celerity_y = get_celerity(h0, dzph_x, dzph_y)
    ub_ct, vb_ct, ub_cx, ub_cy, vb_cx, vb_cy = vector_trans_2d(ub, vb)
    ub_cx, vb_cy = get_vel_vis_2d(celerity_x, celerity_y, h0, ub_cx, vb_cy)
    flux_hu, flux_hv = flux_calculation(h0, ub_cx, vb_cy, dzph_x, dzph_y)
    
    # Python `if` 在这里可以安全使用，因为 is_laststep 在外层是静态 bool 字面量
    if is_laststep:
        flux_hu, flux_hv = jax.pure_callback(
            FMS_chtholly.communication2d,
            (jax.ShapeDtypeStruct(flux_hu.shape, flux_hu.dtype), jax.ShapeDtypeStruct(flux_hv.shape, flux_hv.dtype)),
            flux_hu, flux_hv
        )
        
    div_out = agrid_div(flux_hu, flux_hv)
    new_h0 = h0p - div_out * dt
    
    new_h0 = jax.pure_callback(
        FMS_chtholly.ext_scalar,
        jax.ShapeDtypeStruct(new_h0.shape, new_h0.dtype),
        new_h0
    )
    
    h0_tem = fb_scheme(new_h0, h0, beta_d)
    pgf_u, pgf_v = calculate_pgf(h0_tem)  # 原来的 pax, pxb 好像被你注释掉了，如果需要加回，在这修改
    pgf_u, pgf_v = get_pgf_vis_2d(celerity_x, celerity_y, rdx, rdy, ub_ct, vb_ct, pgf_u, pgf_v)
    
    advx, advy = calculate_advection(vb_cx, ub_cy, ub_cx, vb_cy, vb_ct, ub_ct, rdx, rdy)
    rhs_u = pgf_u + advx + a_f * vb_ct
    rhs_v = pgf_v + advy - a_f * ub_ct
    
    new_ub = ubp + rhs_u * dt
    new_vb = vbp + rhs_v * dt
    
    new_ub, new_vb = jax.pure_callback(
        FMS_chtholly.ext_vector,
        (jax.ShapeDtypeStruct(new_ub.shape, new_ub.dtype), jax.ShapeDtypeStruct(new_vb.shape, new_vb.dtype)),
        new_ub, new_vb
    )
    
    return new_h0, new_ub, new_vb, celerity_x, celerity_y, ub_ct, vb_ct, ub_cx, ub_cy, vb_cx, vb_cy, advx, advy

# =====================================================================
# 3. 完整循环 JIT 编译 (利用 jax.lax.fori_loop)
# =====================================================================

@jax.jit
def _barotr_rk2_jit_loop(state, consts, nbb, dtb):
    """将整个 RK2 nbb 循环吃进 XLA"""
    def body_fun(i, val):
        # 解包状态 (isb 作为 jnp.int32 在内部流转)
        h0, h0p, ub, vb, ubp, vbp, isb, *_ = val
        
        # RK2 Step 1
        res1 = _step_rk_logic(h0, h0p, ub, vb, ubp, vbp, consts, dtb/2.0, 0.0, False)
        h0_1, ub_1, vb_1 = res1[0], res1[1], res1[2]
        
        # RK2 Step 2
        res2 = _step_rk_logic(h0_1, h0p, ub_1, vb_1, ubp, vbp, consts, dtb, 0.0, True)
        h0_2, ub_2, vb_2, cel_x, cel_y, ub_ct, vb_ct, ub_cx, ub_cy, vb_cx, vb_cy, advx, advy = res2
        
        # Postprocess 逻辑
        new_isb = isb + 1
        return (h0_2, h0_2, ub_2, vb_2, ub_2, vb_2, new_isb, cel_x, cel_y, ub_ct, vb_ct, ub_cx, ub_cy, vb_cx, vb_cy, advx, advy)

    return jax.lax.fori_loop(0, nbb, body_fun, state)

@jax.jit
def _barotr_rk3_jit_loop(state, consts, nbb, dtb):
    """将整个 RK3 nbb 循环吃进 XLA"""
    def body_fun(i, val):
        h0, h0p, ub, vb, ubp, vbp, isb, *_ = val
        
        # RK3 Steps
        res1 = _step_rk_logic(h0, h0p, ub, vb, ubp, vbp, consts, dtb/3.0, 1.0, False)
        res2 = _step_rk_logic(res1[0], h0p, res1[1], res1[2], ubp, vbp, consts, dtb/2.0, 1.0, False)
        res3 = _step_rk_logic(res2[0], h0p, res2[1], res2[2], ubp, vbp, consts, dtb, 1.0, True)
        
        h0_3, ub_3, vb_3, cel_x, cel_y, ub_ct, vb_ct, ub_cx, ub_cy, vb_cx, vb_cy, advx, advy = res3
        
        # Postprocess 逻辑
        new_isb = isb + 1
        return (h0_3, h0_3, ub_3, vb_3, ub_3, vb_3, new_isb, cel_x, cel_y, ub_ct, vb_ct, ub_cx, ub_cy, vb_cx, vb_cy, advx, advy)

    return jax.lax.fori_loop(0, nbb, body_fun, state)


# =====================================================================
# 4. 类方法绑定
# =====================================================================

def add_barotropic_methods(cls):
    
    def _pack_state_and_consts(self):
        """将类变量打包成 JAX 可以追踪的 tuple 结构"""
        # 注意：isb 必须包装为 jnp.int32 以适应 lax.fori_loop 的强类型要求
        state = (
            self.h0, self.h0p, self.ub, self.vb, self.ubp, self.vbp, self.isb,
            self.celerity_x, self.celerity_y, self.ub_ct, self.vb_ct, self.ub_cx, 
            self.ub_cy, self.vb_cx, self.vb_cy, self.advx, self.advy
        )
        consts = (
            Dg.dzph_x, Dg.dzph_y, self.pax, self.pxb, self.whx, 
            self.pay, self.pyb, self.why, self.wgp, Dg.rdx, Dg.rdy, Dg.a_f
        )
        return state, consts

    def _unpack_state(self, state):
        """将 JIT 算完的结果解包覆写回 self"""
        (
            self.h0, self.h0p, self.ub, self.vb, self.ubp, self.vbp, self.isb,
            self.celerity_x, self.celerity_y, self.ub_ct, self.vb_ct, self.ub_cx, 
            self.ub_cy, self.vb_cx, self.vb_cy, self.advx, self.advy
        ) = state

    def barotr_rk2(self) -> None:
        state, consts = self._pack_state_and_consts()
        # 一发入魂：调用全量 JIT 循环
        final_state = _barotr_rk2_jit_loop(state, consts, self.nbb, self.dtb)
        self._unpack_state(final_state)

    def barotr_rk3(self) -> None:
        state, consts = self._pack_state_and_consts()
        final_state = _barotr_rk3_jit_loop(state, consts, self.nbb, self.dtb)
        self._unpack_state(final_state)

    cls.barotr_rk2 = barotr_rk2
    cls.barotr_rk3 = barotr_rk3
    cls._pack_state_and_consts = _pack_state_and_consts
    cls._unpack_state = _unpack_state
    
    return cls