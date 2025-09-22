"""
File: barotropic.py
Description: High-performance barotropic time stepping methods for Momentum class.
    Ported from Fortran barotr_mod.F90 with JAX JIT compilation for maximum performance.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-22 (Ported from Fortran)

REVISION HISTORY:
    30/12/2024 - Initial Fortran Version
    17/03/2025 - Fortran revisions for barotropic advection
    22/09/2025 - Python port with high-performance JAX implementation
"""

# Third-party imports
import jax
import jax.numpy as jnp
import functools
from typing import Tuple

# Local application imports
from backend.calculation.field import Field
from backend.cube_grid.use_mpp import FMS_chtholly
from duogrid.duogrid import Duogrid as Dg
from operators.agrid import agrid_div, agrid_grad, agrid_vorticity
from operators.remap import vector_trans_2d
from operators.poly import vector_interpolation


# JIT-compiled pure functions (module level for better performance)
@jax.jit
def _flux_calculation_jit(h0: jax.Array, ub_cx: jax.Array, vb_cy: jax.Array, 
                         dzph_x: jax.Array, dzph_y: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Calculate flux of <hu> & <hv> using upwind algorithm"""
    # Get interpolated h values
    he, hw, hn, hs = vector_interpolation(h0)
    
    # Upwind scheme for flux calculation
    # X-direction flux
    h_upwind_x = jnp.zeros_like(ub_cx).at[1:, :].set(
        jnp.where(
            ub_cx[1:, :] > 0.0, he[:-1, :], hw[1:, :]
        )
    )
    flux_hu = jnp.zeros_like(ub_cx).at[:, :].set(
        (h_upwind_x + dzph_x) * ub_cx
    )
    
    # Y-direction flux  
    h_upwind_y = jnp.zeros_like(vb_cy).at[:, 1:].set(
        jnp.where(vb_cy[:, 1:] > 0.0, hn[:, :-1], hs[:, 1:])
    )
    flux_hv = jnp.zeros_like(vb_cy).at[:, :].set(
        (h_upwind_y + dzph_y) * vb_cy
    )
    
    return flux_hu, flux_hv


@jax.jit
def _calculate_pgf_jit(h0: jax.Array, pax: jax.Array, pxb: jax.Array, whx: jax.Array,
                      pay: jax.Array, pyb: jax.Array, why: jax.Array, wgp: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Calculate pressure gradient force"""
    # Calculate SSH gradient
    gradx, grady = agrid_grad(h0)
    
    # PGF calculation
    grav = 9.8
    pgf_u = (wgp - 1.0) * grav * gradx + pax + pxb - h0 * whx
    pgf_v = (wgp - 1.0) * grav * grady + pay + pyb - h0 * why
    
    return pgf_u, pgf_v


@jax.jit
def _calculate_rhs_jit(pgf_u: jax.Array, pgf_v: jax.Array, advx: jax.Array, advy: jax.Array,
                      a_f: float, vb_ct: jax.Array, ub_ct: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Calculate right-hand side of momentum equations"""
    rhs_u = pgf_u + advx + a_f * vb_ct
    rhs_v = pgf_v + advy - a_f * ub_ct
    
    return rhs_u, rhs_v


@jax.jit
def _calculate_advection_jit(vb_cx: jax.Array, ub_cy: jax.Array, ub_cx: jax.Array, 
                            vb_cy: jax.Array, vb_ct: jax.Array, ub_ct: jax.Array,
                            rdx: jax.Array, rdy: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Calculate 2D advection terms"""
    # Calculate vorticity
    vort = agrid_vorticity(vb_cx, ub_cy)
    
    # Calculate kinetic energy terms
    kinetic_u = 0.5 * (ub_cx**2 + vb_cx**2)
    kinetic_v = 0.5 * (vb_cy**2 + ub_cy**2)
    
    # Calculate advection
    advx = jnp.zeros_like(vb_cx).at[:-1, :].set(
        vort[:-1, :] * vb_ct[:-1, :] - (
            kinetic_u[1:, :] - kinetic_u[:-1, :]
        ) * rdx[:-1, :]
    )
    advy = jnp.zeros_like(vb_cy).at[:, :-1].set(
        -vort[:, :-1] * ub_ct[:, :-1] - (
            kinetic_v[:, 1:] - kinetic_v[:, :-1]
        ) * rdy[:, :-1]
    )
    
    return advx, advy


@jax.jit
def _update_ssh_jit(h0p: jax.Array, div_out: jax.Array, dt: float) -> jax.Array:
    """Update SSH field"""
    return h0p - div_out * dt


@jax.jit
def _update_velocities_jit(ubp: jax.Array, vbp: jax.Array,
                          rhs_u: jax.Array, rhs_v: jax.Array, 
                          dt: float) -> Tuple[jax.Array, jax.Array]:
    """Update velocity fields"""
    return ubp + rhs_u * dt, vbp + rhs_v * dt


@jax.jit
def _fb_scheme_jit(h0: jax.Array, h0_tem: jax.Array, beta_d: float) -> jax.Array:
    """Forward-Backward scheme"""
    return (1.0 - beta_d) * h0 + beta_d * h0_tem


def add_barotropic_methods(cls):
    """
    Decorator to add high-performance barotropic methods to Momentum class.
    All methods are JIT-compiled for maximum performance.
    """
    
    # Core timestepping methods
    def barotr_rk2(self) -> None:
        """Barotropic Time Stepping Using 2nd-order Runge-Kutta"""
        beta_d = 0.0
        
        for nc in range(1, self.nbb + 1):
            is_nc = (nc == self.nbb)
            
            # RK2 steps
            self._step_rk(self.dtb/2.0, beta_d, False, is_nc)
            self._step_rk(self.dtb, beta_d, True, is_nc)
            
            # Post-process
            self._postprocess()

            if Dg.mp.pe == 6:
                print(jnp.average(self.ub), jnp.average(self.vb), jnp.average(self.h0))
                print(jnp.max(self.ub), jnp.max(self.vb), jnp.max(self.h0))
    
    def barotr_rk3(self) -> None:
        """Barotropic Time Stepping Using 3rd-order Runge-Kutta"""
        beta_d = 1.0
        
        for nc in range(1, self.nbb + 1):
            is_nc = (nc == self.nbb)
            
            # RK3 steps
            self._step_rk(self.dtb/3.0, beta_d, False, is_nc)
            self._step_rk(self.dtb/2.0, beta_d, False, is_nc)
            self._step_rk(self.dtb, beta_d, True, is_nc)
            
            # Post-process
            self._postprocess()
    
    def _step_rk(self, dt: float, beta_d: float, is_laststep: bool, isnc: bool) -> None:
        """Kernel of Barotropic Time Stepping in RK"""
        # Store h0 for FB scheme
        h0_tem = self.h0
        
        # LMARS celerity
        self._lmars_get_celerity()
        
        # Vector transformation and communication
        self._vector_transform_and_comm()
        
        # Add LMARS velocity viscosity
        self._lmars_add_vel_vis()
        
        # Predict SSH
        self._predict_ssh(dt, is_laststep)
        
        # Extend scalar boundary
        self.h0 = FMS_chtholly.ext_scalar(self.h0)
        
        # Forward-Backward scheme
        h0_tem = _fb_scheme_jit(self.h0, h0_tem, beta_d)
        
        # Predict velocities
        self._predict_uv(dt, h0_tem)
        
        # Extend vector boundary
        self.ub, self.vb = FMS_chtholly.ext_vector(self.ub, self.vb)
        
        # Update advection (last step only)
        if is_laststep and isnc:
            self.advx, self.advy = self._calculate_advection()

    def _predict_ssh(self, dt: float, is_laststep: bool) -> None:
        """SSH prediction"""
        # Calculate flux
        flux_hu, flux_hv = self._flux_calculation()
        
        # Communication (non-JIT part handled separately)
        if is_laststep:
            flux_hu, flux_hv = FMS_chtholly.communication2d(flux_hu, flux_hv)
        
        # Calculate divergence and update SSH
        div_out = agrid_div(flux_hu, flux_hv)
        self.h0 = _update_ssh_jit(self.h0p, div_out, dt)
    
    def _predict_uv(self, dt: float, h_old: jax.Array) -> None:
        """Velocity prediction"""
        # Calculate pressure gradient force
        pgf_u, pgf_v = self._calculate_pgf(h_old)
        
        # Add LMARS PGF viscosity
        pgf_u, pgf_v = self._lmars_add_pgf_vis(pgf_u, pgf_v)
        
        # Calculate advection
        self.advx, self.advy = self._calculate_advection()
        
        # Calculate RHS and update velocities
        rhs_u, rhs_v = self._calculate_rhs(pgf_u, pgf_v)
        self.ub, self.vb = _update_velocities_jit(
            self.ubp, self.vbp, rhs_u, rhs_v, dt
        )
    
    def _flux_calculation(self) -> Tuple[jax.Array, jax.Array]:
        """Calculate flux using JIT-compiled function"""
        return _flux_calculation_jit(
            self.h0, self.ub_cx, self.vb_cy, 
            Dg.dzph_x, Dg.dzph_y
        )
    
    def _calculate_pgf(self, h0: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Calculate pressure gradient force using JIT-compiled function"""
        return _calculate_pgf_jit(
            h0, self.pax, self.pxb, self.whx,
            self.pay, self.pyb, self.why, self.wgp
        )
    
    def _calculate_rhs(self, pgf_u: jax.Array, pgf_v: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Calculate RHS using JIT-compiled function"""
        return _calculate_rhs_jit(
            pgf_u, pgf_v, self.advx, self.advy,
            Dg.a_f, self.vb_ct, self.ub_ct
        )
    
    def _calculate_advection(self) -> Tuple[jax.Array, jax.Array]:
        """Calculate advection using JIT-compiled function"""
        return _calculate_advection_jit(
            self.vb_cx, self.ub_cy, self.ub_cx, self.vb_cy,
            self.vb_ct, self.ub_ct, Dg.rdx, Dg.rdy
        )
    
    def _vector_transform_and_comm(self) -> None:
        """Vector transformation"""
        (self.ub_ct, self.vb_ct, self.ub_cx, self.ub_cy,
         self.vb_cx, self.vb_cy) = vector_trans_2d(self.ub, self.vb)
        
    def _postprocess(self) -> None:
        """Post-processing after completing one barotropic step"""
        # Update time counter
        self.isb += 1
        
        # Update previous step values
        self.ubp = self.ub
        self.vbp = self.vb  
        self.h0p = self.h0
        
        # Accumulate for time averaging (commented out as in original)
        # self.h0f = self.h0f + self.h0
        # self.h0bf = self.h0bf + self.h0
    
    # LMARS methods (non-JIT wrapper methods)
    def _lmars_get_celerity(self) -> None:
        """Get LMARS celerity"""
        self.lmars.get_celerity(self.h0)
    
    def _lmars_add_vel_vis(self) -> None:
        """Add LMARS velocity viscosity"""
        self.lmars.get_vel_vis_2d(self.h0)
        self.lmars.add_vel_vis(self.ub_cx, self.vb_cy)
    
    def _lmars_add_pgf_vis(self, pgf_u: jax.Array, pgf_v: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Add LMARS PGF viscosity"""
        self.lmars.get_pgf_vis_2d(self.ub, self.vb)
        self.lmars.add_pgf_vis(pgf_u, pgf_v)
        return pgf_u, pgf_v
    
    # Add all methods to the class
    cls.barotr_rk2 = barotr_rk2
    cls.barotr_rk3 = barotr_rk3
    cls._step_rk = _step_rk
    cls._predict_ssh = _predict_ssh
    cls._predict_uv = _predict_uv
    cls._flux_calculation = _flux_calculation
    cls._calculate_pgf = _calculate_pgf
    cls._calculate_rhs = _calculate_rhs
    cls._calculate_advection = _calculate_advection
    cls._vector_transform_and_comm = _vector_transform_and_comm
    cls._postprocess = _postprocess
    cls._lmars_get_celerity = _lmars_get_celerity
    cls._lmars_add_vel_vis = _lmars_add_vel_vis
    cls._lmars_add_pgf_vis = _lmars_add_pgf_vis
    
    return cls