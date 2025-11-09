"""
File: barotropic.py
Description: High-performance barotropic time stepping methods for Momentum class.
    Ported from Fortran barotr_mod.F90 with JAX JIT compilation for maximum performance.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-22
Updated: 2025-09-24

REVISION HISTORY:
    22/09/2025 - Python port with high-performance JAX implementation
    23/09/2025 - split into barotr_jit.py and barotr.py
    25/09/2025 - Chtholly debug and add LMARS
"""

# Third-party imports
import jax
import jax.numpy as jnp

# Standard library imports
from typing import Tuple

# Local application imports
from mymodule.timer import timed, Timer
from backend.cube_grid.use_mpp import FMS_chtholly
from duogrid.duogrid import Duogrid as Dg
from operators.agrid import agrid_div
from operators.remap import vector_trans_2d
from ._barotr_jit import (
    _flux_calculation_jit,
    _calculate_pgf_jit,
    _calculate_rhs_jit,
    _calculate_advection_jit,
    _update_ssh_jit,
    _update_velocities_jit,
    _fb_scheme_jit,
    _horizontal_diffusion_jit,
)
from ._lmars_jit import (
    get_celerity_jit,
    get_vel_vis_2d_jit,
    get_pgf_vis_2d_jit,
)


def add_barotropic_methods(cls):
    """
    Decorator to add high-performance barotropic methods to Momentum class.
    All methods are JIT-compiled for maximum performance.
    """
    
    # Core timestepping methods
    #@timed('total')
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
            # if Dg.mp.pe == 0:
            #     print("h0*",nc,jnp.max(self.h0[3:-3,3:-3]), jnp.min(self.h0[3:-3,3:-3]), jnp.sum(self.h0[3:-3,3:-3]))        

    
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
        _h0_tem = self.h0
        
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
        _h0_tem = _fb_scheme_jit(self.h0, _h0_tem, beta_d)
        
        # Predict velocities
        self._predict_uv(dt, _h0_tem)
        
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
        
        # Calculate divergence
        div_out = agrid_div(flux_hu, flux_hv)

        # Update SSH
        self.h0 = _update_ssh_jit(self.h0p, div_out, dt)

    def _predict_uv(self, dt: float, h_old: jax.Array) -> None:
        """Velocity prediction"""
        # Calculate pressure gradient force
        pgf_u, pgf_v = self._calculate_pgf(h_old)
        
        # Add LMARS PGF viscosity
        pgf_u, pgf_v = self._lmars_add_pgf_vis(pgf_u, pgf_v)
        
        # Calculate advection
        self.advx, self.advy = self._calculate_advection()
        
        # Calculate RHS 
        rhs_u, rhs_v = self._calculate_rhs(pgf_u, pgf_v)

        # Update velocities
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
        
        # Accumulate for time averaging
        # self.h0f = self.h0f + self.h0
        # self.h0bf = self.h0bf + self.h0
    
    # LMARS methods (non-JIT wrapper methods)
    def _lmars_get_celerity(self) -> None:
        """Get LMARS celerity"""
        self.celerity_x, self.celerity_y = get_celerity_jit(self.h0, Dg.dzph_x, Dg.dzph_y)
    
    def _lmars_add_vel_vis(self) -> None:
        """Add LMARS velocity viscosity"""
        self.ub_cx, self.vb_cy = get_vel_vis_2d_jit(
            self.celerity_x, self.celerity_y, self.h0, 
            self.ub_cx, self.vb_cy
        )

    def _lmars_add_pgf_vis(self, pgf_u: jax.Array, pgf_v: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Add LMARS PGF viscosity"""

        return get_pgf_vis_2d_jit(
            self.celerity_x, self.celerity_y, Dg.rdx, Dg.rdy, 
            self.ub_ct, self.vb_ct, pgf_u, pgf_v
        )

    # explicit viscosity now not used
    # def _add_hmix(self, pgf_u: jax.Array, pgf_v: jax.Array) -> Tuple[jax.Array, jax.Array]:
    #     hduk, hdvk = _horizontal_diffusion_jit(self.ub, self.vb,
    #         Dg.c_dx[:-1,:], Dg.d_dy[:,:-1],Dg.rdx, Dg.rdy,0
    #         )
    #     return pgf_u + hduk, pgf_v + hdvk
    
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
    #cls._add_hmix = _add_hmix
    return cls