"""
File: barotr.py
Description: High-performance barotropic time stepping methods for Momentum class.
    Consolidated with JIT compatibility for sharded JAX execution.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-22
Updated: 2026-04-07

REVISION HISTORY:
    22/09/2025 - Initial implementation of barotropic solver
    15/03/2026 - Unified JIT and sharding compatibility
    19/03/2026 - Refactor imports to package-level paths
    04/04/2026 - Fix JAX multidimensional UnspecifiedValue sharding issue and strip isb from jit state
    07/04/2026 - Optimized JIT static bindings and moved logic details to markdown
"""

import jax
import jax.numpy as jnp
from jax import jit

# Local application imports
from licom.mesh import Communication, GPU_Mesh, Cube, make_spmd_jit, auto_pack, auto_unpack
from licom.duogrid import Dg
from licom.operators import AGrid, Poly, Remap
from .barotr_pure import _step_rk_logic

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
        return (h0_2, h0_2, ub_2, vb_2, ub_2, vb_2, *res2[3:])

    return jax.lax.fori_loop(0, nbb, body_fun, state)

def _barotr_rk3_core(state, consts, nbb, dtb):
    def body_fun(i, val):
        h0, h0p, ub, vb, ubp, vbp, *others = val
        res1 = _step_rk_logic(h0, h0p, ub, vb, ubp, vbp, consts, dtb/3.0, 1.0, False)
        res2 = _step_rk_logic(res1[0], h0p, res1[1], res1[2], ubp, vbp, consts, dtb/2.0, 1.0, False)
        res3 = _step_rk_logic(res2[0], h0p, res2[1], res2[2], ubp, vbp, consts, dtb, 1.0, True)
        return (res3[0], res3[0], res3[1], res3[2], res3[1], res3[2], *res3[3:])

    return jax.lax.fori_loop(0, nbb, body_fun, state)

_barotr_rk2_jit = None
_barotr_rk3_jit = None

# =====================================================================
# 4. Field Declarations (defined once, auto pack/unpack)
# =====================================================================

# State fields: mutable, both input and output of the JIT graph
_STATE_KEYS = (
    'h0', 'h0p', 'ub', 'vb', 'ubp', 'vbp',
    'celerity_x', 'celerity_y', 'ub_ct', 'vb_ct', 'ub_cx',
    'ub_cy', 'vb_cx', 'vb_cy', 'advx', 'advy',
)

# Const fields: read-only inputs. (source_object, attribute_name)
_CONST_SOURCES = (
    (Dg,   'dzph_x'), (Dg,   'dzph_y'),
    (None, 'pax'),    (None, 'pxb'),    (None, 'whx'),
    (None, 'pay'),    (None, 'pyb'),    (None, 'why'),
    (None, 'wgp'),
    (Dg,   'rdx'),    (Dg,   'rdy'),    (Dg,   'a_f'),
)

# =====================================================================
# 5. Class Methods
# =====================================================================

def add_barotropic_methods(cls):
    def setup_barotropic_jit(self, rk_type):
        """
        初始化时调用一次：按需打包状态，并一键编译 SPMD 算子图，缓存至实例。
        """
        self._barotr_state, self._barotr_consts = auto_pack(self, _STATE_KEYS, _CONST_SOURCES)
        
        # 选择不同阶数的物理核心
        core_fn = _barotr_rk2_core if rk_type == 2 else _barotr_rk3_core
        
        with GPU_Mesh.mesh:
            self._barotr_jit_fn = make_spmd_jit(core_fn, self._barotr_state, self._barotr_consts, static_argnums=(2, 3))

    def barotr(self):
        """
        统一的时间步进接口，根据编译期挂载的不同维度的 JIT 图执行。
        """
        with GPU_Mesh.mesh:
            self._barotr_state = self._barotr_jit_fn(self._barotr_state, self._barotr_consts, self.nbb, self.dtb)
            auto_unpack(self, _STATE_KEYS, self._barotr_state)

    cls.setup_barotropic_jit = setup_barotropic_jit
    cls.barotr = barotr
    return cls