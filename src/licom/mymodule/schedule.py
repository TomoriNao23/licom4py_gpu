"""
File: schedule.py
Description: Global scheduling system for LICOM model execution.
    Owns the SPMD JIT compilation, state/const field declarations,
    and executes the simulation loop via JAX fori_loop.

    Architecture:
    ┌─────────────── Python (host) ────────────────────────────────┐
    │  for chunk in range(n_chunks):          # 仅诊断边界才回到 host │
    │    ┌── single JIT dispatch ───────────────────────────────┐  │
    │    │  fori_loop(0, diag_interval):     # 子 JAX 循环       │  │
    │    │    ├─ barotropic(nbb sub-steps)   # 正压              │  │
    │    │    ├─ baroclinic()                # 斜压 (TODO)       │  │
    │    │    └─ tracer()                    # 示踪 (TODO)       │  │
    │    └──────────────────────────────────────────────────────┘  │
    │    auto_unpack + print_global_diag     # 诊断输出           │
    └──────────────────────────────────────────────────────────────┘

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2026-05-14

REVISION HISTORY:
    03/09/2025 - Initial implementation of Schedule class
    07/01/2026 - Refactored execution strategy for performance optimization
    19/03/2026 - Refactor imports to package-level paths
    13/04/2026 - Lifted SPMD compilation to Schedule; Python loop → JAX fori_loop
    14/05/2026 - Using async to print global diagnostics and future IO
"""

from __future__ import annotations

# Standard library imports
from typing import TYPE_CHECKING

# Third-party imports
import jax

# Local application imports
from licom.duogrid import Dg
from licom.kernel import GPU_Mesh, auto_pack, auto_unpack, make_spmd_jit
from licom.readnamelist import Namelist, Timer

if TYPE_CHECKING:
    # Local application imports
    from licom.momentum import Momentum


# =====================================================================
# Global State & Const Field Declarations
# =====================================================================
#
# STATE_KEYS:   mutable fields, both input and output of the JIT graph.
# CONST_SOURCES: read-only fields. (source_object, attribute_name)
#                source_object=None → read from the momentum instance.
#
# When adding baroclinic / tracer modules, extend these tuples:
#   STATE_KEYS += ("ua", "va", "uap", "vap", ...)
#   CONST_SOURCES += ((Dg, "new_grid_field"), ...)
# =====================================================================

STATE_KEYS = (
    # --- Barotropic core state (0 ~ 15) ---
    "h0",
    "h0p",
    "ub",
    "vb",
    "ubp",
    "vbp",
    "celerity_x",
    "celerity_y",
    "ub_ct",
    "vb_ct",
    "ub_cx",
    "ub_cy",
    "vb_cx",
    "vb_cy",
    "advx",
    "advy",
    # --- Cross-module coupling fields (barotropic reads / baroclinic writes) ---
    "pax",
    "pxb",
    "whx",
    "pay",
    "pyb",
    "why",
    "wgp",
    # --- Baroclinic state (TODO) ---
    # "ua", "va", "uap", "vap",
    # --- Tracer state (TODO) ---
    # "temp", "salt", ...
)

# Consts: truly immutable grid geometry from Dg (never modified by any module)
CONST_SOURCES = (
    # --- Grid geometry ---
    (Dg, "dzph_x"),
    (Dg, "dzph_y"),
    (Dg, "rdx"),
    (Dg, "rdy"),
    (Dg, "a_f"),
    # --- Cube remapping / vector transform arrays ---
    (Dg, "k2e_coef"),
    (Dg, "loc_i_local"),
    (Dg, "loc_j_local"),
    (Dg, "a_c2l"),
    (Dg, "a_l2c"),
    (Dg, "inner"),
    (Dg, "outer"),
    # --- AGrid / Remap operator arrays ---
    (Dg, "a_gct"),
    (Dg, "a_sina"),
    (Dg, "rda"),
    (Dg, "c_dy"),
    (Dg, "d_dx"),
)

# Slice boundaries for bc_body field isolation
N_BAROTR_STATE = 23    # state[0:23]   → barotropic (16 core + 7 coupling)
N_BAROTR_CONSTS = 17   # consts[0:17]  → barotropic (5 geometry + 7 cube + 5 operators)
# N_BCLINIC_STATE = ?   # state[23:23+?] → baroclinic (TODO)
# N_BCLINIC_CONSTS = ?  # consts[17:17+?] → baroclinic (TODO)


# =====================================================================
# Schedule-Level Core Factory
# =====================================================================


def _make_schedule_core(rk_type_barotr):
    """
    Factory: builds and returns the pure computation kernel for the schedule.
    The returned function wraps `chunk_size` baroclinic steps in a single
    JAX fori_loop, each step containing barotropic + (future) baroclinic + tracer.

    This function is called ONCE at configure() time. The returned kernel is
    then compiled via make_spmd_jit into a single SPMD JIT function.
    """
    # Deferred import to avoid circular dependency at module load time
    from licom.momentum.barotr import _barotr_rk2_core, _barotr_rk3_core

    rk_core = _barotr_rk2_core if rk_type_barotr == 2 else _barotr_rk3_core

    # Bind mesh layout as Python constants (never changes during simulation)
    px = GPU_Mesh.mesh.shape["x"]
    py = GPU_Mesh.mesh.shape["y"]

    def schedule_core(state, consts, chunk_size, nbb, dtb):
        """
        Pure function: chunk_size baroclinic steps, each containing:
          1. Barotropic: nbb sub-steps via fori_loop (RK2/RK3)
          2. Baroclinic: (TODO)
          3. Tracer:     (TODO)

        Each core receives only its own slice of (state, consts).
        XLA optimizes untouched fields as identity — zero overhead.
        """
        # Slice boundaries (update when adding new modules)
        n_s_bt = N_BAROTR_STATE                         # 正压 state 字段数
        n_c_bt = N_BAROTR_CONSTS                        # 正压 consts 字段数
        # n_s_bc = N_BCLINIC_STATE                      # 斜压 state (TODO)
        # n_c_bc = N_BCLINIC_CONSTS                     # 斜压 consts (TODO)

        barotr_consts = consts[:n_c_bt]
        # bclinic_consts = consts[n_c_bt:n_c_bt+n_c_bc]  # (TODO)

        def bc_body(i, s):
            # ── Step 1: Barotropic ───────────────────────
            barotr_s = s[:n_s_bt]
            barotr_s = rk_core(barotr_s, barotr_consts, nbb, dtb, px, py)
            s = barotr_s + s[n_s_bt:]  # preserve extra fields

            # ── Step 2: Baroclinic (TODO) ────────────────
            # bclinic_s = s[n_s_bt:n_s_bt+n_s_bc]
            # bclinic_s = bclinic_core(bclinic_s, bclinic_consts)
            # s = s[:n_s_bt] + bclinic_s + s[n_s_bt+n_s_bc:]

            # ── Step 3: Tracer (TODO) ────────────────────
            # tracer_s = s[n_s_bt+n_s_bc:]
            # tracer_s = lax.cond(
            #     i % tracer_interval == 0,
            #     lambda t: tracer_core(t, tracer_consts),
            #     lambda t: t,
            #     tracer_s,
            # )
            # s = s[:n_s_bt+n_s_bc] + tracer_s

            return s

        return jax.lax.fori_loop(0, chunk_size, bc_body, state)

    return schedule_core


# =====================================================================
# Schedule Class
# =====================================================================


class Schedule:
    """
    Global simulation scheduler.

    Lifecycle:
        1. configure(namelist, momentum) — pack state, compile SPMD JIT (once)
        2. run(momentum)                — execute simulation in chunks
    """

    @classmethod
    def configure(cls, namelist: Namelist, momentum: Momentum, routines: list = None):
        """
        Pack state/consts from momentum, build the schedule core, and compile
        the global SPMD JIT function. Called ONCE during initialization.
        """
        cls.current_time: Timer = Timer(
            namelist._start_datetime, namelist.baroclinic_dt
        )
        cls.total_baroclinic_steps: int = namelist._total_baroclinic_steps
        cls.routines: list = (
            ["barotropic", "baroclinic", "tracer"] if routines is None else routines
        )

        # Diagnostics: diag_freq > 0 → print every diag_freq hours; 0/None → final only
        cls.diag_interval: int = (
            int(3600 / namelist.baroclinic_dt * namelist.diag_freq)
            if namelist.diag_freq is not None and namelist.diag_freq > 0
            else None
        )
        cls.async_threads: int = getattr(namelist, 'async_threads', 8)

        # Static scalars (compiled into JIT graph as constants)
        cls._nbb = momentum.nbb
        cls._dtb = momentum.dtb

        # Pack: momentum instance → (state_tuple, consts_tuple)
        cls._state, cls._consts = auto_pack(momentum, STATE_KEYS, CONST_SOURCES)

        # Build core and compile ONCE
        core_fn = _make_schedule_core(momentum.rk_barotr)
        with GPU_Mesh.mesh:
            cls._jit_fn = make_spmd_jit(
                core_fn, cls._state, cls._consts, static_argnums=(2, 3, 4)
            )

    @classmethod
    def _async_io_and_diag(cls, time_str, momentum, state_futures):
        """
        Background task:
        1. Block and gather sharded arrays from GPU to CPU memory.
        2. Unpack into momentum object.
        3. Execute diagnostics and future IO logic on CPU.
        """
        # Third-party imports
        import jax
        
        # Gather sharded array to CPU memory (blocks only this background thread)
        cpu_state = jax.device_get(state_futures)
        
        # Unpack NumPy arrays back to momentum object
        auto_unpack(momentum, STATE_KEYS, cpu_state)
        
        # Perform pure-CPU diagnostics (or future IO)
        momentum.print_global_diag(time_str)

    @classmethod
    def _dispatch_async_task(cls, executor, time_prefix, momentum):
        """Helper to format time and submit async IO/diag task to the background pool."""
        time_str = f"{time_prefix}: {cls.current_time.prev_dt.strftime('%Y-%m-%d-%H')}"
        executor.submit(cls._async_io_and_diag, time_str, momentum, cls._state)

    @classmethod
    def run(cls, momentum: Momentum) -> None:
        """
        Execute the full simulation.

        Data flow:
            cls._state ──→ _jit_fn ──→ cls._state ──→ _jit_fn ──→ ...
                 ↑            (fori_loop: diag_interval bc steps)      ↓
              pack (once)                                        async unpack/IO

        - diag_freq=0:  1 JIT dispatch for the entire simulation
        - diag_freq=24: 1 JIT dispatch per day (24 bc steps each)
        """
        # Standard library imports
        import concurrent.futures
        
        total = cls.total_baroclinic_steps
        chunk = cls.diag_interval if cls.diag_interval is not None else total

        # Start a single-worker thread pool for orderly async IO & diagnostics
        with concurrent.futures.ThreadPoolExecutor(max_workers=cls.async_threads) as executor:
            cls._dispatch_async_task(executor, "Initial time", momentum)

            for start in range(0, total, chunk):
                n = min(chunk, total - start)

                # ── Single JIT dispatch: n bc steps in fori_loop ──
                # This returns futures immediately
                with GPU_Mesh.mesh:
                    cls._state = cls._jit_fn(
                        cls._state, cls._consts, n, cls._nbb, cls._dtb
                    )

                # ── Host-side: advance time + dispatch async IO ──
                cls.current_time.advance(n)
                if cls.diag_interval is not None:
                    cls._dispatch_async_task(executor, "Time", momentum)

            # Final diagnostic
            if cls.diag_interval is None:
                cls._dispatch_async_task(executor, "Final time", momentum)
            
            # The context manager automatically waits for all async threads to complete before exiting
