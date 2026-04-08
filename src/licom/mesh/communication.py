"""
File: communication.py
Description: Defines communication topology, scheduling, and core halo exchange routines for meshes.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-03-14
Updated: 2026-03-14

REVISION HISTORY:
    14/03/2026 - Formatting and header updates
"""
# Third-party imports
import jax
import jax.numpy as jnp
from jax import lax
from jax import jit
from jax.sharding import PartitionSpec as P

# ==========================================
# 1. Topology Class
# ==========================================
class Topology:
    """
    Responsible for handling the topology connection rules of the grid, 
    with built-in direction and rotation constant configurations.
    """
    # Direction constants (East, West, North, South)
    E, W, N, S = 0, 1, 2, 3
    
    # Transformation rule constants
    ID = 0              # Identity (no transformation)
    FLIP_I_TRANS = 1    # Flip and transpose
    
    NTILE = 6           # Number of tiles (e.g., for a cubed-sphere grid)

    @classmethod
    def get_transform(cls, dir_a, dir_b):
        """
        Determines the data transformation rule when joining two adjacent faces.
        """
        table = {
            (cls.E, cls.W): cls.ID,
            (cls.W, cls.E): cls.ID,
            (cls.N, cls.S): cls.ID,
            (cls.S, cls.N): cls.ID,
            
            (cls.E, cls.S): cls.FLIP_I_TRANS,
            (cls.W, cls.N): cls.FLIP_I_TRANS,
            (cls.S, cls.E): cls.FLIP_I_TRANS,
            (cls.N, cls.W): cls.FLIP_I_TRANS,
        }
        return table[(dir_a, dir_b)]

    @classmethod
    def generate_edges(cls):
        """
        Generates topological edges: records the geometric rules of interconnected faces.
        Returns a list of tuples: (tile_A, dir_A, tile_B, dir_B, transform_id).
        """
        edges = []
        for tile in range(cls.NTILE):
            # Abstract adjacency rules for even and odd faces on a cubed sphere
            if tile % 2 == 0:  
                rules = {
                    cls.S: ((tile + 4) % 6, cls.E),
                    cls.E: ((tile + 2) % 6, cls.S),
                    cls.N: ((tile + 1) % 6, cls.S),
                    cls.W: ((tile + 5) % 6, cls.E),
                }
            else:  
                rules = {
                    cls.E: ((tile + 1) % 6, cls.W),
                    cls.N: ((tile + 2) % 6, cls.W),
                    cls.W: ((tile + 4) % 6, cls.N),
                    cls.S: ((tile + 5) % 6, cls.N),
                }

            # Generate NSEW rotation rules and topological connection edges
            for d_a, (nb, d_b) in rules.items():
                # Only add the edge if tile < nb to avoid duplicating bidirectional edges
                if tile < nb:
                    tid = cls.get_transform(d_a, d_b)
                    edges.append((tile, d_a, nb, d_b, tid))
        return edges


# ==========================================
# 3. Communication Class
# ==========================================
class Communication:
    """
    Communication core responsible for domain updates and boundary mean synchronization.
    """
    
    # Physical edge communication index configuration specific to C-grids
    IDX_W = 3
    IDX_E = -3
    IDX_S = 3
    IDX_N = -3

    halo = 0
    nx_local = 0
    ny_local = 0
    _n_pad = 0
    mesh = None

    _routing = None

    _update_domain_jit = None
    _boundary_communication_jit = None

    @classmethod
    def configure(cls, halo, nx_local, ny_local, mesh):
        cls.halo = halo
        cls.nx_local = nx_local
        cls.ny_local = ny_local
        cls.mesh = mesh
        cls._n_pad = max(nx_local, ny_local)

        edges = Topology.generate_edges()
        
        # Build static routing map for single-round communication
        routing_list = [[[0, 0, 0] for _ in range(4)] for _ in range(Topology.NTILE)]
        for tA, dA, tB, dB, tid in edges:
            routing_list[tA][dA] = [tB, dB, tid]
            routing_list[tB][dB] = [tA, dA, tid]
        cls._routing = jnp.array(routing_list, dtype=jnp.int32)

        # Vectorize over the 'tile' axis and use jit to preserve sharding.
        # NOTE: Do NOT specify in_shardings/out_shardings here.
        # These functions may be called as nested jits inside _barotr_rk2_jit
        # (which has no in_shardings). Mixing explicit sharding constraints in
        # inner jits with no-constraint outer jits causes XLA to produce
        # UnspecifiedValue sharding objects, which crash at runtime with:
        #   AttributeError: 'UnspecifiedValue' object has no attribute
        #   'addressable_devices_indices_map'
        # Sharding is instead propagated through the active Mesh context.
        pass

    # -------------------------------------------------
    # Shared Logic
    # -------------------------------------------------
    @classmethod
    def _apply_transform(cls, data, recv_dir, tid, nx_eval, ny_eval):
        """
        Unified data rotation/flip application layer.
        """
        is_ew = recv_dir in (Topology.E, Topology.W)
        n = nx_eval if is_ew else ny_eval
        raw = data[..., :, :n]
        
        # Make tid broadcastably compatible for batch arrays
        if hasattr(tid, 'ndim') and tid.ndim > 0:
            tid_mask = jnp.expand_dims(tid, axis=tuple(range(tid.ndim, raw.ndim)))
        else:
            tid_mask = tid

        if is_ew:
            # Expected shape: (..., ny_local, halo) or (ny_local, halo)
            swapped = jnp.swapaxes(raw, -1, -2)
            return lax.cond(
                tid == Topology.FLIP_I_TRANS,
                lambda x: jnp.flip(x, axis=-2),
                lambda x: x,
                swapped
            ) if not hasattr(tid, 'ndim') or tid.ndim == 0 else jnp.where(tid_mask == Topology.FLIP_I_TRANS, jnp.flip(swapped, axis=-2), swapped)
        else:
            # Expected shape: (..., halo, nx_local) or (halo, nx_local)
            return lax.cond(
                tid == Topology.FLIP_I_TRANS,
                lambda x: jnp.flip(x, axis=-1),
                lambda x: x,
                raw
            ) if not hasattr(tid, 'ndim') or tid.ndim == 0 else jnp.where(tid_mask == Topology.FLIP_I_TRANS, jnp.flip(raw, axis=-1), raw)

    # -------------------------------------------------
    # Domain Update (Halo Exchange)
    # -------------------------------------------------
    @classmethod
    def _pack_edge(cls, u, d):
        """Extracts the boundary data to be sent based on the direction."""
        h = cls.halo
        nx = u.shape[-2] - 2 * h
        ny = u.shape[-1] - 2 * h
        n = max(nx, ny)

        if d == Topology.E:   
            raw = jnp.swapaxes(u[..., h:-h, -2*h:-h], -1, -2)
        elif d == Topology.W: 
            raw = jnp.swapaxes(u[..., h:-h, h:2*h], -1, -2)
        elif d == Topology.N: 
            raw = u[..., -2*h:-h, h:-h]
        else:                 
            raw = u[..., h:2*h, h:-h]

        cur = raw.shape[-1]
        if cur < n:
            pad_width = [(0, 0)] * raw.ndim
            pad_width[-1] = (0, n - cur)
            raw = jnp.pad(raw, pad_width)
        return raw

    @classmethod
    def _gather_halo(cls, send):
        """Universally pulls 4 boundary halos natively bounded by mesh index."""
        try:
            recv_x = lax.all_gather(send, "x", tiled=False)
            recv_xy = lax.all_gather(recv_x, "y", tiled=False)
            recv_global = lax.all_gather(recv_xy, "tile", tiled=False)
            
            py = recv_global.shape[1]
            px = recv_global.shape[2]
            
            if send.ndim > 3:
                # recv_global is (mesh_tile, py, px, 4, vtile, halo, n)
                # Properly pull vtile (axis 4) out next to mesh_tile (axis 0)
                recv = jnp.transpose(recv_global, (0, 4, 1, 2, 3, 5, 6)) # (mesh_tile, vtile, py, px, 4, halo, n)
                
                n_pad = send.shape[-1]
                recv = recv.reshape((Topology.NTILE, py, px, 4, cls.halo, n_pad))
                
                mesh_t = lax.axis_index("tile")
                vtile = send.shape[1]
                t = mesh_t * vtile + jnp.arange(vtile)
            else:
                recv = recv_global
                t = lax.axis_index("tile")
                
            x = lax.axis_index("x")
            y = lax.axis_index("y")
        except NameError:
            py, px = 1, 1
            if send.ndim > 3:
                recv = jnp.swapaxes(send, 0, 1) # (vtile, 4, halo, n)
                recv = jnp.expand_dims(recv, axis=(1, 2)) # (vtile, 1, 1, 4, halo, n)
                t = jnp.arange(recv.shape[0])
            else:
                recv = jnp.expand_dims(send, axis=(0, 1, 2))
                t = 0
            x, y = 0, 0
            
        return recv, t, x, y, px, py

    @classmethod
    def _update_domain_single(cls, u):
        """Applies single-round boundary exchange using all_gather."""
        # Pack all 4 edges into an array of shape (4, halo, n)
        send = jnp.stack([
            cls._pack_edge(u, Topology.E),
            cls._pack_edge(u, Topology.W),
            cls._pack_edge(u, Topology.N),
            cls._pack_edge(u, Topology.S),
        ])
        
        recv, t, x, y, px, py = cls._gather_halo(send)

        def process_dir(d):
            nb = cls._routing[t, d, 0]
            dB = cls._routing[t, d, 1]
            tid_ext = cls._routing[t, d, 2]
            
            is_EW = jnp.logical_or(d == Topology.E, d == Topology.W)
            my_idx = jnp.where(is_EW, y, x)
            
            idx_b_flip = jnp.where(dB <= 1, (py - 1) - my_idx, (px - 1) - my_idx)
            idx_b = jnp.where(tid_ext == Topology.FLIP_I_TRANS, idx_b_flip, my_idx)
            
            x_b_ext = jnp.where(dB == Topology.W, 0, jnp.where(dB == Topology.E, px - 1, idx_b))
            y_b_ext = jnp.where(dB == Topology.S, 0, jnp.where(dB == Topology.N, py - 1, idx_b))
            
            if d == Topology.E:
                is_internal = x < px - 1
                int_x_b, int_y_b, int_d_b = x + 1, y, Topology.W
            elif d == Topology.W:
                is_internal = x > 0
                int_x_b, int_y_b, int_d_b = x - 1, y, Topology.E
            elif d == Topology.N:
                is_internal = y < py - 1
                int_x_b, int_y_b, int_d_b = x, y + 1, Topology.S
            else:
                is_internal = y > 0
                int_x_b, int_y_b, int_d_b = x, y - 1, Topology.N
                
            t_b = jnp.where(is_internal, t, nb)
            x_b = jnp.where(is_internal, int_x_b, x_b_ext)
            y_b = jnp.where(is_internal, int_y_b, y_b_ext)
            d_b = jnp.where(is_internal, int_d_b, dB)
            tid = jnp.where(is_internal, Topology.ID, tid_ext)
            
            nx_eval = u.shape[-2] - 2 * cls.halo
            ny_eval = u.shape[-1] - 2 * cls.halo
            return cls._apply_transform(recv[t_b, y_b, x_b, d_b], d, tid, nx_eval, ny_eval)

        h = cls.halo
        u = u.at[..., h:-h, -h:].set(process_dir(Topology.E))
        u = u.at[..., h:-h, :h].set(process_dir(Topology.W))
        u = u.at[..., -h:, h:-h].set(process_dir(Topology.N))
        u = u.at[..., :h, h:-h].set(process_dir(Topology.S))
        
        return u

    @classmethod
    def update_domain(cls, u):
        """Public API to trigger domain update within the defined Mesh."""
        with cls.mesh:
            return cls._update_domain_single(u)

    # -------------------------------------------------
    # C-Grid Boundary Edge Synchronization
    # -------------------------------------------------
    @classmethod
    def _pack_boundary_edge(cls, u, v, d):
        """Extracts boundary lines for U/V vector fields based on direction."""
        h = cls.halo
        nx = u.shape[-2] - 2 * h
        ny = u.shape[-1] - 2 * h
        n = max(nx, ny)

        if d == Topology.E:   
            line = v[..., h:-h, cls.IDX_E]
        elif d == Topology.W: 
            line = v[..., h:-h, cls.IDX_W]
        elif d == Topology.N: 
            line = u[..., cls.IDX_N, h:-h]
        else:                 
            line = u[..., cls.IDX_S, h:-h]

        line = jnp.expand_dims(line, -2)
        pad_h = [(0, 0)] * line.ndim
        pad_h[-2] = (0, h - 1)
        raw = jnp.pad(line, pad_h)

        cur = raw.shape[-1]
        if cur < n:
            pad_width = [(0, 0)] * raw.ndim
            pad_width[-1] = (0, n - cur)
            raw = jnp.pad(raw, pad_width)
        return raw

    @classmethod
    def _boundary_communication_single(cls, u, v):
        """Applies single-round boundary synchronization for vector fields."""
        send = jnp.stack([
            cls._pack_boundary_edge(u, v, Topology.E),
            cls._pack_boundary_edge(u, v, Topology.W),
            cls._pack_boundary_edge(u, v, Topology.N),
            cls._pack_boundary_edge(u, v, Topology.S),
        ])
        
        recv, t, x, y, px, py = cls._gather_halo(send)
            
        def process_dir(d):
            nb = cls._routing[t, d, 0]
            dB = cls._routing[t, d, 1]
            tid_ext = cls._routing[t, d, 2]
            
            is_EW = jnp.logical_or(d == Topology.E, d == Topology.W)
            my_idx = jnp.where(is_EW, y, x)
            
            idx_b_flip = jnp.where(dB <= 1, (py - 1) - my_idx, (px - 1) - my_idx)
            idx_b = jnp.where(tid_ext == Topology.FLIP_I_TRANS, idx_b_flip, my_idx)
            
            x_b_ext = jnp.where(dB == Topology.W, 0, jnp.where(dB == Topology.E, px - 1, idx_b))
            y_b_ext = jnp.where(dB == Topology.S, 0, jnp.where(dB == Topology.N, py - 1, idx_b))
            
            if d == Topology.E:
                is_internal = x < px - 1
                int_x_b, int_y_b, int_d_b = x + 1, y, Topology.W
            elif d == Topology.W:
                is_internal = x > 0
                int_x_b, int_y_b, int_d_b = x - 1, y, Topology.E
            elif d == Topology.N:
                is_internal = y < py - 1
                int_x_b, int_y_b, int_d_b = x, y + 1, Topology.S
            else:
                is_internal = y > 0
                int_x_b, int_y_b, int_d_b = x, y - 1, Topology.N
                
            t_b = jnp.where(is_internal, t, nb)
            x_b = jnp.where(is_internal, int_x_b, x_b_ext)
            y_b = jnp.where(is_internal, int_y_b, y_b_ext)
            d_b = jnp.where(is_internal, int_d_b, dB)
            tid = jnp.where(is_internal, Topology.ID, tid_ext)
            
            nx_eval = u.shape[-2] - 2 * cls.halo
            ny_eval = u.shape[-1] - 2 * cls.halo
            return cls._apply_transform(recv[t_b, y_b, x_b, d_b], d, tid, nx_eval, ny_eval)

        h = cls.halo
        
        arr_E = process_dir(Topology.E)
        val_E = ((v[..., h:-h, cls.IDX_E] + arr_E[..., :, 0]) / 2.0).astype(v.dtype)
        v = lax.cond(x == px - 1, lambda val: val.at[..., h:-h, cls.IDX_E].set(val_E), lambda val: val, v)
        
        arr_W = process_dir(Topology.W)
        val_W = ((v[..., h:-h, cls.IDX_W] + arr_W[..., :, 0]) / 2.0).astype(v.dtype)
        v = lax.cond(x == 0, lambda val: val.at[..., h:-h, cls.IDX_W].set(val_W), lambda val: val, v)
        
        arr_N = process_dir(Topology.N)
        val_N = ((u[..., cls.IDX_N, h:-h] + arr_N[..., 0, :]) / 2.0).astype(u.dtype)
        u = lax.cond(y == py - 1, lambda val: val.at[..., cls.IDX_N, h:-h].set(val_N), lambda val: val, u)
        
        arr_S = process_dir(Topology.S)
        val_S = ((u[..., cls.IDX_S, h:-h] + arr_S[..., 0, :]) / 2.0).astype(u.dtype)
        u = lax.cond(y == 0, lambda val: val.at[..., cls.IDX_S, h:-h].set(val_S), lambda val: val, u)
        
        return u, v

    @classmethod
    def boundary_communication(cls, u, v):
        """Public API to trigger boundary synchronization within the defined Mesh."""
        with cls.mesh:
            return cls._boundary_communication_single(u, v)
    
    # -------------------------------------------------
    # Print Scheduling Info
    # -------------------------------------------------
    @classmethod
    def print_schedule_info(cls):
        """Prints the calculated topology routing."""
        if cls._routing is None:
            print("Error: Communication is not yet configured. Please call configure() first.")
            return

        print("Communication routing (Single Round All-Gather):")
        dn = {Topology.E: 'E', Topology.W: 'W', Topology.N: 'N', Topology.S: 'S'}
        for t in range(Topology.NTILE):
            routes = []
            for d in (Topology.E, Topology.W, Topology.N, Topology.S):
                nb, dB, _ = cls._routing[t, d]
                routes.append(f"{dn[d]}<-T{nb+1}:{dn[int(dB)]}")
            print(f"  Tile {t+1}: " + ", ".join(routes))