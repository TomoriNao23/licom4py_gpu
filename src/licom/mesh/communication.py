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
from jax.sharding import PartitionSpec as P, NamedSharding

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

        # Vectorize over the 'tile' axis and use jit to preserve sharding
        # in_shardings and out_shardings enforce logic
        sharding = NamedSharding(cls.mesh, P('tile', 'x', 'y'))
        cls._update_domain_jit = jit(
            jax.vmap(cls._update_domain_single, axis_name="tile"),
            in_shardings=(sharding,),
            out_shardings=sharding
        )
        
        cls._boundary_communication_jit = jit(
            jax.vmap(cls._boundary_communication_single, axis_name="tile"),
            in_shardings=(sharding, sharding),
            out_shardings=(sharding, sharding)
        )

    # -------------------------------------------------
    # Shared Logic
    # -------------------------------------------------
    @classmethod
    def _apply_transform(cls, data, recv_dir, tid):
        """
        Unified data rotation/flip application layer.
        """
        is_ew = recv_dir in (Topology.E, Topology.W)
        n = cls.ny_local if is_ew else cls.nx_local
        raw = data[:, :n]

        if is_ew:
            # Expected shape: (ny_local, halo)
            return lax.cond(
                tid == Topology.FLIP_I_TRANS,
                lambda x: x.T[::-1, :],
                lambda x: x.T,
                raw
            )
        else:
            # Expected shape: (halo, nx_local)
            return lax.cond(
                tid == Topology.FLIP_I_TRANS,
                lambda x: x[:, ::-1],
                lambda x: x,
                raw
            )

    # -------------------------------------------------
    # Domain Update (Halo Exchange)
    # -------------------------------------------------
    @classmethod
    def _pack_edge(cls, u, d):
        """Extracts the boundary data to be sent based on the direction."""
        h, n = cls.halo, cls._n_pad

        if d == Topology.E:   
            raw = u[h:-h, -2*h:-h].T
        elif d == Topology.W: 
            raw = u[h:-h, h:2*h].T
        elif d == Topology.N: 
            raw = u[-2*h:-h, h:-h]
        else:                 
            raw = u[h:2*h, h:-h]

        # [Optimization] Use jnp.pad instead of concatenate for better XLA compilation friendliness
        cur = raw.shape[1]
        if cur < n:
            raw = jnp.pad(raw, ((0, 0), (0, n - cur)))
        return raw

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
        
        # Gather all edges from all tiles. recv shape: (NTILE, 4, halo, n)
        recv = lax.all_gather(send, "tile", tiled=False) 
        
        tile_id = lax.axis_index("tile")
        
        def process_dir(d):
            nb, dB, tid = cls._routing[tile_id, d]
            return cls._apply_transform(recv[nb, dB], d, tid)

        h = cls.halo
        u = u.at[h:-h, -h:].set(process_dir(Topology.E))
        u = u.at[h:-h, :h].set(process_dir(Topology.W))
        u = u.at[-h:, h:-h].set(process_dir(Topology.N))
        u = u.at[:h, h:-h].set(process_dir(Topology.S))
        
        return u

    @classmethod
    def update_domain(cls, u):
        """Public API to trigger domain update within the defined Mesh."""
        with cls.mesh:
            return cls._update_domain_jit(u)

    # -------------------------------------------------
    # C-Grid Boundary Edge Synchronization
    # -------------------------------------------------
    @classmethod
    def _pack_boundary_edge(cls, u, v, d):
        """Extracts boundary lines for U/V vector fields based on direction."""
        h, n = cls.halo, cls._n_pad

        if d == Topology.E:   
            line = v[h:-h, cls.IDX_E]
        elif d == Topology.W: 
            line = v[h:-h, cls.IDX_W]
        elif d == Topology.N: 
            line = u[cls.IDX_N, h:-h]
        else:                 
            line = u[cls.IDX_S, h:-h]

        raw = jnp.zeros((h, len(line)), dtype=u.dtype).at[0, :].set(line)

        # [Optimization] Use jnp.pad instead of concatenate
        cur = raw.shape[1]
        if cur < n:
            raw = jnp.pad(raw, ((0, 0), (0, n - cur)))
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
        
        # Gather all edges from all tiles. recv shape: (NTILE, 4, halo, n)
        recv = lax.all_gather(send, "tile", tiled=False)
        tile_id = lax.axis_index("tile")
        
        def process_dir(d):
            nb, dB, tid = cls._routing[tile_id, d]
            return cls._apply_transform(recv[nb, dB], d, tid)

        h = cls.halo
        
        arr_E = process_dir(Topology.E)
        val_E = ((v[h:-h, cls.IDX_E] + arr_E[:, 0]) / 2.0).astype(v.dtype)
        v = v.at[h:-h, cls.IDX_E].set(val_E)
        
        arr_W = process_dir(Topology.W)
        val_W = ((v[h:-h, cls.IDX_W] + arr_W[:, 0]) / 2.0).astype(v.dtype)
        v = v.at[h:-h, cls.IDX_W].set(val_W)
        
        arr_N = process_dir(Topology.N)
        val_N = ((u[cls.IDX_N, h:-h] + arr_N[0, :]) / 2.0).astype(u.dtype)
        u = u.at[cls.IDX_N, h:-h].set(val_N)
        
        arr_S = process_dir(Topology.S)
        val_S = ((u[cls.IDX_S, h:-h] + arr_S[0, :]) / 2.0).astype(u.dtype)
        u = u.at[cls.IDX_S, h:-h].set(val_S)
        
        return u, v

    @classmethod
    def boundary_communication(cls, u, v):
        """Public API to trigger boundary synchronization within the defined Mesh."""
        with cls.mesh:
            return cls._boundary_communication_jit(u, v)
    
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