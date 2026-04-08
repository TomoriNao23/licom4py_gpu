"""
File: duogrid.py
Description: Duogrid data structure for grid management.
    Now uses the refactored Global2Local for intelligent auto-sharding 
    of pre-padded arrays.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-03-15
Updated: 2026-03-19

REVISION HISTORY:
    15/03/2026 - Initial implementation with Global2Local-based sharding
    16/03/2026 - Refine field distribution logic
    19/03/2026 - Refactor imports to package-level paths
"""

# Standard library imports
import os

# Third-party imports
import numpy as np
import jax.numpy as jnp

# Local application imports
from licom.mesh import Global2Local
from jax import jit
from jax.sharding import NamedSharding


class Duogrid:
    """
    Duogrid data structure for grid management in LICOM ocean model.
    """
    # Grid parameters
    ntile: int = 6
    halo: int = 3
    nx_local: int = 0
    ny_local: int = 0
    px: int = 1
    py: int = 1

    @classmethod
    def configure(cls, namelist, gpu_mesh) -> 'Duogrid':
        cls.ntile = gpu_mesh.ntile
        cls.halo = gpu_mesh.halo
        cls.nx_local = gpu_mesh.nx_local
        cls.ny_local = gpu_mesh.ny_local
        cls.px = gpu_mesh.px
        cls.py = gpu_mesh.py

        # 获取 NPZ 路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        npz_path = os.path.join(project_root, 'field', f'duogrid_C{namelist.nx}.npz')

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Duogrid NPZ file not found: {npz_path}")

        data = np.load(npz_path)
        print(f"Loading pre-padded duogrid from: {npz_path}")

        # 使用重构后的分配函数
        cls._distribute_fields(data)

        # 初始化需要计算生成的屏蔽场和深度场
        # 这些函数内部会调用 Global2Local.zeros/array 并保持 sharding
        cls._init_calculations()
        
        from licom.mesh import Cube
        Cube.configure(cls.k2e_coef, cls.k2e_loc_i, cls.k2e_loc_j, cls.a_c2l, cls.a_l2c, cls.inner, cls.outer,
                       cls.nx_local, cls.ny_local, cls.px, cls.py)

        print("Duogrid initialized successfully.")
        return cls

    @classmethod
    def _distribute_fields(cls, data) -> None:
        """
        分发自带 Halo 区的网格场。
        注意：统一调用 distribute_pre_padded，
        它会根据 (6, [npz], nx+h, ny+h, [2,2]) 的形状自动判断 'x' 和 'y' 的轴。
        """
        dist_fixed = Global2Local.distribute_pre_padded

        # ---- 所有网格场自动分片 ----
        # 包含了原本 list 中的 a, b, c, d grid 的各种分量
        fields_to_load = [
            'a_x', 'a_y', 'a_kik_x', 'a_kik_y', 'a_sina', 'a_cosa', 
            'a_dx', 'a_dy', 'a_da', 'rda', 'rdx', 'rdy', 'a_f', 'ub', 'vb',
            'a_pt', 'k2e_coef', 'a_gco', 'a_gct', 'a_c2l', 'a_l2c',
            'k2e_loc_i', 'k2e_loc_j',
            'b_pt', 
            'c_gco', 'c_gct', 'c_ct2ort_x', 'c_ort2ct_x', 'c_sina', 'c_cosa', 'c_dy', 'c_dx',
            'd_gco', 'd_gct', 'd_ct2ort_y', 'd_ort2ct_y', 'd_sina', 'd_cosa', 'd_dx', 'd_dy'
        ]

        for name in fields_to_load:
            if name in data:
                # 显式转换为 jnp 数组后直接按照已有形状分片
                setattr(cls, name, dist_fixed(jnp.array(data[name], dtype=Global2Local.dtype)))

        # 特殊处理：整数索引字段
        if 'k2e_loc' in data:
            cls.k2e_loc = dist_fixed(jnp.array(data['k2e_loc'], dtype=jnp.int32))

    @classmethod
    def _init_calculations(cls) -> None:
        """初始化计算生成的衍生字段"""
        cls._init_inner_outer_fields()
        cls._ocean_depth()

    @classmethod
    def _init_inner_outer_fields(cls) -> None:
        """使用逻辑计算填充 Mask 场"""
        # Global2Local.zeros 会根据 '2d' 模板创建 (6, nx_h, ny_h) 的正确 sharding 场
        inner = Global2Local.zeros('2d')

        from licom.mesh import GPU_Mesh
        from jax.experimental.shard_map import shard_map
        with GPU_Mesh.mesh:
            ones_full = jnp.ones_like(inner)
            
            spec = Global2Local.get_spec(inner.shape)
            
            def _set_mask_logic(in_arr, out_arr):
                h = cls.halo
                in_arr = in_arr.at[..., h:-h, h:-h].set(1.0)
                out_arr = out_arr.at[..., h:-h, h:-h].set(0.0)
                return in_arr, out_arr

            sharded_fn = shard_map(
                _set_mask_logic,
                mesh=GPU_Mesh.mesh,
                in_specs=(spec, spec),
                out_specs=(spec, spec),
                check_rep=False
            )
            
            cls.inner, cls.outer = jit(sharded_fn)(inner, ones_full)

    @classmethod
    def _ocean_depth(cls) -> None:
        """初始化海深相关的 2D/3D 场"""
        
        # 预先分配具备正确 Sharding 的显存空间
        dzph_init = Global2Local.zeros('2d')
        kmt_init = Global2Local.zeros('2d')
        vit_init = Global2Local.zeros('3d') 

        from licom.mesh import GPU_Mesh
        from jax.experimental.shard_map import shard_map
        with GPU_Mesh.mesh:
            spec_2d = Global2Local.get_spec(dzph_init.shape)
            spec_3d = Global2Local.get_spec(vit_init.shape)

            def _init_depth_logic(dzph, kmt, vit):
                # 将全场初始化为恒定值 (dzph=5600, kmt=30层, vit=1.0)
                dzph = dzph.at[:].set(5600.0)
                kmt = kmt.at[:].set(30.0)
                vit = vit.at[:].set(1.0)
                return dzph, kmt, vit

            sharded_fn = shard_map(
                _init_depth_logic,
                mesh=GPU_Mesh.mesh,
                in_specs=(spec_2d, spec_2d, spec_3d),
                out_specs=(spec_2d, spec_2d, spec_3d),
                check_rep=False
            )
            
            cls.dzph, cls.kmt, cls.vit = jit(sharded_fn)(dzph_init, kmt_init, vit_init)
        # B/C/D grid 的对应场由于在 stencil 计算中可能通过插值得到，这里按需补充
        cls.dzph_x = cls.dzph
        cls.dzph_y = cls.dzph