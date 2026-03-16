"""
File: g2l.py
Description: Refactored Global-to-Local distribution logic with dynamic 
             dimension detection and halo handling for different grid structures.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental.pjit import pjit
from typing import Any, Dict, Tuple


class Global2Local:
    # 全局配置参数
    mesh: Mesh = None
    halo: int = 3
    nx_local: int = 0
    ny_local: int = 0
    npz: int = 30
    ntile: int = 6
    dtype: jnp.dtype = jnp.float64

    @classmethod
    def configure(cls, mesh, halo: int, nx_local: int, ny_local: int, 
                  npz: int, ntile: int = 6) -> None:
        """从 GPU_Mesh 获取基础配置"""
        cls.mesh = mesh
        cls.halo = halo
        cls.nx_local = nx_local
        cls.ny_local = ny_local
        cls.npz = npz
        cls.ntile = ntile

    @classmethod
    def get_spec(cls, shape: Tuple[int, ...]) -> P:
        """
        核心逻辑：动态检测维度并返回 PartitionSpec
        规则：
        1. 维度 0 (size=6) 始终是 'tile'
        2. 跳过维度 0 后，找到前两个 size > nx_local 的维度分别映射为 'x' 和 'y'
        3. 其他维度映射为 None (不剖分)
        """
        spec_list = [None] * len(shape)
        if len(shape) > 0 and shape[0] == cls.ntile:
            spec_list[0] = 'tile'
        
        found_spatial = 0
        for i in range(1, len(shape)):
            # 这里 nx_local 是不含 halo 的 local 宽度。
            # 如果是已经带 halo 的数组，其维度 size 必然大于 nx_local
            if shape[i] > cls.nx_local and found_spatial < 2:
                if found_spatial == 0:
                    spec_list[i] = 'x'
                else:
                    spec_list[i] = 'y'
                found_spatial += 1
            else:
                spec_list[i] = None
        
        return P(*spec_list)

    @classmethod
    def distribute_pre_padded(cls, global_data: jnp.ndarray) -> jnp.ndarray:
        """
        将【已经包含 Halo 区】的数据分配到显卡。
        自动识别逻辑维度并放置，不需要额外填充边界。
        """
        spec = cls.get_spec(global_data.shape)
        sharding = NamedSharding(cls.mesh, spec)
        
        # 将数据转换成 JAX 数组并按照指定的 sharding 分片
        return jax.device_put(global_data, sharding)

    @classmethod
    def distribute(cls, global_data: jnp.ndarray) -> jnp.ndarray:
        """
        将【不包含 Halo 区】的原始数据分配到显卡并【补全全零边缘】。
        """
        # 1. 探测原始数据的分片方式
        # 先按原始形状探测，这通常是用于放置初始全局数组
        temp_spec = cls.get_spec(global_data.shape)
        temp_sharding = NamedSharding(cls.mesh, temp_spec)
        data_sharded = jax.device_put(global_data, temp_sharding)

        # 2. 定义带 Halo 的最终形状和分片规格
        h = cls.halo
        # 探测哪些维是空间维 (x, y)，对应的维需要增加 2*h
        spatial_axes = []
        for i, name in enumerate(temp_spec):
            if name in ('x', 'y'): spatial_axes.append(i)

        final_shape = list(global_data.shape)
        for axis in spatial_axes:
            final_shape[axis] += 2 * h
        
        final_spec = cls.get_spec(tuple(final_shape))
        final_sharding = NamedSharding(cls.mesh, final_spec)

        # 3. 执行 Padding。
        # 定义一个 pjit 内部函数，利用 vmap 或 slice 实现高效填充
        def _pad_core(data):
            # 构造计算 pad 的 slice 数组，其余维度补 : (full slice)
            padding = [(0, 0)] * data.ndim
            for axis in spatial_axes:
                padding[axis] = (h, h)
            return jnp.pad(data, padding, mode='constant', constant_values=0)

        pjit_pad = pjit(
            _pad_core,
            in_shardings=(temp_sharding,),
            out_shardings=final_sharding
        )

        return pjit_pad(data_sharded)

    # ---- 快捷工厂方法 ----

    @classmethod
    def zeros(cls, shape_key: str) -> jnp.ndarray:
        """
        根据 key 创建带 Halo 的 sharded zero 数组
        """
        # 这里可以使用简单的固化映射
        h = cls.halo
        nx_h, ny_h = cls.nx_local + 2*h, cls.ny_local + 2*h
        
        # 简单定义一些常用的 shape
        shapes = {
            '2d':       (cls.ntile, nx_h, ny_h),
            '3d':       (cls.ntile, cls.npz, nx_h, ny_h),
            '3d1':      (cls.ntile, cls.npz + 1, nx_h, ny_h),
            '3d_agrid': (cls.ntile, nx_h, ny_h, 2),
            '4d_agrid': (cls.ntile, nx_h, ny_h, 2, 2),
            # 也可保留 4d 作为一个通用别名
            '4d':       (cls.ntile, nx_h, ny_h, 2, 2)
        }
        
        target_shape = shapes[shape_key]
        spec = cls.get_spec(target_shape)
        sharding = NamedSharding(cls.mesh, spec)
        
        return jax.device_put(jnp.zeros(target_shape, dtype=cls.dtype), sharding)

    @classmethod
    def allocate(cls, owner: Any, field_groups: Dict[str, Any]):
        """批量分配字段"""
        for shape_key, names in field_groups.items():
            for name in names:
                setattr(owner, name, cls.zeros(shape_key))