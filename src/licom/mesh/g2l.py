"""
File: g2l.py
Description: Refactored Global-to-Local distribution logic with dynamic
             dimension detection and halo handling for different grid structures.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from typing import Any, Dict, Tuple


class Global2Local:
    # 全局配置参数
    mesh: Mesh = None
    halo: int = 3
    nx_local: int = 0
    ny_local: int = 0
    nx: int = 0
    ny: int = 0
    npz: int = 30
    ntile: int = 6
    dtype: jnp.dtype = jnp.float64

    @classmethod
    def configure(
        cls,
        mesh,
        halo: int,
        nx_local: int,
        ny_local: int,
        nx: int,
        ny: int,
        npz: int,
        ntile: int = 6,
    ) -> None:
        """从 GPU_Mesh 获取基础配置"""
        cls.mesh = mesh
        cls.halo = halo
        cls.nx_local = nx_local
        cls.ny_local = ny_local
        cls.nx = nx
        cls.ny = ny
        cls.npz = npz
        cls.ntile = ntile

    @classmethod
    def get_spec(cls, shape: Tuple[int, ...]) -> P:
        """
        核心逻辑：动态检测维度并返回 PartitionSpec。
        规则：
          1. 维度 0 (size == ntile) 始终映射为 'tile'
          2. 跳过维度 0 后，前两个 size > nx_local 的维度依次映射为 'x' 和 'y'
          3. 其余维度映射为 None（不剖分）
        """
        spec_list = [None] * len(shape)
        if len(shape) > 0 and shape[0] == cls.ntile:
            spec_list[0] = "tile"

        found_spatial = 0
        for i in range(1, len(shape)):
            # nx_local 是不含 halo 的 local 宽度；
            # 带 halo 的数组其空间维 size 必然大于 nx_local。
            if shape[i] >= cls.nx_local and found_spatial < 2:
                spec_list[i] = "x" if found_spatial == 0 else "y"
                found_spatial += 1

        return P(*spec_list)

    @classmethod
    def distribute_pre_padded(cls, global_data: jnp.ndarray) -> jnp.ndarray:
        """
        将【已包含最外侧 Halo 区】的全局数组分配到各显卡。
        每块设备恰好获得 (nx_local + 2h) × (ny_local + 2h) 的数据。
        """
        h = cls.halo
        spec = cls.get_spec(global_data.shape)
        spatial_axes = [i for i, s in enumerate(spec) if s in ("x", "y")]

        if len(spatial_axes) != 2:
            return jax.device_put(global_data, NamedSharding(cls.mesh, spec))

        x_axis, y_axis = spatial_axes
        nx_h = cls.nx_local + 2 * h
        ny_h = cls.ny_local + 2 * h
        px = cls.mesh.shape["x"]
        py = cls.mesh.shape["y"]

        target_shape = list(global_data.shape)
        target_shape[x_axis] = px * nx_h
        target_shape[y_axis] = py * ny_h
        target_shape = tuple(target_shape)

        sharding = NamedSharding(cls.mesh, spec)

        def slice_fn(idx):
            slc = list(idx)
            x_start = 0 if slc[x_axis].start is None else slc[x_axis].start
            y_start = 0 if slc[y_axis].start is None else slc[y_axis].start

            ix = x_start // nx_h
            iy = y_start // ny_h

            slc[x_axis] = slice(ix * cls.nx_local, ix * cls.nx_local + nx_h)
            slc[y_axis] = slice(iy * cls.ny_local, iy * cls.ny_local + ny_h)
            return global_data[tuple(slc)]

        return jax.make_array_from_callback(target_shape, sharding, slice_fn)

    @classmethod
    def distribute(cls, global_data: jnp.ndarray) -> jnp.ndarray:
        """
        将【不包含 Halo 区】的原始全局数组在 CPU 上整体 pad 后，
        复用 distribute_pre_padded 分卡。
        """
        h = cls.halo
        spec = cls.get_spec(global_data.shape)
        spatial_axes = [i for i, s in enumerate(spec) if s in ("x", "y")]

        padding = [(0, 0)] * global_data.ndim
        for axis in spatial_axes:
            padding[axis] = (h, h)
        padded = jnp.pad(global_data, padding, mode="constant", constant_values=0)

        return cls.distribute_pre_padded(padded)

    # ---- 快捷工厂方法 ----

    @classmethod
    def zeros(cls, shape_key: str) -> jnp.ndarray:
        """
        根据 key 创建带 Halo 的 sharded 零数组。
        使用【不含 halo】的全局形状构造零数组，调用 distribute 完成 pad 与分卡。
        """
        # 全局形状（不含 halo）
        shapes = {
            "2d":       (cls.ntile, cls.nx, cls.ny),
            "3d":       (cls.ntile, cls.npz, cls.nx, cls.ny),
            "4d":       (cls.ntile, cls.nx, cls.ny, 2, 2),
            "3d1":      (cls.ntile, cls.npz + 1, cls.nx, cls.ny),
            "3d_agrid": (cls.ntile, cls.nx, cls.ny, 2),
            "4d_agrid": (cls.ntile, cls.nx, cls.ny, 2, 2),
        }

        return cls.distribute(jnp.zeros(shapes[shape_key], dtype=cls.dtype))

    @classmethod
    def allocate(cls, owner: Any, field_groups: Dict[str, Any]) -> None:
        """批量分配字段"""
        for shape_key, names in field_groups.items():
            for name in names:
                setattr(owner, name, cls.zeros(shape_key))