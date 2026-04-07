"""
File: spmd.py
Description: Generalized Single Program Multiple Data (SPMD) compilation wrapper for LICOM.
             Provides the core translation layer between global tracer tensors and local block execution.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2026-04-07
"""

import jax
from jax import jit
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

# Local application imports
from licom.mesh import GPU_Mesh

def _sh(x):
    return getattr(x, 'sharding', None)

def _spec(x):
    sh = _sh(x)
    return sh.spec if sh is not None else P()

def make_spmd_jit(core_fn, state, consts, static_argnums=(2, 3)):
    """
    通用分布式的 JIT 并行编译包装器。
    在 Tracing 期间执行静态 Monkey Patch，在真实运算期间进行零损耗 Local Shard 调用。
    
    Args:
        core_fn: 需要被分布式并行的纯函数，签名需为 (state, consts, *static_args)
        state: 由 JAX Distributed Arrays 组成的状态输入 (包含 Sharding 布局)
        consts: 静态只读的参数输入
        static_argnums: 指定在 wrapper 中哪些参数为静态常量 (默认 2 开始)
    """
    from licom.duogrid import Dg
    from licom.mesh import Cube

    # 1. 提取 state 和 consts 的 shard specs（防止 UnspecifiedValue 引起 JIT 崩溃）
    state_specs = jax.tree_util.tree_map(_spec, state)
    consts_specs = jax.tree_util.tree_map(_spec, consts)

    # 2. 提取 sharding 定义以用于最终的 jit 装饰器
    state_sh = jax.tree_util.tree_map(_sh, state)
    consts_sh = jax.tree_util.tree_map(_sh, consts)

    # 3. 从单例对象 (Dg, Cube) 中静态提取 patch 映射表
    patch_keys = []
    patch_specs = []
    for obj in (Dg, Cube):
        for k, v in vars(obj).items():
            if hasattr(v, 'shape') and hasattr(v, 'sharding'):
                patch_keys.append((obj, k))
                patch_specs.append(_spec(v))
    patch_specs = tuple(patch_specs)

    # 4. 定义暴露给外部的模型封装器 wrapper，使用变长参数支持不同物理步的自有常数
    def wrapper(s, c, *static_args):
        # 4.1 提取待映射的全局 array 的局部 tracer 引用
        patch_vals = tuple(getattr(obj, k) for obj, k in patch_keys)
        
        # 4.2 定义 shard_map 并行核心算子
        def map_fn(s_inner, c_inner, patch_inner):
            # 保存旧全局属性
            old_vals = [getattr(obj, k) for obj, k in patch_keys]
            
            # 将传进来的局部切片挂靠给模块级单例对象
            for (obj, k), mapped_v in zip(patch_keys, patch_inner):
                setattr(obj, k, mapped_v)
            
            try:
                # 4.3 核心：执行 JIT 数学步进
                return core_fn(s_inner, c_inner, *static_args)
            finally:
                # 4.4 从单例对象中剥离局部引用，恢复全局单例属性
                for (obj, k), orig_v in zip(patch_keys, old_vals):
                    setattr(obj, k, orig_v)
            
        # 4.5 启动 shard_map 提交给 Mesh 执行
        sharded_fn = shard_map(
            map_fn,
            mesh=GPU_Mesh.mesh,
            in_specs=(state_specs, consts_specs, patch_specs),
            out_specs=state_specs,
            check_rep=False
        )
        return sharded_fn(s, c, patch_vals)

    return jit(
        wrapper,
        static_argnums=static_argnums,
        in_shardings=(state_sh, consts_sh),
        out_shardings=state_sh,
    )

# =====================================================================
# 通用打包解包逻辑
# =====================================================================

def auto_pack(instance, state_keys, const_sources):
    """
    通用打包函数：将类实例中的字段按需组装为 state 和 consts tuples，供 SPMD JIT 消费。
    """
    state = tuple(getattr(instance, k) for k in state_keys)
    consts = tuple(getattr(src, k) if src is not None else getattr(instance, k)
                   for src, k in const_sources)
    return state, consts

def auto_unpack(instance, state_keys, state):
    """
    通用解包函数：将 JIT 返回的 state tuple 写回给实例。
    """
    for k, v in zip(state_keys, state):
        setattr(instance, k, v)
