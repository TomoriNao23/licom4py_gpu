# JAX 分布式并行 Sharding 核心解析：UnspecifiedValue 错误复盘

## 现象描述
在代码以高分辨率（如 C1536）并设定多卡（如 pdev=2）运行 JAX 程序时，抛出了如下异常：
```python
AttributeError: 'UnspecifiedValue' object has no attribute 'addressable_devices_indices_map'
```
而在低分辨率（C768 及以下）或单卡（pdev=1）运行时则完全正常。

## 根本原因 (Root Cause)

该问题的核心源头来自于 **多设备 Sharding（分片策略）的不兼容组合** 以及 **JAX 编译器 (XLA) 对于常量的处理机制**。

### 1. 设备集不兼容 (Incompatible Devices)
在 `barotr_rk2_core` 被 `jit` 编译并调用时，它接收两类状态参数：
1. **气象/海洋场（如 $u_b$, $h_0$ 等 2D 数组）**：这些是在 `Global2Local` 中通过 `NamedSharding(mesh, P('tile', 'x', 'y'))` 正确分配的，跨了多张显卡（如 GPU:0, GPU:1）。
2. **标量参数（如计算步数累加器 `isb`）**：默认情况下，Python 标量或 `jnp.int32(0)` 会被放置在**单张默认设备**上（通常带有 `SingleDeviceSharding(GPU:0)`）。

当这两种拥有完全不同设备拓扑的参数同时输入给同一个未受限的 `jit` 函数时发生冲突。

### 2. 为什么是 UnspecifiedValue？
当 JAX `jit` 在编译阶段发现多输入参数分布在不同、且不兼容的设备拓扑组合时，它尝试建立输入参数和目标设备的映射关系，并在内部退化回一种无规则的标记（即 `UnspecifiedValue` 占位符）。
而在执行计算 (dispatch) 阶段尝试将真实的分布式数组分配过去时，发现目标 Sharding 并不是一个合法的拓扑规则，从而导致报错。

### 3. 为什么低分辨率 (C768 以下) 不报错？
在低分辨率场景，全局的一些常数和小型参数（比如 `consts` 和 `isb`）其总显存占用小于 XLA 的截断限制（报错首行提到 `2.74GB constants captured` 警告证明了 C1536 突破了阈值）。
- **小分辨率**：标量和小矩阵会在编译时被 XLA 作为 `captured constants` 直接内联编译进 Graph 中，这就**绕开了运行时的 Sharding 检查**。
- **大分辨率 (C1536)**：由于捕获常量过大（>2GB），XLA 放弃常量内联，将其转化为在运行时需要传入的核心张量。此时严格的 Sharding 检查开始工作，彻底暴露了上述设备集不兼容的致命问题。

## 解决方案与核心机制 (Solutions & Core Mechanisms)

解决该问题分为两步，分别对应 JAX 在多卡并行环境下的两个核心陷阱：

### 1. 为什么必须保留“薄包裹”的延迟绑定？ (Why Lazy Binding is Required)
如果直接使用原生的纯 `jit` 而不明确指定输出约束（`out_shardings`），当矩阵经过 JIT 完成运算并把结果返回至 Python 时，JAX (v0.4.x) 出于默认策略会丢失张量的显式网格分片实例，导致该数组在 Python 层的 `.sharding` 属性退化为无规则的 `UnspecifiedValue`。
- **崩溃触发点**：由于主程序（如 `Schedule.run`）在连续多个时间步中循环，上一阶段出来的变量会在下一轮重新砸进同一个 JIT。第二步的 XLA 编译器在解析输入布局（`_resolve_in_layouts`）时，只要一碰到 `UnspecifiedValue` 标签就会报错崩溃。
- **解决方案**：保留原有的几行薄延迟绑定（即 `_make_barotr_jit`），其唯一的目的就是在进入主循环的每一次初始，**提取所有变量原生的 `NamedSharding` 然后原封不动地设为 `out_shardings=state_sh`**。这就强迫 JAX 能够老老实实把切片状态带回给外部结构。
  *(注：这种构建只能在运行时调用，因为项目启动模块被加载时，网格上下文 `GPU_Mesh.mesh` 甚至还没被初始化)*

### 2. 像 `isb` 这类的计数量如果真要进出 JIT，应该额外怎么处理？
最初的架构里，仅仅用来当做外部计数的 0 维标量 `isb` 也被包裹在 `state` tuple 里强行和多卡阵列组队：
- **可能面临的陷阱**：标量的最初默认分片状态（或由 JAX 内部 for 循环所动态生成的 `jnp.int32` 分片状态）通常默认驻留于单卡，或是在多卡退化为 `UnspecifiedValue`。如果直接随意调用 `jax.device_put` 与多卡对齐，极易因这层残留的脏标签触发异常。
- **需要额外怎么处理**：如果你真的要把标量强行放进循环参与分布式状态同步，唯一安全的办法是在 `_pack()` 时做到**两步走切断污染**：
  1. 先安全拉回主机洗白：`isb_clean = jax.device_get(self.isb) if hasattr(self.isb, 'sharding') else self.isb`（彻底剥离从上次 JIT 里带出来的任何异常的 Sharding 标签）
  2. 再向全网格打上干净的显式分身：`isb_replicated = jax.device_put(jnp.int32(isb_clean), NamedSharding(GPU_Mesh.mesh, P()))`，随后打包进 Tuple。
- **目前的最终更优实践**：由于深入审视物理后发现 `isb` 只是一个统计步数的外部纯计数器。在目前修复版的代码中，我们直接**把它彻底从 Tuple `state` 传参中清除了**，仅在 JIT 函数外独立的执行 `self.isb += self.nbb`，彻底把非纯数学场的东西隔绝出 XLA，也清除了源头隐患。
