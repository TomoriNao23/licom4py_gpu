# JAX SPMD 并行架构原理与核心机制深度解析

**所属模块**：`src/licom/mesh/spmd.py: make_spmd_jit`

在 LICOM 模式向多 GPU 高性能计算架构演进过程中，为实现物理代码在分布式算力上的无损映射并保持低运行时开销，本系统设计了 `make_spmd_jit` 作为核心分发接口。该模块统一接管多设备边界约束（Sharding）、抽象作用域替换及动态代码向 C++ 静态图编译（JIT Tracing）的底层逻辑。

本文档系统性论述该架构的底层运行机理，分为两大核心板块：基于 `shard_map` 的分布式数据流控设计，以及基于 JAX 追踪器的闭包变量穿越机制。

---

## 第一部分：基于 `shard_map` 的分布式映射与状态代理设计

针对复杂物理算子直接调用全局上下文（如 `Dg`, `Cube` 单例模块）会导致的并行越界漏洞，以及 JAX 编译器内部的分布式元数据擦除隐患，本框架实施了“静态反射先验证 - 并行图内沙盒挂载”的完整防御链条设计。

### 1.1 核心设计思路 (Architecture Strategy)

#### 1. 差异化的分片解析策略 (Selective Sharding Resolution)
在单程序多数据流（SPMD）范式下，显式通过函数形参传递的物理系统张量集合（由外层主动传入的 `state` 与 `consts` 结构）能完美对接 `shard_map` 的内部边界隔离特性。XLA 生成映射过程时，会自动为这类形式传入的阵列裁剪出正确的局部维度（映射至 `s_inner` 等变量）。
反之，大量底层微分以及平流运算为了保证地学算子的模块化与整洁性，大量保留了纯硬编码的特征常数绝对访问（例如 `AGrid.grad` 内部硬编码调取 `Dg.rdx` 分辨率常数）。该类游离域的调用在进入独立于设备的片场内存中时，极易诱发维度垮塌违例。

#### 2. 静态反射前置规避 (Static Reflection Ahead-of-Time)
基于字典映射（`vars()` 或 `getattr()`）的重反射动作如果暴露于 JAX 等图级编译器执行管线下，必激怒 Interpreter C-API 引发致命串行化堵塞。系统的策略在于，利用顶层封装框架运行期外部独立沙箱时间，对需要进行沙盒投射的单例池先行提取全部常量键对及分布指标结构（Sharding Metadata），以此构建不可变缓存序列。

#### 3. 命名空间蒙版挂载 (Abstract Namespace Monkey-Patching)
核心目标是为纯算子函数提供一个屏蔽跨卡网络边界的受限作用空间视角。借助上述提取的索引图腾，在卡内循环执行过程中对全局单例环境实施 `try...finally` 生命周期监控型临时引用替换。由此实现深层基础物理函数无代码侵入，同时无缝连接本地空间环境的映射奇点。

### 1.2 结构化实施层 (Implementation Steps)

#### 步骤一：编译期保护伞结构建立 (Sharding Specification Fallback Constraint)
**阻碍与对策**：分布式的 `Array` 对象离开 `shard_map` 的管辖出口时，若无直接约束申明，引擎极易擦除底层切分布局（退回不合法之 `UnspecifiedValue` 占位符）。这会导致高时间步迭代循环场景下（如 RK 显式步内或跨步传输），该张量随之下探下一批管线时诱发验证阶段即刻崩溃。
**操作规程**：通过在全局切面进行树形解包（Tree Map），将提取至顶端的合法外边界元数据强制配置给最外层 `jit`，实现类型体系闭环兜底：
```python
state_sh = jax.tree_util.tree_map(_sh, state)
consts_sh = jax.tree_util.tree_map(_sh, consts)
# ...强制重锁定输出特征...
in_shardings=(state_sh, consts_sh),
out_shardings=state_sh,
```

#### 步骤二：独立空间的变量拦截与伪装投递 (Variables Interception & Deployment)
将 `Dg, Cube` 单例模块下的重型张量资源化为不可变静态参数管线集，以独立通道（`patch_inner`）传入真正承接本地空间算子编译的 `map_fn`；以局部 Tracer（微缩占位符对象）劫持全局模块指针，以构筑本地算力独立视图：
```python
patch_vals = tuple(getattr(obj, k) for obj, k in patch_keys)
# 在片上分配核算子内:
for (obj, k), mapped_v in zip(patch_keys, patch_inner):
    setattr(obj, k, mapped_v)
```

#### 步骤三：空间回收协议的硬性执行 (Lifecycle Restitution Protocol)
**阻碍与对策**：用于拦截视图重定向的单例驻留在整个宿主系统顶层。倘使 C++ 后段追踪完成解耦，这些 Tracer 垃圾对象滞留将招致其后对流与辐射算法模块因引用伪影爆出严重之 `UnexpectedTracerError`。
**操作规程**：计算过程由 `finally` 关键字锁死还原程序，确保跨边界并行图编织竣工并释放资源引用后，全局对象同步回退为原生广域张量。
```python
finally:
    for (obj, k), orig_v in zip(patch_keys, old_vals):
        setattr(obj, k, orig_v)
```

---

## 第二部分：JIT 环境下的闭包与动静态切分设计

当任务深度下探进入包含显式计算时间推演流程的主循环结构（见诸 `barotr_rk2_core` 及内置 `jax.lax.fori_loop`），内存语义被严格平分为二元化设计：作为管道输入载体的**可变传导流 (State Pipe)**，以及依托高层级封闭捕获的**只读闭包穿越群 (Lexical Constant Register)**。

### 2.1 基于显式状态传输的高频迭代 (Explicit Propagating State Pipeline)
作为计算主体的物理预报场（例如表层水深偏移状态等），被归类为单次循环内外重度非对称改写的重构态。这种特征变量天然必须遵循“单入单出互斥”规范，以正式参数管道流模型（由 `val` 入射并作为返回值闭合更新），提供 C++ 外层引擎判定张量内存覆写或新分配之凭据准则。

### 2.2 构建零开销挂载的闭包穿越极简范式 (Zero-Cost Lexical Closures Binding)
**阻碍与对策**：如果地脉特征与科里奥利常数场参量如同动态层叠状态一样经 `fori_loop` 进出函数帧栈，巨量的静态载荷往复赋值会极易穿透编译器堆栈优化策略，产生大规模不必要的显存通讯和重定位冗余中继。

**操作规程**：通过消除这些变量在 `body_fun` 中的形式接管，促发 Python 对于高深嵌套块函数执行环境默认向更高结构域外（Enclosing Scope 即 `_barotr_rk2_core` 原函数作用域的已解算张量引用）开启的词法闭包含捕：
```python
def _barotr_rk2_core(state, consts, nbb, dtb): # 安全的闭包挂靠根区
    def body_fun(i, val):
        # 抛弃常量接力，交予执行域捕获并下沉
        ...调用... dtb, consts ... 
```
由于 JAX Tracer 的设计极度贴合纯函数法则，当追踪编译器侦测到这些未遭任何形式写赋值的外部透传读取，将精准定义其为无副作用特征量（Side-effect Free Feature），并自动化地将它们在底层的 XLA 运算图中剥离出来，转存为挂靠在循环节点背部、直接高速读取的 **单向静态常量缓冲板（Immutable Read-Only Cache buffers）**。此行为全面终结了参数分配拷贝的时间复杂度，达成架构最优化实现。
