# 多GPU数据分片与SPMD并行机制

### 1. 全局物理量数据结构
全局数据遵循行优先存储, 从左到右通常分别是：面(tile)数据、层(z)数据、x方向数据、y方向数据以及其他附加的网格/分维数据。  
全局数据需要记录tile最外侧halo区的数据。每个tile内部如果划分了子区域，则每个子区域内部halo区无需额外在全局内存中记录隔离带，而是通过网格划分的数据重叠直接获取相邻区域数据（即按 `nx_local + 2*halo` 重叠切片）。

根据 `g2l` 模块，目前基础全局变量形状包含以下几类（以**包含最外层halo区的内存真实存储形状**为主）：
#### 2维标量变量 (2d, 如ub/vb)
```
tensor = (ntile, nx+2*halo, ny+2*halo)
```
#### 3维标量变量 (3d, 如u/v/ws)
```
tensor = (ntile, npz, nx+2*halo, ny+2*halo)
```
#### 3维扩展一层变量 (3d1)
```
tensor = (ntile, npz+1, nx+2*halo, ny+2*halo)
```
#### 3维A-grid变量 (3d_agrid)
```
tensor = (ntile, nx+2*halo, ny+2*halo, 2)
```
#### 4维变量 (4d 及 4d_agrid)
```
tensor = (ntile, nx+2*halo, ny+2*halo, 2, 2)
```


### 2. 将所有GPU组织为3维拓扑网络
```python
Mesh = Mesh(devices, ('tile', 'x', 'y'))
```
配置与拆分完全遵循 `namelist` 设定，网格总卡数必须符合 `n_mesh = pdev * px * py`。
例如：  
(1) 6张GPU（pdev=6, px=1, py=1），`Mesh=(6, 1, 1)`，每个GPU获取1个tile的所有数据。  
(2) 24张GPU（pdev=6, px=2, py=2），`Mesh=(6, 2, 2)`，每个GPU获取每个tile的1/4块区域的数据。


### 3. 全局数据映射GPU拓扑的分片与重叠切片逻辑（Global-to-Local）

分片机制的核心是**必须将后续所有计算全部限定在 `shard_map` 的局部数据上进行**，防止复杂的模板计算与全局自动运算触发无谓的全图同步与通信：
1. **真实物理量形状**：第一步，物理量实际长和宽起步就是 `nx + 2*halo`，**最外层halo数据是必要且固定的**。
2. **虚拟放大维度与重叠获取**：为了拆解到各卡并自然获取内部相邻点的 halo 区，内部自动根据块划分 `px, py` 构建了一个用于向 JAX 解释的虚拟扩展全局数组目标大小——等于 `(nx/px + 2*halo) * px`。
3. **单卡分配实现**：通过 `slice_fn` 映射，**每次均在真实全局数组内按固定的物理偏移获取重叠数据。最终分片落到每张卡上的都是完整带 halo 的子数据块（`nx/px + 2*halo`），而 tile 内部的相邻 halo 数据在此过程中即实现了重叠借用获取。**

```python
# 核心分配与重叠获取代码 (摘录)
nx_h = cls.nx_local + 2 * h
ny_h = cls.ny_local + 2 * h

# 构建用于分片的逻辑(虚拟)放大全局数组维度
target_shape[x_axis] = px * nx_h
target_shape[y_axis] = py * ny_h

def slice_fn(idx):
    slc = list(idx)
    ...
    ix = x_start // nx_h
    iy = y_start // ny_h
    
    # 核心重叠切取：按 nx/px 切分出局部偏移，但每次往后切取足足 nx/px + 2*halo 个长度！
    slc[x_axis] = slice(ix * cls.nx_local, ix * cls.nx_local + nx_h)
    slc[y_axis] = slice(iy * cls.ny_local, iy * cls.ny_local + ny_h)
    return global_data[tuple(slc)]

# 在全局 Mesh 自动收集分布 array
jax.make_array_from_callback(tuple(target_shape), sharding, slice_fn)
```

#### 动态检测维度自动分配 PartitionSpec
除了上述切片外，将变量与并行 `Mesh` 绑定的 `PartitionSpec` 挂载规则为在 `Global2Local.get_spec` 中实现：
1. **面级别**：维度 0 如果 `size == ntile` (默认6)，始终映射为 `'tile'`。
2. **空间划分**：跳过维度 0 后，找到前两个 `size >= nx_local` 的维度依次分别映射为 `'x'` 和 `'y'`。
3. **其他维度**：其余均映射为 `None` (在所有GPU上做完整复制，不剖分)。


### 4. SPMD JIT 并行编译与局部执行视图
实际运算在 `spmd.py` 内部被封装转化为单程序多数据流（SPMD）模式：
1. **零损耗拆解**：通过 JAX 的 `shard_map` 直接让核函数消费分片后的本地视图数组。
2. **动态切面注入**：利用 `Monkey Patching` 在 Tracing 期间静态截获单例对象（如 `Dg`, `Cube`），将其引用的全局网格参数透明替换为分片后的局部引用；并在计算跳出时安全还原。
3. **无需关注全局索引**：物理公式内无需处理复杂的跨块索引换算，纯粹通过局部包含 halo 的视窗（`nx_local + 2*halo`）完成网格计算。


### 总结
1. **自动将所有物理量和网格参数分配到准确对应的GPU**：每块GPU仅获取并维护自己局地计算相关的核心区以及 `halo` 拓展区。
2. **计算高度独立**：除手动显式调用的 `halo` 通信外，完全避免跨 GPU 读写数据或返回主机（Host）内存进行串行同步处理。
3. **多维缩放对等**：数据在 `Mesh = (tile, x, y)` 的轴划分上完全正交对等，对于 `tile` 轴的解耦与 `x/y` 上的任意二维剖分具备形式上完全一致的 GPU 可扩展性。