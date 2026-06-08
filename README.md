# KDA算子分析
以下性能数据来源于FLA仓pytest用例，执行指令为`pytest tests/ops/test_kda_chunk.py::test_megatron_varlen`，用例为`(32, [0, 2048, 4096, 8192], True, -5.0, torch.bfloat16)`。

通过msprof工具抓取KDA算子性能，可以发现该算子由十余个triton kernel算子拼接实现，单轮用时260ms，以下两个算子为主要瓶颈：

+ chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2（159ms）
+ chunk_kda_bwd_kernel_intra（45ms）

因此下文主要优化这两个瓶颈算子。



# chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2优化
agent分析出的优化点以及收益如下，总时间由159ms优化到84ms

## <font style="color:#000000;">优化点 1：Autotune 配置扩展（AutoTune 自动调优）</font><font style="color:#DF2A3F;">（159->110）</font>
**<font style="color:#000000;">v0</font>**<font style="color:#000000;">：</font>

```python
@triton.autotune(
    configs=[triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=3)],
    key=['BT', 'TRANSPOSE_STATE'],
)
```

**<font style="color:#000000;">v1</font>**<font style="color:#000000;">：</font>

```python
def get_autotune_configs():
    return [
        triton.Config({'BK': 32, 'BV': 32, 'multibuffer': False}),
        triton.Config({'BK': 32, 'BV': 32, 'multibuffer': True}),
        triton.Config({'BK': 64, 'BV': 64, 'multibuffer': False}),
        triton.Config({'BK': 64, 'BV': 64, 'multibuffer': True}),
        triton.Config({'BK': 128, 'BV': 64, 'multibuffer': False}),
        triton.Config({'BK': 128, 'BV': 64, 'multibuffer': True}),
        triton.Config({'BK': 64, 'BV': 128, 'multibuffer': False}),
        triton.Config({'BK': 64, 'BV': 128, 'multibuffer': True}),
    ]
```

**<font style="color:#000000;">收益原因</font>**<font style="color:#000000;">：</font>

+ **<font style="color:#000000;">Tiling 参数搜索空间扩展</font>**<font style="color:#000000;">：v0 只有 1 个固定配置 </font>`<font style="color:#000000;">BK=32, BV=32</font>`<font style="color:#000000;">，无法适配不同 K/V 维度下的最优分块。v1 用 8 个配置覆盖 </font>`<font style="color:#000000;">BK/BV ∈ {32, 64, 128}</font>`<font style="color:#000000;"> 的组合，autotune 会自动选出实际延迟最低的配置。</font>
+ **<font style="color:#000000;">不同 K/V 维度对最优 BLOCK_SIZE 敏感</font>**<font style="color:#000000;">：当 K 较大时，增大 BK 可以减少外层循环 </font>`<font style="color:#000000;">i_k</font>`<font style="color:#000000;"> 的迭代次数；当 V 较大时，增大 BV 减少内层循环 </font>`<font style="color:#000000;">i_v</font>`<font style="color:#000000;"> 的迭代次数。不同 shape 的最优 Tiling 差异巨大，固定参数会严重劣化部分场景。</font>

---

## <font style="color:#000000;">优化点 2：移除 </font>`<font style="color:#000000;">num_warps</font>`<font style="color:#000000;"> 和 </font>`<font style="color:#000000;">num_stages</font>`<font style="color:#000000;">（GPU 概念不适用）</font>
**<font style="color:#000000;">v0</font>**<font style="color:#000000;"> </font><font style="color:#000000;">→</font><font style="color:#000000;"> </font>**<font style="color:#000000;">v1</font>**<font style="color:#000000;">：</font>`<font style="color:#000000;">num_warps=2, num_stages=3</font>`<font style="color:#000000;"> </font><font style="color:#000000;">被完全移除。</font>

**<font style="color:#000000;">收益原因</font>**<font style="color:#000000;">：</font>

+ `<font style="color:#000000;">num_warps</font>`<font style="color:#000000;"> </font><font style="color:#000000;">和</font><font style="color:#000000;"> </font>`<font style="color:#000000;">num_stages</font>`<font style="color:#000000;"> </font><font style="color:#000000;">是</font><font style="color:#000000;"> </font>**<font style="color:#000000;">GPU CUDA 编程模型</font>**<font style="color:#000000;">下的概念（warp 调度、软件流水线级数），在 Ascend NPU 的 Vector/Cube 架构下</font>**<font style="color:#000000;">不具备等价语义</font>**<font style="color:#000000;">。</font>
+ <font style="color:#000000;">保留这些参数不会带来收益，反而可能干扰 Ascend 后端的编译策略。移除后编译器可使用 Ascend 原生的分核和流水线策略。</font>

---

## <font style="color:#000000;">优化点 3：</font>`<font style="color:#000000;">multibuffer</font>`<font style="color:#000000;"> </font><font style="color:#000000;">编译参数（Double Buffer 优化）</font>
**<font style="color:#000000;">v1 新增</font>**<font style="color:#000000;">：每个 Config 都包含</font><font style="color:#000000;"> </font>`<font style="color:#000000;">multibuffer: True/False</font>`<font style="color:#000000;"> </font><font style="color:#000000;">两种变体。</font>

**<font style="color:#000000;">收益原因</font>**<font style="color:#000000;">：</font>

+ **<font style="color:#000000;">双缓冲（Double Buffer）</font>**<font style="color:#000000;">：开启</font><font style="color:#000000;"> </font>`<font style="color:#000000;">multibuffer</font>`<font style="color:#000000;"> </font><font style="color:#000000;">后，编译器在 UB（Unified Buffer）中分配双倍空间，当前数据块的计算与下一数据块的加载可以</font>**<font style="color:#000000;">重叠执行</font>**<font style="color:#000000;">，隐藏访存延迟。</font>
+ <font style="color:#000000;">本 kernel 有大量 </font>`<font style="color:#000000;">tl.load</font>`<font style="color:#000000;"> + </font>`<font style="color:#000000;">tl.dot</font>`<font style="color:#000000;"> 的循环模式，属于典型的 Cube 计算与 Vector 加载可并行场景，双缓冲带来的流水线并行收益显著。</font>

---

## <font style="color:#000000;">优化点 4：整数比较 → FP32 比较（向量比较优化 / Vector Compare）</font><font style="color:#DF2A3F;">（110->80）</font>
**<font style="color:#000000;">v0</font>**<font style="color:#000000;">：</font>

```python
m_t = o_t < T                              # int comparison
m_A = (o_t[:, None] > o_t[None, :]) & ...  # int comparison
```

**<font style="color:#000000;">v1</font>**<font style="color:#000000;">：</font>

```python
m_t = o_t.to(tl.float32) < T.to(tl.float32)                              # fp32 comparison
m_A = (o_t[:, None].to(tl.float32) > o_t[None, :].to(tl.float32)) & ...  # fp32 comparison
```

**<font style="color:#000000;">收益原因</font>**<font style="color:#000000;">：</font>

+ <font style="color:#000000;">Ascend NPU 的 Vector 处理器</font>**<font style="color:#000000;">不支持整数的向量比较指令</font>**<font style="color:#000000;">（仅支持 i32 的 EQ/NE 和浮点比较），</font>`<font style="color:#000000;">i64</font>`<font style="color:#000000;">/整数的大小比较会被</font>**<font style="color:#000000;">标量降级</font>**<font style="color:#000000;">（scalar lowering），逐元素串行执行，性能损失</font><font style="color:#000000;"> </font>**<font style="color:#000000;">10-100 倍</font>**<font style="color:#000000;">。</font>
+ <font style="color:#000000;">转为 fp32 后可使用 VPFCMP 指令进行向量化并行比较，一条指令处理整个向量块。</font>

---

## <font style="color:#000000;">优化点 5：冗余</font><font style="color:#000000;"> </font>`<font style="color:#000000;">>= 0</font>`<font style="color:#000000;"> </font><font style="color:#000000;">mask 消除（边界运算简化）</font>
**<font style="color:#000000;">v0</font>**<font style="color:#000000;">：</font>

```python
BK_mask = (BK_arange < K) & (BK_arange >= 0)
BV_mask = (BV_arange < V) & (BV_arange >= 0)
BT_mask = (BT_arange < T) & (BT_arange >= 0)
```

**<font style="color:#000000;">v1</font>**<font style="color:#000000;">：</font>

```python
BK_mask = BK_arange < K
BV_mask = BV_arange < V
BT_mask = BT_arange.to(tl.float32) < T.to(tl.float32)
```

**<font style="color:#000000;">收益原因</font>**<font style="color:#000000;">：</font>

+ `<font style="color:#000000;">tl.arange(0, BK)</font>`<font style="color:#000000;"> </font><font style="color:#000000;">产生的值</font>**<font style="color:#000000;">恒 ≥ 0</font>**<font style="color:#000000;">，</font>`<font style="color:#000000;">BK_arange >= 0</font>`<font style="color:#000000;"> </font><font style="color:#000000;">永远为 True，属于冗余条件。</font>
+ <font style="color:#000000;">冗余 mask 带来</font>**<font style="color:#000000;">两次额外开销</font>**<font style="color:#000000;">：(1) 多一次比较运算；(2)</font><font style="color:#000000;"> </font>`<font style="color:#000000;">&</font>`<font style="color:#000000;"> </font><font style="color:#000000;">操作将两个 mask 合并时产生额外的逐元素与运算。</font>
+ <font style="color:#000000;">移除后减少运算量，同时也减少了标量降级的风险（整数</font><font style="color:#000000;"> </font>`<font style="color:#000000;">&</font>`<font style="color:#000000;"> </font><font style="color:#000000;">操作可能在特定条件下降级）。</font>
+ <font style="color:#000000;">参考</font><font style="color:#000000;"> </font>`<font style="color:#000000;">latency-optimizer/references/redundant_boundary_operation.md</font>`<font style="color:#000000;">：KVR（Known-Value Region）分析框架——已知的恒真条件不应参与 mask 计算。</font>

---

## <font style="color:#000000;">优化点 6：</font>`<font style="color:#000000;">BT_mask_zero_offset</font>`<font style="color:#000000;"> </font><font style="color:#000000;">完全消除</font>
**<font style="color:#000000;">v0</font>**<font style="color:#000000;">：</font>

```python
BT_mask_zero_offset = (BT_arange_zero_offset < BT) & (BT_arange_zero_offset >= 0)
# 用于 b_A 加载: mask = BT_mask_zero_offset[:, None] & BT_mask[None, :]
```

**<font style="color:#000000;">v1</font>**<font style="color:#000000;">：</font>`<font style="color:#000000;">BT_mask_zero_offset</font>`<font style="color:#000000;"> </font><font style="color:#000000;">变量完全删除，</font>`<font style="color:#000000;">b_A</font>`<font style="color:#000000;"> </font><font style="color:#000000;">加载仅用</font><font style="color:#000000;"> </font>`<font style="color:#000000;">BT_mask[None, :]</font>`<font style="color:#000000;">。</font>

**<font style="color:#000000;">收益原因</font>**<font style="color:#000000;">：</font>

+ `<font style="color:#000000;">BT_arange_zero_offset = tl.arange(0, BT)</font>`<font style="color:#000000;">，而 BT 是</font><font style="color:#000000;"> </font>`<font style="color:#000000;">tl.constexpr</font>`<font style="color:#000000;">，所以</font><font style="color:#000000;"> </font>`<font style="color:#000000;">tl.arange(0, BT) < BT</font>`<font style="color:#000000;"> </font>**<font style="color:#000000;">恒为 True</font>**<font style="color:#000000;">。</font>
+ `<font style="color:#000000;">>= 0</font>`<font style="color:#000000;"> </font><font style="color:#000000;">同理恒真。整个</font><font style="color:#000000;"> </font>`<font style="color:#000000;">BT_mask_zero_offset</font>`<font style="color:#000000;"> </font><font style="color:#000000;">是全 True mask，与任何 mask 做</font><font style="color:#000000;"> </font>`<font style="color:#000000;">&</font>`<font style="color:#000000;"> </font><font style="color:#000000;">不改变结果。</font>
+ <font style="color:#000000;">消除后：(1) 减少一次比较 + 一次与运算；(2)</font><font style="color:#000000;"> </font>`<font style="color:#000000;">b_A</font>`<font style="color:#000000;"> </font><font style="color:#000000;">加载的 mask 更简单，编译器更容易优化。</font>

---

## <font style="color:#000000;">优化点 7：Load 指令重排序（Load Order Optimization）</font>
**<font style="color:#000000;">v0</font>**<font style="color:#000000;">（inner loop 内的加载顺序）：</font>

```python
# 先加载手动指针数据
b_v_new = tl.load(p_v_new, mask=...)  # manual ptr
b_do = tl.load(p_do, mask=...)        # manual ptr
# 再加载 block_ptr 数据
b_h = tl.load(p_h, ...)               # block_ptr
b_dh = tl.load(p_dh, ...)             # block_ptr
b_dv = tl.load(p_dv, mask=...)        # manual ptr
```

**<font style="color:#000000;">v1</font>**<font style="color:#000000;">：</font>

```python
# 先加载 block_ptr 数据（与上一轮 store 无依赖）
b_h = tl.load(p_h, ...)               # block_ptr
b_dh = tl.load(p_dh, ...)             # block_ptr
# 再加载手动指针数据
b_v_new = tl.load(p_v_new, mask=...)  # manual ptr
b_do = tl.load(p_do, mask=...)        # manual ptr
b_dv = tl.load(p_dv, mask=...)        # manual ptr
```

**<font style="color:#000000;">收益原因</font>**<font style="color:#000000;">：</font>

+ <font style="color:#000000;">内层循环</font><font style="color:#000000;"> </font>`<font style="color:#000000;">for i_v</font>`<font style="color:#000000;"> </font><font style="color:#000000;">中，</font>`<font style="color:#000000;">b_h</font>`<font style="color:#000000;">/</font>`<font style="color:#000000;">b_dh</font>`<font style="color:#000000;"> </font><font style="color:#000000;">的地址仅依赖外层变量</font><font style="color:#000000;"> </font>`<font style="color:#000000;">i_k</font>`<font style="color:#000000;">、</font>`<font style="color:#000000;">i_v</font>`<font style="color:#000000;">，与上一轮</font><font style="color:#000000;"> </font>`<font style="color:#000000;">i_v-1</font>`<font style="color:#000000;"> </font><font style="color:#000000;">的 store 操作</font>**<font style="color:#000000;">无 RAW（Read-After-Write）依赖</font>**<font style="color:#000000;">。</font>
+ <font style="color:#000000;">而</font><font style="color:#000000;"> </font>`<font style="color:#000000;">b_v_new</font>`<font style="color:#000000;">/</font>`<font style="color:#000000;">b_do</font>`<font style="color:#000000;"> </font><font style="color:#000000;">的地址虽然也不依赖上一轮的写，但 v0 中先加载它们会使</font><font style="color:#000000;"> </font>`<font style="color:#000000;">b_h</font>`<font style="color:#000000;">/</font>`<font style="color:#000000;">b_dh</font>`<font style="color:#000000;"> </font><font style="color:#000000;">的加载被延迟。</font>
+ **<font style="color:#000000;">将无依赖的 load 提前</font>**<font style="color:#000000;">，可以让它与上一轮的 </font>`<font style="color:#000000;">tl.store</font>`<font style="color:#000000;">（写 </font>`<font style="color:#000000;">dv2</font>`<font style="color:#000000;">）在硬件流水线上并行执行，隐藏访存延迟。</font>

---

## <font style="color:#000000;">优化点 8：</font>`<font style="color:#000000;">tl.where</font>`<font style="color:#000000;"> </font><font style="color:#000000;">消除 +</font><font style="color:#000000;"> </font>`<font style="color:#000000;">other</font>`<font style="color:#000000;"> </font><font style="color:#000000;">参数（冗余边界运算消除 / KVR）</font>
**<font style="color:#000000;">v0</font>**<font style="color:#000000;">：</font>

```python
b_v_new = tl.load(p_v_new, mask=BT_mask[:,None] & BV_mask[None, :])
b_v_new = tl.where(BT_mask[:,None] & BV_mask[None, :], b_v_new, 0)

b_do = tl.load(p_do, mask=BT_mask[:,None] & BV_mask[None, :])
b_do = tl.where(BT_mask[:,None] & BV_mask[None, :], b_do, 0)
```

**<font style="color:#000000;">v1</font>**<font style="color:#000000;">：</font>

```python
b_v_new = tl.load(p_v_new, mask=BT_mask[:,None] & BV_mask[None, :], other=0)
b_do = tl.load(p_do, mask=BT_mask[:,None] & BV_mask[None, :], other=0)
```

**<font style="color:#000000;">收益原因</font>**<font style="color:#000000;">：</font>

+ `<font style="color:#000000;">tl.load(..., mask=M, other=0)</font>`<font style="color:#000000;"> </font>**<font style="color:#000000;">已经保证</font>**<font style="color:#000000;">了</font><font style="color:#000000;"> </font>`<font style="color:#000000;">M=False</font>`<font style="color:#000000;"> </font><font style="color:#000000;">位置值为 0（KVR：Known-Value Region）。</font>
+ <font style="color:#000000;">随后再执行</font><font style="color:#000000;"> </font>`<font style="color:#000000;">tl.where(M, val, 0)</font>`<font style="color:#000000;"> </font><font style="color:#000000;">是</font>**<font style="color:#000000;">完全冗余的</font>**<font style="color:#000000;">——它把已知为 0 的区域再次设为 0。</font>
+ <font style="color:#000000;">消除 </font>`<font style="color:#000000;">tl.where</font>`<font style="color:#000000;"> 后：(1) 省去一次逐元素条件选择运算；(2) 减少一次 mask 计算和广播；(3) 减少寄存器压力。</font>

---

## <font style="color:#000000;">优化点 9：</font>`<font style="color:#000000;">b_gn</font>`<font style="color:#000000;"> 加载位置调整 + </font>`<font style="color:#000000;">tl.where</font>`<font style="color:#000000;"> 消除</font><font style="color:#DF2A3F;">（80->70）</font>
**<font style="color:#000000;">v0</font>**<font style="color:#000000;">（</font>`<font style="color:#000000;">b_gn</font>`<font style="color:#000000;"> </font><font style="color:#000000;">在</font><font style="color:#000000;"> </font>`<font style="color:#000000;">for i_v</font>`<font style="color:#000000;"> </font><font style="color:#000000;">循环之前加载）：</font>

```python
# 在 for i_k 内部、for i_v 之前
p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * H*K + o_k
b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)
b_gn = tl.where(m_k, b_gn, 0)  # 冗余 tl.where

for i_v in range(tl.cdiv(V, BV)):
    ...  # b_gn 未在内层循环中使用
```

**<font style="color:#000000;">v1</font>**<font style="color:#000000;">（</font>`<font style="color:#000000;">b_gn</font>`<font style="color:#000000;"> </font><font style="color:#000000;">在</font><font style="color:#000000;"> </font>`<font style="color:#000000;">for i_v</font>`<font style="color:#000000;"> </font><font style="color:#000000;">循环之后加载）：</font>

```python
for i_v in range(tl.cdiv(V, BV)):
    ...  # 内层循环不涉及 b_gn

# 在 for i_k 内部、for i_v 之后
p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * H*K + o_k
b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)
# 无冗余 tl.where
```

**<font style="color:#000000;">收益原因</font>**<font style="color:#000000;">：</font>

+ **<font style="color:#000000;">位置调整（延迟加载）</font>**<font style="color:#000000;">：</font>`<font style="color:#000000;">b_gn</font>`<font style="color:#000000;"> </font><font style="color:#000000;">仅在</font><font style="color:#000000;"> </font>`<font style="color:#000000;">for i_v</font>`<font style="color:#000000;"> </font><font style="color:#000000;">循环之后的 exp2 计算中使用，提前加载会</font>**<font style="color:#000000;">占据 UB/寄存器空间</font>**<font style="color:#000000;">长达整个内层循环。移到使用点附近，减少 UB 压力，为</font><font style="color:#000000;"> </font>`<font style="color:#000000;">multibuffer</font>`<font style="color:#000000;"> </font><font style="color:#000000;">双缓冲腾出空间。</font>
+ `**<font style="color:#000000;">tl.where</font>**`**<font style="color:#000000;"> </font>****<font style="color:#000000;">消除</font>**<font style="color:#000000;">：同优化点 8，</font>`<font style="color:#000000;">tl.load(..., mask=m_k, other=0)</font>`<font style="color:#000000;"> </font><font style="color:#000000;">已建立 KVR（m_k=False 区域值为 0），</font>`<font style="color:#000000;">tl.where(m_k, b_gn, 0)</font>`<font style="color:#000000;"> </font><font style="color:#000000;">完全冗余。</font>

---

## <font style="color:#000000;">优化点 10：</font>`<font style="color:#000000;">b_dv</font>`<font style="color:#000000;"> </font><font style="color:#000000;">加载添加</font><font style="color:#000000;"> </font>`<font style="color:#000000;">other=0</font>`
**<font style="color:#000000;">v0</font>**<font style="color:#000000;">：</font>

```python
b_dv = tl.load(p_dv, mask=BT_mask[:,None] & BV_mask[None, :])
```

**<font style="color:#000000;">v1</font>**<font style="color:#000000;">：</font>

```python
b_dv = tl.load(p_dv, mask=BT_mask[:,None] & BV_mask[None, :], other=0)
```

**<font style="color:#000000;">收益原因</font>**<font style="color:#000000;">：</font>

+ <font style="color:#000000;">v0 中未指定</font><font style="color:#000000;"> </font>`<font style="color:#000000;">other</font>`<font style="color:#000000;">，mask 外的值</font>**<font style="color:#000000;">未定义</font>**<font style="color:#000000;">（可能是 UB 中的残留数据），后续</font><font style="color:#000000;"> </font>`<font style="color:#000000;">b_dw += tl.dot(b_dv.to(...), b_h.to(...))</font>`<font style="color:#000000;"> </font><font style="color:#000000;">会将未定义值卷入累加器，导致精度问题。</font>
+ <font style="color:#000000;">添加</font><font style="color:#000000;"> </font>`<font style="color:#000000;">other=0</font>`<font style="color:#000000;"> </font><font style="color:#000000;">后，边界区域值为确定的 0，不参与 dot 乘积累加，</font>**<font style="color:#000000;">保证数值正确性</font>**<font style="color:#000000;">的同时也让编译器可以</font>**<font style="color:#000000;">跳过 mask 外区域的计算</font>**<font style="color:#000000;">（dead code elimination）。</font>

---

## <font style="color:#000000;">优化点 11：</font>`<font style="color:#000000;">reset_to_zero</font>`<font style="color:#000000;"> 声明（累加器正确初始化）</font>
**<font style="color:rgb(204, 204, 204);">v1 新增</font>**<font style="color:rgb(204, 204, 204);">：</font>

```python
reset_to_zero=['dq', 'dk', 'dv2', 'db', 'dg', 'dA'],
```

**收益原因**：

+ `dq`、`dk`、`dg` 等输出张量在 kernel 内部通过 `b_dq = tl.zeros(...)` 初始化，但多个 program 可能写入同一全局内存区域（尤其是多 shape 场景下），**autotune 重试不同 Config 时未清零会导致残留数据污染**。
+ `reset_to_zero` 确保 autotune 切换 Config 时自动将这些 buffer 清零，保证**不同 Config 之间的结果独立、正确**，避免 autotune 选出"看起来快但结果错误"的配置。

# chunk_kda_bwd_kernel_intra优化（50ms->20ms）
## 优化 1：NT 维度并行分裂（NSPLIT=3）
**v0**：所有 NT 个 chunk 在同一个 program 内串行循环处理

```python
# v0: chunk_kda_bwd_intra
for i_t_chunk in range(NT):    # 一个 program 处理全部 NT 个 chunk
```

**v1**：引入 `NSPLIT=3`，将 NT 个 chunk 均分到多个 program 实例并行执行

```python
# v1: chunk_kda_bwd_kernel_intra_glm
NT_per_split = tl.cdiv(NT, NSPLIT)
NT_start = i_split * NT_per_split
NT_end = min((i_split + 1) * NT_per_split, NT)
for i_t_chunk in range(NT_start, NT_end):   # 每个 program 只处理约 NT/3 个 chunk
```

Grid 维度变化：

+ v0: `(NK * NC, B * H)`
+ v1: `(NK * NC * 3, B * H)` — program 数量 3×

新增 constexpr 参数：

+ `NSPLIT`: 分裂数（host 端设为 3）
+ `NK_NC_CONST`: 预计算的 `NK * NC`，用于编译期确定 program ID → `(i_split, i_k, i_i)` 的映射

```python
# v1 program ID 解码
i_split = i_kc // NK_NC
i_kc_inner = i_kc - i_split * NK_NC
i_k, i_i = i_kc_inner // NC, i_kc_inner - (i_kc_inner // NC) * NC
```

**收益**：长序列场景下，v0 单 program 串行处理所有 chunk 是瓶颈；v1 将 workload 分散到 3 倍 program 上，在128k case下仅使用8个计算核心，因此这里切分成3块，将所有core都用上。

---

## 优化 2：BC 子块大小从 32 缩小到 16
```python
# v0
BC = min(32, BT)
# v1
BC = min(16, BT)
```

**效果**：

+ NC 翻倍，可以多用一倍的核心

---

## 优化 3：Load 指令重排序（ILP / 延迟隐藏）
### 下三角部分（i_i > 0 分支内的循环）
**v0**：load k → load gk → compute b_kg → load dA（compute 阻塞后续 load）

```python
# v0
b_k = tl.load(p_k)                # load 1
b_gk = tl.load(p_gk)              # load 2
b_kg = b_k * exp2(b_gn - b_gk)    # compute（依赖 load 1,2 结果）← 阻塞
b_dAqk = tl.load(p_dAqk)          # load 3（被 compute 阻塞）
b_dAkk = tl.load(p_dAkk)          # load 4
```

**v1**：先发射全部独立 load，再执行依赖 compute

```python
# v1
b_k = tl.load(p_k)                # load 1
b_dAqk = tl.load(p_dAqk)          # load 2（与 load 1 独立，可并行发射）
b_dAkk = tl.load(p_dAkk)          # load 3（与 load 1,2 独立）
b_gk = tl.load(p_gk)              # load 4
b_kg = b_k * exp2(b_gn - b_gk)    # compute（在 4 个 load 均已发射后执行）
```

### 上三角部分（i_i < NC_local - 1 分支内的循环）
**v0**：`b_b → b_q → b_kb(b_k*b_b) → b_gk → b_dAqk → b_dAkk`  
**v1**：`b_q → b_dAqk → b_dAkk → b_b → b_kb(b_k*b_b) → b_gk`

v1 将 `b_q`, `b_dAqk`, `b_dAkk` 三个独立 load 提前，避免 `b_kb` 计算阻塞后续 load 发射。

**收益**：增加 memory pipeline 中 in-flight 请求数，减少 load-compute bubble，提升内存带宽利用率。

---
