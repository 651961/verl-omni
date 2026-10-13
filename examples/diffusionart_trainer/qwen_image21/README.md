# Qwen-Image 2.1 DiffusionART

集成入口为 `run_qwen_image21_async.sh`，用 `ART_MODEL_PATH` 指定 checkpoint。
这是新增实现，尚未完成真实模型和 8 卡 FSDP／多节点 HSDP／NCCL 运行验证，不提供实测吞吐承诺。
运行环境必须包含提供 `QwenImage21Pipeline`、
`QwenImage21Transformer2DModel`、`AutoencoderKLQwenImage21` 的 Diffusers 和 Qwen3VL Transformers，
以及提供原生 `vllm_omni.diffusion.models.qwen_image_21` 的 vLLM-Omni、verl、TransferQueue 和 delta 后端依赖。
当前 pyproject 固定的旧 vLLM-Omni 版本不包含该模型，运行本例须使用含原生模型的版本。

```bash
ART_MODEL_PATH=/path/to/Qwen-Image-2.1 \
TRAIN_FILE=/data/train.parquet VAL_FILE=/data/val.parquet \
REWARD_FUNCTION_PATH=/path/to/reward.py REWARD_FUNCTION_NAME=compute_score \
bash examples/diffusionart_trainer/qwen_image21/run_qwen_image21_async.sh
```

默认使用 **8 张训练卡 + 2 张独立 rollout 卡**。训练固定每节点 8 卡、
`fsdp_size=8`、SP=1：单节点为 FSDP，多节点为 HSDP，跨节点复制、节点内分片。
`ACTOR_GPUS` 表示训练总卡数，16／32 会自动对应 2／4 个训练节点；
也可以用 `ACTOR_NNODES` 设置节点数。`ROLLOUT_GPUS` 表示 rollout 总卡数，
超过 8 时按每节点 8 卡拆分，可用 `ROLLOUT_GPUS_PER_NODE` 和 `ROLLOUT_NNODES` 显式指定。
两种 pool 的总卡数必须与节点数乘以每节点卡数一致。
reward GPU 需要另行计入总预算；本脚本默认调用你的 reward function，
该函数可以访问本地 GPU LLM 服务。异步重叠需要独立的 rollout GPU。本 recipe 的 rollout 使用 TP=SP=CFG-parallel=1，
多个独立副本和原生 batch 提供并发。训练使用微批次 1 和逐步 CPU 数据搬运，避免同时将完整轨迹放进显存。
`torch.compile` 和延迟梯度同步关闭。
FSDP2 默认开启下一层参数的前向预取，以重叠参数通信与计算；这会增加临时显存占用。
可在启动命令末尾添加 `actor_rollout_ref.actor.fsdp_config.forward_prefetch=false` 关闭。

例如，2 个训练节点与 1 个 rollout 节点（合计 24 卡，不含 reward GPU）：

```bash
ACTOR_GPUS=16 ROLLOUT_GPUS=8 \
ART_MODEL_PATH=/path/to/Qwen-Image-2.1 \
TRAIN_FILE=/data/train.parquet VAL_FILE=/data/val.parquet \
REWARD_FUNCTION_PATH=/path/to/reward.py REWARD_FUNCTION_NAME=compute_score \
bash examples/diffusionart_trainer/qwen_image21/run_qwen_image21_async.sh
```

Ray 负责 worker 调度和 RPC；本 recipe 使用 TransferQueue 的 SimpleStorage（ZMQ 内存存储）
保存待训练样本。训练队列在异步打分完成后只保留 `uid`、`rm_scores`、`art_cache` 和少量统计，
不再保存图像、prompt token、attention mask 或原始数据集字段；
配置 `trainer.rollout_data_dir` 时保留图像及 dump 所需字段。尚未异步打分的样本保留 reward 所需输入。
`art_cache` 独立于 `extra_fields`，统计读取不拉取轨迹。组内完全相同的条件编码、
参考 latent 和时间表只在首行保存一次，读取完整组时共享引用恢复；不同条件不合并。
验证不采集训练轨迹，保留真实图像用于打分和展示，默认每次验证读取 8 个 prompt，
可用 `VAL_PROMPTS_PER_BATCH` 和 `VAL_GROUP_SIZE` 调整，避免一次生成整个验证集。

存储单元默认按 actor 和 rollout 的节点数之和设置（默认 2 个），通过 `TQ_STORAGE_UNITS` 调整。
单元分片存储数据，不表示多份数据复制。`TQ_STORAGE_SIZE` 默认只容纳几批活动样本
（默认配置 320 个槽位，包含 prompt 控制行与验证余量），不是预留的内存字节数；
提高 warmup、并发在途批数或验证组大小时应同步提高容量。
每步统计／可选 dump 完成后清理训练行，采样时清理 prompt 控制行，过期／失败组沿用 replay buffer 清理。

轨迹采集直接写入最终 FP32 CPU tensor，避免快照列表与 `stack` 结果共存，
条件 hook 只复制参考 latent，不额外复制目标 latent。
actor 先裁掉其他样本引入的 padding，再按时间步传入 GPU。
必要的 states 和 old velocities 仍保留 FP32：1024²、28 步约 56 MiB／样本，
这是两类张量的理论大小，尚不含 endpoint／条件／通信临时内存；不通过降精度改变 ART 数值契约。

配置 `reward.models` 的本地 reward 模型若要流式打分，需要
`reward.reward_model.enable_resource_pool=true`，并为所有 named models 设置 `offload=false`。
新路由会让每个样本经过全部 reward term 分组，并合并分数；每个 worker 同时最多接收一个请求，
避免随机选择单个分组漏算 reward、重复补齐样本或逐样本唤醒／休眠模型。
共享 GPU 或 `offload=true` 的模型仍走整批生命周期，不与 actor 并发使用同一显存。

权重同步返回父级统计，ART MFU 按 Qwen 2.1 的单流 SwiGLU 和 block-causal 布局、
实际参考图／文本长度、所选训练步和 CFG 分支估算。MFU 不包含 reward、VAE、文本编码和 checkpoint 重算。

Qwen-Image 2.1 按官方预期**不开 CFG**：训练和验证 recipe 均为 `true_cfg_scale=1.0`，
rollout 不编码负条件，actor 不做负条件前向。可参考本地 Diffusers 的
`src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py` 中参数说明。

rollout adapter 继承 vLLM-Omni 原生 `QwenImage21Pipeline`，复用请求批处理、
逐步执行、CFG、KV 和权重加载。默认 `step_execution=true`、`max_num_seqs=4`，
可用 `ROLLOUT_STEP_EXECUTION`、`ROLLOUT_MAX_NUM_SEQS` 和 `ROLLOUT_ENFORCE_EAGER` 调整。
引擎按尺寸及条件布局组织 batch；不同 prompt 组和同组样本均沿用原有负载均衡。
每个样本生成完成后立即进入 reward，不等待同组其他样本生成完成。
等整组 reward 返回后才计算 ART covariance。

KV 只使用原生实现：每个请求第一次去噪建立 prefix KV，后续去噪步骤读取。
当前原生代码没有同 prompt 跨请求共享 prefix 的查找接口，本集成不补充该能力，
不固定 group 到一个 replica，也没有自建条件缓存、LRU、内容 hash 或缓存预算配置。
actor replay 不保存或读取 rollout KV，但保留 prefill 补齐后的条件长度，以匹配 RoPE。
step 模式下条件布局必须在同一轨迹内保持不变；关闭 prefix KV 后若布局变化会明确报错。

agent loop 使用 checkpoint 的 Qwen3-VL processor 和原生 prompt 模板完成分词与视觉预处理，
发送 token IDs、attention mask 和视觉输入；rollout 直接编码这些输入，不重新分词。
RGBA 条件先按原生尺寸规则缩放，VLM 使用白底 RGB，VAE 保留 RGBA。

actor 使用单样本微批次（`ppo_micro_batch_size_per_gpu=1`），支持不同分辨率、
参考图数量和文本长度；每条样本按其原始布局独立执行前向和反向。

数据使用普通 RL parquet 字段 `prompt`、`data_source`、`reward_model`、`extra_info`。
`prompt` 支持字符串或消息列表。编辑数据在文本中用 `<image>` 引用 `images` 列，
也可直接使用结构化 `{"type": "image", "image": ...}` 内容块。
图像支持本地路径、PIL 或 parquet 的 bytes/path 记录；原始 alpha 保留给 VAE。
文本编码器对 alpha 的处理遵循官方 Qwen3VL pipeline。
`negative_prompt` 可保留在数据里；默认无 CFG 路径不使用它。

每条数据可在 `extra_info` 中设置 `height`、`width`、`output_resolution`。
不同 prompt 可混合分辨率和参考图数量，同一 prompt 组的目标布局必须相同。
样本的 `output_resolution` 覆盖全局默认宽高；样本显式的 `height` / `width` 优先级最高。
整批奖励在打分前根据 `response_shape` 裁掉传输补齐，保留原始 RGBA 尺寸。
原生 pipeline 将目标宽高向下对齐到 32 的倍数；未指定尺寸时，T2I 默认 1024×1024，
编辑按最后一张参考图的宽高比确定画布。输出保留 RGBA 四通道。
奖励函数收到原始尺寸的单样本图像，应自行决定是否合成白底，不能假定只有 RGB 三通道。

精度策略：FP32 主参数、FP32 梯度累积/同步、`torch.optim.AdamW` 的 FP32 状态，
FSDP gathered 参数和计算用 BF16。首次发布与后续 delta 均导出 BF16；
delta 对比已发布副本，FP32 中未跨越 BF16 舍入边界的小更新继续保留。
变化比例由实际 BF16 发布结果决定，不假设固定 1%–2%；
`CKPT_BACKEND=nccl` 可运行相同配置的全量同步对照。

ART 按完整 prompt 组先计算标量 reward / endpoint covariance，然后分卡。
默认采用源实现的 `velocity_linear_sigma` 损失、RMS 方向、scheduler-reference corrective mask，
精确处理常量 reward 的零信号。没有 EMA 模型或旧策略训练前向。
旧速度直接保存实际 rollout 的无 CFG 预测，同时保存选定 ODE 状态、sigma、dt、
prompt 条件、参考图 latent 和最终 endpoint。**ART 需要传输这些缓存**，其传输成本应计入吞吐。
`ART_STEP_INDICES='[0,4,8,12]'` 可以只训练明确选择的步骤；默认训练所有步骤。
调整此项会改变训练目标的采样，不是无损的系统加速。

默认**不加载**独立的冻结 base/reference；这与官方 DiffusionART 主目标一致。
如确实需要额外的速度 MSE 正则化，可显式设置 `USE_REFERENCE=true`，并用
`REFERENCE_MSE_COEF` 设置系数；这不是 PPO log-prob KL。开启后会额外占用一个
reference 模型及其一次无梯度前向。
每个选中步骤、每条样本：actor 1 次前向 + 1 次反向，reference 开启时另加 1 次无梯度前向。
actor 和 reference 都逐条执行；通用代码保留显式 CFG 兼容分支，本 recipe 使用无 CFG。
梯度 checkpointing 的重计算还会额外执行 block 前向。

`separate_async` 保持训练与独立 rollout 重叠，只在发布阶段协调权重更新。
模型版本在完整权重流校验/应用完成后提交，并在实际生成时写入缓存。
每条样本保留实际生成版本，沿用 replay-buffer 的 staleness 策略处理异步滞后，
不因组内版本不同重新生成整组。默认允许有限滞后，不能当作严格 on-policy。

创建 venv 后先运行核心测试与配置检查，再按独立 actor/rollout pool 的资源要求短跑：

```bash
python tests/trainer/diffusion/test_art_core_on_cpu.py
python tests/trainer/diffusion/test_art_wiring_on_cpu.py
python tests/pipelines/test_qwen_image21_diffusion_art_on_cpu.py
python -m unittest discover -s tests/trainer/diffusion -p 'test_art_*_on_cpu.py'
python -m unittest discover -s tests/utils -p 'test_qwen_image21_mfu_on_cpu.py'
python -m unittest discover -s tests/reward_loop -p 'test_streaming_reward_router_on_cpu.py'
# 使用与上面相同的数据/奖励环境变量检查 Hydra 配置
bash examples/diffusionart_trainer/qwen_image21/run_qwen_image21_async.sh --cfg job
# 对真实模型做两步集成验证；如需快速验证可显式减少 ART_STEP_INDICES
TRAIN_STEPS=2 bash examples/diffusionart_trainer/qwen_image21/run_qwen_image21_async.sh
# 单步 smoke 入口，沿用每节点 8 卡的 actor pool 与独立 rollout pool
bash tests/gpu_smoke/run_qwen_image21_diffusion_art.sh
```

集成验证应包含 T2I、RGB 编辑、RGBA 编辑和混合宽高比，检查 reference 开/关、
FP32 optimizer checkpoint 恢复，以及 NCCL 全量发布和 delta 发布的权重一致性。
对比 step、sample wait、weight sync、轨迹传输耗时后，再判断异步与 delta 的实际收益。

CPU 测试覆盖数学、真实小型 diffusers DiT 的 replay 梯度、TQ 数据边界，
以及本地 vLLM-Omni 源码的 full-forward/step 调用与权重名称映射。
组件与分布式运行时使用测试替身；这些结果不能替代实际 GPU 引擎、NCCL/delta 发布和 HSDP 验证。
