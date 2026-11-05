# MiniMind 从零学习计划

> 适合对象：会基础 Python，但不是大模型从业者，希望通过 MiniMind 理解小型语言模型的完整训练流程。  
> 建议周期：8～10 周，每周 5 天，每天 1～2 小时。  
> 当前设备：NVIDIA GeForce RTX 5060 Laptop GPU（8GB 显存）。

## 一、总目标

完成本计划后，应该能够：

1. 解释语言模型为什么可以通过“预测下一个 token”生成文本。
2. 说清楚 Tokenizer、Embedding、Attention、MLP、Logits 和交叉熵的作用。
3. 看懂 MiniMind 的数据加载、模型前向传播和训练循环。
4. 独立完成一次小规模 `Pretrain → SFT → Eval` 闭环。
5. 理解 LoRA、DPO、知识蒸馏和强化学习分别解决什么问题。
6. 能够加载、转换和部署自己训练的模型。
7. 知道小模型的能力边界，不把 loss 下降等同于真实能力提升。

## 二、学习原则

### 1. 先主线，后扩展

第一轮只关注：

```text
Tokenizer → Pretrain → Full SFT → Eval
```

完成主线后，再学习：

```text
LoRA → DPO → Distillation → GRPO/PPO → Agentic RL
```

### 2. 每个知识点都完成三件事

1. 用自己的语言解释它。
2. 在代码中找到它。
3. 运行或修改一个最小实验验证它。

### 3. 不以模型回答质量作为唯一标准

MiniMind 主线模型约 64M 参数，主要价值是教学和实验。学习时重点观察：

- 数据格式是否正确；
- 张量形状是否正确；
- loss 是否能计算和下降；
- 权重能否保存和重新加载；
- 不同训练阶段是否产生可解释的行为变化。

## 三、开始前的准备

### 已完成

- 项目已经克隆到本地。
- 已创建 `.venv` Python 3.10 环境。
- 已安装 PyTorch CUDA 和项目依赖。
- 已下载 Transformers 格式的 `minimind-3` 模型。
- CUDA 已识别 RTX 5060 Laptop GPU。

### 每次学习前激活环境

```powershell
.venv\Scripts\Activate.ps1
```

如果 PowerShell 禁止执行激活脚本，也可以始终使用完整路径：

```powershell
.venv\Scripts\python.exe
```

### 验证环境

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期结果：

- `torch.cuda.is_available()` 为 `True`；
- GPU 名称为 RTX 5060 Laptop GPU。

---

## 阶段 0：建立大模型全局认识

建议时间：2～3 天。

### 阶段目标

- 知道语言模型并不是搜索答案，而是在预测下一个 token。
- 能说清楚训练和推理的区别。
- 理解 Pretrain、SFT、LoRA、DPO、RL 的大致位置。

### 必学概念

- 文本、字符、token、token id；
- 参数、权重、梯度；
- 训练集和验证集；
- 前向传播、loss、反向传播、优化器；
- 基座模型和指令模型；
- 训练、微调和推理。

### 学习任务

1. 阅读 `README.md` 的“项目介绍”“快速开始”“数据介绍”和“模型结构”。
2. 用一张纸画出：

```text
文本
  ↓ Tokenizer
token id
  ↓ Transformer
下一个 token 的概率
  ↓ 采样
生成文本
```

3. 用自己的语言回答：
   - 模型为什么会产生幻觉？
   - 参数量代表什么？
   - 训练为什么需要大量数据？

### 阶段产出

写一页学习笔记，标题为“我理解的语言模型工作过程”。

### 验收标准

不看资料，能够在 5 分钟内向别人解释：

> 语言模型如何从输入文本生成下一个词。

---

## 阶段 1：运行和观察现成模型

建议时间：3～4 天。

### 阶段目标

- 成功运行 CLI 推理。
- 理解 Transformers 模型目录中的主要文件。
- 观察温度、随机采样和提示词对输出的影响。

### 学习文件

- `eval_llm.py`
- `minimind-3/config.json`
- `minimind-3/generation_config.json`
- `minimind-3/tokenizer_config.json`

### 运行模型

```powershell
python eval_llm.py --load_from ./minimind-3
```

### 建议测试问题

准备一份固定问题集，每次模型变化后重复测试：

1. `请介绍一下你自己。`
2. `为什么天空是蓝色的？`
3. `计算 17 × 23。`
4. `用三句话总结什么是机器学习。`
5. `写一个 Python 二分查找函数。`
6. `如果不知道答案，请明确说不知道：中国第一位登上月球的人是谁？`

### 代码阅读重点

在 `eval_llm.py` 中找到：

- Tokenizer 如何加载；
- Transformers 模型如何加载；
- 对话模板如何应用；
- prompt 如何变成 tensor；
- `model.generate()` 如何调用；
- 生成结果如何 decode。

### 小实验

分别修改或传入不同参数，观察输出：

- `temperature`；
- `top_p`；
- `max_new_tokens`；
- `open_thinking`。

每个参数至少重复生成 3 次，不要根据单次输出下结论。

### 阶段产出

记录一份“推理参数实验”：

- 使用的提示词；
- 参数设置；
- 输出差异；
- 你的解释。

### 验收标准

- 能独立启动和退出推理程序；
- 能解释为什么相同问题可能产生不同回答；
- 能说清楚 Transformers 格式目录中配置、Tokenizer 和权重文件的作用。

---

## 阶段 2：学习 Tokenizer 和训练数据

建议时间：1 周。

### 阶段目标

- 理解文本如何转换为 token id。
- 理解 padding、truncation、BOS、EOS 和 attention mask。
- 看懂 Pretrain、SFT、DPO 三种数据格式的差异。

### 学习文件

- `model/tokenizer.json`
- `model/tokenizer_config.json`
- `dataset/lm_dataset.py`
- `trainer/train_tokenizer.py`（只阅读，暂时不要重训）

### Tokenizer 实验

在 Python 交互环境中运行：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./model")
text = "你好，MiniMind！Transformer is useful."
ids = tokenizer.encode(text)

print(ids)
print(tokenizer.convert_ids_to_tokens(ids))
print(tokenizer.decode(ids))
```

测试以下文本：

- 常见中文；
- 生僻中文；
- 英文；
- 数字和公式；
- 中英文混合文本；
- 一段代码。

### 数据集阅读顺序

1. `PretrainDataset`
2. `SFTDataset`
3. `DPODataset`
4. `RLAIFDataset`
5. `AgentRLDataset`

第一轮只需要吃透前两个。

### 必须理解的问题

#### PretrainDataset

- 为什么文本前后要添加 BOS 和 EOS？
- 为什么 padding 对应的 label 被设为 `-100`？
- `input_ids` 和 `labels` 为什么看起来几乎相同？

#### SFTDataset

- `apply_chat_template()` 做了什么？
- system、user、assistant、tool 分别是什么角色？
- 为什么通常只让 assistant 部分参与 loss？
- 如果所有 token 都参与 loss，会出现什么问题？

### 数据格式

预训练数据：

```json
{"text": "一段用于预测下一个 token 的文本。"}
```

SFT 数据：

```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么可以帮助你的？"}
  ]
}
```

DPO 数据：

```json
{
  "chosen": [
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "较好的回答"}
  ],
  "rejected": [
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "较差的回答"}
  ]
}
```

### 阶段产出

画出一条 SFT 样本经过下面步骤后的变化：

```text
JSON 消息 → Chat Template → Tokenizer → input_ids → labels → loss mask
```

### 验收标准

- 给出一条对话数据，能够指出哪些 token 应参与 SFT loss；
- 能解释 `-100` 的作用；
- 能解释预训练数据和 SFT 数据解决的问题有什么不同。

---

## 阶段 3：吃透 MiniMind 模型结构

建议时间：1～2 周。

### 阶段目标

- 看懂 Decoder-Only Transformer 的前向传播。
- 能跟踪主要张量形状。
- 理解模型如何从 hidden states 得到词表概率。

### 学习文件

主要阅读：

- `model/model_minimind.py`

辅助阅读：

- `minimind-3/config.json`

### 推荐阅读顺序

1. `MiniMindConfig`
2. `RMSNorm`
3. `precompute_freqs_cis`
4. `apply_rotary_pos_emb`
5. `repeat_kv`
6. `Attention`
7. `FeedForward`
8. `MiniMindBlock`
9. `MiniMindModel`
10. `MiniMindForCausalLM`
11. `generate`

第一轮跳过 `MOEFeedForward`，等 Dense 模型完全理解后再回来。

### 每个模块需要回答的问题

#### Embedding

- token id 如何变成向量？
- 输入形状如何从 `[batch, sequence]` 变成 `[batch, sequence, hidden]`？

#### Attention

- Q、K、V 分别从哪里来？
- 为什么要除以 `sqrt(head_dim)`？
- causal mask 为什么能防止模型看到未来 token？
- GQA 中 Q heads 与 KV heads 为什么数量不同？

#### RoPE

- 为什么模型需要位置信息？
- RoPE 为什么应用在 Q 和 K 上？

#### FeedForward

- SwiGLU 与普通激活层的区别是什么？
- MLP 为什么通常比 Attention 占用更多参数？

#### Residual 和 RMSNorm

- 残差连接为什么有助于训练深层网络？
- Pre-Norm 表示归一化发生在哪里？

#### LM Head 和 Loss

- hidden states 如何映射到 6400 维词表？
- 为什么 logits 需要错一位与 labels 比较？
- 交叉熵在鼓励模型做什么？

### 张量形状练习

假设：

```text
batch_size = 2
sequence_length = 128
hidden_size = 768
num_attention_heads = 8
num_key_value_heads = 4
head_dim = 96
vocab_size = 6400
```

手工写出以下张量形状：

- `input_ids`
- embedding 输出
- Q、K、V
- attention scores
- Attention 输出
- logits
- labels

### 动手实验

写一个小脚本，只创建随机输入并执行一次 forward：

```python
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

config = MiniMindConfig(hidden_size=256, num_hidden_layers=4)
model = MiniMindForCausalLM(config)
input_ids = torch.randint(0, config.vocab_size, (2, 32))
labels = input_ids.clone()

output = model(input_ids, labels=labels)
print(output.logits.shape)
print(output.loss)
```

### 阶段产出

画一张 MiniMind Block 结构图，并为每一步标注输入、输出形状。

### 验收标准

- 能从 `input_ids` 一直解释到 `loss`；
- 能说明 Attention、MLP、残差和归一化各自作用；
- 修改 batch size 或 sequence length 后，能预测主要张量形状。

---

## 阶段 4：理解预训练代码

建议时间：1 周。

### 阶段目标

- 看懂一个完整的 PyTorch 训练循环。
- 理解梯度累积、混合精度、梯度裁剪和学习率调度。
- 完成小规模预训练冒烟测试。

### 学习文件

- `trainer/train_pretrain.py`
- `trainer/trainer_utils.py`

### 阅读主线

按照下面顺序在代码中定位：

```text
解析参数
  ↓
初始化随机种子和设备
  ↓
创建模型和 Tokenizer
  ↓
创建 Dataset 和 DataLoader
  ↓
前向传播
  ↓
计算 loss
  ↓
backward
  ↓
梯度裁剪和 optimizer.step
  ↓
保存权重和 checkpoint
```

### 必须理解的训练参数

- `epochs`
- `batch_size`
- `learning_rate`
- `max_seq_len`
- `accumulation_steps`
- `grad_clip`
- `save_interval`
- `hidden_size`
- `num_hidden_layers`
- `dtype`
- `from_weight`
- `from_resume`

### 8GB 显存的首次建议配置

不要直接使用默认参数做第一次实验。先运行一个小模型：

```powershell
cd trainer
python train_pretrain.py `
  --hidden_size 256 `
  --num_hidden_layers 4 `
  --max_seq_len 128 `
  --batch_size 2 `
  --accumulation_steps 8 `
  --epochs 1 `
  --num_workers 0 `
  --log_interval 10 `
  --save_interval 100
```

说明：

- `num_workers=0` 在 Windows 上更容易排查 DataLoader 问题；
- 首次目标只是跑通，不是得到高质量模型；
- 如果显存充足，再逐步增加 `batch_size` 或 `max_seq_len`；
- 如果出现 OOM，优先减小 `batch_size`，然后减小 `max_seq_len`。

### 数据准备

完整学习前下载：

- `dataset/pretrain_t2t_mini.jsonl`
- `dataset/sft_t2t_mini.jsonl`

第一次冒烟测试可以从 mini 数据中截取少量样本：

```powershell
python -c "import itertools; src='dataset/pretrain_t2t_mini.jsonl'; dst='dataset/pretrain_tiny.jsonl'; f=open(src,encoding='utf-8'); o=open(dst,'w',encoding='utf-8'); o.writelines(itertools.islice(f,2000)); f.close(); o.close()"
```

然后增加：

```powershell
--data_path ../dataset/pretrain_tiny.jsonl
```

### 观察指标

- 初始 loss；
- loss 是否总体下降；
- GPU 显存占用；
- 每 step 时间；
- 学习率变化；
- 是否生成 `out/pretrain_256.pth`；
- 是否生成可恢复的 checkpoint。

### 阶段产出

保存一份训练记录：

```text
模型配置：
数据量：
训练时长：
显存峰值：
初始 loss：
最终 loss：
遇到的问题：
解决方式：
```

### 验收标准

- 能独立解释一次 `optimizer.step()` 前发生了什么；
- 能成功训练至少 100 steps；
- 能从 checkpoint 恢复训练；
- 能解释为什么 loss 降低不等于模型一定能正确回答问题。

---

## 阶段 5：完成 SFT 和完整闭环

建议时间：1 周。

### 阶段目标

- 理解 SFT 如何让基座模型学会对话和遵循指令。
- 完成 `Pretrain → SFT → Eval`。
- 比较预训练权重和 SFT 权重的行为差异。

### 学习文件

- `trainer/train_full_sft.py`
- `dataset/lm_dataset.py` 中的 `SFTDataset`
- `eval_llm.py`

### 准备小型 SFT 数据

```powershell
python -c "import itertools; src='dataset/sft_t2t_mini.jsonl'; dst='dataset/sft_tiny.jsonl'; f=open(src,encoding='utf-8'); o=open(dst,'w',encoding='utf-8'); o.writelines(itertools.islice(f,2000)); f.close(); o.close()"
```

### 小模型 SFT 命令

必须让 SFT 配置和预训练权重结构保持一致：

```powershell
cd trainer
python train_full_sft.py `
  --hidden_size 256 `
  --num_hidden_layers 4 `
  --max_seq_len 128 `
  --batch_size 2 `
  --epochs 1 `
  --num_workers 0 `
  --log_interval 10 `
  --save_interval 100 `
  --data_path ../dataset/sft_tiny.jsonl `
  --from_weight pretrain
```

预期产物：

```text
out/full_sft_256.pth
```

### 测试自己训练的模型

回到项目根目录：

```powershell
python eval_llm.py `
  --load_from ./model `
  --weight full_sft `
  --hidden_size 256 `
  --num_hidden_layers 4
```

### 对比实验

使用阶段 1 中相同的固定问题集，分别测试：

- `pretrain_256.pth`
- `full_sft_256.pth`
- 下载的正式 `minimind-3`

重点观察：

- 是否能遵循提问格式；
- 是否能正常结束回答；
- 是否出现机械重复；
- 回答是否相关；
- 事实是否可靠。

### 阶段产出

写一份对比报告：

```text
Pretrain 模型更像什么？
SFT 后发生了什么变化？
小数据 SFT 的局限是什么？
正式模型为什么明显更好？
```

### 验收标准

- 成功完成从随机初始化到 SFT 权重的完整流程；
- 能解释 Pretrain 和 SFT 的目标函数有什么相同与不同；
- 能正确加载自己训练的权重进行推理。

---

## 阶段 6：学习 LoRA 和 DPO

建议时间：1～2 周。

## 6.1 LoRA

### 阶段目标

- 理解为什么冻结基础模型也可以完成领域适配。
- 理解低秩矩阵近似的直觉。
- 能训练并加载一个简单 LoRA。

### 学习文件

- `model/model_lora.py`
- `trainer/train_lora.py`

### 建议实验

创建少量“自我认知”数据，让模型稳定回答：

- 它叫什么名字；
- 它的用途；
- 它不能做什么。

测试时比较：

- 基础 `full_sft`；
- `full_sft + LoRA`。

### 验收标准

- 能指出哪些参数被冻结、哪些参数参与训练；
- 能解释 LoRA 为什么显著减少可训练参数；
- 能说明 LoRA 和 Full SFT 的适用场景。

## 6.2 DPO

### 阶段目标

- 理解 chosen/rejected 偏好数据。
- 理解策略模型和参考模型的作用。
- 知道 DPO 为什么不需要单独训练 Reward Model。

### 学习文件

- `dataset/lm_dataset.py` 中的 `DPODataset`
- `trainer/train_dpo.py`

### 第一轮要求

第一轮以读懂代码和数据流为主，不要求完整训练。画出：

```text
prompt
 ├─ chosen   → policy/ref log probability
 └─ rejected → policy/ref log probability
                    ↓
                  DPO loss
```

### 验收标准

- 能解释 DPO 与 SFT 的区别；
- 能解释参考模型为什么保持固定；
- 能解释偏好变强可能带来的“对齐税”。

---

## 阶段 7：知识蒸馏与强化学习

建议时间：2 周以上，可按兴趣选择。

### 学习顺序

1. `trainer/train_distillation.py`
2. `trainer/rollout_engine.py`
3. `trainer/train_grpo.py`
4. `trainer/train_ppo.py`
5. `trainer/train_agent.py`

不要同时学习全部脚本。

## 7.1 知识蒸馏

### 目标

- 区分黑盒蒸馏和白盒蒸馏；
- 理解学生模型为什么同时使用 CE 和 KL loss；
- 理解温度参数对概率分布的影响。

### 验收标准

能够解释：

> 学生模型除了学习正确 token，为什么还要学习教师模型对其他 token 的相对偏好。

## 7.2 GRPO

### 目标

- 理解 rollout、reward、advantage、policy update；
- 理解同一问题生成多个答案的原因；
- 理解组内奖励相同为什么会导致学习信号消失。

### 验收标准

能画出：

```text
prompt → 多个回答 → 奖励评分 → 组内优势 → 更新策略
```

## 7.3 PPO

### 目标

- 理解 Actor、Critic、Reference Model；
- 理解 GAE、概率比裁剪和 KL 约束；
- 知道 PPO 为什么比 GRPO 更复杂、更占显存。

RTX 5060 8GB 不适合直接运行默认完整 PPO。此阶段优先阅读代码，实际训练可使用云端大显存 GPU。

## 7.4 Agentic RL

### 目标

- 理解多轮 Tool Call rollout；
- 理解环境反馈和延迟奖励；
- 理解训练侧与 rollout 侧解耦。

学习重点：

```text
模型生成工具调用
  ↓
执行工具
  ↓
把结果加入上下文
  ↓
继续生成
  ↓
整条轨迹评分
```

### 阶段验收

- 能区分 SFT、DPO 和 On-Policy RL；
- 能解释 Reward Model 与规则奖励的差异；
- 能解释奖励稀疏和 reward hacking；
- 能根据显存条件判断哪些实验适合本地、哪些适合云端。

---

## 阶段 8：评测、转换和部署

建议时间：1 周。

### 阶段目标

- 建立正确的模型评估意识。
- 学会模型格式转换。
- 能启动本地 API 或 WebUI。

### 学习文件

- `eval_llm.py`
- `scripts/eval_toolcall.py`
- `scripts/convert_model.py`
- `scripts/serve_openai_api.py`
- `scripts/chat_api.py`
- `scripts/web_demo.py`

### 评测原则

至少包含：

- 固定提示词人工对比；
- 事实准确性；
- 指令遵循；
- 格式遵循；
- 重复和退化；
- 训练前后对比；
- 独立验证数据上的 loss。

不要只展示训练数据中的问题，也不要只选择表现好的案例。

### 启动 OpenAI 兼容 API

```powershell
cd scripts
python serve_openai_api.py
```

在另一个终端中：

```powershell
cd scripts
python chat_api.py
```

### 阶段产出

完成一个小型最终项目：

1. 准备一个特定主题的小数据集；
2. 训练 LoRA 或小型 Full SFT；
3. 使用固定测试集评估；
4. 转换为 Transformers 格式；
5. 启动 API；
6. 写一份实验报告。

### 验收标准

- 能通过 API 调用本地模型；
- 能说明 Transformers 格式和原生 `.pth` 格式的区别；
- 能用实验数据而不是主观感受评价一次训练。

---

## 四、推荐每周节奏

每周学习 5 天时，可以采用：

### 第 1 天：概念

- 阅读 README 或基础材料；
- 记录不理解的术语；
- 用自己的话写解释。

### 第 2 天：源码

- 每次只读一个核心文件；
- 标记输入、输出和关键函数；
- 不要求一次理解所有细节。

### 第 3 天：运行

- 运行现有示例；
- 保存命令、日志和结果；
- 先保证可复现。

### 第 4 天：修改

一次只改一个变量，例如：

- batch size；
- sequence length；
- temperature；
- 数据量；
- 模型层数。

### 第 5 天：复盘

回答：

1. 本周学会了什么？
2. 哪段代码仍然不理解？
3. 哪个实验支持了我的理解？
4. 下周要解决什么问题？

---

## 五、学习记录模板

建议在项目外或自己的笔记目录中，每次实验记录：

```markdown
# 实验名称

## 目的

## 环境
- GPU：
- PyTorch：
- 数据：
- Git commit：

## 模型配置
- hidden_size：
- num_hidden_layers：
- max_seq_len：

## 训练配置
- batch_size：
- accumulation_steps：
- learning_rate：
- epochs：

## 结果
- 初始 loss：
- 最终 loss：
- 训练时间：
- 显存峰值：

## 固定问题测试

## 观察与结论

## 遇到的问题

## 下一步
```

---

## 六、本地设备使用建议

RTX 5060 Laptop GPU 只有 8GB 显存，建议：

- 推理正式 64M 模型：适合；
- 小规模 Pretrain/SFT：适合；
- LoRA：适合；
- 默认参数完整 Pretrain/SFT：可能需要降低 batch 和序列长度；
- PPO、完整 GRPO、Agentic RL：可能因多个模型同时驻留而显存不足；
- MoE 训练：先不要作为入门实验；
- 大规模完整复现：按需租用 24GB 或更大显存 GPU。

显存不足时按以下顺序调整：

1. 降低 `batch_size`；
2. 降低 `max_seq_len`；
3. 增大 `accumulation_steps` 补偿有效 batch；
4. 降低 `hidden_size`；
5. 降低 `num_hidden_layers`；
6. 关闭非必要的并行模型或程序。

注意：模型结构变化后，后续阶段必须使用相同的 `hidden_size`、`num_hidden_layers` 和 `use_moe`，否则权重无法加载。

---

## 七、最终检查清单

完成学习后，逐项检查：

- [ ] 我能解释 next-token prediction。
- [ ] 我能展示一段文本的 token 化结果。
- [ ] 我能解释 causal attention。
- [ ] 我能跟踪模型前向传播的张量形状。
- [ ] 我能解释交叉熵和 labels shift。
- [ ] 我能看懂 PyTorch 训练循环。
- [ ] 我完成过小规模预训练。
- [ ] 我完成过 SFT。
- [ ] 我能加载自己训练的权重推理。
- [ ] 我能解释 LoRA 与 Full SFT 的区别。
- [ ] 我能解释 SFT、DPO、GRPO/PPO 的区别。
- [ ] 我知道模型格式如何转换和部署。
- [ ] 我能客观记录一次实验，而不是只展示最好案例。

## 八、当前最应该做的下一步

按顺序执行：

1. 激活 `.venv`。
2. 运行下载好的 `minimind-3`。
3. 建立固定测试问题集。
4. 阅读 `eval_llm.py` 的模型加载和生成流程。
5. 开始阶段 2 的 Tokenizer 实验。

当前不需要立即下载全部训练数据，也不需要立即学习 PPO、GRPO 或 Agentic RL。
