"""
MiniMind 的核心模型实现。

阅读时可统一使用以下张量符号：
    B：batch size，一批样本的数量
    T：sequence length，当前序列的 token 数
    C：hidden size，每个 token 的隐藏向量维度（默认 768）
    H：query attention heads 数量（默认 8）
    H_kv：key/value heads 数量（默认 4）
    D：head dim，每个注意力头的维度（默认 768 / 8 = 96）
    V：vocab size，词表大小（默认 6400）

主干数据流：
    input_ids [B, T]
        -> token embedding [B, T, C]
        -> N 个 Transformer Block [B, T, C]
        -> final RMSNorm [B, T, C]
        -> lm_head [B, T, V]
"""

import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    """集中保存模型结构和推理配置，并兼容 Hugging Face PretrainedConfig。"""

    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        # Transformer 主体规模：隐藏维度、层数以及是否使用 MoE 前馈网络。
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe

        # 词表与特殊 token。模型最终为 vocab_size 个 token 分别输出一个 logit。
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)

        # GQA 配置：Q 有 8 个头，K/V 只有 4 个头，K/V 会在计算前复制到 8 个头。
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)

        # FFN 使用 SiLU/SwiGLU。中间维度向上取整到 64 的倍数，便于硬件计算。
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)

        # RoPE 位置编码与 RMSNorm 参数。
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)

        # 输入 Embedding 与输出 lm_head 共享权重，可减少参数量。
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)

        # 推理时可用 YaRN 调整 RoPE 频率，将上下文外推到训练长度之外。
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Normalization。

    对每个 token 的最后一维 C 独立归一化：
        y = x / sqrt(mean(x^2) + eps) * weight
    与 LayerNorm 不同，RMSNorm 不减去均值；输入输出形状保持不变。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 每个隐藏维度都有一个可学习的缩放系数，初始值均为 1。
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        # 在最后一维计算均方根的倒数，再与原向量逐元素相乘。
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # float32 中计算可提高数值稳定性，最后转换回原始 dtype。
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """
    预计算 RoPE 所需的 cos/sin 位置频率表。

    返回形状均为 [end, dim]，模型前向时只切取当前 token 位置对应的部分。
    这些值不是训练参数，因此可以一次算好并注册为 buffer。
    """
    # 不同通道使用不同旋转频率；相邻的两个通道组成一个二维旋转平面。
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    # 外积把每个位置 t 与每个通道频率组合，得到该位置的旋转角度。
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # 复制一次以恢复到完整 head_dim，方便直接和 Q/K 逐元素运算。
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """把 RoPE 旋转位置编码应用到 Q 和 K；V 不需要旋转。"""

    # 将向量后半段取负并放到前面，实现二维向量旋转中的正交分量。
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将较少的 K/V heads 复制到与 Q heads 相同的数量。

    例如默认 H=8、H_kv=4，则每个 K/V head 复制 2 次：
        [B, T, 4, D] -> [B, T, 8, D]
    这就是 GQA 相比标准多头注意力节省 K/V 参数和缓存的关键。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    """带 Q/K Norm、RoPE、因果掩码和 KV Cache 的分组查询注意力（GQA）。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True

        # 输入 x 为 [B, T, C]。Q 使用全部 H 个头，K/V 只使用 H_kv 个头。
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 拼回所有 Q heads 后，将注意力结果映射回隐藏维度 C。
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # 每个注意力头独立对 D 维 Q/K 做归一化。
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        # 1. 线性投影：[B,T,C] -> Q [B,T,H*D]，K/V [B,T,H_kv*D]。
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 2. 拆分 heads，便于各个头独立计算注意力。
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # 3. Q/K Norm 后注入旋转位置信息。
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # 4. 推理时拼接历史 K/V，避免每生成一个 token 都重算完整上下文。
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        # 5. 复制 K/V heads，并转成注意力算子需要的 [B,H,T,D]。
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        # 6. 优先使用 PyTorch 的高效 SDPA（通常会选择 Flash Attention 内核）。
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # 手写注意力：scores [B,H,T_query,T_key]，Softmax 在 T_key 维分配注意力。
            scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
            # 上三角位置设为 -inf，使当前 token 无法看到未来 token。
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # padding 位置同样设为极小值，Softmax 后概率接近 0。
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq)) @ xv
        # 7. 合并 heads：[B,H,T,D] -> [B,T,H*D]，再投影回 [B,T,C]。
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    """
    SwiGLU 前馈网络。

    两条上投影分支从 C 扩展到 intermediate_size：
        SiLU(gate_proj(x)) * up_proj(x)
    再通过 down_proj 投影回 C，因此可与残差相加。
    """

    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    """
    Mixture of Experts 前馈网络。

    每个 token 先由 gate 计算各专家概率，只发送给 top-k 专家。
    Dense 模型不会进入此模块；仅当 config.use_moe=True 时使用。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # Router：为每个 token 的 C 维向量输出 num_experts 个路由分数。
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # 每个 expert 都是一个独立的 SwiGLU FFN。
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        # 将 batch 与序列维展平，让每个 token 独立选择专家。
        x_flat = x.view(-1, hidden_dim)
        # scores: [B*T, num_experts]，表示每个 token 选择各专家的概率。
        scores = F.softmax(self.gate(x_flat), dim=-1)
        # 只保留概率最大的 num_experts_per_tok 个专家。
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1: 
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                # top-1 前向权重固定为 1，但用 straight-through 保留 gate 的梯度。
                top1 = torch.topk(F.softmax(self.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False)[0]
                topk_weight = top1 - top1.detach() + 1.0 # k=1: 1.0 forward on purpose, gradient via straight-through
        y = torch.zeros_like(x_flat)
        # 按专家收集 token，分别执行 FFN，再加回对应 token 的输出位置。
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 即使本批没有 token 进入该专家，也让参数留在计算图中，兼容分布式训练。
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        # 负载均衡辅助损失：避免所有 token 长期集中到少数专家。
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    """一个 Pre-Norm Transformer Decoder Block：Attention + FFN/MoE。"""

    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 第一条残差支路：x = x + Attention(RMSNorm(x))。
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        # 第二条残差支路：x = x + FFN(RMSNorm(x))。
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    """不包含词表输出头的 Transformer 主体，负责产生每个 token 的隐藏状态。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # 查表将 token id [B,T] 转成隐藏向量 [B,T,C]。
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 默认堆叠 8 个结构相同但参数独立的 Transformer Block。
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # RoPE 频率表不参与梯度更新；persistent=False 表示不写入权重文件。
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        # 某些 Transformers 版本会传入专用 Cache 对象，本实现统一使用每层 (K,V) 列表。
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # 已有 KV Cache 时，新 token 的位置从历史长度之后开始。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        # 仅取当前输入 token 所在位置的 RoPE cos/sin。
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        # 隐藏状态依次通过所有 Block，同时保存每层的新 KV Cache。
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        # 所有 Block 之后再做一次最终 RMSNorm。
        hidden_states = self.norm(hidden_states)
        # Dense 模型中没有 MOEFeedForward，因此 aux_loss 为 0。
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """在 MiniMindModel 之上增加词表输出头、训练损失与自回归生成逻辑。"""

    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        # 将每个 token 的 C 维隐藏状态映射为 V=6400 个词表 logits。
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 共享输入/输出矩阵：同一份参数既负责 token->向量，也负责向量->token 分数。
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        # 推理时可只保留末尾若干位置的 logits，减少显存占用；0 表示保留全部。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # [B,T,C] -> [B,T,V]；这里输出原始 logits，不提前做 Softmax。
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            # next-token prediction：位置 t 的输出预测位置 t+1 的真实 token。
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # CrossEntropy 内部已包含 LogSoftmax；label=-100 的位置不参与损失。
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """
        最小化实现的自回归生成循环。

        每轮只完成一件事：根据已有序列得到最后位置的 6400 个 logits，
        经过 temperature / repetition penalty / top-k / top-p 处理后选择
        一个 next_token，并追加到 input_ids，直到 EOS 或达到长度上限。
        """
        # repeat 支持一次为同一输入采样多条不同回复。
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        # 记录每个序列是否已经生成 EOS。
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            # 有 Cache 时只把尚未计算的新 token 送入模型。
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # 只使用最后一个位置预测下一个 token；低温度使分布更集中，高温度更随机。
            logits = outputs.logits[:, -1, :] / temperature
            # 已经在上下文中出现过的 token 会被适当降权，减少机械重复。
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            # top-k：仅保留分数最高的 k 个候选 token。
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            # top-p：仅保留累计概率达到 p 的最小候选集合（nucleus sampling）。
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            # 采样模式按 Softmax 概率随机选择；非采样模式直接选择最大 logit。
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            # 已结束的序列后续始终补 EOS，避免继续生成有效内容。
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids