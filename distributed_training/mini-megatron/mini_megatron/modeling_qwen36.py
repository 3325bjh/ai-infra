# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


from mini_megatron.config import Qwen36ModelConfig


@dataclass
class CausalLMOutput:
    """Output object for the mini causal language model."""

    logits: torch.Tensor
    loss: torch.Tensor | None
    aux_loss: torch.Tensor | None = None
    lm_loss: torch.Tensor | None = None


class RMSNorm(nn.Module):
    """Root mean square normalization skeleton used by Qwen-family models."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        rms=torch.rsqrt(hidden_states.pow(2).mean(dim=-1)+self.eps)
        return hidden_states*rms*self.weight


class RotaryEmbedding(nn.Module):
    """RoPE frequency skeleton for Qwen-style attention."""

    def __init__(self, head_dim: int, rope_theta: float) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0
            / (
                    rope_theta
                    ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            ),
            persistent=False,
        )

    def forward(self, seq_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cosine and sine RoPE tensors for a sequence length."""
        positions = torch.arange(
            seq_length, device=device, dtype=self.inv_freq.dtype
        )
        angles = torch.outer(positions, self.inv_freq.to(device))
        cos = angles.cos()
        sin = angles.sin()

        return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a `[batch, seq, heads, head_dim]` tensor."""
    #cos [seq,head_dim/2]
    # cos=cos.unsqueeze(1).unsqueeze(0) #[1,seq,1,head_dim/2]
    # sin=sin.unsqueeze(1).unsqueeze(0)
    cos = cos.to(device=x.device, dtype=x.dtype)[None, :, None, :]
    sin = sin.to(device=x.device, dtype=x.dtype)[None, :, None, :]
    x1=x[...,0::2] #[batch,seq,heads,head_dim/2]
    x2=x[...,1::2]
    output=torch.empty_like(x,dtype=x.dtype,device=x.device)
    output[...,0::2]=x1*cos-x2*sin
    output[...,1::2]=x1*sin+x2*cos
    return output



class Qwen36Attention(nn.Module):
    """Grouped-query causal self-attention skeleton."""

    def __init__(self, config: Qwen36ModelConfig) -> None:
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        assert (
                self.num_attention_heads % self.num_key_value_heads == 0
        ), "num_attention_heads must be divisible by num_key_value_heads"

        assert (
                self.num_attention_heads * self.head_dim == self.hidden_size
        ), "num_attention_heads * head_dim must equal hidden_size"

        self.group_size = (
                self.num_attention_heads // self.num_key_value_heads
        )

        self.q = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            rope_theta=config.rope_theta,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape

        # [B, S, Nq, D]
        q = self.q(hidden_states).view(
            batch,
            seq,
            self.num_attention_heads,
            self.head_dim,
        )

        # [B, S, Nkv, D]
        k = self.k(hidden_states).view(
            batch,
            seq,
            self.num_key_value_heads,
            self.head_dim,
        )
        v = self.v(hidden_states).view(
            batch,
            seq,
            self.num_key_value_heads,
            self.head_dim,
        )

        # RoPE expects [B, S, H, D]
        cos, sin = self.rope(
            seq_length=seq,
            device=hidden_states.device,
        )

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # GQA:
        # [B, S, Nkv, D] -> [B, S, Nq, D]
        k = k.repeat_interleave(self.group_size, dim=2)
        v = v.repeat_interleave(self.group_size, dim=2)

        # [B, S, H, D] -> [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # [B, H, S, D] @ [B, H, D, S]
        # -> [B, H, S, S]
        attention_scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(
                seq,
                seq,
                dtype=torch.bool,
                device=attention_scores.device,
            ),
            diagonal=1,
        )

        attention_scores = attention_scores.masked_fill(
            causal_mask,
            torch.finfo(attention_scores.dtype).min,
        )

        attention_probs = F.softmax(
            attention_scores,
            dim=-1,
        )
        attention_probs = self.dropout(attention_probs)

        # [B, H, S, S] @ [B, H, S, D]
        # -> [B, H, S, D]
        output = torch.matmul(attention_probs, v)

        # [B, H, S, D] -> [B, S, H, D]
        output = output.transpose(1, 2).contiguous()

        # [B, S, H, D] -> [B, S, hidden_size]
        output = output.view(
            batch,
            seq,
            self.hidden_size,
        )

        return self.o(output)





class Qwen36MLP(nn.Module):
    """SwiGLU feed-forward block skeleton."""

    def __init__(self, config: Qwen36ModelConfig) -> None:
        super().__init__()
        self.gate=nn.Linear(config.hidden_size,config.intermediate_size,bias=False)
        self.up=nn.Linear(config.hidden_size,config.intermediate_size,bias=False)
        self.down=nn.Linear(config.intermediate_size,config.hidden_size,bias=False)
        self.silu=nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(self.silu(self.gate(hidden_states))*(self.up(hidden_states)))


class Qwen36SparseMoE(nn.Module):
    """Simple top-k sparse MoE block skeleton for learning the routing idea."""

    def __init__(self, config: Qwen36ModelConfig) -> None:
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        if self.top_k > self.num_experts:
            raise ValueError("top_k must not exceed num_experts")

        self.router = nn.Linear(
            self.hidden_size,
            self.num_experts,
            bias=False,
        )

        self.experts = nn.ModuleList(
            Qwen36MLP(config)
            for _ in range(self.num_experts)
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, hidden_size = hidden_states.shape

        # [B, S, H] -> [T, H]
        # T = B * S
        flat_hidden_states = hidden_states.reshape(-1, hidden_size)
        num_tokens = flat_hidden_states.shape[0]

        # Router logits: [T, E]
        router_logits = self.router(flat_hidden_states)

        # Router probabilities: [T, E]
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts for each token
        # topk_weights:  [T, K]
        # topk_indices:  [T, K]
        topk_weights, topk_indices = torch.topk(
            router_probs,
            k=self.top_k,
            dim=-1,
        )

        # Switch-style load balancing loss.
        # f_i is the fraction of hard routing assignments sent to expert i,
        # while P_i is the mean probability assigned to expert i by the full
        # router distribution.  Keeping router_probs dense is important: the
        # soft term must include the probability mass of non-top-k experts.
        routing_map = F.one_hot(
            topk_indices,
            num_classes=self.num_experts,
        ).any(dim=1).to(dtype=router_probs.dtype)
        tokens_per_expert = routing_map.sum(dim=0)
        routing_fraction = tokens_per_expert / (num_tokens * self.top_k)
        router_probability = router_probs.mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(
            routing_fraction * router_probability
        )
        aux_loss = load_balance_loss * self.config.router_aux_loss_coef
        # Normalize weights among selected experts
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1,
            keepdim=True,
        )

        # Output buffer: [T, H]
        output = torch.zeros_like(flat_hidden_states)

        # Dispatch tokens to experts
        for expert_id, expert in enumerate(self.experts):
            # [T, K] -> [T]
            token_mask = (topk_indices == expert_id).any(dim=-1)

            if not token_mask.any():
                continue

            token_indices = token_mask.nonzero(
                as_tuple=True
            )[0]

            # Selected input tokens: [N, H]
            expert_input = flat_hidden_states[token_mask]

            # Expert output: [N, H]
            expert_output = expert(expert_input)

            # Get this expert's routing weight for each selected token
            expert_topk_indices = topk_indices[token_mask]
            expert_topk_weights = topk_weights[token_mask]

            expert_mask = expert_topk_indices == expert_id

            routing_weights = expert_topk_weights[expert_mask]
            routing_weights = routing_weights.unsqueeze(-1)

            # Weighted expert output: [N, H]
            weighted_output = expert_output * routing_weights

            # Accumulate back into the corresponding token positions
            output.index_add_(
                dim=0,
                index=token_indices,
                source=weighted_output,
            )

        output = output.reshape(batch, seq, hidden_size)

        return output, aux_loss




class Qwen36DecoderLayer(nn.Module):
    """One pre-norm Qwen-style decoder layer."""

    def __init__(
        self,
        config: Qwen36ModelConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx

        self.norm1 = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.attention = Qwen36Attention(config)

        self.norm2 = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.is_moe_layer = (
            layer_idx % config.moe_layer_frequency == 0
        )

        if self.is_moe_layer:
            self.feed_forward = Qwen36SparseMoE(config)
        else:
            self.feed_forward = Qwen36MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_output = self.attention(
            self.norm1(hidden_states)
        )
        hidden_states = hidden_states + attention_output

        feed_forward_output = self.feed_forward(
            self.norm2(hidden_states)
        )
        if self.is_moe_layer:
            feed_forward_output, aux_loss = feed_forward_output
        else:
            aux_loss = hidden_states.new_zeros(())
        hidden_states = hidden_states + feed_forward_output

        return hidden_states, aux_loss


class Qwen36ForCausalLM(nn.Module):
    """Compact Qwen3.6-style causal LM skeleton."""

    def __init__(self, config: Qwen36ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            Qwen36DecoderLayer(config, i)
            for i in range(config.num_layers)
        )

        self.final_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # 先初始化独立模块，再绑定权重
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            # Weight tying
            self.head.weight = self.embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear and embedding weights."""

        initializer_range = 0.02

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=initializer_range,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=initializer_range,
            )

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> CausalLMOutput:
        # [batch, seq] -> [batch, seq, hidden_size]
        hidden_states = self.embedding(input_ids)

        # 逐层执行 Decoder Layer
        aux_loss = hidden_states.new_zeros(())
        for layer in self.layers:
            hidden_states, layer_aux_loss = layer(hidden_states)
            aux_loss = aux_loss + layer_aux_loss

        # Final RMSNorm
        hidden_states = self.final_norm(hidden_states)

        # [batch, seq, hidden_size]
        # -> [batch, seq, vocab_size]
        logits = self.head(hidden_states)

        loss = None
        lm_loss = None

        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=0,
            )
            loss = lm_loss + aux_loss

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
            lm_loss=lm_loss,
        )
