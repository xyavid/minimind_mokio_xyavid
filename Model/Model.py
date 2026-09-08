from typing import Any

from transformers import PretrainedConfig

#hugging face的类 类似定义模型参数
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
import math

#继承nn.Module类
class RMSNorm(nn.Module):
    #初始化
    def __init__(self, dim:int,eps:float=1e-5):
        super(RMSNorm,self).__init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
    #_norm
    def _norm(self,x):
        #翻译RMSNorm的公式
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    #forward 每一个module都要有一个forward 这里也就是乘上权重，可学习参数，来适应训练需要的尺度
    def forward(self,x):
        #这里的.type_as是为了恢复x的16位，本身x是float16，使用了.float()后变为了float32（为了不漏算数据），最后使用这个变回float16
        return self.weight*self._norm(x.float()).type_as(x)
#yarn编写
def precompute_freqs_cis(dim:int,end:int(32*1024),rope_base,rope_scaling:Optional[dict]=None):
    #初始化RoPE频率
    freqs,attn_factor=(1.0/(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))),1.0
    #判断是否要使用yarn以及yarn的参数配置
    if rope_scaling is not None:
        orig_max,factor,beta_fast,beta_slow ={
            rope_scaling["orginal_max_position_embeddings"],
            rope_scaling["factor"],#要扩展的倍数
            rope_scaling["beta_fast"],#高频边界
            rope_scaling["beta_slow"]#低频边界
        }

    if end > orig_max:
        # 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
        inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))

        #定义高低频分界点（以索引大小为low，high区分）即low：不需要缩放的高频部分的最高索引 high：需要完全缩放的低频部分的最低索引
        low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

        #计算缩放因子 即配合分界缩放参数
        ramp = torch.clamp((torch.arange(dim//2,device=freqs.device).float()-low) / max(high-low,0.001),0,1)

        # 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
        # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
        # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
        # ramp在0-1之间时：平滑过渡。
        freqs = freqs * (1 - ramp + ramp / factor)

    #根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    #计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    #计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)即温度
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin
#RoPE相关代码
def apply_rotary_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    #[a,b]->[-b,a]
    def rotate_half(x):
        #x[..., : x.shape[-1] // 2]指取x的后半部分
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )
    #x_rotated=x*cos+rotate_half(x)*sin 半边半边相乘
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed
