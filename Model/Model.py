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
from typing import Optional,Tuple
from torch.nn import functional as F

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

#定义一个用于计算kv复用的函数工具
def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    bs,slen,num_key_value_heads,head_dim=x.shape
    if n_rep==1:
        return x
    #采用shape变换进行对kv的复制
    return (x[:,:,:,None,:].expand(bs,slen,num_key_value_heads,n_rep,head_dim).reshape(bs,slen,num_key_value_heads*n_rep,head_dim))
#Attention类
class Attention(nn.Module):
    def __init__(self,args:MokioMindConfig):
        super().__init__()

        self.num_key_value_heads=args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads

        assert args.num_attention_heads % self.num_key_value_heads == 0 
        "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads=args.num_attention_heads
        self.num_key_value_heads=args.num_key_value_heads
        self.n_rep=self.n_local_heads//self.num_key_value_heads
        self.head_dim=args.hidden_size//args.num_key_value_heads

        #先分别投影成总的 Q/K/V 表示，再 reshape 成多个 head
        self.q_proj=nn.Linear(args.hidden_size,args.num_attention_heads*self.head_dim,bias=False)
        self.k_proj=nn.Linear(args.hidden_size,args.num_key_value_heads*self.head_dim,bias=False)
        self.v_proj=nn.Linear(args.hidden_size,args.num_key_value_heads*self.head_dim,bias=False)
        #output时拼接回来
        self.o_proj=nn.Linear(args.num_key_value_heads*self.head_dim,args.hidden_size,bias=False)

        #dropout
        self.attn_dropout=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)
        self.dropout=args.dropout

        #flash attention
        self.flash=hasattr(torch.nn.functional,'scaled_dot_product_attention') and args.flash_attentionS

    def forward(
        self,
        x: torch.Tensor,
        position_embedding: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        #投影，计算qkv
        bsz,seq_len,_=x.shape
        xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)
        #把输入拆成多个头，使用view
        xq=xq.view(bsz,seq_len,self.n_local_heads,self.head_dim)
        xk=xk.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)
        xv=xv.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)

        #对于q和k，使用RoPE
        cos,sin=position_embedding
        xq,xk=apply_rotary_pos_emb(xq,xk,cos[::seq_len],sin[::seq_len])

        #kv cache实现，存取先前存取的token的k,v值，然后在seq_len处拼接（也即是加了一个seq_len）
        if past_key_value is not None:
            xk=torch.cat([past_key_value[0],xk],dim=1)
            xv=torch.cat([past_key_value[1],xv],dim=1)
        past_kv=(xk,xv) if use_cache else None

        xq,xk,xv=(
            #transpose交换seq_len和head，为了方便后续的attention计算，因为需要每个head单独做attention计算，因此交换更便利
            xq.transpose(1,2),
            #repeatkv将kv扩展到与q同等头数
            repeat_kv(xk,self.n_rep).transpose(1,2),
            repeat_kv(xv,self.n_rep).transpose(1,2)
        )

        #attention计算
        if self.flash and seq_len>1 and (attention_mask is None or torch.all(attention_mask==1)):
            #padding mask，判断是否参与attention计算
            attn_mask=(
                None
                if attention_mask is None
                #如果需要mask，那么把attn_mask扩展到与attention score一样的维度，一一对齐，[bsz,head,q_seq_len,k_seq_len]S
                else attention_mask.view(bsz,1,1,-1).expand(bsz,self.n_local_heads,seq_len,-1).bool()
            )
            output=F.scaled_dot_product_attention(
                xq,xk,xv,attn_mask=attn_mask,dropout_p=self.dropout if self.training else 0.0,is_causal=True
            )
        else:
            #不使用flash attention，手写实现attention计算
            scores=(xq@xk.transpose(-2,-1))/math.sqrt(self.head_dim)
            #加上causal mask，-inf表示负无穷，e的负无穷次方为0，则在softmax处会被忽略
            scores=scores+torch.triu(#取上三角部分操作
                torch.full((seq_len,seq_len),float('-inf'),device=scores.device),
                diagonal=1#保留主对角线
            ).unsqueeze(0).unsqueeze(0)#填上两个维度[seq_len,seq_len]->[1,1,seq_len.seq_len]

            #屏蔽pad等等无效位置
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask)*-1e9
                scores=scores+extended_attention_mask
        #softmax计算
        scores=F.softmax(scores.float(),dim=-1).type_as(xq)
        #执行dropout
        scores=self.attn_dropout(scores)
        #输出结果
        output=scores@xv
        #完成后的初始状态是[bsz,n_local_heads,seq_len,head_dim],下面是下一步处理
        output=output.transpose(1,2).reshape(bsz,seq_len,-1)#拼接各头
        #进行残差架构下的dropout也即对x+f(x)里的f(x)
        output=self.resid_dropout(self.o_proj(output))
        return output,past_kv