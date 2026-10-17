import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import math

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number queries
    # Optional[x] = None tuc la bien nhan gia tri x hoac None
    n_kv_heads: Optional[int] = None # Nếu None thì dùng MHA (= n_heads), nếu nhỏ hơn thì dùng GQA/MQA
    vocab_size: int = -1 # Will be set when we load tokenizers
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norms_eps: float = 1e-5


    # Needeed for KV_Cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None # Point the device which run this: CUDA, CPU, GPU, ....


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0 # d phai la boi cua hai thi moi duoc
    # Build the theta parameter
    # fomula of theta_i = 10000^-2(i - 1)/d for i in range(1 to head_dim / 2)
    # Shape theta = head_dim / 2
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # theta ben phai la 10000.0, ap dung mot phat cho toan bo theta trong numerator luon

    # Construct the position the parameters
    # shape: seq_len
    m = torch.arange(seq_len, device=device) # [0, 1, 2, ..., seq_len - 1]
    # shape: (seq_len) outer product * (head_dim / 2) ->  (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float() # dau ra se la ma tran m.theta_i -> kich thuoc (seq_len, head_dim / 2)
    # Compute complex number by polar, c = R.exp(1.m.theta), where R = 1 as follows
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs) # fomular: z = r.exp(i.theta) == torch.polar(abs, angle)

    return freqs_complex # frequencies: tần số góc

def apply_rotary_encoding(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Transform from 1 to 2
    # tao matrix so ao voi index le la phan thuc, chan la phan ao
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)    
    # * la unpacking: hoat dong cua luong se nhu sau x.shape = (B, seq_len, h, head_dim) -> tach ra thanh rieng biet bo cai chieu cuoi cung di
    # -> (B, seq_len, h) (den -1)  
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # chuyen no thanh mot ma tran co hai cot va so hang bang tong so phan tu chia cho 2

    # them chieu vao cho freqs_complex
    #(seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # Implement rotary
    x_rotate = x_complex * freqs_complex

    # convert back to real number
    # (B, seq_len, H, head_dim / 2) -> (B, seq_len, H, head_dim / 2, 2) 
    x_real = torch.view_as_real(x_rotate)

    # flatten it
    x_out = x_real.reshape(*x.shape) # reshape no lai cho giong x
    return x_out.type_as(x).to(device)
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else: 
        return(
            # (B, seq_len, n_kv_heads, 1, head_dim)
            x[:,:,:,None,:]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )
class RMSNorm(nn.Module): 
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Gamma paramater
        self.weight = nn.Parameter(torch.ones(dim)) # tensor 1 chieu vi chi co mot tham so truyen vao
        # Boc trong khoi parameter == bien nay can tinh gradient

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1,  keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # code da bo qua song song hoa cho nhieu gpu

        # Indicates the number of heads for the keys and values, (2 truong hop Multihead and Group query)
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and the Values should be repeateds
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each (Kich thuoc cua mot head)
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim,  args.n_heads * self.head_dim, bias = False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias = False)    

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # (b, 1, dim)

        # (B, 1, dim) -> (B, 1, H_Q * Head_dim)
        xq = self.wq(x)
        # (B, 1, dim) => (B, 1, H_KV * Head_dim)
        xk = self.wk(x)
        # (B, 1, dim) -> (B, 1, H_KV * Head_dim)
        xv = self.wv(x)

        # (B, 1, H_Q * Head_dim) -> (B, 1, H_Q, Head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_dim) -> (B, 1, H_KV, Head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_dim) -> (B, 1, H_KV, Head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary, which does not change the size of the vector
        xq = apply_rotary_encoding(xq, freqs_complex, device=x.device)
        xk = apply_rotary_encoding(xk, freqs_complex, device=x.device)

        # Replace the entry in the cache    
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        # Retrive all the cached keys and values so far
        keys = self.cache_k[:batch_size, 0 : start_pos + seq_len]    
        values = self.cache_v[:batch_size, 0 : start_pos + seq_len]

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, Head_dim) -> (B, H_Q, 1, Head_dim)
        xq = xq.transpose(1,2)

        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_len_KV, Head_Dim)
        keys = keys.transpose(1,2)

        # (B, Seq_len_kv, H_Q, Head_dim) -> (B, H_Q, Seq_len_kv, Head_dim)
        values = values.transpose(1,2)

        # (B, H_Q, 1, Head_dim) @ (B, H_Q, Head_dim, Seq_len_KV) -> (B, H_Q, 1, Seq_len_KV)
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim = -1).type_as(xq)

        # (B, H_Q, 1, Seq_len_kv) @ (B, H_Q, Seq_len_kv, Head_dim) -> (B, H_Q, 1, Head_dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1,2).contiguous().view(batch_size, seq_len, -1)) # Ham contigous se kiem tra xem 
        # cac phan tu con lien ke nhau khong, neu khong thi se tao ban copy
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim   
        hidden_dim = int(2 * hidden_dim / 3)

        # lam tron hidden_dim den mot boi so gan nhat cua multiple_of
        # ki thuat nay hay phet
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        # Lý do: phần cứng (đặc biệt GPU) chạy hiệu quả hơn khi kích thước tensor là bội số đẹp (ví dụ 256)
        # nên người ta ép hidden_dim về một số "đẹp" thay vì để một con số lẻ.


        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))

        x_V = self.w3(x)

        x = swish * x_V

        x = self.w2(x)
        return x
class EncoderBlock(nn.Module): 
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE attention
        self.attention_norm = RMSNorm(args.dim, eps = args.norms_eps)
        # Normalization BEFORE feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps = args.norms_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # Apply resnet and self attention
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )
        # Apply resnet and feed forward
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()

        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norms_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        # Nhan hai o cho max_seq_len de tang khoang dem an toan tranh loi chi so vi tri
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # Embedding: (B, seq_len) -> (B, seq_len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        # den cuoi moi ap dung norm 
        h = self.norm(h)
        output = self.output(h).float()
        return output

    