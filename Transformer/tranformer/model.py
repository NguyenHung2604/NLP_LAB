import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import math
class inputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    # d_model la so chieu cua mo hinh, vocab_size la kich thoc tu vung
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) # dropout voi ti le la ...

        # create a matrix with size = (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term_trick = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000))/d_model)
        # Apply to each row, sin and cos position
        pe[:, 0::2] = torch.sin(position * div_term_trick)
        pe[:, 1::2] = torch.cos(position * div_term_trick)

        pe = pe.unsqueeze(0) # them mot chieu o vi tri 0 cho batch size -> (1, seq_len, d_model)

        """pe la tai san cua mo hinh, khi chuyen mo hinh len gpu thi phai mang theo, khong duoc phep train cai nay"""
        self.register_buffer("pe", pe) # giup luu tru pe vao bo dem, khong lam mat di khi dua len gpu

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # khong thay doi cua gradients, cong them pe vao x
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1)) # alpha = 1, tham so truyen vao la chi co 1 gia tri
        self.bias = nn.Parameter(torch.zeros(1)) # khoi tao bias = 0

        # cong thuc: y = alpha*(x - nguy) / sqet(epsilon + sigma**2) + bias
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha*(x - mean) / torch.sqrt(self.epsilon + std**2) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 B2

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))    

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        # Kiem tra dieu kien bat buoc truoc khi thuc hien ben duoi
        assert d_model % h == 0, "d_model is not dividable by h"
        
        self.d_k = d_model // h # chia lay thuong

        # Q, K ,V

        # (seq, d_model) @ (d_model, d_model) -> (seq, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Masking
        if mask is not None:
            # thay the tat ca cac cho mask bang mot so cuc nho
            attention_scores.masked_fill_(mask == 0, -1e9) 
            # khi cho vao e mu thi no se tien den 0
        attention_scores = attention_scores.softmax(dim = -1) 

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq, d_model) -> (batch, seq, d_model)
        key = self.w_k(k) # (batch, seq, d_model) -> (batch, seq, d_model)
        value = self.w_v(v) # (batch, seq, d_model) -> (batch, seq, d_model)

        # (batch, seq, d_model) -> (batch, seq, h, d_k) -> (batch, h, seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_score = MultiheadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Chuyen nguoc lai ve (batch, seq, h, d_k) sau do --> (batch, seq, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # -1 de no tu tinh lai 
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


"""Encoder"""
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask): # src_mask la ma tran mat na (che di pad)
        # self_attention
        # Dùng lambda để gom self_attention_block (4 tham số) thành một hàm nhận 1 tham số (x)
        # nhằm tương thích với sublayer(norm(x)) trong ResidualConnection
        # Nhan 3 tham so x de lam q, k, v
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # python thuc thi ve phai truoc roi moi truyen sang bien ben trai, do do, x duoc bien doi
        # Truyền trực tiếp self.feed_forward_block (callable object) vào sublayer; 
        # PyTorch sẽ tự động gọi phương thức .forward(norm(x)) bên dưới --> __call__ cua nn.Module
        x = self.residual_connections[1](x, self.feed_forward_block)    # truyen feed_forward_block(x) thi se bi sai ngay lap tuc do chua norm x
        return x

# Combine Nx Encoder Block lai tro thanh mot block lon hon
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

"""Decoder"""
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadAttentionBlock, cross_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float)->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # selfattention for decoder
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))

        # Cross attention for decoder with query from decoder and key and value from encoder
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))

        # Feed forward 
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

# Phep chieu
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)



"""Transformer"""

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: inputEmbedding, tgt_embed: inputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # h = head, N = Nx tuc so block, d_ff la kich thuoc an cua feed forward block

    # Create embedding layers
    src_embed = inputEmbedding(d_model, src_vocab_size)
    tgt_embed = inputEmbedding(d_model, tgt_vocab_size)

    # Create positional encoding layers (co the dung chung vi chung giong nhau, toi uu hoa se oke hon)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []

    for _ in range(N):
        encoder_self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block,feed_forward_block, dropout))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size) # cai nay nam o decoder

    # Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Xavier parameter initialization co nhieu cach khac nhu he initialization, truncated, scaled 
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


