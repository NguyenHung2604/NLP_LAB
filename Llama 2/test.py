from model import ModelArgs, Transformer
import torch

args = ModelArgs(dim=64, n_layers=2, n_heads=2, n_kv_heads=2,
                  vocab_size=100, max_seq_len=16,
                  max_batch_size=1, device="cpu")
model = Transformer(args)

tokens = torch.randint(0, 100, (1, 1))   # 1 token gia, batch=1
out = model.forward(tokens, 0)
print(out.shape)   # ky vong: torch.Size([1, 1, 100])