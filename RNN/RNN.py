import numpy as np
import os
# Thu vien os dung de thao tac truc tiep voi he dieu hanh may tinh

# Direct path
DATA_PATH = "tinyshakespeare_100k.txt" # test witch 100k first


def load_dataset(path = DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find the file '{path}', should check data_path "
            "Or put right path into load_dataset"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_vocab(text):
    chars = sorted(list(set(text)))
    char_to_idx = {ch : i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    # tuong duong voi doan code sau 
    # char_to_idx = {}
    # for i, ch in enumerate(chars):
    #     char_to_idx[ch] = i
    return chars, char_to_idx, idx_to_char

"""RNN from scratch"""

class RNN:
    def __init__(self, vocab_size, hidden_size = 100, seg_length = 25, learning_rate= 0.1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seg_length = seg_length
        self.learning_rate = learning_rate

        # Nhan voi 0.01 giup cho gia tri cua tung diem tren ma tran gan voi so 0, cho ma tanh co dao ham doc nhat
        self.Wxh = np.random.randn(hidden_size, vocab_size)*0.01
        self.Whh = np.random.randn(vocab_size, hidden_size)*0.01
        self.Why = np.random.rand(vocab_size, hidden_size)*0.01
        # su dung tuple ben trong de zeros khong hieu nham thanh hai tham so (vi du shape va dtype)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))


        # Using bo nho cong don, cho Adagrad zeros_like = tao ra mot ma tran giong nhu cai ma tran truyen vao nhung toan so 0
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

    def foward_backward(self, inputs, targets, h_prev):
        xs, hs, ys, xs = {}, {}, {}, {}
        
        