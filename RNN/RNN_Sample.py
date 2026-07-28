"""
RNN CÀI ĐẶT TỪ ĐẦU (FROM SCRATCH) BẰNG NUMPY
==============================================
Dataset: Tiny Shakespeare (char-level language model)

Mục tiêu của file này là để ANH HIỂU RÕ TỪNG BƯỚC bên trong một RNN,
không dùng PyTorch/TensorFlow, không dùng autograd — mọi đạo hàm
(backpropagation through time - BPTT) được viết tay bằng NumPy.

Cấu trúc file:
    1. Lý thuyết tóm tắt (đọc comment bên dưới)
    2. Chuẩn bị dữ liệu (tải Tiny Shakespeare, xây vocab ở mức ký tự)
    3. Class RNN (forward, backward/BPTT, cập nhật tham số)
    4. Training loop + sinh văn bản mẫu (sampling) để theo dõi tiến độ
    5. Hàm main để chạy toàn bộ

Chạy:
    python rnn_from_scratch.py
"""

import numpy as np
import os


# ============================================================
# PHẦN 1: LÝ THUYẾT TÓM TẮT
# ============================================================
#
# RNN (Recurrent Neural Network) xử lý dữ liệu dạng chuỗi (sequence)
# bằng cách duy trì một "hidden state" h_t, được cập nhật liên tục
# qua từng bước thời gian t, dựa trên input hiện tại x_t và hidden
# state trước đó h_{t-1}.
#
# FORWARD PASS (tại mỗi bước t):
#
#   h_t = tanh( Wxh @ x_t + Whh @ h_{t-1} + bh )
#   y_t = Why @ h_t + by
#   p_t = softmax(y_t)          # phân phối xác suất trên vocab
#
# Trong đó:
#   x_t   : vector one-hot của ký tự tại bước t     (vocab_size, 1)
#   h_t   : hidden state tại bước t                  (hidden_size, 1)
#   y_t   : logits (chưa qua softmax)                (vocab_size, 1)
#   Wxh   : ma trận input -> hidden                  (hidden_size, vocab_size)
#   Whh   : ma trận hidden -> hidden (recurrence)    (hidden_size, hidden_size)
#   Why   : ma trận hidden -> output                 (vocab_size, hidden_size)
#   bh, by: bias tương ứng
#
# LOSS: cross-entropy giữa p_t và ký tự đúng (target) tại bước t.
#
#   L = - sum_t log( p_t[target_t] )
#
# BACKPROPAGATION THROUGH TIME (BPTT):
# Vì h_t phụ thuộc vào h_{t-1}, gradient phải được "truyền ngược"
# qua toàn bộ chuỗi thời gian, cộng dồn gradient tại mỗi bước:
#
#   dy_t        = p_t - one_hot(target_t)          (đạo hàm softmax+CE)
#   dWhy       += dy_t @ h_t.T
#   dby        += dy_t
#   dh          = Why.T @ dy_t + dh_next            (dh_next: gradient
#                                                     truyền từ bước t+1)
#   dh_raw      = (1 - h_t^2) * dh                  (đạo hàm của tanh)
#   dWxh       += dh_raw @ x_t.T
#   dWhh       += dh_raw @ h_{t-1}.T
#   dbh        += dh_raw
#   dh_next     = Whh.T @ dh_raw                    (truyền tiếp về bước t-1)
#
# VANISHING / EXPLODING GRADIENT:
# Vì dh_next được nhân lặp lại qua Whh.T qua nhiều bước, gradient có
# thể co lại về 0 (vanishing) hoặc phình to bất thường (exploding).
# Kỹ thuật đơn giản để giảm exploding gradient là GRADIENT CLIPPING:
# giới hạn giá trị mỗi phần tử gradient trong khoảng [-clip, clip]
# trước khi cập nhật tham số.
#
# CẬP NHẬT THAM SỐ: dùng Adagrad (thay vì SGD thường), vì Adagrad tự
# điều chỉnh learning rate theo lịch sử gradient của từng tham số,
# giúp huấn luyện RNN ổn định hơn:
#
#   mem   += grad * grad
#   param += -learning_rate * grad / sqrt(mem + eps)


# ============================================================
# PHẦN 2: CHUẨN BỊ DỮ LIỆU
# ============================================================

DATA_PATH = "tinyshakespeare.txt"  # đổi thành "tinyshakespeare_100k.txt" nếu muốn dùng file rút gọn


def load_dataset(path=DATA_PATH):
    """Đọc dataset trực tiếp từ file .txt có sẵn trong thư mục project."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy file '{path}'. Anh kiểm tra lại tên file/đường dẫn "
            "hoặc truyền đúng path vào load_dataset()."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text):
    """Xây vocabulary ở mức ký tự (character-level)."""
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return chars, char_to_idx, idx_to_char


# ============================================================
# PHẦN 3: RNN CÀI ĐẶT TỪ ĐẦU
# ============================================================

class RNN:
    def __init__(self, vocab_size, hidden_size=100, seq_length=25, learning_rate=0.1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        # Khởi tạo tham số: giá trị nhỏ ngẫu nhiên để phá vỡ tính đối xứng
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

        # Bộ nhớ cộng dồn gradient bình phương, dùng cho Adagrad
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

    def forward_backward(self, inputs, targets, h_prev):
        """
        Chạy forward + backward (BPTT) trên MỘT đoạn chuỗi (chunk).

        inputs, targets: list các chỉ số ký tự (độ dài = seq_length)
        h_prev: hidden state cuối cùng của chunk trước (để nối liền mạch)

        Trả về: loss, gradient của tất cả tham số, hidden state cuối cùng
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0.0

        # ---- FORWARD PASS: đi qua từng ký tự trong chunk ----
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1  # one-hot encoding

            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t - 1] + self.bh)
            ys[t] = self.Why @ hs[t] + self.by

            # softmax ổn định số học (trừ max trước khi exp)
            exp_y = np.exp(ys[t] - np.max(ys[t]))
            ps[t] = exp_y / np.sum(exp_y)

            loss += -np.log(ps[t][targets[t], 0] + 1e-12)  # cross-entropy

        # ---- BACKWARD PASS (BPTT): đi ngược từ cuối chunk về đầu ----
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])

        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # đạo hàm softmax + cross-entropy

            dWhy += dy @ hs[t].T
            dby += dy

            dh = self.Why.T @ dy + dh_next
            dh_raw = (1 - hs[t] * hs[t]) * dh  # đạo hàm của tanh

            dWxh += dh_raw @ xs[t].T
            dWhh += dh_raw @ hs[t - 1].T
            dbh += dh_raw

            dh_next = self.Whh.T @ dh_raw

        # Gradient clipping: chống exploding gradient
        for grad in (dWxh, dWhh, dWhy, dbh, dby):
            np.clip(grad, -5, 5, out=grad)

        grads = (dWxh, dWhh, dWhy, dbh, dby)
        return loss, grads, hs[len(inputs) - 1]

    def update_params(self, grads):
        """Cập nhật tham số bằng Adagrad."""
        dWxh, dWhh, dWhy, dbh, dby = grads
        params = [self.Wxh, self.Whh, self.Why, self.bh, self.by]
        mems = [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]

        for param, dparam, mem in zip(params, grads, mems):
            mem += dparam * dparam
            param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)

    def sample(self, h, seed_idx, n, idx_to_char):
        """
        Sinh văn bản (text generation) bằng cách "tự hồi quy":
        lấy output của bước hiện tại làm input cho bước sau.
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1
        indices = []

        for _ in range(n):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            exp_y = np.exp(y - np.max(y))
            p = exp_y / np.sum(exp_y)

            # lấy mẫu ngẫu nhiên theo phân phối xác suất (không lấy argmax
            # để văn bản sinh ra đa dạng hơn thay vì lặp lại 1 từ)
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            indices.append(idx)

        return "".join(idx_to_char[i] for i in indices)


# ============================================================
# PHẦN 4: TRAINING LOOP
# ============================================================

def train(n_iterations=20000, hidden_size=100, seq_length=25,
          learning_rate=0.1, print_every=500, sample_len=200):
    text = load_dataset()
    chars, char_to_idx, idx_to_char = build_vocab(text)
    vocab_size = len(chars)
    data_size = len(text)

    print(f"Dataset có {data_size} ký tự, vocab có {vocab_size} ký tự khác nhau.")

    rnn = RNN(vocab_size, hidden_size, seq_length, learning_rate)

    pointer = 0
    h_prev = np.zeros((hidden_size, 1))
    smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss kỳ vọng ban đầu

    for iteration in range(n_iterations):
        # Nếu chạy hết dữ liệu hoặc mới bắt đầu -> reset hidden state
        if pointer + seq_length + 1 >= data_size or iteration == 0:
            h_prev = np.zeros((hidden_size, 1))
            pointer = 0

        inputs = [char_to_idx[ch] for ch in text[pointer: pointer + seq_length]]
        targets = [char_to_idx[ch] for ch in text[pointer + 1: pointer + seq_length + 1]]

        loss, grads, h_prev = rnn.forward_backward(inputs, targets, h_prev)
        rnn.update_params(grads)

        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        if iteration % print_every == 0:
            sample_text = rnn.sample(h_prev, inputs[0], sample_len, idx_to_char)
            print("=" * 60)
            print(f"Iteration {iteration}, smooth loss: {smooth_loss:.4f}")
            print("-" * 60)
            print(sample_text)

        pointer += seq_length

    return rnn, char_to_idx, idx_to_char


# ============================================================
# PHẦN 5: SO SÁNH NHANH VỚI PYTORCH (chỉ để đối chiếu, KHÔNG chạy)
# ============================================================
#
# Nếu dùng PyTorch, phần forward + backward + update ở trên chỉ gói gọn
# trong vài dòng, vì autograd tự tính đạo hàm và optimizer tự cập nhật:
#
#   rnn_cell = nn.RNNCell(vocab_size, hidden_size)
#   output_layer = nn.Linear(hidden_size, vocab_size)
#   optimizer = torch.optim.Adagrad(params, lr=0.1)
#   ...
#   loss.backward()          # <- thay cho toàn bộ hàm forward_backward
#   optimizer.step()         # <- thay cho update_params
#
# Việc tự viết BPTT bằng tay (như file này) giúp anh hiểu rõ autograd
# thực chất "làm hộ" mình những gì.
#
# HƯỚNG MỞ RỘNG TIẾP THEO:
#   - Thay ô nhớ tanh bằng LSTM/GRU để giảm vanishing gradient
#   - Thay one-hot bằng character embedding (ma trận embedding học được)
#   - Dùng batch training (xử lý nhiều chuỗi song song) thay vì 1 chuỗi/lần


if __name__ == "__main__":
    trained_rnn, char_to_idx, idx_to_char = train(
        n_iterations=20000,
        hidden_size=100,
        seq_length=25,
        learning_rate=0.1,
        print_every=500,
        sample_len=200,
    )