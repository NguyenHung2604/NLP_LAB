import os
import struct
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
""" Data link initiation"""

#Initiate absolute direct of NLP_myself.py
BASE_DIRECT = os.path.dirname(__file__)

TRAIN_IMGS = os.path.join(BASE_DIRECT, "train-images.idx3-ubyte")
TRAIN_LABELS = os.path.join(BASE_DIRECT, "train-labels.idx1-ubyte")
TEST_IMGS = os.path.join(BASE_DIRECT, "t10k-images.idx3-ubyte")
TEST_LABELS = os.path.join(BASE_DIRECT, "t10k-labels.idx1-ubyte")

""" Necessary funtion """
# Create function read binary
def read_idx_imgs(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        # take 16 first bytes by struct.unpack
        # >IIII: Compling rules in big-endian , 4 bytes
        # 16 byte dau tien chua thong tin ve du lieu do
        magic_num, num_imgs, rows, cols = struct.unpack(">IIII", f.read(16)) # 4 bytes
        if magic_num != 2051: # Check safe password
            raise ValueError(f"Invalid images file {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8) # split buffer into bytes = 8 bits
    return data.reshape(num_imgs, rows, cols) # return 3 dims array

def read_idx_labels(path: str) -> np.ndarray:
    with open(path, 'rb') as f: # translate into binary
        magic_num, num_labels = struct.unpack('>II', f.read(8)) # 2 bytes
        if magic_num != 2049: # check safe pass
            raise ValueError(f"Invalid labels file{path}")
        labels_data = np.frombuffer(f.read(), dtype=np.uint8)
    return labels_data

def relu_activation(x: np.ndarray) -> np.ndarray: # x la features
    return np.maximum(x, 0.0)

def one_hot_encoding(y: np.ndarray, num_classes: int = 10) -> np.ndarray: # Default: 10 digits, y la danh sach cac nhan
    # int 10: cac anh tu 0 den 9
    # using advanced indexing
    res = np.zeros((y.shape[0], num_classes), dtype=np.float32) # tao mot mang 2 chieu toan so 0
    res[np.arange(y.shape[0]), y] = 1.0 # danh so chi muc 1 vao dung vi tri
    return res

def relu_gradient(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)   # x < 0 - > 0.0 x > 0 -> 1.0

def softmax(x: np.ndarray) -> np.ndarray:

    "Rất may mắn, toán học đã chứng minh rằng"
    "nếu bạn cộng hoặc trừ cùng một hằng số c"
    "vào tất cả các phần tử trước khi đưa vào Softmax"
    "kết quả cuối cùng không hề thay đổi."

    # Max scaling normalization
    shifted = x - np.max(x, axis = 1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis = 1, keepdims=True)

def cross_entropy(soft_max: np.ndarray, y_true: np.ndarray) -> float:
    epsilon = 1e-12
    return -np.mean(np.log(soft_max[np.arange(y_true.shape[0]), y_true] + epsilon))

def accurracy(y_pred:np.ndarray, y_true:np.ndarray)->float:
    return float(np.mean(y_pred == y_true)) # true = 1, false = 0

@dataclass
class MLP_defaultConfig:
        input_dim: int = 784 # 28*28
        hidden_dims:tuple[int, ...] = (128, 64)
        output_dims: int = 10
        learning_rate: float = 0.01
        seed: int = 42

class MLP:
    def __init__(self, config: MLP_defaultConfig):
        self.config = config
        random_number_gen = np.random.default_rng(config.seed)
        layer_dims = [config.input_dim, *config.hidden_dims, config.output_dims] # * de tach cai tuple ra, ghep lai thanh mot layers dims
        self.biases = []
        self.weights = []

        for in_dims, out_dims in zip(layer_dims[:-1], layer_dims[1:]): # (784, 128), (128, 64), (64, 10)
            # He initialization 
            weights = random_number_gen.normal(0.0, np.sqrt(2.0/in_dims), size = (in_dims, out_dims)).astype(np.float32) # weight matrix initialization 
            biases = np.zeros((1, out_dims),dtype=np.float32)
            self.biases.append(biases)
            self.weights.append(weights)

    def forwarding(self, x: np.ndarray):
        activations = [x]
        pre_activations = [] # z
        cur = x 

        output_layer = len(self.weights) - 1 # index cua ma tran weight cuoi cung tuc last layer
        # su duhg enumerate de lay ca idx va value
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = cur @ w + b
            pre_activations.append(z)
            if idx == output_layer:
                cur = softmax(z) # prediction
            else:
                cur = relu_activation(z)
            activations.append(cur)
        return cur, activations, pre_activations # den buoc cuoi cung, cur se thanh output
    
    def backwarding(self, probs: np.ndarray, y_batch: np.ndarray, activations, pre_activations):
        # Using mini batch gradient descend
        batch_size = y_batch.shape[0]
        y_batch_one_hot = one_hot_encoding(y_batch, self.config.output_dims) # y_batchs = so luong mau sample cho vao

        dWs = [None]*len(self.weights) # Create an array filled with None values ​​with a length of len(self.weight)
        dBs = [None]*len(self.biases) # Create an array filled with None values ​​with a length of len(self.bias)

        dZ = (probs - y_batch_one_hot) / batch_size
        for layer_index in reversed(range(len(self.weights))): # lon nguoc tu cuoi len dau
            a_prev = activations[layer_index]
            dWs[layer_index] = a_prev.T @ dZ
            dBs[layer_index] = np.sum(dZ, axis=0, keepdims=True) # broadcasting

            # If not input layers
            if layer_index > 0:
                da_prev = dZ @ self.weights[layer_index].T
                dZ = da_prev * relu_gradient(pre_activations[layer_index - 1])
        return dWs, dBs

    # Update W and b
    def update(self, grad_weights, grad_biasas):
        for idx in range(len(grad_weights)):
            self.weights[idx] -= self.config.learning_rate*grad_weights[idx]
            self.biases[idx] -= self.config.learning_rate*grad_biasas[idx]
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        probs,_,_ = self.forwarding(x) # throw away dumb result, khong can hai cai ket qua phia sau
        return np.argmax(probs, axis=1) # return ra vi tri cua probs la max, theo chieu ngang, probs la mot ma tran (do co nhieu con vat)
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 5, batch_size: int = 64):
        num_samples = x_train.shape[0]

        # Using dictionary for following loss and accuracy
        history = {"loss": [], "accuracy": []}

        epoch_losses = [] # luu epoch 
        epoch_preds = [] # luu y du doan 
        epoch_targets = [] # gia tri y real

        for epoch in range(epochs):
            epoch_losses = [] # luu epoch 
            epoch_preds = [] # luu y du doan 
            epoch_targets = [] # gia tri y real

            # Shuffle the array of x and y for each epochs
            shuffle_indices = np.random.permutation(num_samples)

            x_train = x_train[shuffle_indices]
            y_train = y_train[shuffle_indices]

            for start in range(0, num_samples, batch_size):
                # chay tu 0 den num_samples, buoc nhay la batch size
                end = start + batch_size
                x_batch = x_train[start:end]
                y_batch = y_train[start:end]

                # forward propagation
                probs, activations, pre_activations = self.forwarding(x_batch)
                # Compute loss
                loss = cross_entropy(probs, y_batch)
                # Backpropagation
                grad_weights, grad_biases = self.backwarding(probs, y_batch, activations, pre_activations)
                # Update parameter
                self.update(grad_weights, grad_biases)
    
                epoch_losses.append(loss)
                epoch_preds.append(np.argmax(probs, axis=1))
                epoch_targets.append(y_batch)

            # Concat into a long array
            epoch_preds = np.concatenate(epoch_preds)
            epoch_targets = np.concatenate(epoch_targets)
            avg_loss = float(np.mean(epoch_losses))
            avg_acc = accurracy(epoch_preds, epoch_targets)

            history['loss'].append(avg_loss)
            history['accuracy'].append(avg_acc)
            print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - acc: {avg_acc:.4f}")
        return history


# Loading data

def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = read_idx_imgs(TRAIN_IMGS).astype(np.float32) /255.0   
    y_train = read_idx_labels(TRAIN_LABELS).astype(np.int64)
    x_test = read_idx_imgs(TEST_IMGS).astype(np.float32) / 255.0
    y_test = read_idx_labels(TEST_LABELS).astype(np.int64)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    return x_train, y_train, x_test, y_test


def plot_history(history):
    epochs = np.arange(1, len(history['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize = (10,4)) # 1 dong, 2 cot, figsize = 10,4

    # Chieu ngang
    axes[0].plot(epochs, history['loss'], marker = 'o')
    axes[0].set_title("Training loss")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('loss')

    # Chieu doc
    axes[1].plot(epochs, history['accuracy'], marker = "x") 
    axes[1].set_title("Training accurracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accurracy")

    plt.tight_layout()
    plt.show()    

def main():
    x_train, y_train, x_test, y_test = load_mnist()

    model = MLP(
        MLP_defaultConfig(
            hidden_dims=(128,64),
            learning_rate=0.01,
        )
    )

    history = model.fit(x_train, y_train, epochs=5, batch_size=64)
    test_preds = model.predict(x_test)
    test_acc = accurracy(test_preds, y_test)
    print(f"Test accuracy:{test_acc: .4f}")

    plot_history(history)

if __name__ == "__main__":
    main()