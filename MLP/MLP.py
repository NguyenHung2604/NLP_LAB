"""MLP from scratch with NumPy for MNIST raw IDX files.

This script:
- reads MNIST .ubyte files directly
- normalizes images to [0, 1]
- trains a small fully connected neural network from scratch
- reports training and test accuracy
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMAGES = os.path.join(BASE_DIR, "train-images.idx3-ubyte")
TRAIN_LABELS = os.path.join(BASE_DIR, "train-labels.idx1-ubyte")
TEST_IMAGES = os.path.join(BASE_DIR, "t10k-images.idx3-ubyte")
TEST_LABELS = os.path.join(BASE_DIR, "t10k-labels.idx1-ubyte")


def read_idx_images(path: str) -> np.ndarray:
	with open(path, "rb") as f:
		magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
		if magic != 2051:
			raise ValueError(f"Invalid image file: {path}")
		data = np.frombuffer(f.read(), dtype=np.uint8)
	return data.reshape(num_images, rows, cols)


def read_idx_labels(path: str) -> np.ndarray:
	with open(path, "rb") as f:
		magic, num_labels = struct.unpack(">II", f.read(8))
		if magic != 2049:
			raise ValueError(f"Invalid label file: {path}")
		data = np.frombuffer(f.read(), dtype=np.uint8)
	return data


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
	result = np.zeros((y.shape[0], num_classes), dtype=np.float32)
	result[np.arange(y.shape[0]), y] = 1.0
	return result


def relu(x: np.ndarray) -> np.ndarray:
	return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
	return (x > 0).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
	shifted = x - np.max(x, axis=1, keepdims=True)
	exp_x = np.exp(shifted)
	return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y_true: np.ndarray) -> float:
	eps = 1e-12
	return -np.mean(np.log(probs[np.arange(y_true.shape[0]), y_true] + eps))


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
	return float(np.mean(y_pred == y_true))


@dataclass
class MLPConfig:
	input_dim: int = 784
	hidden_dims: tuple[int, ...] = (128, 64)
	output_dim: int = 10
	learning_rate: float = 0.01
	seed: int = 42


class MLP:
	def __init__(self, config: MLPConfig):
		self.config = config
		rng = np.random.default_rng(config.seed)

		layer_dims = [config.input_dim, *config.hidden_dims, config.output_dim]
		self.weights = []
		self.biases = []

		for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
			weight = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim)).astype(np.float32)
			bias = np.zeros((1, out_dim), dtype=np.float32)
			self.weights.append(weight)
			self.biases.append(bias)

	def forward(self, x: np.ndarray):
		activations = [x]
		pre_activations = []

		current = x
		last_layer = len(self.weights) - 1
		for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
			z = current @ weight + bias
			pre_activations.append(z)
			if idx == last_layer:
				current = softmax(z)
			else:
				current = relu(z)
			activations.append(current)

		return current, activations, pre_activations

	def backward(self, probs: np.ndarray, y: np.ndarray, activations, pre_activations):
		batch_size = y.shape[0]
		y_one_hot = one_hot(y, self.config.output_dim)

		grad_weights = [None] * len(self.weights)
		grad_biases = [None] * len(self.biases)

		dz = (probs - y_one_hot) / batch_size

		for layer_idx in reversed(range(len(self.weights))):
			a_prev = activations[layer_idx]
			grad_weights[layer_idx] = a_prev.T @ dz
			grad_biases[layer_idx] = np.sum(dz, axis=0, keepdims=True)

			if layer_idx > 0:	
				da_prev = dz @ self.weights[layer_idx].T
				dz = da_prev * relu_grad(pre_activations[layer_idx - 1])

		return grad_weights, grad_biases

	def update(self, grad_weights, grad_biases):
		for idx in range(len(self.weights)):
			self.weights[idx] -= self.config.learning_rate * grad_weights[idx]
			self.biases[idx] -= self.config.learning_rate * grad_biases[idx]

	def predict(self, x: np.ndarray) -> np.ndarray:
		probs, _, _ = self.forward(x)
		return np.argmax(probs, axis=1)

	def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 5, batch_size: int = 64):
		num_samples = x_train.shape[0]
		history = {"loss": [], "acc": []}

		for epoch in range(epochs):
			indices = np.random.permutation(num_samples)
			x_train = x_train[indices]
			y_train = y_train[indices]

			epoch_losses = []
			epoch_preds = []
			epoch_targets = []

			for start in range(0, num_samples, batch_size):
				end = start + batch_size
				x_batch = x_train[start:end]
				y_batch = y_train[start:end]

				probs, activations, pre_activations = self.forward(x_batch)
				loss = cross_entropy(probs, y_batch)
				grad_weights, grad_biases = self.backward(probs, y_batch, activations, pre_activations)
				self.update(grad_weights, grad_biases)

				epoch_losses.append(loss)
				epoch_preds.append(np.argmax(probs, axis=1))
				epoch_targets.append(y_batch)

			epoch_preds = np.concatenate(epoch_preds)
			epoch_targets = np.concatenate(epoch_targets)
			avg_loss = float(np.mean(epoch_losses))
			avg_acc = accuracy(epoch_preds, epoch_targets)

			history["loss"].append(avg_loss)
			history["acc"].append(avg_acc)
			print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - acc: {avg_acc:.4f}")

		return history


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	x_train = read_idx_images(TRAIN_IMAGES).astype(np.float32) / 255.0
	y_train = read_idx_labels(TRAIN_LABELS).astype(np.int64)
	x_test = read_idx_images(TEST_IMAGES).astype(np.float32) / 255.0
	y_test = read_idx_labels(TEST_LABELS).astype(np.int64)

	x_train = x_train.reshape(x_train.shape[0], -1)
	x_test = x_test.reshape(x_test.shape[0], -1)

	return x_train, y_train, x_test, y_test


def plot_history(history):
	epochs = np.arange(1, len(history["loss"]) + 1)
	fig, axes = plt.subplots(1, 2, figsize=(10, 4))

	axes[0].plot(epochs, history["loss"], marker="o")
	axes[0].set_title("Training Loss")
	axes[0].set_xlabel("Epoch")
	axes[0].set_ylabel("Loss")

	axes[1].plot(epochs, history["acc"], marker="o")
	axes[1].set_title("Training Accuracy")
	axes[1].set_xlabel("Epoch")
	axes[1].set_ylabel("Accuracy")

	plt.tight_layout()
	plt.show()


def main():
	x_train, y_train, x_test, y_test = load_mnist()

	# first_img = x_train[0].reshape(28,28)

	# plt.imshow(first_img, cmap= "gray")
	# plt.title(f"Nhan cua anh nay la:{y_train[0]}")
	# plt.axis('off')
	# plt.show()

	model = MLP(
		MLPConfig(
			hidden_dims=(128, 64),
			learning_rate=0.01,
		)
	)

	history = model.fit(x_train, y_train, epochs=5, batch_size=64)
	test_preds = model.predict(x_test)
	test_acc = accuracy(test_preds, y_test)
	print(f"Test accuracy: {test_acc:.4f}")

	plot_history(history)


if __name__ == "__main__":
	main()

