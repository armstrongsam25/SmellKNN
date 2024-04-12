from collections import Counter

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

class KNearestNeighbors:
	def __init__(self, k=5):
		"""
		Initialize the k-Nearest Neighbors classifier.

		Parameters:
		- k: int, the number of nearest neighbors to consider (default is 5)
		"""
		self.k = k
		self.X_train = None
		self.y_train = None
		self.scaler = StandardScaler()

	def fit(self, X_train, y_train):
		"""
		Store the training data and labels.

		Parameters:
		- X_train: torch.Tensor, the training data features
		- y_train: torch.Tensor, the training data labels
		"""
		# Standardize the features
		X_train = self.scaler.fit_transform(X_train)

		self.X_train = torch.tensor(X_train, dtype=torch.float32)
		self.y_train = y_train.clone().detach().long()

	def predict(self, X_new):
		"""
		Classify new data points based on the k-nearest neighbors.

		Parameters:
		- X_new: torch.Tensor, the new data to classify

		Returns:
		- predictions: np.array, the predicted class labels for X_new
		"""
		# Standardize the new data using the same scaler
		X_new = self.scaler.transform(X_new)
		X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

		# Compute the distance matrix between X_train and X_new
		distances = torch.cdist(X_new_tensor, self.X_train)

		# Get the indices of the k nearest neighbors for each data point in X_new
		k_nearest_indices = torch.topk(distances, self.k, largest=False).indices

		# Get the labels of the k nearest neighbors for each data point in X_new
		k_nearest_labels = self.y_train[k_nearest_indices]

		# Use a majority vote to determine the predicted class for each data point in X_new
		predictions = []
		for neighbors in k_nearest_labels:
			# Count the occurrences of each class label in the k nearest neighbors
			label_counts = Counter(neighbors.tolist())
			# Find the class label with the maximum count (majority vote)
			predicted_label = max(label_counts, key=label_counts.get)
			predictions.append(predicted_label)

		return np.array(predictions)