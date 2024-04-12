import torch.nn as nn

class MultiClassClassifier(nn.Module):
	def __init__(self, features):
		super(MultiClassClassifier, self).__init__()
		self.fc1 = nn.Linear(features.shape[1], 128)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(128, 64)
		self.relu2 = nn.ReLU()
		self.fc3 = nn.Linear(64, 5)  # Output layer with 5 units (one for each class)
		self.softmax = nn.Softmax(dim=1)  # Softmax activation function

	def forward(self, x):
		x = self.relu(self.fc1(x))
		x = self.relu2(self.fc2(x))
		x = self.fc3(x)
		return self.softmax(x)