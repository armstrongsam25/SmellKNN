import numpy as np
import pandas as pd

from smell_utils import *
from multiclass_classifier import MultiClassClassifier
from knearestneighbors_imp import KNearestNeighbors

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot_over_time(channels_to_plot, df):
	# Plot each selected channel over time
	for channel in channels_to_plot:
		plt.plot(df['#header:timestamp'], df[channel], label=channel)

	# Add labels and a legend
	plt.xlabel('Timestamp')
	plt.ylabel('Value')
	plt.title('Channels over Time')
	plt.legend()

	# plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

	# Show the plot
	plt.show()

if __name__ == '__main__':

	read_from_robot('./datasets/Robot/')
	exit(0)



	smell_labels = {
		"Cow": 0,
		"Dirt": 1,
		"Bacon": 2,
		"Garbage": 3,
		"Hot Wings": 4
	}
	cow_baseline, cow_data = read_raw_csv('./datasets/Cow Sticker.csv')
	bacon_baseline, bacon_data = read_raw_csv('./datasets/Bacon Sticker.csv')
	dirt_baseline, dirt_data = read_raw_csv('./datasets/Dirt Sticker.csv')
	garbage_baseline, garbage_data = read_raw_csv('./datasets/Garbage Sticker.csv')
	hot_wings_baseline, hot_wings_data = read_raw_csv('./datasets/Hot Wings Sticker.csv')

	cow_data_normal = normalize_to_baseline(cow_baseline, cow_data)
	bacon_data_normal = normalize_to_baseline(bacon_baseline, bacon_data)
	dirt_data_normal = normalize_to_baseline(dirt_baseline, dirt_data)
	garbage_data_normal = normalize_to_baseline(garbage_baseline, garbage_data)
	hot_wings_data_normal = normalize_to_baseline(hot_wings_baseline, hot_wings_data)

	# random channels, no idea if they even mean anything
	# plot_over_time(['ch3', 'ch6', 'ch22', 'ch38', 'ch44', 'ch62'], cow_data_normal)
	# plot_over_time(['ch3', 'ch6', 'ch22', 'ch38', 'ch44', 'ch62'], bacon_data_normal)
	# plot_over_time(['ch3', 'ch6', 'ch22', 'ch38', 'ch44', 'ch62'], dirt_data_normal)
	# plot_over_time(['ch3', 'ch6', 'ch22', 'ch38', 'ch44', 'ch62'], garbage_data_normal)
	# plot_over_time(['ch3', 'ch6', 'ch22', 'ch38', 'ch44', 'ch62'], hot_wings_data_normal)

	cow_data_normal['class'] = smell_labels["Cow"]
	bacon_data_normal['class'] = smell_labels["Bacon"]
	dirt_data_normal['class'] = smell_labels["Dirt"]
	garbage_data_normal['class'] = smell_labels["Garbage"]
	hot_wings_data_normal['class'] = smell_labels["Hot Wings"]

	all_data = pd.concat([cow_data_normal, bacon_data_normal, dirt_data_normal, garbage_data_normal, hot_wings_data_normal], ignore_index=True)

	features = all_data.iloc[:, 1:-1].values  # Exclude timestamp and last column (assuming last column is the target)
	labels = all_data.iloc[:, -1].values  # Last column as target

	print(features.shape)
	print(labels.shape)

	# Split the data into training and validation sets
	X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

	# Standardize features
	# scaler = StandardScaler()
	# X_train = scaler.fit_transform(X_train)
	# X_val = scaler.transform(X_val)

	# Convert data to PyTorch tensors
	# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
	# y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(
	# 	1)  # Add an extra dimension for binary classification
	# X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
	# y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
	X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
	y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
	X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
	y_val_tensor = torch.tensor(y_val, dtype=torch.int64)

	# Create PyTorch datasets and dataloaders
	train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
	val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

############### TRAINING #################
	# Create an instance of the KNearestNeighbors class
	k = 5  # Number of nearest neighbors
	knn = KNearestNeighbors(k)

	# Fit the model with the training data
	knn.fit(X_train_tensor, y_train_tensor)

	# Predict the validation set
	predictions = knn.predict(X_val_tensor)

	# Calculate accuracy
	accuracy = np.mean(predictions == y_val_tensor.numpy())
	print(f"Validation accuracy: {accuracy:.4f}")
	# # Create model instance
	# model = MultiClassClassifier(features)
	#
	# # Define loss function and optimizer
	# criterion = nn.CrossEntropyLoss()
	# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	#
	# # Training loop
	# num_epochs = 2000
	#
	# for epoch in range(num_epochs):
	# 	model.train()  # Set model to training mode
	# 	running_loss = 0.0
	#
	# 	for X_batch, y_batch in train_loader:
	# 		y_batch	= y_batch.squeeze()
	# 		optimizer.zero_grad()  # Zero the parameter gradients
	# 		outputs = model(X_batch)  # Forward pass
	# 		loss = criterion(outputs, y_batch.long())  # Calculate loss
	# 		loss.backward()  # Backward pass
	# 		optimizer.step()  # Update parameters
	#
	# 		running_loss += loss.item() * X_batch.size(0)
	#
	# 	# Calculate average loss
	# 	avg_loss = running_loss / len(train_loader.dataset)
	#
	# 	print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
	#
	# 	# Validation
	# 	model.eval()  # Set model to evaluation mode
	# 	with torch.no_grad():
	# 		correct = 0
	# 		total = 0
	# 		for X_batch, y_batch in val_loader:
	# 			outputs = model(X_batch)
	# 			_, predicted = torch.max(outputs, 1)  # Get class with the highest probability
	# 			total += y_batch.size(0)
	# 			correct += (predicted == y_batch.long()).sum().item()
	#
	# 		val_accuracy = correct / total
	# 		print(f"Validation accuracy: {val_accuracy:.4f}")


##################### INFERENCE ######################
	# # Sample new data for prediction
	# new_data = [[6600.0, 6601.0, ..., 4100.0]]  # Replace with your new data
	#
	# # Standardize new data using the same scaler
	# new_data = scaler.transform(new_data)
	#
	# # Convert new data to PyTorch tensor
	# new_data_tensor = torch.tensor(new_data, dtype=torch.float32)
	#
	# # Predict
	# model.eval()  # Set model to evaluation mode
	# with torch.no_grad():
	# 	output = model(new_data_tensor)
	# 	_, prediction = torch.max(output, 1)  # Get the index of the class with the highest probability
	#
	# print(f"Prediction: Class {prediction.item()}")