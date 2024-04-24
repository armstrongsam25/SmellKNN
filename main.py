import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from smell_utils import *
from knearestneighbors_imp import KNearestNeighbors
from randomforest_imp import RandomForestClassifier
from nn_imp import NeuralNetClassifier


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

	smell_labels = {
		"nosoda": 0,
		"diet": 1,
		"regular": 2
	}

	normal_prelim_data = []
	normal_15_diet_data = []
	normal_30_diet_data = []
	normal_60_diet_data = []
	normal_15_regular_data = []
	normal_30_regular_data = []
	normal_60_regular_data = []

	for root, dirs, files in os.walk(r'C:\Users\sear234\Desktop\Containers\Breath_Datasets\Blood Glucose\Prelim (No Soda)'):
		for file in files:
			full_path = os.path.join(root, file)
			baseline_prelim, data_prelim = read_raw_csv(full_path)
			normal_prelim = normalize_to_baseline(baseline_prelim, data_prelim)
			normal_prelim_data.append(normal_prelim)

	for root, dirs, files in os.walk(r'C:\Users\sear234\Desktop\Containers\Breath_Datasets\Blood Glucose\15mins'):
		for file in files:
			full_path = os.path.join(root, file)
			baseline_15, data_15 = read_raw_csv(full_path)
			normal_15 = normalize_to_baseline(baseline_15, data_15)

			if 'diet' in full_path.lower():
				normal_15_diet_data.append(normal_15)
			else:
				normal_15_regular_data.append(normal_15)

	for root, dirs, files in os.walk(r'C:\Users\sear234\Desktop\Containers\Breath_Datasets\Blood Glucose\30mins'):
		for file in files:
			full_path = os.path.join(root, file)
			baseline_30, data_30 = read_raw_csv(full_path)
			normal_30 = normalize_to_baseline(baseline_30, data_30)
			if 'diet' in full_path.lower():
				normal_30_diet_data.append(normal_30)
			else:
				normal_30_regular_data.append(normal_30)

	for root, dirs, files in os.walk(r'C:\Users\sear234\Desktop\Containers\Breath_Datasets\Blood Glucose\60mins'):
		for file in files:
			full_path = os.path.join(root, file)
			baseline_60, data_60 = read_raw_csv(full_path)
			normal_60 = normalize_to_baseline(baseline_60, data_60)
			if "diet" in full_path.lower():
				normal_60_diet_data.append(normal_60)
			else:
				normal_60_regular_data.append(normal_60)


	all_prelim_data = pd.concat(normal_prelim_data, axis=0, ignore_index=True)
	all_15_diet_data = pd.concat(normal_15_diet_data, axis=0, ignore_index=True)
	all_15_regular_data = pd.concat(normal_15_regular_data, axis=0, ignore_index=True)
	all_30_diet_data = pd.concat(normal_30_diet_data, axis=0, ignore_index=True)
	all_30_regular_data = pd.concat(normal_30_regular_data, axis=0, ignore_index=True)
	all_60_diet_data = pd.concat(normal_60_diet_data, axis=0, ignore_index=True)
	all_60_regular_data = pd.concat(normal_60_regular_data, axis=0, ignore_index=True)


	# random channels, no idea if they even mean anything
	# plot_over_time(['ch3', 'ch6', 'ch22', 'ch38', 'ch44', 'ch62'], cow_data_normal)

	all_prelim_data['class'] = smell_labels["nosoda"]
	all_15_diet_data['class'] = smell_labels["diet"]
	all_30_diet_data['class'] = smell_labels["diet"]
	all_60_diet_data['class'] = smell_labels["diet"]
	all_15_regular_data['class'] = smell_labels["regular"]
	all_30_regular_data['class'] = smell_labels["regular"]
	all_60_regular_data['class'] = smell_labels["regular"]

	all_prelim_15_data = pd.concat([all_prelim_data, all_15_diet_data, all_15_regular_data], axis=0, ignore_index=True)
	all_prelim_30_data = pd.concat([all_prelim_data, all_30_diet_data, all_30_regular_data], axis=0, ignore_index=True)
	all_prelim_60_data = pd.concat([all_prelim_data, all_60_diet_data, all_60_regular_data], axis=0, ignore_index=True)

	# all_prelim_15_data.to_csv('all_prelim_15_data.csv', index=False)

	# # trying to classify 15 mins
	# features = all_prelim_15_data.iloc[:, 1:-1].values  # Exclude timestamp and last column (assuming last column is the class)
	# labels = all_prelim_15_data.iloc[:, -1].values  # Last column as class

	# trying to classify 30 mins
	features = all_prelim_30_data.iloc[:,1:-1].values  # Exclude timestamp and last column (assuming last column is the class)
	labels = all_prelim_30_data.iloc[:, -1].values  # Last column as class

	# # trying to classify 60 mins
	# features = all_prelim_60_data.iloc[:,1:-1].values  # Exclude timestamp and last column (assuming last column is the class)
	# labels = all_prelim_60_data.iloc[:, -1].values  # Last column as class

	print(f'Feature shape: {features.shape}')
	print(f'Labels shape: {labels.shape}')
	print()

	# Split the data into training and validation sets
	X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

	# Standardize features
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_val = scaler.transform(X_val)

	# Convert data to PyTorch tensors
	# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
	# y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
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

############### KNN TRAINING #################
	print("KNN")
	# Create an instance of the KNearestNeighbors class
	k = len(np.unique(labels))  # Number of nearest neighbors
	knn = KNearestNeighbors(k)

	# Fit the model with the training data
	knn.fit(X_train_tensor, y_train_tensor)

	# Predict the validation set
	predictions = knn.predict(X_val_tensor)

	# Calculate accuracy
	accuracy = np.mean(predictions == y_val_tensor.numpy())
	print(f"Validation accuracy: {accuracy * 100:.2f}%")
	print()



############### RANDOM FOREST TRAINING #################
print("Random Forest")
# Instantiate the random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=10, max_depth=3)

# Train the random forest classifier
rf_classifier.fit(X_train_tensor, y_train_tensor)

# Evaluate the model on the test data
y_pred = rf_classifier.predict(X_val_tensor.numpy())
accuracy = np.mean(y_pred == y_val_tensor.numpy())
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print('\n')

############### CNN TRAINING #################
print("CNN")

# Instantiate your fully connected neural network model
input_size = 44  # Adjust based on your input data shape
output_size = 3  # Number of classes
model = NeuralNetClassifier(input_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define number of epochs
num_epochs = 10000

# Training loop
for epoch in range(num_epochs):
	# Training phase
	model.train()
	train_loss = 0.0
	correct_train = 0
	total_train = 0

	for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		train_loss += loss.item() * inputs.size(0)
		_, predicted = torch.max(outputs, 1)
		total_train += labels.size(0)
		correct_train += (predicted == labels).sum().item()

	train_loss = train_loss / len(train_loader.dataset)
	train_accuracy = correct_train / total_train

	# Validation phase
	model.eval()
	val_loss = 0.0
	correct_val = 0
	total_val = 0

	with torch.no_grad():
		for inputs, labels in val_loader:
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, labels)

			val_loss += loss.item() * inputs.size(0)
			_, predicted = torch.max(outputs, 1)
			total_val += labels.size(0)
			correct_val += (predicted == labels).sum().item()

	val_loss = val_loss / len(val_loader.dataset)
	val_accuracy = correct_val / total_val

	print(f'Epoch {epoch + 1}/{num_epochs}, '
		  f'Train Loss: {train_loss:.4f}, Train Accuracy: {100 * train_accuracy:.2f}%, '
		  f'Val Loss: {val_loss:.4f}, Val Accuracy: {100 * val_accuracy:.2f}%')

