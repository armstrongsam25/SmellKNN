import csv
import pandas as pd
import os

NOISY_COLUMNS = [
		'ch1',
		'ch2',
		'ch15',
		'ch16',
		'ch17',
		'ch18',
		'ch19',
		'ch20',
		'ch21',
		'ch31',
		'ch32',
		'ch33',
		'ch34',
		'ch47',
		'ch48',
		'ch49',
		'ch50',
		'ch63',
		'ch64',
		'humidity',
		'temperature'
	]

# returns baseline row, and pandas dataframe of data
def read_raw_csv(file_path):
	# Open the file in read mode
	with open(file_path, mode='r', newline='') as file:
		# Create a csv.reader object
		csv_reader = csv.reader(file)

		# Iterate through the rows in the CSV file
		for i, row in enumerate(csv_reader):
			if i < 2:
				continue
			elif i == 3:
				_, row[0] = row[0].split('|')
				baseline_raw = [row]
				break

	data = pd.read_csv(file_path, skiprows=4)
	column_names = list(data.columns)

	data = data.drop(columns=NOISY_COLUMNS)

	baseline = pd.DataFrame(baseline_raw, columns=column_names)
	baseline = baseline.drop(columns=NOISY_COLUMNS)

	return baseline, data


def read_from_robot(folder_path):
	# List to store DataFrames
	dataframes = []

	# Get a list of all files in the directory
	csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

	# Read each CSV file and append to the list of DataFrames
	for csv_file in csv_files:
		# Build the full file path
		file_path = os.path.join(folder_path, csv_file)
		# Read the CSV file and append it to the list of DataFrames
		df = pd.read_csv(file_path)
		df = df.drop(columns=NOISY_COLUMNS, errors='ignore')
		data = normalize_to_baseline(df.iloc[[0]], df.iloc[1:])
		dataframes.append(data)

	# Concatenate all DataFrames into a single DataFrame
	combined_dataframe = pd.concat(dataframes, ignore_index=True)

	mask = combined_dataframe['location'] != 'going to ' #TODO need to change this in temi code (first locations don't have a location)

	# Filter the DataFrame using the mask
	filtered_dataframe = combined_dataframe[mask]

	# Display the combined DataFrame (optional)
	print(filtered_dataframe)

def normalize_to_baseline(baseline, data):
	# Step 1: Align columns in data_df and baseline_df
	# Since the timestamp may be included in the dataframes, we assume the actual channel data starts from column index 1
	data_channels = data.iloc[:, 1:-1]  # Exclude the timestamp, location column
	baseline_channels = baseline.iloc[:, 1:-1]  # Exclude the timestamp, location column

	# Step 2: Perform normalization
	# Subtract the baseline values from the data values
	# This is done in a vectorized manner for the data DataFrame
	data_channels.to_csv('./test.csv')
	normalized_data = data_channels.astype(float) - baseline_channels.iloc[0].astype(float)

	# Step 3: If you want to retain the timestamp in the normalized data:
	normalized_data_with_timestamp = pd.concat([data.iloc[:, 0], normalized_data], axis=1)
	normalized_data_with_timestamp_location = pd.concat([data.iloc[:, -1], normalized_data], axis=1)

	# normalized_data_with_timestamp now contains the timestamp column and normalized data
	return normalized_data_with_timestamp
