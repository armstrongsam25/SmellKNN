import csv
import pandas as pd

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

	noisy_columns = [
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

	data = data.drop(columns=noisy_columns)

	baseline = pd.DataFrame(baseline_raw, columns=column_names)
	baseline = baseline.drop(columns=noisy_columns)

	return baseline, data


def normalize_to_baseline(baseline, data):
	# Step 1: Align columns in data_df and baseline_df
	# Since the timestamp may be included in the dataframes, we assume the actual channel data starts from column index 1
	data_channels = data.iloc[:, 1:]  # Exclude the timestamp column
	baseline_channels = baseline.iloc[:, 1:]  # Exclude the timestamp column

	# Step 2: Perform normalization
	# Subtract the baseline values from the data values
	# This is done in a vectorized manner for the data DataFrame
	normalized_data = data_channels - baseline_channels.iloc[0].astype(float)

	# Step 3: If you want to retain the timestamp in the normalized data:
	normalized_data_with_timestamp = pd.concat([data.iloc[:, 0], normalized_data], axis=1)

	# normalized_data_with_timestamp now contains the timestamp column and normalized data
	return normalized_data_with_timestamp
