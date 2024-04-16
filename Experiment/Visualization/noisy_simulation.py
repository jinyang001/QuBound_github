import csv

import numpy as np
import pandas as pd
import supermarq
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit_aer.noise import NoiseModel
from qiskit_experiments.library import StandardRB
from qiskit_ibm_provider import IBMProvider
from torch.utils.data import DataLoader, TensorDataset
from qiskit import QuantumCircuit, execute, Aer
import time
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

provider = IBMProvider()
backend = provider.get_backend('ibmq_kolkata')

g_nrows = 2
g_cols = 1

num_epochs = 20
batch_size = 64

# x_values = [0.875, 0.9, 0.925, 0.95, 0.975, 0.999]
# x_values = [0.975]
x_values = [0.999]


def create_quantum_circuit():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.measure_all()
    return qc


qc = create_quantum_circuit()

CARE_STATES = [format(i, '04b') for i in range(2 ** 4)]
num_qubits = 4

print(CARE_STATES)

num_states = 2 ** qc.num_qubits

N_STATES = len(CARE_STATES)

# Parameters to control sizes
num_train_samples = 2600
num_val_samples = 200
num_test_samples = 200
# 1. Load Data from CSV
data = pd.read_csv('../../datasets/ibmq_kolkata/10000shots/GHZ4_ibmq_kolkata.csv')

# Removing outliers
Q1 = data[[f'state_{state}' for state in CARE_STATES]].quantile(0.25)
Q3 = data[[f'state_{state}' for state in CARE_STATES]].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data[[f'state_{state}' for state in CARE_STATES]] < (Q1 - 1.5 * IQR)) |
              (data[[f'state_{state}' for state in CARE_STATES]] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Load metadata
num_qubits_info = data['num_qubits_info'].iloc[0]
num_gates_info = data['num_gates_info'].iloc[0]
num_params_info = data['num_params_info'].iloc[0]
circuit_depth_info = data['circuit_depth_info'].iloc[0]
num_qubits_after_transpile = data['num_qubits_after_transpile'].iloc[0]
sim_times = data['sim_times'].values
date = data['date'].values
metadata_cols = 7

# Identifying gate error/readout error columns
error_cols = list(range(4, data.shape[1] - (num_states + metadata_cols), 5))

# data.iloc[:, error_cols] = 0  # Set gate error/readout error columns to 0

X_data_all = data.iloc[:, :- (num_states + metadata_cols)].values

care_state_cols = [f'state_{state}' for state in CARE_STATES]
y_data_all = data[care_state_cols].values

# 2. Directly slice data to get test samples
X_test = X_data_all[-num_test_samples:]
y_test = y_data_all[-num_test_samples:]
sim_times_test = sim_times[-num_test_samples:]
date_test = date[-num_test_samples:]
# print(date_test.shape)
# print(date_test)

# Convert date strings to datetime objects
date_times = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in date_test])

# Get unique days
unique_days = np.unique(date_times.astype('datetime64[D]'))

# Check if the unique days are more than 50 and trim if necessary
if unique_days.shape[0] > 50:
    unique_days = unique_days[:50]

# Now, find the index of the first occurrence of each unique day in date_times
indices = [np.where(date_times.astype('datetime64[D]') == day)[0][0] for day in unique_days]

# Select the unique days and corresponding indices for X_test
new_date_test = date_times[indices]
new_X_test = X_test[indices]

new_y_test = y_test[indices]
# Verify shapes
print(new_date_test.shape)  # Should be (50,)
print(new_date_test)
print(new_X_test.shape)
# ss
# 3. From the remaining data, slice again to get the training samples
X_train = X_data_all[-(num_test_samples + num_train_samples):-num_test_samples]
y_train_original = y_data_all[-(num_test_samples + num_train_samples):-num_test_samples]

result = sm.tsa.seasonal_decompose(y_train_original, model='additive', period=7)

y_train = result.trend
seasonal_train = result.seasonal
residual_train = result.resid

sim_times_train = sim_times[-(num_test_samples + num_train_samples):-num_test_samples]

if y_train.ndim == 1:
    y_train = y_train[:, np.newaxis]

if residual_train.ndim == 1:
    residual_train = residual_train[:, np.newaxis]
valid_indices = ~np.isnan(y_train).any(axis=1)

# Filter X_train, y_train and sim_times_train using valid indices
X_train = X_train[valid_indices]
y_train = y_train[valid_indices]
seasonal_train = seasonal_train[valid_indices]
residual_train = residual_train[valid_indices]

# Generate random indices for shuffling
shuffle_indices = np.arange(X_train.shape[0])
np.random.seed(RANDOM_SEED)  # For reproducibility
np.random.shuffle(shuffle_indices)

# Shuffle X_train and y_train using these indices
X_train = X_train[shuffle_indices]
y_train = y_train[shuffle_indices]

# Split out validation set from training set
X_val = X_train[:num_val_samples]
y_val = y_train[:num_val_samples]

X_train = X_train[num_val_samples:]
y_train = y_train[num_val_samples:]

# Reshape X_data
X_train = X_train.reshape(X_train.shape[0], -1, num_qubits_after_transpile, 5)
X_val = X_val.reshape(num_val_samples, -1, num_qubits_after_transpile, 5)
new_X_test = new_X_test.reshape(50, -1, num_qubits_after_transpile, 5)

# Prepare PyTorch Datasets for training, validation, and testing
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(new_X_test, dtype=torch.float32),
                             torch.tensor(new_y_test, dtype=torch.float32))

# Data Loaders for training, validation, and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

simulator = Aer.get_backend('qasm_simulator')
shots = 2000


def calculate_expectation_value(counts, shots):
    return sum((-1) ** bin(int(state, 2)).count('1') * count for state, count in counts.items()) / shots


def simulate_for_date(date):
    expectation_values = []
    for _ in range(40):  # Run the simulation 20 times
        properties = backend.properties(datetime=date)
        noise_model = NoiseModel.from_backend_properties(properties)
        coupling_map = backend.configuration().coupling_map
        basis_gates = noise_model.basis_gates
        result = execute(qc, simulator, coupling_map=coupling_map,
                         basis_gates=basis_gates, initial_layout=[0, 1, 4, 7], noise_model=noise_model,
                         shots=shots).result()
        counts = result.get_counts(qc)
        # state_probability = counts.get(CARE_STATES[0], 0) / shots
        # probabilities.append(state_probability)
        expectation_value = calculate_expectation_value(counts, shots)
        expectation_values.append(expectation_value)
    return np.min(expectation_values), np.max(expectation_values), expectation_values


# target_values = new_y_test[:, 0]
target_values = new_y_test
actual_Z = np.zeros(target_values.shape[0])
coeffs = np.array([+1 if bin(state).count('1') % 2 == 0 else -1 for state in range(2 ** num_qubits)])
for i in range(target_values.shape[0]):
    # For each state, determine whether to add or subtract its probability bound
    actual_Z[i] = np.sum(coeffs * target_values[i, :])
# print(new_date_test.shape)
# print(target_values.shape)
# print(actual_Z.shape)
# ss
lower_bounds = []
upper_bounds = []
probabilities_list = []
# Iterate over the date list and get bounds for each date
# for date in new_date_test:
#     lower_bound, upper_bound, probability = simulate_for_date(date)
#     lower_bounds.append(lower_bound)
#     upper_bounds.append(upper_bound)
#     probabilities_list.append(probability)

df = pd.read_csv('noise_simulation_bounds_data.csv')
# Plotting the bounds
plt.figure(figsize=(10, 6))
plt.grid(axis='y', ls="--")
plt.scatter(new_date_test, actual_Z, color='black', label='Target Values (First Quantum State)')
plt.plot(new_date_test, df['Upper Bound'], label='Upper Bounds', linestyle='--', color='red')
plt.plot(new_date_test, df['Lower Bound'], label='Lower Bounds', linestyle='--', color='green')

plt.xlabel('Date')
plt.ylabel('Probability Bounds')
plt.title('Predicted Probability Bounds Over Time')
start_date = min(new_date_test)
end_date = max(new_date_test)
delta = (end_date - start_date) / 10
dates = mdates.drange(start_date, end_date + timedelta(days=1), delta)

plt.xticks(dates, [mdates.num2date(d).strftime('%Y-%m-%d') for d in dates])
plt.ylim(0.78, 0.92)
plt.yticks(np.arange(0.78, 0.92, 0.02))
# Rotate date labels for better readability
plt.xticks(rotation=45)
# plt.legend()

plt.savefig('noise_simulation.eps', format='eps', dpi=1000)
plt.show()
sss
new_date_test_str = [date.strftime("%Y-%m-%d %H:%M:%S") if isinstance(date, datetime) else str(date) for date in
                     new_date_test]

# Specify the filename
filename = 'noise_simulation_bounds_data.csv'

# Writing data to CSV
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['Date', 'Lower Bound', 'Upper Bound'])

    # Write the data
    for date, lower, upper in zip(new_date_test_str, lower_bounds, upper_bounds):
        writer.writerow([date, lower, upper])

print(f'Data saved to {filename}')

######
# Convert datetime objects in new_date_test to string
new_date_test_str = [date.strftime("%Y-%m-%d %H:%M:%S") if isinstance(date, datetime) else str(date) for date in
                     new_date_test]

filename2 = 'noise_simulation_dates_with_probabilities.csv'

max_probabilities = max(len(p) for p in probabilities_list)  # Find the maximum length of probabilities list

with open(filename2, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    header = ['Date'] + [f'Probability_{i + 1}' for i in range(max_probabilities)]
    writer.writerow(header)

    # Write the data
    for date, probabilities in zip(new_date_test_str, probabilities_list):
        row = [date] + probabilities + [None] * (
                max_probabilities - len(probabilities))  # Pad shorter lists with None
        writer.writerow(row)

print(f'Data saved to {filename2}')
