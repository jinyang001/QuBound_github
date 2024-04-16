import numpy as np
import pandas as pd
import supermarq
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit_experiments.library import StandardRB
from torch.utils.data import DataLoader, TensorDataset
from qiskit import QuantumCircuit, execute, Aer
import time
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib.pyplot as plt

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

g_nrows = 2
g_cols = 2

num_epochs = 200
batch_size = 64

# x_values = [0.875, 0.9, 0.925, 0.95, 0.975, 0.999]
x_values = [0.75]


# x_values = [0.999]


def get_top_states_from_circuit(qc, num_shots=10000, num_top=4):
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=num_shots).result()
    counts = result.get_counts(qc)
    print(counts)
    # Sort the states by their counts in descending order
    sorted_states = sorted(counts.keys(), key=lambda state: counts[state], reverse=True)
    print(sorted_states)
    # Get the top 'num_top' states
    top_states = sorted_states[:num_top]
    print(top_states)
    # ss
    return top_states


def create_quantum_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc
# def create_quantum_circuit():
#     qc = QuantumCircuit(12)
#     qc.h(0)
#     qc.cx(0, 1)
#     qc.cx(1, 2)
#     qc.cx(2, 3)
#     qc.cx(3, 4)
#     qc.cx(4, 5)
#     qc.cx(5, 6)
#     qc.cx(6, 7)
#     qc.cx(7, 8)
#     qc.cx(8, 9)
#     qc.cx(9, 10)
#     qc.cx(10, 11)
#     qc.measure_all()
#     return qc


def create_rb_circuit(num_qubits=2, rb_pattern=[[0, 1]], lengths=[1], num_samples=1, seed=42):
    # Generate RB circuits (standard)
    rb_exp = StandardRB(
        physical_qubits=list(range(num_qubits)),
        lengths=lengths,
        num_samples=num_samples,
        seed=seed
    )
    rb_circs = rb_exp.circuits()

    return rb_circs


# 1. Simulate noiseless results
ghz_circuit = create_quantum_circuit()

qc = ghz_circuit

# CARE_STATES = ['000', '111']
CARE_STATES = [format(i, '03b') for i in range(2 ** 3)]
# CARE_STATES = [format(i, '012b') for i in range(2 ** 12)]
print(CARE_STATES)

num_states = 2 ** qc.num_qubits

N_STATES = len(CARE_STATES)

# Parameters to control sizes
num_train_samples = 2600
num_val_samples = 200
num_test_samples = 200
# 1. Load Data from CSV
data = pd.read_csv('../../datasets/ibmq_mumbai/GHZ_3_ibmq_mumbai.csv')
data_real = pd.read_csv('../../datasets/real_30_mumbai_new_ghz3.csv')
# Extract X_data for the given sample
X_data_real = data_real.iloc[:, :- (num_states + 7)].values
y_data_real = data_real.iloc[:, -num_states:].values
date = data_real['date'].values
# print(date.shape)
# print(date)
# ss
# print(y_data_real.shape)
# print(y_data_real)
# ss
# Reshape X_data for model input
X_data_real = X_data_real.reshape(X_data_real.shape[0], -1, 3, 5)

# print(X_data_real)
# print(X_data_real.shape)
# ss
X_tensor_real = torch.tensor(X_data_real, dtype=torch.float32)
y_tensor_real = torch.tensor(y_data_real, dtype=torch.float32)
print(data)
print(data.shape)
# Removing outliers
# Q1 = data[[f'state_{state}' for state in CARE_STATES]].quantile(0.25)
# Q3 = data[[f'state_{state}' for state in CARE_STATES]].quantile(0.75)
# IQR = Q3 - Q1
# data = data[~((data[[f'state_{state}' for state in CARE_STATES]] < (Q1 - 1.5 * IQR)) |
#               (data[[f'state_{state}' for state in CARE_STATES]] > (Q3 + 1.5 * IQR))).any(axis=1)]
# print(data)
# print(data.shape)
# ss
# Load metadata
num_qubits_info = data['num_qubits_info'].iloc[0]
num_gates_info = data['num_gates_info'].iloc[0]
num_params_info = data['num_params_info'].iloc[0]
circuit_depth_info = data['circuit_depth_info'].iloc[0]
num_qubits_after_transpile = data['num_qubits_after_transpile'].iloc[0]
sim_times = data['sim_times'].values

metadata_cols = 7

# Identifying gate error/readout error columns
error_cols = list(range(4, data.shape[1] - (num_states + metadata_cols), 5))

# data.iloc[:, error_cols] = 0  # Set gate error/readout error columns to 0

X_data_all = data.iloc[:, :- (num_states + metadata_cols)].values

care_state_cols = [f'state_{state}' for state in CARE_STATES]
print(care_state_cols)
y_data_all = data[care_state_cols].values
print(y_data_all)
print(y_data_all.shape)
# 2. Directly slice data to get test samples
X_test = X_data_all[-num_test_samples:]
y_test = y_data_all[-num_test_samples:]
sim_times_test = sim_times[-num_test_samples:]

# 3. From the remaining data, slice again to get the training samples
X_train = X_data_all[-(num_test_samples + num_train_samples):-num_test_samples]
y_train_original = y_data_all[-(num_test_samples + num_train_samples):-num_test_samples]
print(y_train_original)
print(y_train_original.shape)
# ss
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
X_test = X_test.reshape(num_test_samples, -1, num_qubits_after_transpile, 5)

# Prepare PyTorch Datasets for training, validation, and testing
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

# Data Loaders for training, validation, and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the RNN model with output units
class EnhancedRNNModel(nn.Module):
    def __init__(self, num_qubits):
        super(EnhancedRNNModel, self).__init__()

        self.hidden_size = 64
        self.num_layers = 1
        self.rnn = nn.LSTM(input_size=num_qubits_after_transpile * 5,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True,
                           bidirectional=False)
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(64, N_STATES)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)  # New shape: (batch_size, num_stages, num_qubits * 5)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, (h0, c0))

        out = out[:, -1, :]
        out = self.fc(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc4(out)
        return out


from torchmetrics import MeanSquaredLogError

model = EnhancedRNNModel(num_qubits_info)

weights = 1.0 / (y_data_all.mean(axis=0) + 1e-8)
weights = torch.tensor(weights)

criterion = nn.MSELoss(reduction='none')

criterion3 = MeanSquaredLogError()
optimizer = optim.Adam(model.parameters(), lr=0.008)

# Training loop
train_losses = []
save_model = model
best_model_state = None
minimum_loss = 999999
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        # loss = (criterion(outputs, y_batch) * weights).mean()
        loss = criterion3(outputs, y_batch)
        if loss.item() < minimum_loss:
            minimum_loss = loss.item()
            save_model = model
            best_model_state = model.state_dict()
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    train_losses.append(loss.item())
model = save_model
# After training
if best_model_state:
    torch.save(best_model_state, 'best_model_state_GHZ3_test_all_two.pth')
    print('Best model state saved.')
else:
    print('No model state to save.')

print('Training finished.')

# Plotting the training loss
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# # Evaluate the validation dataset
train_predicted_results = []
train_actual_results = []
#
# model.load_state_dict(torch.load('best_model_state_GHZ3_test_all_two.pth'))

model.eval()

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        loss = criterion3(outputs, y_batch)
        train_predicted_results.extend(outputs.numpy())
        train_actual_results.extend(y_batch.numpy())
        # break  # Only evaluate one batch for inspection
train_predicted_results = np.array(train_predicted_results)
train_actual_results = np.array(train_actual_results)

# Test the model
model.eval()
test_loss = 0
predicted_results = []
actual_results = []
prediction_times = []
with torch.no_grad():
    for X_test_batch, y_test_batch in test_loader:
        start_time = time.time()
        outputs = model(X_test_batch)

        end_time = time.time()
        avg_prediction_time_per_sample = (end_time - start_time) / len(X_test_batch)
        prediction_times.extend([avg_prediction_time_per_sample] * len(X_test_batch))
        loss = criterion3(outputs, y_test_batch)
        test_loss += loss.item()
        predicted_results.extend(outputs.numpy())
        actual_results.extend(y_test_batch.numpy())

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# Convert to numpy arrays for easier handling
predicted_results = np.array(predicted_results)
actual_results = np.array(actual_results)

average_baseline_latency = np.mean(sim_times_test)
average_our_method_latency = np.mean(prediction_times)
improvement_latency = (average_baseline_latency - average_our_method_latency) / average_baseline_latency * 100


def calculate_bcr_bdp(actual, lower_bound, upper_bound):
    # Number of samples where actual values are within the bounds
    within_bounds = np.sum((lower_bound <= actual) & (actual <= upper_bound))

    # Calculate BCR
    bcr = (within_bounds / len(actual)) * 100

    # Calculate BDP
    deviation = np.where(actual < lower_bound, lower_bound - actual,
                         np.where(actual > upper_bound, actual - upper_bound, 0))

    # Normalize the BDP
    max_possible_deviation = np.max(actual) - np.min(actual)
    normalized_bdp = np.sum(deviation) / (len(actual) * max_possible_deviation) * 100

    return bcr, normalized_bdp


def calculate_bcr_bdp_for_each_state(predicted_upper, predicted_lower, actual_values):
    statewise_bcr = []
    statewise_bdp = []

    for idx in range(predicted_upper.shape[1]):  # Loop over states
        bcr, bdp = calculate_bcr_bdp(actual_values[:, idx], predicted_lower[:, idx], predicted_upper[:, idx])
        statewise_bcr.append(bcr)
        statewise_bdp.append(bdp)

    return statewise_bcr, statewise_bdp


def calculate_expectation_bounds(predicted_upper_bounds, predicted_lower_bounds, actual_values, num_qubits):
    # Initialize arrays to hold the expectation bounds for each sample
    predicted_upper_Z = np.zeros(predicted_upper_bounds.shape[0])
    predicted_lower_Z = np.zeros(predicted_lower_bounds.shape[0])
    actual_Z = np.zeros(actual_values.shape[0])

    # Coefficients for the Z expectation value: +1 for even parity, -1 for odd parity
    # For a 3-qubit system, the parity for '00' and '11' is even (0 and 2 1s respectively), and for '01' and '10' is odd (1 1)
    coeffs = np.array([+1 if bin(state).count('1') % 2 == 0 else -1 for state in range(2 ** num_qubits)])

    # Calculate expectation bounds and actual for each sample
    for i in range(predicted_upper_bounds.shape[0]):
        # For each state, determine whether to add or subtract its probability bound
        predicted_upper_Z[i] = np.sum(
            coeffs * predicted_upper_bounds[i, :] * (coeffs > 0) + coeffs * predicted_lower_bounds[i, :] * (coeffs < 0))
        predicted_lower_Z[i] = np.sum(
            coeffs * predicted_lower_bounds[i, :] * (coeffs > 0) + coeffs * predicted_upper_bounds[i, :] * (coeffs < 0))
        actual_Z[i] = np.sum(coeffs * actual_values[i, :])

    return predicted_upper_Z.reshape(-1, 1), predicted_lower_Z.reshape(-1, 1), actual_Z.reshape(-1, 1)


for x in x_values:
    print(f"Test result for xvalues{x}\n")
    z_score = norm.ppf(x)

    residual_upper_bounds = []
    residual_lower_bounds = []
    for i in range(residual_train.shape[1]):
        residual_state = residual_train[:, i]
        residual_mean = np.nanmean(residual_state)
        residual_std = np.nanstd(residual_state)

        residual_ci = z_score * residual_std

        residual_upper_bounds.append(residual_mean + residual_ci)
        residual_lower_bounds.append(residual_mean - residual_ci)
        # print(residual_mean + residual_ci)
        # print(residual_mean + residual_ci)
        # ss
    # Feed the processed data to the trained model
    with torch.no_grad():
        model.eval()
        predictions = model(X_tensor_real)

    states_to_plot = ["000", "001", "010", "011","100","101","110","111"]
    indices_to_plot = [CARE_STATES.index(state) for state in states_to_plot]

    print(predictions)
    predictions = np.array(predictions)
    predicted_upper_bound_real = predictions + residual_upper_bounds
    predicted_lower_bound_real = predictions + residual_lower_bounds

    predicted_upper_bound_selected_real = predicted_upper_bound_real[:, indices_to_plot]
    predicted_lower_bound_selected_real = predicted_lower_bound_real[:, indices_to_plot]
    actual_results = np.array(y_tensor_real)
    actual_selected = actual_results[:, indices_to_plot]
    print(predicted_upper_bound_selected_real)
    print(predicted_lower_bound_selected_real)
    print(actual_selected)
    # predicted_upper_Z, predicted_lower_Z, actual_Z = calculate_expectation_bounds(
    #     predicted_upper_bound_selected_real, predicted_lower_bound_selected_real, actual_selected, qc.num_qubits
    # )
    # print(predicted_upper_Z)
    # print(predicted_lower_Z)
    # print(actual_Z)

    # Set confidence level for the plot title (replace x with your confidence value)
    confidence_level = (1 - 2 * (1 - x)) * 100
    # print(confidence_level)
    # ss
    # Create a single plot for the expectation value
    plt.figure(figsize=(10, 6))

    print(date.shape)
    # print(predicted_upper_Z.shape)

    plt.plot(date, predicted_upper_bound_selected_real, linestyle='--', label='Predicted Upper Bound_Z', color='red')
    plt.plot(date, predicted_lower_bound_selected_real, linestyle='--', label='Predicted Lower Bound_Z', color='green')
    plt.plot(date, actual_selected, label='Test Actual_Z', color='blue')

    plt.xlabel('Sample')
    plt.ylabel('Expectation Value')
    plt.legend()
    plt.title(f'Test Data_Z: Actual vs Predicted with Bounds (Confidence: {confidence_level}%)')

    plt.tight_layout()
    plt.show()

    if len(date.shape) == 1:
        date = date.reshape(-1, 1)
    print(predicted_upper_bound_selected_real.shape)
    # Combine the arrays into a DataFrame
    data_to_save = pd.DataFrame({
        'Date': date.flatten(),  # Flatten to convert (10, 1) to (10,)
        'Lower Bound_0x0': predicted_lower_bound_selected_real[:, 0],
        'Upper Bound_0x0': predicted_upper_bound_selected_real[:, 0],
        'Lower Bound_0x1': predicted_lower_bound_selected_real[:, 1],
        'Upper Bound_0x1': predicted_upper_bound_selected_real[:, 1],
        'Lower Bound_0x2': predicted_lower_bound_selected_real[:, 2],
        'Upper Bound_0x2': predicted_upper_bound_selected_real[:, 2],
        'Lower Bound_0x3': predicted_lower_bound_selected_real[:, 3],
        'Upper Bound_0x3': predicted_upper_bound_selected_real[:, 3],

        'Lower Bound_0x4': predicted_lower_bound_selected_real[:, 4],
        'Upper Bound_0x4': predicted_upper_bound_selected_real[:, 4],
        'Lower Bound_0x5': predicted_lower_bound_selected_real[:, 5],
        'Upper Bound_0x5': predicted_upper_bound_selected_real[:, 5],
        'Lower Bound_0x6': predicted_lower_bound_selected_real[:, 6],
        'Upper Bound_0x6': predicted_upper_bound_selected_real[:, 6],
        'Lower Bound_0x7': predicted_lower_bound_selected_real[:, 7],
        'Upper Bound_0x7': predicted_upper_bound_selected_real[:, 7],



    })

    # Write the DataFrame to a CSV file
    data_to_save.to_csv('output_3_test_all.csv', index=False)

    SSS

    # Convert the lists to numpy arrays for vectorized operations
    residual_upper_bounds = np.array(residual_upper_bounds)
    residual_lower_bounds = np.array(residual_lower_bounds)

    # Adjust the upper and lower bound computation
    predicted_upper_bound = predicted_results + residual_upper_bounds
    predicted_lower_bound = predicted_results + residual_lower_bounds
    bound_distance = residual_upper_bounds - residual_lower_bounds
    # print(predicted_upper_bound.shape)
    # print(bound_distance.shape)
    from numpy import mean

    print(f"average bound distance{mean(bound_distance)}\n")

    train_predicted_selected = train_predicted_results[:, indices_to_plot]
    train_actual_selected = train_actual_results[:, indices_to_plot]

    predicted_selected = predicted_results[:, indices_to_plot]
    actual_selected = actual_results[:, indices_to_plot]
    predicted_upper_bound_selected = predicted_upper_bound[:, indices_to_plot]
    predicted_lower_bound_selected = predicted_lower_bound[:, indices_to_plot]
    # bound_distance_selected=bound_distance[indices_to_plot]
    # Number of states to plot
    num_states = len(states_to_plot)

    # Create subplots for the train data
    fig, axes = plt.subplots(nrows=g_nrows, ncols=g_cols, figsize=(15, 12))
    for idx, (ax, state) in enumerate(zip(axes.ravel(), states_to_plot)):
        ax.plot(train_predicted_selected[:, idx], label=f'Train Predicted {state}', color='green')
        ax.plot(train_actual_selected[:, idx], label=f'Train Actual {state}', color='blue')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.legend()
        ax.set_title(f'Train Data: Actual vs Predicted for {state}')
    plt.tight_layout()
    plt.show()

    # Create subplots for the test data
    fig, axes = plt.subplots(nrows=g_nrows, ncols=g_cols, figsize=(15, 12))
    for idx, (ax, state) in enumerate(zip(axes.ravel(), states_to_plot)):
        ax.plot(predicted_upper_bound_selected[:, idx], linestyle='--', label='Predicted Upper Bound', color='red')
        ax.plot(predicted_lower_bound_selected[:, idx], linestyle='--', label='Predicted Lower Bound', color='green')
        # ax.plot(predicted_selected[:, idx], label=f'Test Predicted {state}', color='orange')
        ax.plot(actual_selected[:, idx], label=f'Test Actual {state}', color='blue')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.legend()
        ax.set_title(f'Test Data: Actual vs Predicted for {state} with Bounds (Confidence: {(1 - 2 * (1 - x)) * 100}%)')

        print(f'bound distance for {state}={bound_distance[idx]}')
    plt.tight_layout()
    plt.show()

    # Using the function with your data
    statewise_bcr, statewise_bdp = calculate_bcr_bdp_for_each_state(predicted_upper_bound_selected,
                                                                    predicted_lower_bound_selected, actual_selected)

    # Printing the results
    total_bcr = sum(statewise_bcr)
    average_bcr = total_bcr / len(statewise_bcr)

    total_bdp = sum(statewise_bdp)
    average_bdp = total_bdp / len(statewise_bdp)

    # Print the statewise BCR and BDP
    for state, bcr, bdp in zip(states_to_plot, statewise_bcr, statewise_bdp):
        print(f"For state {state}:")
        print(f"Bound-Compliance Rate (BCR): {bcr:.2f}%")
        print(f"Bound-Deviation Penalty (BDP): {bdp:.2f}%")
        print("----------")

    # Print the average BCR and BDP
    print(f"Average Bound-Compliance Rate (BCR) for all states: {average_bcr:.2f}%")
    print(f"Average Bound-Deviation Penalty (BDP) for all states: {average_bdp:.2f}%")


    def simulate_noiseless(qc: QuantumCircuit, num_shots: int = 10000):
        """Simulates a given quantum circuit without any noise."""
        simulator = Aer.get_backend('qasm_simulator')
        result = execute(qc, simulator, shots=num_shots).result()
        counts = result.get_counts(qc)

        num_qubits = qc.num_qubits
        num_possible_states = 2 ** num_qubits
        all_states_counts = [counts.get(bin(i)[2:].zfill(num_qubits), 0) for i in range(num_possible_states)]
        total_counts = sum(all_states_counts)
        all_states_probabilities = [count / total_counts for count in all_states_counts]
        target_probs = [all_states_probabilities[int(state, 2)] for state in CARE_STATES]

        return target_probs


    noiseless_simulation_results = [simulate_noiseless(qc) for _ in range(num_test_samples)]
    noiseless_simulation_results = np.array(noiseless_simulation_results)

    # 2. Calculate the average value for each state
    avg_values = np.mean(noiseless_simulation_results, axis=0)

    # 3. Generate straight lines using the average value and residuals
    noiseless_upper_bound = avg_values + residual_upper_bounds[:num_test_samples]
    noiseless_lower_bound = avg_values + residual_lower_bounds[:num_test_samples]

    fig, axes = plt.subplots(nrows=g_nrows, ncols=g_cols, figsize=(15, 12))
    for idx, (ax, state) in enumerate(zip(axes.ravel(), CARE_STATES)):
        # ax.plot([avg_values[idx]] * 80, label=f'Average {state}', color='purple')  # Straight line for average
        ax.plot(actual_selected[:, idx], label=f'Test Actual {state}', color='blue')
        ax.plot([noiseless_upper_bound[idx]] * num_test_samples, linestyle='--', label='Upper Bound', color='red')
        ax.plot([noiseless_lower_bound[idx]] * num_test_samples, linestyle='--', label='Lower Bound', color='green')

        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.legend()
        ax.set_title(f'Average and Bounds for {state}')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    predicted_upper_agnostic = np.array([noiseless_upper_bound] * num_test_samples)
    predicted_lower_agnostic = np.array([noiseless_lower_bound] * num_test_samples)

    # Using the function with your data
    statewise_bcr_agnostic, statewise_bdp_agnostic = calculate_bcr_bdp_for_each_state(predicted_upper_agnostic,
                                                                                      predicted_lower_agnostic,
                                                                                      actual_selected)

    total_bcr_agnostic = sum(statewise_bcr_agnostic)
    average_bcr_agnostic = total_bcr_agnostic / len(statewise_bcr_agnostic)

    total_bdp_agnostic = sum(statewise_bdp_agnostic)
    average_bdp_agnostic = total_bdp_agnostic / len(statewise_bdp_agnostic)

    # Printing the results
    for state, bcr, bdp in zip(states_to_plot, statewise_bcr_agnostic, statewise_bdp_agnostic):
        print(f"For state {state}:")
        print(f"Bound-Compliance Rate for agnostic (BCR): {bcr:.2f}%")
        print(f"Bound-Deviation Penalty for agnostic (BDP): {bdp:.2f}%")
        print("----------")

    # Print the average BCR and BDP
    print(f"Average Bound-Compliance Rate for agnostic (BCR) for all states: {average_bcr_agnostic:.2f}%")
    print(f"Average Bound-Deviation Penalty for agnostic (BDP) for all states: {average_bdp_agnostic:.2f}%")
