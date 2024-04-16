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
from numpy import mean
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

g_nrows = 4
g_cols = 2

num_epochs = 200
batch_size = 64

# x_values = [0.875, 0.9, 0.925, 0.95, 0.975, 0.999]
# x_values = [0.875]
x_values = [0.975]

CARE_STATES = [format(i, '04b') for i in range(2 ** 4)]
num_qubits=4

print(CARE_STATES)

num_states = 2 ** num_qubits

N_STATES = len(CARE_STATES)

# Parameters to control sizes
num_train_samples = 2600
num_val_samples = 200
num_test_samples = 200
# 1. Load Data from CSV
# data = pd.read_csv('../../datasets/ibmq_kolkata/10000shots/GHZ4_ibmq_kolkata.csv')
data = pd.read_csv('../../datasets/ibmq_kolkata/1000shots/GHZ4_ibmq_kolkata.csv')
# data = pd.read_csv('../../datasets/ibm_hanoi/GHZ_3_ibm_hanoi.csv')
# ss
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
# X_test = X_test.reshape(num_test_samples, -1, num_qubits_after_transpile, 5) ### 200 test samples
new_X_test = new_X_test.reshape(50, -1, num_qubits_after_transpile, 5)  ### 50 test samples

# Prepare PyTorch Datasets for training, validation, and testing
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
# test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)) ### 200 test samples

test_dataset = TensorDataset(torch.tensor(new_X_test, dtype=torch.float32),
                             torch.tensor(new_y_test, dtype=torch.float32))  ### 50 test samples
# Data Loaders for training, validation, and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the RNN model with output units
class EnhancedRNNModel(nn.Module):
    def __init__(self, num_qubits):
        super(EnhancedRNNModel, self).__init__()
        self.hidden_size = 64  # Increased hidden units
        self.num_layers = 1  # Number of LSTM layers
        # Bi-directional
        self.rnn = nn.LSTM(input_size=num_qubits_after_transpile * 5,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True,
                           bidirectional=False)
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, N_STATES)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Reshape x to flatten the last two dimensions
        x = x.view(x.size(0), x.size(1), -1)  # New shape: (batch_size, num_stages, num_qubits * 5)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        # Take the output from the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


from pytorch_forecasting.metrics.point import MAPE
from torchmetrics import MeanSquaredLogError

# Initialize the model, loss function, and optimizer
model = EnhancedRNNModel(num_qubits_info)
criterion3 = MeanSquaredLogError()
optimizer = optim.Adam(model.parameters(), lr=0.008)

# Training loop
train_losses = []
save_model = model
best_model_state = None
minimum_loss = 999999
start_times=time.time()
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
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
# if best_model_state:
#     torch.save(best_model_state, 'best_model_state_GHZ4.pth')
#     print('Best model state saved.')
# else:
#     print('No model state to save.')
end_times = time.time()

print('Training finished.')
print(end_times-start_times)

# Plotting the training loss
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the validation dataset
train_predicted_results = []
train_actual_results = []

# model.load_state_dict(torch.load('best_model_state_GHZ4.pth'))

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
time1 = 0
with torch.no_grad():
    for X_test_batch, y_test_batch in test_loader:
        start_time = time.time()
        outputs = model(X_test_batch)

        end_time = time.time()
        avg_prediction_time_per_sample = (end_time - start_time) / len(X_test_batch)
        time1 = avg_prediction_time_per_sample
        print(f"Average Prediction Time per Sample: {avg_prediction_time_per_sample:.6f} seconds")
        # ss
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


def format_number(num):
    """Formats number in scientific notation if it's too small, otherwise in normal float format."""
    return "{:.4e}".format(num) if num < 1e-4 else "{:.4f}".format(num)


# Print headers (column names) first. You can run this only once.
print("Metric\tBaseline\tOur Method\t% Improvement")

# Values
print(f"Qubits\t{num_qubits_info}\t-\t-")
print(f"Gates\t{num_gates_info}\t-\t-")
print(f"Parameters\t{num_params_info}\t-\t-")
print(f"Circuit Depth\t{circuit_depth_info}\t-\t-")
print(
    f"Average Time (s)\t{average_baseline_latency:.4f}\t{format_number(average_our_method_latency)}\t{improvement_latency:.2f}%")


# def calculate_bcr_bdp(actual, lower_bound, upper_bound):
#     # Number of samples where actual values are within the bounds
#     within_bounds = np.sum((lower_bound <= actual) & (actual <= upper_bound))
#
#     # Calculate BCR
#     bcr = (within_bounds / len(actual)) * 100
#
#     # Calculate BDP
#     deviation = np.where(actual < lower_bound, lower_bound - actual,
#                          np.where(actual > upper_bound, actual - upper_bound, 0))
#
#     # Normalize the BDP
#     max_possible_deviation = np.max(actual) - np.min(actual)
#     normalized_bdp = np.sum(deviation) / (len(actual) * max_possible_deviation) * 100
#
#     return bcr, normalized_bdp
def calculate_bcr_bdp(actual, lower_bound, upper_bound):
    # Number of samples where actual values are within the bounds
    within_bounds = np.sum((lower_bound <= actual) & (actual <= upper_bound))

    # Calculate BCR
    bcr = (within_bounds / len(actual)) * 100

    # Calculate raw BDP (handle cases where max_possible_deviation is zero)
    deviation = np.where(actual < lower_bound, lower_bound - actual,
                         np.where(actual > upper_bound, actual - upper_bound, 0))
    max_possible_deviation = np.max(actual) - np.min(actual)

    if max_possible_deviation > 0:
        normalized_bdp = np.sum(deviation) / (len(actual) * max_possible_deviation) * 100
    else:
        # Handle the case where all actual values are the same
        normalized_bdp = np.mean(deviation) if np.any(deviation) else 0  # This could be zero if there are no deviations

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


# def simulate_noiseless(qc: QuantumCircuit, num_shots: int = 10000):
#     """Simulates a given quantum circuit without any noise."""
#     simulator = Aer.get_backend('qasm_simulator')
#     result = execute(qc, simulator, shots=num_shots).result()
#     counts = result.get_counts(qc)
#
#     num_qubits = qc.num_qubits
#     num_possible_states = 2 ** num_qubits
#     all_states_counts = [counts.get(bin(i)[2:].zfill(num_qubits), 0) for i in range(num_possible_states)]
#     total_counts = sum(all_states_counts)
#     all_states_probabilities = [count / total_counts for count in all_states_counts]
#     target_probs = [all_states_probabilities[int(state, 2)] for state in CARE_STATES]
#
#     return target_probs
#
#
# noiseless_simulation_results = [simulate_noiseless(qc) for _ in range(num_test_samples)]
# noiseless_simulation_results = np.array(noiseless_simulation_results)

for x in x_values:
    print(f"Test result for xvalues{x}\n")
    z_score = norm.ppf(x)
    print(z_score)
    ss
    residual_upper_bounds = []
    residual_lower_bounds = []
    for i in range(residual_train.shape[1]):
        residual_state = residual_train[:, i]
        residual_mean = np.nanmean(residual_state)
        residual_std = np.nanstd(residual_state)

        residual_ci = z_score * residual_std

        residual_upper_bounds.append(residual_mean + residual_ci)
        residual_lower_bounds.append(residual_mean - residual_ci)

    # Convert the lists to numpy arrays for vectorized operations
    residual_upper_bounds = np.array(residual_upper_bounds)
    residual_lower_bounds = np.array(residual_lower_bounds)

    # Adjust the upper and lower bound computation
    predicted_upper_bound = predicted_results + residual_upper_bounds
    predicted_lower_bound = predicted_results + residual_lower_bounds
    bound_distance = residual_upper_bounds - residual_lower_bounds
    # print(predicted_upper_bound.shape)
    # print(bound_distance.shape)

    print(f"average bound distance{mean(bound_distance)}\n")
    states_to_plot = CARE_STATES
    indices_to_plot = [CARE_STATES.index(state) for state in states_to_plot]

    train_predicted_selected = train_predicted_results[:, indices_to_plot]
    train_actual_selected = train_actual_results[:, indices_to_plot]

    predicted_selected = predicted_results[:, indices_to_plot]
    print(actual_results.shape)
    actual_selected = actual_results[:, indices_to_plot]
    print(actual_results.shape)
    predicted_upper_bound_selected = predicted_upper_bound[:, indices_to_plot]
    predicted_lower_bound_selected = predicted_lower_bound[:, indices_to_plot]
    # bound_distance_selected=bound_distance[indices_to_plot]
    start = time.time()
    # Calculate the bounds and actual value for the expectation of Z observable
    predicted_upper_Z, predicted_lower_Z, actual_Z = calculate_expectation_bounds(
        predicted_upper_bound_selected, predicted_lower_bound_selected, actual_selected, num_qubits
    )
    end = time.time()
    time2 = (end - start) / 50
    print(f"Whole Average Prediction Time per Sample: {time1 + time2:.6f} seconds")
    # ss
    statewise_bcr_Z, statewise_bdp_Z = calculate_bcr_bdp_for_each_state(
        np.array([predicted_upper_Z]), np.array([predicted_lower_Z]), np.array([actual_Z])
    )

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

        # print(f'bound distance for {state}={bound_distance[idx]}')
    plt.tight_layout()
    plt.show()

    # Set confidence level for the plot title (replace x with your confidence value)
    confidence_level = (1 - 2 * (1 - x)) * 100

    # Create a single plot for the expectation value
    plt.figure(figsize=(10, 6))

    plt.plot(predicted_upper_Z, linestyle='--', label='Predicted Upper Bound_Z', color='red')
    plt.plot(predicted_lower_Z, linestyle='--', label='Predicted Lower Bound_Z', color='green')
    plt.plot(actual_Z, label='Test Actual_Z', color='blue')

    plt.xlabel('Sample')
    plt.ylabel('Expectation Value')
    plt.legend()
    plt.title(f'Test Data_Z: Actual vs Predicted with Bounds (Confidence: {confidence_level}%)')

    plt.tight_layout()
    plt.show()

    ###############
    # Plotting the bounds
    plt.figure(figsize=(10, 6))
    plt.grid(axis='y', ls="--")
    plt.scatter(new_date_test, actual_Z, color='black', label='Target Values (ZZZ Observable)')
    plt.plot(new_date_test, predicted_upper_Z, label='Upper Bounds', linestyle='--', color='red')
    plt.plot(new_date_test, predicted_lower_Z, label='Lower Bounds', linestyle='--', color='green')

    plt.xlabel('Date')
    plt.ylabel('Probability Bounds')
    plt.title('Predicted Probability Bounds Over Time')
    start_date = min(new_date_test)
    end_date = max(new_date_test)
    delta = (end_date - start_date) / 10
    dates = mdates.drange(start_date, end_date + timedelta(days=1), delta)

    plt.xticks(dates, [mdates.num2date(d).strftime('%Y-%m-%d') for d in dates])

    # Rotate date labels for better readability
    plt.xticks(rotation=45)

    # plt.savefig('figure1_4.eps', format='eps', dpi=1000)
    plt.show()

    # Desired date for extraction
    desired_date = datetime(2023, 9, 13)

    # Find the index of the desired date
    date_index = None
    for i, date in enumerate(new_date_test):
        # Compare only the year, month, and day
        if date.year == desired_date.year and date.month == desired_date.month and date.day == desired_date.day:
            date_index = i
            break

    if date_index is not None:
        # Extract the upper and lower bounds for the desired date
        upper_bound = predicted_upper_Z[date_index]
        lower_bound = predicted_lower_Z[date_index]
        actual = actual_Z[date_index]
        # Calculate the range
        bound_range = upper_bound - lower_bound

        # Print the results
        print(f"Date: {desired_date.strftime('%Y-%m-%d')}")
        print(f"Upper Bound: {upper_bound}")
        print(f"Lower Bound: {lower_bound}")
        print(f"Bound Range: {bound_range}")
        print(f"Actual Result: {actual}")
    else:
        print(f"Date {desired_date.strftime('%Y-%m-%d')} not found in the dataset.")

    ###############

    # Using the function with your data
    statewise_bcr, statewise_bdp = calculate_bcr_bdp_for_each_state(predicted_upper_bound_selected,
                                                                    predicted_lower_bound_selected, actual_selected)

    # Printing the results
    total_bcr = sum(statewise_bcr)
    average_bcr = total_bcr / len(statewise_bcr)

    total_bdp = sum(statewise_bdp)
    average_bdp = total_bdp / len(statewise_bdp)

    # Print the statewise BCR and BDP
    # for state, bcr, bdp in zip(states_to_plot, statewise_bcr, statewise_bdp):
    #     print(f"For state {state}:")
    #     print(f"Bound-Compliance Rate (BCR): {bcr:.2f}%")
    #     print(f"Bound-Deviation Penalty (BDP): {bdp:.2f}%")
    #     print("----------")

    # Print the average BCR and BDP
    print(f"Average Bound-Compliance Rate (BCR) for all states: {average_bcr:.2f}%")
    print(f"Average Bound-Deviation Penalty (BDP) for all states: {average_bdp:.2f}%")

    # Print the average BCR and BDP
    average_bcr_Z = np.mean(statewise_bcr_Z)
    non_zero_bdp = [value for value in statewise_bdp_Z if value != 0]
    average_non_zero_bdp = sum(non_zero_bdp) / len(non_zero_bdp) if non_zero_bdp else 0
    print("----------")
    print(f"Average Bound-Compliance Rate (BCR) for Z Observable: {average_bcr_Z:.2f}%")
    print(f"Average Bound-Deviation Penalty (BDP) for Z Observable: {average_non_zero_bdp:.5f}%")

    # # 2. Calculate the average value for each state
    # avg_values = np.mean(noiseless_simulation_results, axis=0)
    #
    # # 3. Generate straight lines using the average value and residuals
    # noiseless_upper_bound = avg_values + residual_upper_bounds[:num_test_samples]
    # noiseless_lower_bound = avg_values + residual_lower_bounds[:num_test_samples]
    #
    # fig, axes = plt.subplots(nrows=g_nrows, ncols=g_cols, figsize=(15, 12))
    # for idx, (ax, state) in enumerate(zip(axes.ravel(), CARE_STATES)):
    #     # ax.plot([avg_values[idx]] * 80, label=f'Average {state}', color='purple')  # Straight line for average
    #     ax.plot(actual_selected[:, idx], label=f'Test Actual {state}', color='blue')
    #     ax.plot([noiseless_upper_bound[idx]] * num_test_samples, linestyle='--', label='Upper Bound', color='red')
    #     ax.plot([noiseless_lower_bound[idx]] * num_test_samples, linestyle='--', label='Lower Bound', color='green')
    #
    #     ax.set_xlabel('Sample')
    #     ax.set_ylabel('Value')
    #     ax.legend()
    #     ax.set_title(f'Average and Bounds for {state}')
    #     ax.grid(True)
    #
    # plt.tight_layout()
    # plt.show()
    #
    # predicted_upper_agnostic = np.array([noiseless_upper_bound] * num_test_samples)
    # predicted_lower_agnostic = np.array([noiseless_lower_bound] * num_test_samples)
    #
    # # Using the function with your data
    # statewise_bcr_agnostic, statewise_bdp_agnostic = calculate_bcr_bdp_for_each_state(predicted_upper_agnostic,
    #                                                                                   predicted_lower_agnostic,
    #                                                                                   actual_selected)
    #
    # predicted_upper_Z_agnostic, predicted_lower_Z_agnostic, actual_Z_agnostic = calculate_expectation_bounds(
    #     predicted_upper_agnostic, predicted_lower_agnostic, actual_selected, qc.num_qubits
    # )
    #
    # statewise_bcr_Z_agnostic, statewise_bdp_Z_agnostic = calculate_bcr_bdp_for_each_state(
    #     np.array([predicted_upper_Z_agnostic]), np.array([predicted_lower_Z_agnostic]), np.array([actual_Z_agnostic])
    # )
    #
    #
    # total_bcr_agnostic = sum(statewise_bcr_agnostic)
    # average_bcr_agnostic = total_bcr_agnostic / len(statewise_bcr_agnostic)
    #
    # total_bdp_agnostic = sum(statewise_bdp_agnostic)
    # average_bdp_agnostic = total_bdp_agnostic / len(statewise_bdp_agnostic)
    #
    # # Printing the results
    # # for state, bcr, bdp in zip(states_to_plot, statewise_bcr_agnostic, statewise_bdp_agnostic):
    # #     print(f"For state {state}:")
    # #     print(f"Bound-Compliance Rate for agnostic (BCR): {bcr:.2f}%")
    # #     print(f"Bound-Deviation Penalty for agnostic (BDP): {bdp:.2f}%")
    # #     print("----------")
    #
    # # Print the average BCR and BDP
    # print(f"Average Bound-Compliance Rate for agnostic (BCR) for all states: {average_bcr_agnostic:.2f}%")
    # print(f"Average Bound-Deviation Penalty for agnostic (BDP) for all states: {average_bdp_agnostic:.2f}%")
    #
    # average_bcr_Z_agnostic = np.mean(statewise_bcr_Z_agnostic)
    # non_zero_bdp_agnostic = [value for value in statewise_bdp_Z_agnostic if value != 0]
    # average_non_zero_bdp_agnostic = sum(non_zero_bdp_agnostic) / len(non_zero_bdp_agnostic) if non_zero_bdp_agnostic else 0
    # print("----------")
    # print(f"Average Bound-Compliance Rate (BCR) for Z Observable for agnostic: {average_bcr_Z_agnostic:.2f}%")
    # print(f"Average Bound-Deviation Penalty (BDP) for Z Observable for agnostic: {average_non_zero_bdp_agnostic:.5f}%")
