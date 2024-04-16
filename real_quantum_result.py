# from qiskit import IBMQ, transpile, Aer, assemble
from qiskit_ibm_provider import IBMProvider
from qiskit.providers import JobStatus
# from qiskit.providers.ibmq import least_busy
import time
import pandas as pd
import numpy as np
import pandas as pd
from qiskit import Aer, transpile, assemble
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from qiskit.dagcircuit import DAGCircuit
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
from torch.utils.data import DataLoader, TensorDataset
from qiskit import QuantumCircuit, execute, Aer
from qiskit.converters import circuit_to_dag
from qiskit_ibm_provider import IBMProvider
from qiskit.providers.exceptions import BackendPropertyError
from qiskit_aer.noise import NoiseModel
from datetime import datetime, timedelta
from typing import List
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from qiskit.compiler import transpile
from datetime import datetime, timedelta
import time

# Load your IBMQ account
# IBMQ.load_account()
provider = IBMProvider(
    token='2dd6975873f543e81927587054bd90f83a5fd36f5c08c0bb9628387244baa7a46131a268e9452d121fb1820d52aef423b7902587c2cf97a2f213807e82308af2')


# Create the quantum circuit
# def create_quantum_circuit():
#     qc = QuantumCircuit(3)
#     qc.h(0)
#     qc.cx(0, 1)
#     qc.cx(1, 2)
#     qc.measure_all()
#     return qc

def create_quantum_circuit():
    qc = QuantumCircuit(12)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 4)
    qc.cx(4, 5)
    qc.cx(5, 6)
    qc.cx(6, 7)
    qc.cx(7, 8)
    qc.cx(8, 9)
    qc.cx(9, 10)
    qc.cx(10, 11)
    qc.measure_all()
    return qc


qc = create_quantum_circuit()

provider = IBMProvider(instance='ibm-q-lanl/lanl/quantum-optimiza')
# backend = provider.get_backend('ibmq_qasm_simulator')
# backend = provider.get_backend('ibm_auckland')

backend = provider.get_backend('ibm_cairo')
# backend = provider.get_backend('ibm_hanoi')
# backend = provider.get_backend('ibm_algiers')
# backend = provider.get_backend('ibmq_kolkata')
# backend = provider.get_backend('ibmq_mumbai')


# backend = provider.get_backend('ibmq_guadalupe')
# backend = provider.get_backend('ibm_nairobi')
# backend = provider.get_backend('ibm_lagos')
# backend = provider.get_backend('ibm_perth')
circuits = [qc for _ in range(80)]
# Execute the list of circuits in a single job
print("Executing 80 circuits...")

# job = execute(circuits, backend, optimization_level=0, shots=10000)
#
# # Monitor the job status
# while job.status() not in [JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR]:
#     print(f"Status @ {time.strftime('%Y-%m-%d %H:%M:%S')}: {job.status().name},"
#           f" status={job.status()}")
#     time.sleep(60)  # check every 10 seconds
#
# # After executing the job, retrieve the results
# results = job.result()
# print(results)
# job = execute(circuits, backend, initial_layout=[0, 1, 4], optimization_level=0, shots=10000)
job = execute(circuits, backend, initial_layout=[0, 1, 4, 7, 10, 12, 13, 14, 16, 19, 22, 25], optimization_level=0,
              shots=10000)

# Monitor the job status
while job.status() not in [JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR]:
    print(f"Status @ {time.strftime('%Y-%m-%d %H:%M:%S')}: {job.status().name},"
          f" status={job.status()}")
    time.sleep(60)  # check every 10 seconds

# After executing the job, retrieve the results
results = job.result()
print(results)
sss
quasi_dists = results.quasi_dists

# Prepare a data structure for the desired CSV format
num_qubits = 3
possible_states = [format(i, f'0{num_qubits}b') for i in range(2 ** num_qubits)]
data_for_csv = {state: [] for state in possible_states}

for q_dist in quasi_dists:
    for state in possible_states:
        # Convert numeric state key to binary format (e.g., 0 to '000')
        state_key = int(state, 2)
        prob = q_dist.get(state_key, 0)  # Get the probability, default to 0 if not present
        data_for_csv[state].append(prob)

df = pd.DataFrame(data_for_csv)

# Save to CSV
df.to_csv('ibmq_results.csv', index=False)

print("Results saved to ibmq_results.csv!")
