import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from qiskit_aer.noise import NoiseModel
from qiskit_experiments.library import StandardRB
from qiskit_ibm_provider import IBMProvider
from qiskit import QuantumCircuit, execute, Aer
from datetime import datetime
import time
import supermarq


# Function to calculate the expectation value
def calculate_expectation_value(counts, shots):
    return sum((-1) ** bin(int(state, 2)).count('1') * count for state, count in counts.items()) / shots


# Convert the SuperMarq VQE circuit to Qiskit format
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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
# def create_quantum_circuit():
#     qc = QuantumCircuit(3)
#     qc.h(0)
#     qc.cx(0, 1)
#     qc.cx(1, 2)
#     qc.measure_all()
#     return qc


# def create_quantum_circuit():
#     qc = QuantumCircuit(6)
#     qc.h(0)
#     qc.cx(0, 1)
#     qc.cx(1, 2)
#     qc.cx(2, 3)
#     qc.cx(3, 4)
#     qc.cx(4, 5)
#     qc.measure_all()
#     return qc

# def create_quantum_circuit():
#     qc = QuantumCircuit(9)
#     qc.h(0)
#     qc.cx(0, 1)
#     qc.cx(1, 2)
#     qc.cx(2, 3)
#     qc.cx(3, 4)
#     qc.cx(4, 5)
#     qc.cx(5, 6)
#     qc.cx(6, 7)
#     qc.cx(7, 8)
#     qc.measure_all()
#     return qc

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

def create_quantum_circuit():
    qc = QuantumCircuit(15)
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

    qc.cx(11, 12)
    qc.cx(12, 13)
    qc.cx(13, 14)
    qc.measure_all()
    return qc

# rb_circuits = create_rb_circuit(num_qubits=3, rb_pattern=[[0],[1],[2]])
# hs_qiskit_circuit = supermarq.converters.cirq_to_qiskit(
#     supermarq.hamiltonian_simulation.HamiltonianSimulation(4).circuit())
# vqe_qiskit_circuit = supermarq.converters.cirq_to_qiskit(supermarq.vqe_proxy.VQEProxy(4, 1).circuit()[0])
# qaoa_qiskit_circuit = supermarq.converters.cirq_to_qiskit(
#     supermarq.qaoa_vanilla_proxy.QAOAVanillaProxy(4).circuit())

# qc=hs_qiskit_circuit
# qc=vqe_qiskit_circuit
# qc = qaoa_qiskit_circuit
# qc=rb_circuits[0]
qc = create_quantum_circuit()
provider = IBMProvider()
backend = provider.get_backend('ibmq_mumbai')
t = datetime(day=13, month=9, year=2023, hour=0)

# Define simulation parameters
shots = 1000
total_runs = 10  # Fixed number of runs
data = []  # To store data for each run

simulator = Aer.get_backend('qasm_simulator')
total_exec_time = 0
# [0,1,4,7,10,12,13,14,16,19,22,25,24,23,21]
# Run the simulation
for _ in range(total_runs):
    start_time = time.time()
    properties = backend.properties(datetime=t)
    noise_model = NoiseModel.from_backend_properties(properties)
    coupling_map = backend.configuration().coupling_map
    basis_gates = noise_model.basis_gates
    result = execute(qc, simulator, coupling_map=coupling_map,
                     basis_gates=basis_gates, initial_layout=[0,1,4,7,10,12,13,14,16,19,22,25,24,23,21], noise_model=noise_model,
                     shots=shots, routing_method='basic').result()
    counts = result.get_counts(qc)

    # Calculate execution time and expectation value
    exec_time = time.time() - start_time
    total_exec_time += exec_time
    expectation_value = calculate_expectation_value(counts, shots)

    # Prepare the row data
    row = [exec_time, expectation_value] + [counts.get(bin(i)[2:].zfill(qc.num_qubits), 0) / shots for i
                                            in range(2 ** qc.num_qubits)]
    data.append(row)
print("Total Execution Time:", total_exec_time, "seconds")
print("Total Execution Time:", 2000*total_exec_time, "seconds")
sss
# Create DataFrame and save to CSV
column_names = ['Execution Time', 'Expectation Value'] + [f'State_{i}' for i in
                                                          range(2 ** qc.num_qubits)]
df = pd.DataFrame(data, columns=column_names)
df.to_csv('noise_sim_bound_GHZ4.csv', index=False)

max_expectation_value = df['Expectation Value'].max()
min_expectation_value = df['Expectation Value'].min()
expectation_value_range = max_expectation_value - min_expectation_value
print("Max Expectation Value:", max_expectation_value)
print("Min Expectation Value:", min_expectation_value)
print("Range of Expectation Values:", expectation_value_range)
print("Total Execution Time:", total_exec_time, "seconds")
# No need to plot anything in this case
