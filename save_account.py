from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_provider import IBMProvider

token = "placeholder"
# Save an IBM Quantum account.
QiskitRuntimeService.save_account(channel="ibm_quantum", token=token)
IBMProvider.save_account(token=token)
