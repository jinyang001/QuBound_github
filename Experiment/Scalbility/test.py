import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Assuming your data is in a CSV file
data = pd.DataFrame({
    'Time': [3, 6, 9, 12, 15],
    'Qubound': [0.0009995, 0.0010045, 0.0019755, 0.0020003, 0.0031819],
    'Noisy Simulation': [22946.5308, 22955.352, 23145.72, 35224.935, 143000],
    'nisqR': [2.845, 5.4445, 7.4795, 10.222, 12.6]
})

plt.figure(figsize=(10, 6))

# Plot each series on the same plot
plt.plot(data['Time'], data['Qubound'], label='Qubound', marker='o')
plt.plot(data['Time'], data['Noisy Simulation'], label='Noisy Simulation', marker='o')
plt.plot(data['Time'], data['nisqR'], label='nisqR', marker='o')

plt.xlabel('Time')
plt.ylabel('Time in Seconds (log scale)')
plt.yscale('log')
# plt.xlim(0,18)
plt.xticks(np.arange(3, 18, 3))
plt.title('Comparison of Time in Seconds')
plt.legend()
plt.savefig('time_vs_num_qubits.eps', format='eps', dpi=1000)
plt.show()
