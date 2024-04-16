import matplotlib.pyplot as plt


probabilities = [0.0134, 0.0142, 0.0127, 0.0142, 0.0114, 0.0133, 0.0108, 0.0118, 0.0123, 0.0124, 0.0108, 0.013, 0.0152, 0.0134, 0.0122, 0.011, 0.0117, 0.013, 0.0113, 0.013, 0.0133, 0.0116, 0.0131, 0.0137, 0.0115, 0.0116, 0.0109, 0.0118, 0.0142, 0.0125, 0.0124, 0.0122, 0.0137, 0.0106, 0.0108, 0.0137, 0.0121, 0.0117, 0.012, 0.0117, 0.0123, 0.0113, 0.0127, 0.012, 0.012, 0.0127, 0.0121, 0.0112, 0.0112, 0.0118, 0.0121, 0.0143, 0.0127, 0.0124, 0.0116, 0.0136, 0.0115, 0.0126, 0.0113, 0.0127, 0.0118, 0.014, 0.0111, 0.0136, 0.0137, 0.0119, 0.0115, 0.0125, 0.0123, 0.0108, 0.0115, 0.0118, 0.011, 0.0123, 0.0119, 0.0137, 0.0121, 0.0113, 0.0108, 0.0122]

upper_bound = 0.015303555384241621



lower_bound = 0.005673918







runs = list(range(1, 81))


plt.figure(figsize=(10, 6))
plt.plot(runs, probabilities, marker='o', linestyle='', color='blue', label='Probability')
plt.hlines(upper_bound, xmin=1, xmax=80, colors='red', linestyles='-', label='Upper Bound')
plt.hlines(lower_bound, xmin=1, xmax=80, colors='green', linestyles='-', label='Lower Bound')

# plt.title('Probability vs Number of Runs with Bounds')
# plt.xlabel('Number of Runs')
# plt.ylabel('Probability')
# plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig('ghz3_2.eps', format='eps', dpi=1000)
plt.show()
