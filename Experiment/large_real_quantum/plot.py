import matplotlib.pyplot as plt

# Your provided list of probabilities
probabilities = [0.4775, 0.485, 0.4836, 0.4757, 0.4822, 0.4811, 0.4779, 0.48, 0.4827, 0.4828, 0.4917, 0.483, 0.4662, 0.4774, 0.4747, 0.478, 0.4816, 0.4751, 0.4784, 0.4714, 0.4814, 0.47, 0.4686, 0.4717, 0.4707, 0.4844, 0.4753, 0.4844, 0.4719, 0.4724, 0.4851, 0.4819, 0.4734, 0.477, 0.4847, 0.4716, 0.4709, 0.4774, 0.4789, 0.4804, 0.4784, 0.4806, 0.4832, 0.4754, 0.4762, 0.4854, 0.4714, 0.4788, 0.4858, 0.4835, 0.4798, 0.4768, 0.4683, 0.4664, 0.4843, 0.4846, 0.4817, 0.4855, 0.4831, 0.4725, 0.4755, 0.4778, 0.4804, 0.4793, 0.4829, 0.478, 0.48, 0.4687, 0.4836, 0.4765, 0.4798, 0.4841, 0.477, 0.4735, 0.4725, 0.4703, 0.4747, 0.4789, 0.4813, 0.4788]
# X-axis values: Number of runs (1 to 80)
# Upper and lower bounds
upper_bound = 0.4913611372603395



lower_bound = 0.46639339449076345



runs = list(range(1, 81))

# Plotting
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
plt.savefig('ghz3_1.eps', format='eps', dpi=1000)
plt.show()

