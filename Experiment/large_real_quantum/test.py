import json
import numpy as np

# Load the results (assuming it's stored in a file named "results.txt")
with open('29.txt', 'r') as file:
    data = json.load(file)

# Extract the counts for states 000 and 111 and convert to probabilities
probabilities_000 = []
probabilities_111 = []
probabilities_2 = []
probabilities_3 = []
probabilities_4 = []
probabilities_5 = []
probabilities_6 = []
probabilities_7 = []
probabilities_8 = []
probabilities_9 = []

# expectation_values = []


def calculate_expectation_value(counts, shots):
    return sum((-1) ** bin(int(state, 16)).count('1') * count for state, count in counts.items()) / shots


for result in data["results"]:
    counts = result["data"]["counts"]
    total_shots = result["shots"]
    print(counts)
    print(total_shots)

    # expectation_value = calculate_expectation_value(counts, total_shots)
    # expectation_values.append(expectation_value)
    # print(expectation_value)
    # ss
    # Convert hex representations back to binary states
    state_000_count = counts.get("0x0", 0)
    state_111_count = counts.get("0xfff", 0)
    state_2_count = counts.get("0xe00", 0)
    state_3_count = counts.get("0x200", 0)
    state_4_count = counts.get("0xffb", 0)
    state_5_count = counts.get("0xf7f", 0)
    state_6_count = counts.get("0x3ff", 0)
    state_7_count = counts.get("0x1fb", 0)
    state_8_count = counts.get("0xff", 0)
    state_9_count = counts.get("0x800", 0)
    # state_111_count = counts.get("0xffb", 0)

    probabilities_000.append(state_000_count / total_shots)
    probabilities_111.append(state_111_count / total_shots)

    probabilities_2.append(state_2_count / total_shots)
    probabilities_3.append(state_3_count / total_shots)

    probabilities_4.append(state_4_count / total_shots)
    probabilities_5.append(state_5_count / total_shots)

    probabilities_6.append(state_6_count / total_shots)
    probabilities_7.append(state_7_count / total_shots)
    probabilities_8.append(state_8_count / total_shots)
    probabilities_9.append(state_9_count / total_shots)

print('000')
print(np.max(probabilities_000))
print(np.min(probabilities_000))
print('0xfff')
print(np.max(probabilities_111))
print(np.min(probabilities_111))
print('0xe00')
print(np.max(probabilities_2))
print(np.min(probabilities_2))
print('0x200')
print(np.max(probabilities_3))
print(np.min(probabilities_3))
print('0xffb')
print(np.max(probabilities_4))
print(np.min(probabilities_4))
print('0xf7f')
print(np.max(probabilities_5))
print(np.min(probabilities_5))
print('0x3ff')
print(np.max(probabilities_6))
print(np.min(probabilities_6))
print('0x1fb')
print(np.max(probabilities_7))
print(np.min(probabilities_7))
print('0xff')
print(np.max(probabilities_8))
print(np.min(probabilities_8))

print('0x800')
print(np.max(probabilities_9))
print(np.min(probabilities_9))
print(probabilities_9)
# ss
lower_bound = 0.245
upper_bound = 0.330
probabilities_000_array = np.array(probabilities_000)
within_bounds = np.sum((lower_bound <= probabilities_000_array) & (probabilities_000_array <= upper_bound))
bcr = (within_bounds / len(probabilities_000_array)) * 100
# print(np.max(probabilities_000_array))
# print(np.min(probabilities_000_array))
# print(bcr)
# print(np.max(probabilities_111))
# print(np.min(probabilities_111))
ss
import numpy as np
import scipy.stats


def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    sem = scipy.stats.sem(data)  # Standard error of the mean
    ci = sem * scipy.stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return mean - ci, mean + ci


print(probabilities_000)
print(probabilities_111)
ci_000 = confidence_interval(probabilities_000)
ci_111 = confidence_interval(probabilities_111)

print(f"State 000 Confidence Interval: {ci_000}")
print(f"State 111 Confidence Interval: {ci_111}")
