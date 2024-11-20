import matplotlib.pyplot as plt

# Read data from the file
filename = "results.txt"
data = {}

with open(filename, "r") as file:
    for line in file:
        label, values = line.strip().split(":")
        data[label.strip()] = list(map(int, values.split()))

# Generate x-axis labels
x_labels = [f"2^{i}" for i in range(19)]

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Linear scale plot
axs[0].set_title("Lower order inputs")
for label, values in data.items():
    axs[0].plot(x_labels[:10], values[:10], marker='o', label=label)
axs[0].set_xlabel("Input Size (2^n)")
axs[0].set_ylabel("Time")
axs[0].grid(True, linestyle='--', linewidth=0.5)
axs[0].legend()
axs[0].tick_params(axis='x', rotation=45)

# Logarithmic scale plot
axs[1].set_title("All inputs")
for label, values in data.items():
    axs[1].plot(x_labels, values, marker='o', label=label)
axs[1].set_xlabel("Input Size (2^n)")
axs[1].set_ylabel("Time")
axs[1].grid(True, which="both", linestyle='--', linewidth=0.5)
axs[1].legend()
axs[1].tick_params(axis='x', rotation=45)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
