import numpy as np
import matplotlib.pyplot as plt

from reservoir_lab.reservoir import ESN


# -----------------------------
# Create two similar signals
# -----------------------------

np.random.seed(42)

timesteps = 2000

t = np.linspace(0, 20, timesteps)


signal1 = np.sin(t)

signal2 = np.sin(t) + 0.05 * np.random.randn(timesteps)


# Reservoir expects:
# (timesteps, inputs)

signal1 = signal1.reshape(-1, 1)
signal2 = signal2.reshape(-1, 1)


# -----------------------------
# Create reservoir
# -----------------------------

esn = ESN(
    n_inputs=1,
    n_reservoir=500,
    spectral_radius=1.1,
    sparsity=0.1
)


# -----------------------------
# Run both signals
# -----------------------------

states1 = esn.run(signal1)

states2 = esn.run(signal2)


# -----------------------------
# Calculate separation over time
# -----------------------------

state_distances = np.linalg.norm(
    states1 - states2,
    axis=1
)

# Input separation over time (same per-timestep basis as state_distances,
# so the two are actually comparable)
input_distances = np.linalg.norm(
    signal1 - signal2,
    axis=1
)

average_distance = np.mean(state_distances)
maximum_distance = np.max(state_distances)


print("======================")
print("Reservoir Separation")
print("======================")

print(f"Average input distance: {np.mean(input_distances):.6f}")
print(f"Average state distance: {average_distance:.6f}")
print(f"Maximum state distance: {maximum_distance:.6f}")


# -----------------------------
# Plot separation over time
# -----------------------------

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle("Reservoir state separation")

axes[0].plot(input_distances, color="gray")
axes[0].set_title("Input distance |signal1 - signal2| over time")

axes[1].plot(state_distances, color="C0")
axes[1].set_title("Reservoir state distance |states1 - states2| over time")
axes[1].set_xlabel("timestep")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("experiments/visuals/exp07_separation.png", dpi=120, bbox_inches="tight")
print("Saved plot to experiments/visuals/exp07_separation.png")