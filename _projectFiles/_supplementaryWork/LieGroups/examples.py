import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Define parameters
n = 100          # length of the time series (dimension of the space)
theta = 0.5      # rotation strength between coupled samples
t = np.linspace(0, 4 * np.pi, n)  # time variable
x = np.sin(t) * np.cos(3*t)    # original 1D sine wave signal

### Example 1: Full Adjacent Band (All Upper Values Positive)
S1 = np.zeros((n, n))
for i in range(n - 1):
    S1[i, i+1] = theta      # all upper adjacent entries positive
    S1[i+1, i] = -theta     # skew-symmetry: lower entries negative
R1 = expm(S1)               # rotation matrix via the matrix exponential
y1 = R1 @ x                 # rotated signal

### Example 2: Partial Adjacent Band (Only Half the Upper Values are Positive)
S2 = np.zeros((n, n))
half = n // 2
# Only fill the first half of the adjacent band with nonzero values
for i in range(n - 1):
    S2[i, i+1] = theta * ((i % 2 == 0)*2 - 1) / 2
    S2[i+1, i] = -theta * ((i % 2 == 0)*2 - 1) / 2
# The remaining entries (i >= half) stay zero
R2 = expm(S2)
y2 = R2 @ x

# ### Example 2: Partial Adjacent Band (Only Half the Upper Values are Positive)
# S2 = np.zeros((n, n))
# half = n // 2
# # Only fill the first half of the adjacent band with nonzero values
# for i in range(n - 1):
#     for j in range(i+1, n - 1):
#         if j == 5 or i == 5:
#             S2[i, j] = theta * 2
#             S2[j, i] = -theta * 2
# # The remaining entries (i >= half) stay zero
# R2 = expm(S2)
# y2 = R2 @ x

### Example 3: Anti-Diagonal Band (Band from Lower Left to Upper Right)
S3 = np.zeros((n, n))
# For each row, place a nonzero entry in the column such that i + j = n - 1.
# We only fill entries for i < j so that we have a consistent skew-symmetric pattern.
for i in range(n):
    j = n - 1 - i
    if i < j:
        try:
            S3[i, j] = theta * ((25 < i < 75)*2 - 1)
            # S3[i+1, j+1] = theta
            S3[j, i] = -theta * ((25 < i < 75)*2 - 1)
            # S3[j+1, i+1] = -theta
        except: continue
R3 = expm(S3)
y3 = R3 @ x

# Plotting all three examples
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Example 1 Plot
axs[0].plot(t, x, label="Original Signal", linewidth=2)
axs[0].plot(t, y1, label="Rotated Signal", linestyle="--", linewidth=2)
axs[0].set_title("Example 1: Full Adjacent Band (All Upper Values Positive)")
axs[0].legend()
axs[0].grid(True)

# Example 2 Plot
axs[1].plot(t, x, label="Original Signal", linewidth=2)
axs[1].plot(t, y2, label="Rotated Signal", linestyle="--", linewidth=2)
axs[1].set_title("Example 2: Partial Adjacent Band (First Half Coupled)")
axs[1].legend()
axs[1].grid(True)

# Example 3 Plot
axs[2].plot(t, x, label="Original Signal", linewidth=2)
axs[2].plot(t, y3, label="Rotated Signal", linestyle="--", linewidth=2)
axs[2].set_title("Example 3: Anti-Diagonal Band")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
fig.savefig("LieTransformations.png", dpi=300)
plt.show()
