import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import expm

# If you have shap, import lch2rgb from shap; otherwise define your own LCH to RGB conversion
from shap.plots.colors._colors import lch2rgb

# --- CREATE CUSTOM COLORMAP ---
blue_lch = [54., 70., 4.6588]
red_lch  = [54., 90., 0.35470565 + 2 * np.pi]
blue_rgb = lch2rgb(blue_lch)
red_rgb  = lch2rgb(red_lch)
white_rgb = np.array([1., 1., 1.])

colors = []
num_steps = 200
# Gradient from blue -> white
for alpha in np.linspace(1, 0, num_steps):
    c = blue_rgb * alpha + (1 - alpha) * white_rgb
    colors.append(c)
# Gradient from white -> red
for alpha in np.linspace(0, 1, num_steps):
    c = red_rgb * alpha + (1 - alpha) * white_rgb
    colors.append(c)

custom_cmap = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

# --- PARAMETERS ---
n = 100
theta = 0.5
t = np.linspace(0, 4*np.pi, n)
x = np.sin(t) * np.cos(3*t)  # original 1D signal

# --- S1 (basic adjacent skew-symmetric) ---
S1 = np.zeros((n, n))
for i in range(n-1):
    S1[i, i+1]   =  theta
    S1[i+1, i]   = -theta
R1 = expm(S1)
y1 = R1 @ x

# --- S2 (every other adjacent entry in top half) ---
S2 = np.zeros((n, n))
for i in range(n-1):
    # ((i % 2 == 0)*2 - 1) is +1 for even i, -1 for odd i
    val = theta * ((i % 2 == 0)*2 - 1) / 1.5
    S2[i, i+1]   =  val
    S2[i+1, i]   = -val
R2 = expm(S2)
y2 = R2 @ x

# --- S3 (skew-symmetric with i+j = n-1, for i < j) ---
S3 = np.zeros((n, n))
for i in range(n):
    j = n - 1 - i
    if i < j:
        # ((25 < i < 75)*2 - 1) is +1 for i in (25,75), else -1
        val = theta * ((25 < i < 75)*2 - 1)
        S3[i, j] =  val
        S3[j, i] = -val
R3 = expm(S3)
y3 = R3 @ x

# --- S4 (skew-symmetric with i+j = n-1, for i < j) ---
S4 = np.random.uniform(size=(n, n), low=-1*np.pi/180, high=1*np.pi/180)
S4[np.tril_indices_from(S4, k=-1)] = 0
S4 = S4 - S4.T  # Make it skew-symmetric
R4 = expm(S4)
y4 = R4 @ x

# Convert theta from radians to degrees for the colorbar range
theta_deg = 1 + theta * 180 / np.pi

# --- PLOTTING ---
fig, axs = plt.subplots(4, 2,
                        figsize=(6.4*2, 4.8*4),
                        sharex='col', sharey='col', squeeze=True)

# --- LEFT COLUMN: TIME-SERIES PLOTS ---
axs[0,0].plot(t, x, linewidth=2, color='#231F20', label='Original signal')
axs[0,0].plot(t, y1, '--', linewidth=2, color='#F3757A', label='Transformed signal')
axs[0,0].legend(loc='upper left')

axs[1,0].plot(t, x, linewidth=2, color='#231F20', label='Original signal')
axs[1,0].plot(t, y2, '--', linewidth=2, color='#F3757A', label='Transformed signal')
axs[1,0].legend(loc='upper left')

axs[2,0].plot(t, x, linewidth=2, color='#231F20', label='Original signal')
axs[2,0].plot(t, y3, '--', linewidth=2, color='#F3757A', label='Transformed signal')
axs[2,0].legend(loc='upper left')

axs[3,0].plot(t, x, linewidth=2, color='#231F20', label='Original signal')
axs[3,0].plot(t, y4, '--', linewidth=2, color='#F3757A', label='Transformed signal')
axs[3,0].legend(loc='upper left')

axs[3,0].set_xlabel('Time')
axs[0,0].set_ylabel('Amplitude')
axs[1,0].set_ylabel('Amplitude')
axs[2,0].set_ylabel('Amplitude')
axs[3,0].set_ylabel('Amplitude')

axs[0, 0].set_title('Signal transformations')
axs[0, 1].set_title('Rotational generators')

# Make sure your S1, S2, S3 are 2D NumPy arrays
im1 = axs[0,1].pcolormesh(np.flip(S1, axis=0) * 180/np.pi, cmap=custom_cmap,
                          vmin=-theta_deg, vmax=theta_deg, shading='auto')

im2 = axs[1,1].pcolormesh(np.flip(S2, axis=0) * 180/np.pi, cmap=custom_cmap,
                          vmin=-theta_deg, vmax=theta_deg, shading='auto')

im3 = axs[2,1].pcolormesh(np.flip(S3, axis=0) * 180/np.pi, cmap=custom_cmap,
                          vmin=-theta_deg, vmax=theta_deg, shading='auto')

im4 = axs[3,1].pcolormesh(np.flip(S4, axis=0) * 180/np.pi, cmap=custom_cmap,
                          vmin=-theta_deg, vmax=theta_deg, shading='auto')

axs[0,1].set_ylabel('Signal index (i)')
axs[1,1].set_ylabel('Signal index (i)')
axs[2,1].set_ylabel('Signal index (i)')
axs[3,1].set_xlabel('Signal index (j)')

# Attach a single colorbar to the entire right column
fig.colorbar(im4, ax=axs[:,1], orientation='vertical')
fig.subplots_adjust(wspace=0.1, hspace=0.3)


# Make sure layout doesn’t overlap
fig.set_constrained_layout(True)
fig.savefig("LieTransformations.pdf", transparent=True, dpi=600, format='pdf')
plt.show()
