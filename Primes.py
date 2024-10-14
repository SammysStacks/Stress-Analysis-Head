import matplotlib.pyplot as plt
import sympy
from sympy import factorint
from mpmath import zeta
import numpy as np

# Generate numbers and their prime factorizations for 2 to 1000
numbers = np.arange(2, 100001)

#
# # Create a list of primes and their exponents for each number
# def get_prime_factors_exp(n):
#     return factorint(n)
#
#
# # Prepare lists to store plot data
# x_vals = []
# y_vals = []
# colors = []
#
# # Color map based on the number of prime occurrences
# color_map = {1: 'black', 2: 'blue', 3: 'red', 4: 'green', 5: 'orange', 6: 'purple', 7: 'brown'}
#
# for number in numbers:
#     prime_factors = get_prime_factors_exp(number)
#     for prime, exponent in prime_factors.items():
#         if exponent != 3: continue
#
#         x_vals.append(number)
#         y_vals.append(prime)
#         # Determine color based on the exponent (higher exponents get default purple)
#         colors.append(color_map.get(exponent, 'purple'))
#
# plt.figure(figsize=(10, 6))
#
# for prime in list(sympy.primerange(0, 100)):
#     # plt.plot(numbers, numbers/prime, 'o', c='tab:red', markersize=0.1)
#     plt.plot(numbers, numbers/prime, 'o', c='tab:blue', markersize=0.1)
#
#
# # Create the plot
# plt.scatter(x_vals, y_vals, c=colors, s=5)
# plt.title('Prime Coefficients for Numbers 2 to 1000')
# plt.xlabel('Number')
# plt.ylabel('Prime Coefficient')
# plt.grid(True)
# plt.show()


# Define a random bounded function
def random_bounded_function(x, n_terms=5):
    y = np.zeros_like(x)
    for _ in range(n_terms):
        # Random frequency and amplitude for each term
        freq = np.random.uniform(0.5, 5)
        amplitude = np.random.uniform(0.5, 2)
        phase = np.random.uniform(0, 2 * np.pi)
        # Add a sinusoidal wave to the function
        y += amplitude * np.sin(freq * x + phase)

    # Scale the result to keep it bounded within [-1, 1]
    y = y / np.max(np.abs(y))
    return y


# Create a range of x values
x = np.linspace(0, 10*np.pi, 1000)

# Generate a random bounded function
y = random_bounded_function(x, n_terms=10)
zy = y + x * 1j

zeta_valuesY = np.asarray([zeta(s) for s in zy])

c = 50
realY = []
imaginaryY = []
for zeta_value in zeta_valuesY:
    realY.append(zeta_value.real)
    imaginaryY.append(zeta_value.imag)

# Create the plot
plt.figure(figsize=(10, 6))
for i in range(0, len(realY), c):
    plt.plot(realY[i:i+c], imaginaryY[i:i+c], 'o-', linewidth=2)
plt.title('Prime Coefficients for Numbers 2 to 1000')
plt.xlabel('Number')
plt.ylabel('Prime Coefficient')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for i in range(0, len(x), c):
    plt.plot(x[i:i+c], y[i:i+c], 'o-', linewidth=2)
plt.plot(x, y, 'o-', c='k', linewidth=2, alpha=0.1)
plt.title('Prime y for Numbers 2 to 1000')
plt.xlabel('Number')
plt.ylabel('Prime Coefficient')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o-', c='k', linewidth=2, alpha=0.1)
plt.plot(x, realY, 'o-', c='tab:red', linewidth=2, alpha=0.1)
plt.title('Prime y for Numbers 2 to 1000')
plt.xlabel('Number')
plt.ylabel('Prime Coefficient')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o-', c='k', linewidth=2, alpha=0.1)
plt.plot(x, imaginaryY, 'o-', c='tab:red', linewidth=2, alpha=0.1)
plt.title('Prime y for Numbers 2 to 1000')
plt.xlabel('Number')
plt.ylabel('Prime Coefficient')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(realY, imaginaryY, 'o-', c='tab:red', linewidth=2, alpha=0.1)
plt.title('Prime y for Numbers 2 to 1000')
plt.xlabel('Number')
plt.ylabel('Prime Coefficient')
plt.grid(True)
plt.show()
