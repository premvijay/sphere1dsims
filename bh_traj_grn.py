#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj

# Constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^(-1) s^(-2)
c = 2.998e8      # Speed of light in m/s
M_sun = 1.989e30 # Solar mass in kg
M_bh = 1e6 * M_sun  # Supermassive black hole mass (e.g., 1 million solar masses)

# Schwarzschild radius for a supermassive black hole
Rs = 2 * G * M_bh / c**2

# Specific angular momentum h for a typical orbiting particle near a supermassive black hole
R_ISCO = 3 * Rs  # Innermost stable circular orbit for a Schwarzschild black hole
h = np.sqrt(3 * G * M_bh * Rs)

# Specific energy epsilon (normalized energy ~1)
epsilon = c**2

# Angular momentum-to-energy ratio
L_by_E = h / epsilon

# Coefficients of the cubic equation for u
def cubic_coeffs(M, L_by_E, h, c):
    a = h / c  # Characteristic length scale related to specific angular momentum
    b = c * L_by_E  # Characteristic scale for energy
    A = Rs
    B = -1
    C = Rs / a**2
    D = -1 / a**2 + 1 / b**2
    return A, B, C, D

# Solve cubic equation using NumPy's roots method
def solve_cubic_np(A, B, C, D):
    # The cubic equation is A*u^3 + B*u^2 + C*u + D = 0
    coefficients = [A, B, C, D]
    roots = np.roots(coefficients)  # Solves the cubic equation
    real_roots = np.real(roots[np.isreal(roots)])  # Filter real roots only
    return np.sort(real_roots)  # Return sorted real roots

# Get the coefficients for the cubic equation
A, B, C, D = cubic_coeffs(M_bh, L_by_E, h, c)
turning_points = solve_cubic_np(A, B, C, D)

# Check if we have enough real roots
if len(turning_points) < 3:
    print(f"Only {len(turning_points)} real root(s) found.")
else:
    u1, u2, u3 = turning_points
    print(f"Turning points (real roots): u1={u1}, u2={u2}, u3={u3}")

    # Jacobi Elliptic sn function and trajectory
    def u_phi(phi, u1, u2, u3, Rs):
        # Constants for the sinus amplitudinus solution
        delta = 0  # initial phase angle
        k = np.sqrt(Rs * (u3 - u1))
        sn_val, _, _, _ = ellipj(0.5 * phi * k + delta, (u2 - u1) / (u3 - u1))
        return u1 + (u2 - u1) * sn_val**2

    # Define the angular range for phi
    phi_range = np.linspace(0, 4 * np.pi, 1000)

    # Compute r(phi) using u_phi
    r_values = [1 / u_phi(phi, u1, u2, u3, Rs) for phi in phi_range]

    # Plot r as a function of phi
    plt.figure(figsize=(8, 6))
    plt.plot(phi_range, r_values)
    plt.xlabel(r"$\phi$ (radians)")
    plt.ylabel(r"$r$ (m)")
    plt.title("Radial Distance as a Function of Angular Coordinate in Schwarzschild Metric")
    plt.grid(True)
    plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

# Constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^(-1) s^(-2)
c = 2.998e8      # Speed of light in m/s
M_sun = 1.989e30 # Solar mass in kg
M_bh = 1e6 * M_sun  # Supermassive black hole mass (e.g., 1 million solar masses)

# Schwarzschild radius for a supermassive black hole
Rs = 2 * G * M_bh / c**2

# Innermost stable circular orbit (ISCO)
R_ISCO = 3 * Rs

# Initial radius of the particle
r_in = 8e5 * R_ISCO  

# Calculate the correct orbital velocity for a circular orbit at r_in
v_orb = np.sqrt(G * M_bh / r_in)

# Relativistic energy (normalized)
m = M_sun
gamma = 1 / np.sqrt(1 - (v_orb / c)**2)  # Lorentz factor
E = gamma * m * c**2  # Relativistic energy

# Angular momentum h = v_orb * r_in
h = v_orb * r_in

# Dimensionless parameters
a = h / c
b = c * h * m / E

# Rs=0
# Define the function for dphi/dr
def dphidr(r):
    return -1 / r**2 * np.sqrt((1 / b**2) - (1 - Rs / r) * (1 / a**2 + 1 / r**2))**(-1)

# Generate the radial grid
r = np.linspace(r_in, R_ISCO, 10000)
r = np.logspace(np.log10(r_in), np.log10(R_ISCO), 10000)

# Integrate to get phi as a function of r
phi = cumtrapz(dphidr(r), r, initial=0)

# Plot the orbit in polar coordinates (r, phi)
plt.figure()
plt.plot(r * np.cos(phi), r * np.sin(phi))
plt.xlim(-r_in, r_in)
plt.ylim(-r_in, r_in)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Orbit of a test particle around a Schwarzschild black hole")
plt.grid(True)
# plt.show()







##%%
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import cumtrapz

# # Constants
# G = 6.67430e-11  # Gravitational constant in m^3 kg^(-1) s^(-2)
# M_sun = 1.989e30  # Solar mass in kg
# M_bh = 1e6 * M_sun  # Supermassive black hole mass (1 million solar masses)

# # Initial conditions for the test particle
# r_in = 8e5 * 3 * (2 * G * M_bh / (2.998e8)**2)  # Initial radial distance (8e5 times ISCO)
# R_ISCO = 3 * (2 * G * M_bh / (2.998e8)**2)  # ISCO radius for a Schwarzschild black hole

# # Test particle velocity
# v_orb = 1e5  # Test tangential velocity (in m/s)

# Angular momentum per unit mass (h = v_orb * r_in)
h = v_orb * r_in

# Small inward radial velocity
v_r = -1.9e5  # Small radial inward velocity (in m/s)

# Total energy per unit mass (including radial and tangential velocities)
E = 0.5 * (v_r**2 + v_orb**2) - G * M_bh / r_in

# Effective potential function in Newtonian mechanics
def V_eff(r):
    return -G * M_bh / r + h**2 / (2 * r**2)

# Function for dtheta/dr
def dthetadr(r):
    return -h / (r**2 * np.sqrt(2 * (E - V_eff(r))))

# Generate a radial grid from the initial radius down to ISCO
r = np.logspace(np.log10(r_in), np.log10(R_ISCO), 10000)

# Integrate to get theta as a function of r
theta = cumtrapz(dthetadr(r), r, initial=0)

# Convert polar coordinates to Cartesian for plotting
x = r * np.cos(theta)
y = r * np.sin(theta)

# Plot the orbit
plt.plot(x, y, label= 'Newtonian limit')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.xlim(-r_in, r_in)
plt.ylim(-r_in, r_in)
# plt.title("Newtonian Orbit with Radial Inward Velocity")
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


#%%
plt.figure()
plt.plot(theta,r)
plt.plot(phi,r)
plt.yscale('log')


