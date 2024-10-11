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
from scipy.integrate import cumtrapz, solve_ivp

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

# Angular momentum per unit mass (h = v_orb * r_in)
h = v_orb * r_in

# Small inward radial velocity
v_r = -5.9e5  # Small radial inward velocity (in m/s)

v = np.sqrt(v_r**2 + v_orb**2)

# Relativistic energy (normalized)
m = M_sun
gamma = 1 / np.sqrt(1 - (v / c)**2)  # Lorentz factor
E = gamma * m * c**2  # Relativistic energy

# # Angular momentum h = v_orb * r_in
# h = v_orb * r_in

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




# Total energy per unit mass (including radial and tangential velocities)
E_nl = 0.5 * (v_r**2 + v_orb**2) - G * M_bh / r_in

# Effective potential function in Newtonian mechanics
def V_eff(r):
    return -G * M_bh / r + h**2 / (2 * r**2)

# Function for dtheta/dr
def dthetadr(r):
    return -h / (r**2 * np.sqrt(2 * (E_nl - V_eff(r))))

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

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

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
r_in = 4e1 * R_ISCO  

# Calculate the correct orbital velocity for a circular orbit at r_in
v_orb = np.sqrt(G * M_bh / r_in)/1.02

# Angular momentum per unit mass (h = v_orb * r_in)
h = v_orb * r_in

# Small inward radial velocity
v_r = -v_orb/20  # Small radial inward velocity (in m/s)

v = np.sqrt(v_r**2 + v_orb**2)

# Relativistic energy (normalized)
m = M_sun
gamma = 1 / np.sqrt(1 - (v / c)**2)  # Lorentz factor
E = gamma * m * c**2 - G*M_bh*m/r_in  # Relativistic energy

# Dimensionless parameters for GR case
a = h / c
b = c * h * m / E

# Total energy per unit mass (Newtonian mechanics)
E_nl = 0.5 * m* (v**2) - G * M_bh *m / r_in
E_nl/=m

# Define the effective potential function in Newtonian mechanics
def V_eff(r):
    return -G * M_bh / r + h**2 / (2 * r**2) #* (1-Rs/r)


# Define the function for dr/dphi in GR
def drdphi_gr(phi, r):
    r = np.asarray(r)  # Ensure r is treated as an array
    return -r**2 * np.sqrt((1 / b**2) - (1 - Rs / r) * (1 / a**2 + 1 / r**2))**(1)

# # Solve for the GR case
# solngr = solve_ivp(drdphi_gr, (0, 4*np.pi), [r_in], max_step=0.01)
# r_gr = solngr.y[0]
# phi_gr = solngr.t


# Function for dr/dphi in the Newtonian limit
def drdphi_nl(phi, r):
    r = np.asarray(r)  # Ensure r is treated as an array
    term = 2 * (E_nl - V_eff(r)) /h**2
    # Check if the energy allows for real solutions
    term = np.maximum(term, 0)  # Prevent sqrt of negative numbers
    return - r**2 * np.sqrt(term)

# Event function to detect turning points where dr/dphi = 0
drdphi_gr.terminal = True
drdphi_gr.direction = 0
drdphi_nl.terminal = True
drdphi_nl.direction = 0

# Function to integrate the orbit starting from a middle radius (between periapsis and apoapsis)
def integrate_orbit(r_start, direction='both', drdphi_func=drdphi_nl):
    r_values = []
    phi_values = []

    def integrate_half(direction):
        phi_span = (0, -np.pi) if direction == 'outward' else (0, np.pi)
        sol = solve_ivp(drdphi_func, phi_span, [r_start], max_step=0.001, events=drdphi_func, method='Radau')
        r_vals = sol.y[0]
        phi_vals = sol.t
        return r_vals, phi_vals
    
    # Integrate outward for phi > 0 (dr/dphi > 0)
    if direction == 'both' or direction == 'outward':
        r_out, phi_out = integrate_half(direction='outward')
        r_values.extend(r_out[::-1])
        phi_values.extend(phi_out[::-1])

    # Integrate inward for phi < 0 (dr/dphi < 0)
    if direction == 'both' or direction == 'inward':
        r_in, phi_in = integrate_half(direction='inward')
        r_values.extend(r_in)
        phi_values.extend(phi_in)

    return np.array(r_values)/Rs, np.array(phi_values)

# Initial guess for r_start near periapsis or apoapsis
r_start = r_in * 1

# Integrate the orbit in both directions from the middle
r_gr, phi_gr = integrate_orbit(r_start, direction='both', drdphi_func=drdphi_gr)
r_nl, phi_nl = integrate_orbit(r_start, direction='both', drdphi_func=drdphi_nl)

# Function to extend orbit by repeating solutions
def extend_orbit(r_values, phi_values, num_orbits=4):
    r_extended = np.tile(np.concatenate([r_values, r_values[::-1]]), int(num_orbits/2))
    phi_extended = np.concatenate([phi_values + (phi_values.max()-phi_values.min())*n for n in range(num_orbits)])
    return r_extended, phi_extended

# Extend the half-orbit solution to cover multiple orbits
num_orbits = 40  # Choose how many full orbits to cover
r_gr, phi_gr = extend_orbit(r_gr, phi_gr, num_orbits)
r_nl, phi_nl = extend_orbit(r_nl, phi_nl, num_orbits)


# Plot the orbit
plt.plot(r_gr * np.cos(phi_gr), r_gr * np.sin(phi_gr), label='GR')
plt.plot(r_nl * np.cos(phi_nl), r_nl * np.sin(phi_nl), label='Newtonian limit')
plt.xlabel("$x (R_s)$")
plt.ylabel("$y (R_s)$")
plt.xlim(-2*r_in/Rs, 2*r_in/Rs)
plt.ylim(-2*r_in/Rs, 2*r_in/Rs)
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter([0],[0],)
plt.show()


#%%
plt.figure()
plt.plot(phi_gr, r_gr)
plt.plot(phi_nl,r_nl)
plt.yscale('log')

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import matplotlib.cm as cm
# Set up figure and axis for the animation
fig, ax = plt.subplots()
ax.set_xlim(-2 * r_in / Rs, 2 * r_in / Rs)
ax.set_ylim(-2 * r_in / Rs, 2 * r_in / Rs)
ax.set_aspect('equal')
ax.set_xlabel("$x (R_s)$")
ax.set_ylabel("$y (R_s)$")

# Initialize the orbit plot
nl_line, = ax.plot([], [], label='Newtonian', color='blue', alpha=0.2)
gr_line, = ax.plot([], [], label='GR', color='red', alpha=0.2)

# Markers for current position
nl_marker, = ax.plot([], [], 'bo', markersize=8)
gr_marker, = ax.plot([], [], 'ro', markersize=8)

# Color fading trails
nl_trail, = ax.plot([], [], color='blue', lw=1, alpha=0.5)
gr_trail, = ax.plot([], [], color='red', lw=1, alpha=0.5)

# Initialization function for the animation
def init():
    nl_line.set_data([], [])
    gr_line.set_data([], [])
    nl_marker.set_data([], [])
    gr_marker.set_data([], [])
    nl_trail.set_data([], [])
    gr_trail.set_data([], [])
    return nl_line, gr_line, nl_marker, gr_marker, nl_trail, gr_trail

# Update function for the animation
def update(i):
    i*=200
    # Define the current and trailing positions
    trail_length = 3000  # Adjust the length of the trail
    start = max(0, i - trail_length)
    
    # Fade the trail using color maps
    # fade_values = np.linspace(0.2, 1, trail_length)
    
    # Plot the trail for Newtonian orbit
    # nl_x_trail = r_nl[start:i] * np.cos(phi_nl[start:i])
    # nl_y_trail = r_nl[start:i] * np.sin(phi_nl[start:i])
    # for j in range(0,len(nl_x_trail) - 1,1000):
    #     ax.plot(nl_x_trail[j:j+1000], nl_y_trail[j:j+1000], color=cm.Blues(fade_values[j]), lw=1)
    
    # Plot the trail for GR orbit
    # gr_x_trail = r_gr[start:i] * np.cos(phi_gr[start:i])
    # gr_y_trail = r_gr[start:i] * np.sin(phi_gr[start:i])
    # for j in range(0,len(gr_x_trail) - 1, 1000):
    #     ax.plot(gr_x_trail[j:j+1000], gr_y_trail[j:j+1000], color=cm.Reds(fade_values[j]), lw=1)
    
    # Current positions
    nl_marker.set_data(r_nl[i] * np.cos(phi_nl[i]), r_nl[i] * np.sin(phi_nl[i]))
    gr_marker.set_data(r_gr[i] * np.cos(phi_gr[i]), r_gr[i] * np.sin(phi_gr[i]))

    nl_line.set_data(r_nl[:i] * np.cos(phi_nl[:i]), r_nl[:i] * np.sin(phi_nl[:i]))
    gr_line.set_data(r_gr[:i] * np.cos(phi_gr[:i]), r_gr[:i] * np.sin(phi_gr[:i]))
    # if i<3000:   
    #     print(i, start)
    #     print(r_nl[start:i] * np.cos(phi_nl[start:i]), r_nl[start:i] * np.sin(phi_nl[start:i]))
    nl_trail.set_data(r_nl[start:i] * np.cos(phi_nl[start:i]), r_nl[start:i] * np.sin(phi_nl[start:i]))
    gr_trail.set_data(r_gr[start:i] * np.cos(phi_gr[start:i]), r_gr[start:i] * np.sin(phi_gr[start:i]))
    
    return  nl_line, gr_line, nl_marker, gr_marker, nl_trail, gr_trail

# Create the animation
num_frames = len(r_nl)//200  # Number of frames corresponds to the length of the orbit
ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=10)

# Show the animation
plt.legend()
plt.grid(True)
plt.show()
