#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd


#%%  primitive code with python loop for testing new additions
TotMass = 30
NumShells = 30
shellMass = TotMass/NumShells
pos = np.linspace(.05,1,NumShells)
Hi = 2 # hubble flow
vel = pos*Hi
col = cm.tab20(np.linspace(0,1,NumShells))
t=0

fig, ax = plt.subplots(1,)

TotalTime = 2
dt=.02

Nsteps = int(TotalTime//dt)
for i in range(0,Nsteps):
    t+=dt
    # accel = -shellMass*(pos[:,None]<pos[None]).sum(axis=0)/(pos+1e-2)**2/1e1 # acceleration due to gravity
    accel = -shellMass*(np.tanh(pos[None]-pos[:,None])+1).sum(axis=0)/2/(pos+1e-2)**2/1e1 # acceleration due to gravity
    accel += .05/pos**3 #angular momentum term
    pos += vel*dt
    vel += accel*dt
    #For shells crossing through the centre:
    # vel *= np.sign(pos)
    # pos = np.abs(pos)
    
    
    ax.scatter(pos*0+t,pos,c=col,s=1)

ax.set_ylim(0,1.5)

plt.show()


#%% # Alternate implementation using scipy integrate for faster vectorized computation and robust methods
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the profile for thick shell
def thick_shell_prof(x, size=1):
    """
    Return a profile for thick shells. The profile is symmetric about x=0 
    and smoothly transitions to zero outside the [-1, 1] range.
    """
    x /= size
    return np.where(np.logical_and(x >= -1, x <= 1), (1 + x) / 2, (x > 1).astype(np.int16))

# Black hole mass profile
def Mbh(r, size):
    """
    Black hole mass profile as a function of radius. Returns mass proportional to r inside 
    some size and a constant mass beyond that size.
    """
    return np.where(np.logical_and(r >= 0, r <= size), r / size, (r > size).astype(np.int16))

# Evolve function for the system
def shell_evolve(t, y, L, shell_mass):
    """
    Function representing the evolution of shells. Takes into account self-gravity, angular momentum, 
    and GR correction terms.
    """
    N = len(y) // 2
    pos = y[:N]  # Positions of shells
    vel = y[N:]  # Velocities of shells
    # print(pos, vel)
    
    posprof = pos[:, None].copy()
    posprof.sort(axis=0)  # Sort positions for mass enclosed calculation

    bhsize = 0.001
    # Calculate the mass enclosed within each shell
    mass_enc = shell_mass * np.sum(thick_shell_prof(pos[None] - posprof, 0.02), axis=0) #+ Mbh(pos, bhsize)

    # Gravitational acceleration (self-gravity)
    accel = -mass_enc / (pos + 1e-20)**2 / 1e1
    
    # Angular momentum term
    accel += L**2 / (pos + 1e-20)**3  # Centrifugal acceleration

    # General relativity correction term
    accel += -1e-7 * (3 * mass_enc * L**2) / ((pos + 1e-9)**4)

    vel = np.where(pos>bhsize, vel, 0)
    accel = np.where(pos>bhsize, accel, 0)

    # For shells crossing through the center:
    # Set velocity and position to zero if they reach or cross the center
    # at_center = pos <= 0
    # accel[at_center] = 0
    # vel[at_center] = 0
    # pos[at_center] = 0

    return np.concatenate([vel, accel])

# Initial conditions
TotMass = 30
NumShells = 30
shell_mass = TotMass / NumShells
pos0 = np.linspace(0.05, 1, NumShells)  # Initial positions of shells
vel0 = pos0 * 2  # Initial velocities following Hubble flow

# Angular momentum array (can vary between shells if needed)
L = 0.02 # np.linspace(0.02, 0.020001, NumShells)  # Angular momentum for each shell

# Pack initial conditions
y0 = np.concatenate([pos0, vel0])

# Time span for the simulation
TotalTime = 2
t_span = (0, TotalTime)

# Solve the ODE using solve_ivp with vectorized equations
sol = solve_ivp(lambda t, y: shell_evolve(t, y, L, shell_mass), t_span, y0, method='Radau', rtol=1e-7, max_step=0.1)

##%% Plot the trajectories of the shells
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[:NumShells].T)
plt.xlabel('Time')
plt.ylabel('Position')
plt.ylim(-0.2,3)
plt.title('Trajectories of Shells over Time')
plt.grid(True)
plt.show()




#%%
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

pos = sol.y[:NumShells,1500]

posint = np.linspace(.05, 1, NumShells*10)
plt.plot(posint,savgol_filter(shell_mass * np.sum(thick_shell_prof(posint[None]-pos[:,None],.0002), axis=0),20,1))

Mfn = interp1d(pos0,shell_mass * np.sum(thick_shell_prof(pos0[None]-pos[:,None],.0002), axis=0), kind=1)
plt.plot(pos0,shell_mass * np.sum(thick_shell_prof(pos0[None]-pos[:,None],.0002), axis=0))
posint = np.linspace(.05, 1, NumShells*3)
plt.plot(posint,Mfn(posint))










#%%
# Constants
G = 6.67430e-11  # Gravitational constant
dt = 1e6  # Time step
total_time = 1e9  # Total simulation time
num_shells = 100  # Number of shells
shell_mass = 1e20  # Initial mass of each shell


class SphericalAccretion:
    def __init__(self, r_min, r_max, num_points, initial_mass_profile):
        self.r_min = r_min
        self.r_max = r_max
        self.num_points = num_points
        self.initial_mass_profile = initial_mass_profile
        
        self.dr = (r_max - r_min) / (num_points - 1)
        self.r = np.linspace(r_min, r_max, num_points)
        self.velocities = np.zeros_like(self.r)

    def calculate_gravity(self):
        gravitational_force = np.zeros_like(self.r)
        for i in range(1, self.num_points):
            r_shell = self.r[i]
            mass_enclosed = self.mass_profile[i]
            gravitational_force[i] = mass_enclosed / r_shell**2
        return gravitational_force

    def calculate_mass_profile(self):
        new_mass_profile = np.zeros_like(self.r)
        for i in range(self.num_points):
            new_mass_profile[i] = np.sum(self.initial_mass_profile[self.r <= self.r[i]])
        return new_mass_profile

    def calculate_velocities(self, dt):
        for i in range(1, self.num_points):
            r_shell = self.r[i]
            force = -self.mass_profile[i] / r_shell**2
            self.velocities[i] += force * dt

    def update_positions(self, dt):
        self.r += self.velocities * dt

    def run_simulation(self, total_time, dt):
        num_steps = int(total_time / dt)
        for step in range(num_steps):
            self.mass_profile = self.calculate_mass_profile()
            gravitational_force = self.calculate_gravity()
            self.calculate_velocities(dt)
            self.update_positions(dt)

    def plot_mass_profile(self, ax):
        ax.plot(self.r, self.mass_profile)
        ax.set_xlabel('Radius')
        ax.set_ylabel('Enclosed Mass')
        ax.set_title('Evolution of Enclosed Mass Profile')

# Example usage:
r_min = 10
r_max = 20
num_points = 100
initial_mass_profile = np.ones(100)
# initial_mass_profile[0] = 1  # Initial mass at center
dt = 0.01
total_time = 2

fig1, ax1 = plt.subplots(1,)

accretion = SphericalAccretion(r_min, r_max, num_points, initial_mass_profile)
accretion.run_simulation(total_time, dt)
accretion.plot_mass_profile(ax1)
accretion.run_simulation(total_time, dt)
accretion.plot_mass_profile(ax1)
accretion.run_simulation(total_time, dt)
accretion.plot_mass_profile(ax1)
accretion.run_simulation(total_time, dt)
accretion.plot_mass_profile(ax1)


plt.show()
