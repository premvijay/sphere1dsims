#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd


#%%
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
dt=.002

Nsteps = int(TotalTime//dt)
for i in range(0,Nsteps):
    t+=dt
    accel = -shellMass*(pos[:,None]<pos[None]).sum(axis=0)/pos**2/1e1 # acceleration due to gravity
    # accel += .5/pos**3 #angular momentum term
    pos += vel*dt
    vel += accel*dt
    #For shells crossing through the centre:
    vel *= np.sign(pos)
    pos = np.abs(pos)
    
    
    ax.scatter(pos*0+t,pos,c=col,s=1)

ax.set_ylim(0,1.5)

plt.show()

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