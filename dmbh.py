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

