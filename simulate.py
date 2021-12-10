import numpy as np
import matplotlib.pyplot as plt
# from plot import *

# -------------------- configurable params --------------------
delta_t = 1
num_fragments = 100

# -------------------- physical constants --------------------

mass_sun = 1.989e30 # solar mass in kg
radius_sun = 695500 # solar radius in km
G = 6.67e-20 # gravitational constant in km^3/(kg*s^2)

# -------------------- physical quantities --------------------
# see paper for derivations

# black hole mass in kg
mass_bh = 1e6 * mass_sun 
mass_star = 0.5 * mass_sun
radius_bh = mass_bh / mass_sun * 3
radius_star = 0.5 * radius_sun
# roche limit in km
roche_limit = radius_star * np.power((2 * mass_bh / mass_star), 1/3)
# initial velocity in km/2
velo_orbit = np.array([np.sqrt(1 * G * mass_bh / roche_limit), 0])
velo_e = np.array([np.sqrt(2 * G * mass_bh / roche_limit), 0])
coord_init = np.array([0, -roche_limit])
coord_bh = np.array([0, 0])

# -------------------- code --------------------

rng = np.random.default_rng()
cumu_coord_x_list = [[] for i in range (num_fragments)]
cumu_coord_y_list = [[] for i in range (num_fragments)]
color_list = []

class pointmass:
    def __init__(self, _mass, _velocity, _coord):
        self.m = _mass
        self.v = _velocity.copy()
        self.coord = _coord.copy()
    
    def move(self):
        self.coord += self.v * delta_t

    def apply_force(self):
        d = np.linalg.norm(self.coord - coord_bh) # in km
        accleration_val = G * mass_bh / (d * d) # in kg*km/s^2
        accleration_dir = (coord_bh - self.coord) / d
        self.v += accleration_dir * accleration_val * delta_t

class simulator:
    def __init__(self):
        self.fragments = []
        self.t = 0
        mass_fragment = mass_star / num_fragments
        
        for i in range(num_fragments):
            v = rng.uniform(velo_orbit, velo_e)
            self.fragments.append(
                pointmass(mass_fragment, v, coord_init)
            )
            color_list.append((rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)))

    def save(self):
        for i, f in enumerate(self.fragments):
            cumu_coord_x_list[i].append(f.coord[0])
            cumu_coord_y_list[i].append(f.coord[1])

    def plot_cumu(self):
        # init trivial things
        fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
        plt.xlim([-roche_limit * 2, roche_limit * 5])
        plt.ylim([-roche_limit * 1.5, roche_limit * 3])
        ax.set_xlabel('x coordinate (km)')
        ax.set_ylabel('y coordinate (km)')
        # plot black hole
        bh = plt.Circle((0, 0), radius_bh, color='black')
        ax.add_patch(bh)
        # plot roche limit
        circle = plt.Circle((0, 0), roche_limit, color='green', fill=False)
        ax.add_patch(circle)
        # plot point-mass fragments
        for i in range(num_fragments):
            plt.plot(cumu_coord_x_list[i], cumu_coord_y_list[i], color=color_list[i])
        # save figure
        fig.savefig('cumu.png')

    def run_step(self):
        for f in self.fragments:
            f.apply_force()
            f.move()
        self.t += delta_t

if __name__ == "__main__":
    sim = simulator()
    for i in range(10000):
        if i % 1000 == 0:
            sim.save()
            pass
        sim.run_step()
    sim.plot_cumu()
