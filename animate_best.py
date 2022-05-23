from pylab import *
from matplotlib.animation import FuncAnimation

from evolve import simulate_trial
from evolve import PER_GROUP
from seth_controller import SethController,EntityTypes,ENTITY_RADIUS
import os
plt.switch_backend('TkAgg')

its_per_frame = 1
delay_between_frames = 50 # ms

# loadme = os.path.join('output','best_genome.npy')

genomes = []
for index in range(PER_GROUP):
    genomes.append(np.load("output/{}_genome.npy".format(index)))

controller = [SethController(genome = genome) for genome in genomes]

seed = np.random.randint(1000)
simulate_trial(controller,seed,generating_animation=True)
rxs = []
rys = []
for item in controller:
    rxs.append(item.trial_data['robot'].x_h)
    rys.append(item.trial_data['robot'].y_h)

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'k,')

n_samples, n_entities, n_coords = np.shape( controller[0].trial_data['FOOD_positions'] )
food_positions = np.array(controller[0].trial_data['FOOD_positions'])
#water_positions = np.array(controller[0].trial_data['WATER_positions'])
#trap_positions = np.array(controller[0].trial_data['TRAP_positions'])


## exact radius of entities is approximate. May see glitches where
## robot seems to be close enough to have eaten a thing, but it
## doesn't disappear. That is a bug with the animation, not the
## simulation :-)
fc, = plt.plot(food_positions[0,:,0],
               food_positions[0,:,1],'g.',markersize=35.0,alpha=0.5)
"""wc, = plt.plot(water_positions[0,:,0],
               water_positions[0,:,1],'b.',markersize=35.0,alpha=0.5)
tc, = plt.plot(trap_positions[0,:,0],
               trap_positions[0,:,1],'r.',markersize=35.0,alpha=0.5)"""

robot_pos1, = plt.plot(rxs[0][0],rys[0][0],'k.',markersize=20.0,alpha=0.5)
robot_pos2, = plt.plot(rxs[1][0], rys[1][0], 'k.', markersize=20.0, alpha=0.5)
robot_pos3, = plt.plot(rxs[2][0], rys[2][0], 'k.', markersize=20.0, alpha=0.5)


def init():
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_aspect('equal')
    return ln,fc,robot_pos1, robot_pos2, robot_pos3

def update(frame):
    current_it = frame*its_per_frame 
    xdata=rxs[0:current_it]
    ydata=rys[0:current_it]
    ln.set_data(xdata, ydata)
    fc.set_data(food_positions[current_it,:,0],
                food_positions[current_it,:,1])
    """wc.set_data(water_positions[current_it,:,0],
                water_positions[current_it,:,1])
    tc.set_data(trap_positions[current_it,:,0],
                trap_positions[current_it,:,1])"""
    robot_pos1.set_data(rxs[0][current_it],rys[0][current_it])
    robot_pos2.set_data(rxs[1][current_it],rys[1][current_it])
    robot_pos3.set_data(rxs[2][current_it],rys[2][current_it])
    return ln,fc,robot_pos1,robot_pos2, robot_pos3

ani = FuncAnimation(fig, update, frames=len(rxs[0])//its_per_frame,
                    init_func=init, blit=True, interval=delay_between_frames)
plt.show()



