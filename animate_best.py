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
num_of_controllers=range(len(controller))

seed = np.random.randint(1000)
simulate_trial(controller,seed,generating_animation=True)
rxs = []
rys = []
for item in controller:
    rxs.append(item.trial_data['robot'].x_h)
    rys.append(item.trial_data['robot'].y_h)

fig, ax = plt.subplots()

xy_data=[]
for i in num_of_controllers:
    xdata, ydata = [], []
    xy_data.append([xdata, ydata])

ln_list=[]
for i in num_of_controllers:
    ln_list.append(plt.plot([], [], 'm,'))


TRAIL_LENGTH = 100

n_samples, n_entities, n_coords = np.shape( controller[0].trial_data['FOOD_positions'] )
food_positions = np.array(controller[0].trial_data['FOOD_positions'])

## exact radius of entities is approximate. May see glitches where
## robot seems to be close enough to have eaten a thing, but it
## doesn't disappear. That is a bug with the animation, not the
## simulation :-)
fc, = plt.plot(food_positions[0,:,0],
               food_positions[0,:,1],'g.',markersize=35.0,alpha=0.5)

robot_positions = []
for i in num_of_controllers:
    robot_positions.append(plt.plot(rxs[i][0],rys[i][0],'m.',markersize=20.0,alpha=0.5))

def init():
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    
    ln_tuple=()
    for temp_ln_list in ln_list:
        ln_tuple+=tuple(temp_ln_list)
    
    robot_positions_tuple=()
    for temp_robo_pos_list in robot_positions:
        robot_positions_tuple+=tuple(temp_robo_pos_list)

    Ret_Value = ln_tuple + tuple([fc]) + robot_positions_tuple
    return Ret_Value

def update(frame):
    current_it = frame*its_per_frame

    for robot_index in num_of_controllers:
        xy_data[robot_index][0] =  rxs[robot_index][0 if current_it - TRAIL_LENGTH < 0 else current_it - TRAIL_LENGTH:current_it]
        xy_data[robot_index][1] =  rxs[robot_index][0 if current_it - TRAIL_LENGTH < 0 else current_it - TRAIL_LENGTH:current_it]

    for temp_ln in ln_list:
        for robot_index in num_of_controllers:
            temp_ln[0].set_data(xy_data[robot_index][0],xy_data[robot_index][1])

    fc.set_data(food_positions[current_it, :, 0],
                food_positions[current_it, :, 1])

    for robot_index in num_of_controllers:
        robot_positions[robot_index][0].set_data(rxs[robot_index][current_it], rys[robot_index][current_it])

    ln_tuple=()
    for temp_ln in ln_list:
        ln_tuple+=tuple(temp_ln)

    robot_positions_tuple=()
    for temp_robo_pos_list in robot_positions:
        robot_positions_tuple+=tuple(temp_robo_pos_list)

    Ret_Val = ln_tuple + tuple([fc]) + robot_positions_tuple

    return Ret_Val


ani = FuncAnimation(fig, update, frames=len(rxs[0])//its_per_frame,
                    init_func=init, blit=True, interval=delay_between_frames)
plt.show()



