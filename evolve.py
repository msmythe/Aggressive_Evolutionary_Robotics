import copy

from pylab import *
import os,sys
from scipy.spatial.distance import euclidean
import pyximport; pyximport.install(language_level=3)

from robot import Robot,Light
from seth_controller import SethController, EntityTypes, ENTITY_RADIUS
from plotting import plot_state_history,fitness_plots,plot_population_genepool
import random

from multiprocessing import Pool
plt.switch_backend('agg')

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# where to write simulation output to
savepath = os.path.abspath('./output/')

TEST_GA = False

DRAW_EVERY_NTH_GENERATION = 5

# EVOLUTION PARAMETERS
N_TRIALS = 5
POP_SIZE = 6       #originally 25
SITUATION_DURATION = 15.0
DT = 0.02
PER_GROUP = 4
ROBOT_FOOD_WORTH = 20.0
NATURAL_FOOD_WORTH = 20.0
NUMBER_OF_NATURAL_FOOD = 2
N_STEPS = int(SITUATION_DURATION / DT)  # the maximum number of steps per trial

if TEST_GA:
    N_STEPS = 1
    N_TRIALS = 1

# THE POPULATION
pop = [[SethController() for _ in range(PER_GROUP)] for _ in range(POP_SIZE)]  # the evolving population (a list of SethControllers)

# This keeps track of the fitness of the entire population since the start of the evolution.
# It is plotted in fitness_history.png
pop_fit_history = []

# Book keeping variables:
generation_index = 0



def random_light_position(robots):
    x = np.random.rand() * 2.0 - 1.0
    y = np.random.rand() * 2.0 - 1.0
    robot_close = True
    # make sure new position is not too close to other lights or the robot
    attempts = 0
    while robot_close:
        attempts += 1
        x = np.random.rand() * 2.0 - 1.0
        y = np.random.rand() * 2.0 - 1.0
        for robot in robots:
            robot_close = robot.is_close_to_any_light_or_the_robot(x, y, 2.0 * ENTITY_RADIUS)
            if robot_close:
                break
        if attempts > 100000:
            print("Failed to find new food position, returning a colliding one")
            return x, y
    return x, y


def simulate_trial(controllers, trial_index, generating_animation=False):
    """
    controller       -- the controller we are simulating
    trial_index -- we evaluate fitness by taking the average of N
                   trials. This argument tells us which trial we are
                   currently simulating
    """
    # ## reset the seed to make randomness of environment consistent for this generation
    np.random.seed(generation_index * 10 + trial_index)

    # initialize the simulation
    current_time = 0.0
    score = [0 for _ in range(PER_GROUP)]

    # reset the robot
    robots = [Robot() for _ in range(PER_GROUP)]
    for i in range(len(robots)):
        robots[i] = Robot()
        robots[i].x = random.uniform(0, 1)
        robots[i].y = random.uniform(0, 1)
        robots[i].a = random.uniform(0, 1)
    # get the controller for the robot from the population
    for controller in controllers:
        controller.trial_data = {}

    food_entities = []
    #water_entities = []
    #trap_entities = []

    # reset the environment
    for entity_type in EntityTypes:
        n = {EntityTypes.FOOD: NUMBER_OF_NATURAL_FOOD, EntityTypes.ROBOT: PER_GROUP}  # how many of each entity type to create
            #EntityTypes.WATER: 0,
            #EntityTypes.TRAP: 0,

        for _ in range(n[entity_type]):
            if entity_type == EntityTypes.ROBOT:
                for this_robot in robots:
                    for other_robot in robots:
                        if other_robot!=this_robot:
                            this_robot.add_robot(other_robot)
            else:
                x, y = random_light_position(robots)
                entity_light = Light(x, y, entity_type)
                for robot in robots:
                    robot.add_light(entity_light)
                food_entities.append(entity_light)

            # if entity_type == EntityTypes.FOOD:
            #     food_entities.append(entity_light)
            # if entity_type == EntityTypes.WATER:
            #     water_entities.append(entity_light)
            # if entity_type == EntityTypes.TRAP:
            #     trap_entities.append(entity_light)

    # FOOD battery
    food_b = [1.0 for _ in range(PER_GROUP)]

    # these variables keep track of things we want to plot later
    for controller in controllers:
        controller.trial_data['sample_times'] = []
        # controller.trial_data['water_battery_h'] = []
        controller.trial_data['food_battery_h'] = []
        controller.trial_data['score_h'] = []
        controller.trial_data['eaten_FOOD_positions'] = []
        controller.trial_data['eaten_ROBOT_positions'] = []
        # controller.trial_data['eaten_WATER_positions'] = []
        # controller.trial_data['eaten_TRAP_positions'] = []
        controller.trial_data['FOOD_positions'] = []
        # controller.trial_data['WATER_positions'] = []
        # controller.trial_data['TRAP_positions'] = []

    for iteration in range(N_STEPS):
        for i, controller in enumerate(controllers):
            # keep track of battery states for plotting later
            controller.trial_data['sample_times'].append(current_time)
            # controller.trial_data['water_battery_h'].append(water_b)
            controller.trial_data['food_battery_h'].append(food_b[i])
            controller.trial_data['score_h'].append(score[i])

        if generating_animation:
            # used in animation
            for controller in controllers:
                controller.trial_data[f'FOOD_positions'].append([(l.x, l.y) for l in food_entities])
                # controller.trial_data[f'WATER_positions'].append([(l.x, l.y) for l in water_entities])
                # controller.trial_data[f'TRAP_positions'].append([(l.x, l.y) for l in trap_entities])

        # each iteration, the time moves forward by the time-step, 'DT'
        current_time += DT

        # the battery states steadily drain at a constant rate
        DRAIN_RATE = 0.5
        # water_b = water_b - DT * DRAIN_RATE
        for i, item in enumerate(food_b):
            if robots[i].alive:
                food_b[i] = item - DT * DRAIN_RATE

        # water_b -= (robot.lm**2) * DT * 0.01
        # food_b  -= (robot.rm**2) * DT * 0.01

        for i, item in enumerate(score):
            score[i] += food_b[i] * DT

        # interaction between body and controller
        # tell the controller what the states of its various sensors are
        for entity_type in EntityTypes:
            for i in range(len(controllers)):
                controllers[i].set_sensor_states(entity_type, robots[i].sensors[entity_type])
        
        # tell the robot body what the controller says its motors should be
        for i in range(len(controllers)):
            if robots[i].alive:
                robots[i].lm, robots[i].rm = controllers[i].get_motor_output((food_b[i], 1))
            else:
                robots[i].lm = 0
                robots[i].rm = 0
            robots[i].calculate_derivative()  # calculate the way that the robot's state it changing
            robots[i].euler_update(DT=DT)  # apply that state change (Euler integration step)

        # check if robot has collided with FOOD, WATER or TRAP and
        # update batteries, etc.  accordingly
        
        #
        # check for FOOD collisions
        #
        for i in range(len(controllers)):
            for light in robots[i].lights[EntityTypes.FOOD]:
                if (robots[i].x - light.x) ** 2 + (robots[i].y - light.y) ** 2 < ENTITY_RADIUS ** 2:
                    food_b[i] += NATURAL_FOOD_WORTH * DT
                    controllers[i].trial_data['eaten_FOOD_positions'].append((light.x, light.y))
                    light.x, light.y = random_light_position(robots)  # relocate entity
        
        #
        # check for ROBOT collisions
        #
        for i in range(len(controllers)):
            for other_robot in robots[i].otherRobots:
                #If collision detected initiate combat
                if (robots[i].x - other_robot.x) ** 2 + (robots[i].y - other_robot.y) ** 2 < ENTITY_RADIUS ** 2:
                    if robots[i].alive == False or other_robot.alive == False:
                        continue
                    other_robot_index = robots.index(other_robot)
                    success = np.random.rand()
                    if success <= 0.5:      #50% chance to win confrontation
                        robots[i].alive = False
                        food_b[i] = 0           #dead battery
                        food_b[other_robot_index] += ROBOT_FOOD_WORTH * DT
                    else:
                        other_robot.alive = False
                        food_b[other_robot_index] = 0
                        food_b[i] += ROBOT_FOOD_WORTH * DT
                                
        """
        #
        # check for WATER collisions
        #
        for light in robot.lights[EntityTypes.WATER]:
            if (robot.x - light.x) ** 2 + (robot.y - light.y) ** 2 < ENTITY_RADIUS ** 2:
                water_b += 20.0 * DT
                controller.trial_data['eaten_WATER_positions'].append((light.x, light.y))
                light.x, light.y = random_light_position(robot)  # relocate entity
        #
        # check for TRAP collisions
        #
        for light in robot.lights[EntityTypes.TRAP]:
            if (robot.x - light.x) ** 2 + (robot.y - light.y) ** 2 < ENTITY_RADIUS ** 2:
                food_b -= 50.0 * DT
                water_b -= 50.0 * DT
                score = 0.0
                controller.trial_data['eaten_TRAP_positions'].append((light.x, light.y))
                light.x, light.y = random_light_position(robot)  # relocate entity
        """

        #
        # DEATH -- if all robots batteries reaches 0, the trial is over
        #
        for i in range(len(controllers)):
            if food_b[i] == 0 and robots[i].alive:
                robots[i].alive=False               #robot dies when battery is 0
                for j in range(len(robots)):        #remove dead robot from the other robots vision
                    robots[j].remove_robot(robots[i])
        
        all_dead = True
        for i in range(len(robots)):
            if robots[i].alive:
                all_dead=False
                break
        
        if all_dead:
            break
            
        

    # record the position of the entities still not eaten at end of trial (used for plotting)
    for i in range(len(controllers)):
        controllers[i].trial_data['uneaten_FOOD_positions'] = [(l.x, l.y) for l in robots[i].lights[EntityTypes.FOOD]]
        # controllers[i].trial_data['uneaten_WATER_positions'] = [(l.x, l.y) for l in robots[i].lights[EntityTypes.WATER]]
        # controllers[i].trial_data['uneaten_TRAP_positions'] = [(l.x, l.y) for l in robots[i].lights[EntityTypes.TRAP]]
        controllers[i].trial_data['robot'] = robots[i]

    if TEST_GA:
        score = []
        for controller in controllers:
            score.append(np.mean(controller.genome))

    return score


def evaluate_fitness(controllers):
    """
    An evaluation of an individual's fitness is the average of its
    performance in N_TRIAL simulations.
    ind -- the controller that is being evaluated
    """

    trial_scores = np.array([simulate_trial(controllers, trial_index) for trial_index in range(N_TRIALS)])
    for i, controller in enumerate(controllers):
        controller.fitness = np.mean(trial_scores[:, i])

    return controllers


def generation():
    global pop, generation_index

    # parallel evaluation of fitnesses (in parallel using multiprocessing)
    
    with Pool() as p:
        pop = p.map(evaluate_fitness, pop)

    ## sequential evaluation of fitness

    """for group in range(len(pop)):
        pop[group] = evaluate_fitness(pop[group])"""

    # the fitness of every individual controller in the population
    fitnesses = []
    for item in pop:
        for control in item:
            fitnesses.append(control.fitness)
    
    # we track the fitness of every individual at every generation for plotting
    pop_fit_history.append(sorted(np.array(fitnesses)))
    all_individuals = []
    for item in pop:
        for control in item:
            all_individuals.append(control)
    
    # ## every nth generation, we plot the trajectories of the best and
    # ## worst individual
    if (generation_index % DRAW_EVERY_NTH_GENERATION) == 0:
        all_individuals.sort(key=lambda x: x.fitness)
        fitnesses.sort()
        best_index = np.argmax(fitnesses)

        plot_state_history(savepath, all_individuals[best_index], 'best')
        all_individuals[best_index].plot_links('best')
        for index in range(PER_GROUP):
            np.save(os.path.join(savepath, '{}_genome.npy'.format(index)), all_individuals[-index].genome)

        """worst_index = np.argmin(fitnesses)
        plot_state_history(savepath, all_individuals[worst_index], 'worst')
        all_individuals[worst_index].plot_links('worst')"""

        # ## can plot others too, but actually takes a fair amount of
        # ## time, so leave commented out unless curious
        # for ind_i in np.arange(POP_SIZE)[::5] :
        #     plot_state_history(os.path.join(savepath,'everyone/'),pop[ind_i],f'everyone/i{ind_i}')
        #     pop[worst_index].plot_links(f'everyone/i{ind_i}')

    ######################################################################
    #### USE FITNESS PROPORTIONATE SELECTION TO CREATE THE NEXT GENERATION
    def weighted_choice(weights):
        totals = []
        running_total = 0

        for w in weights:
            running_total += w
            totals.append(running_total)

        rnd = np.random.rand() * running_total
        for i, total in enumerate(totals):
            if rnd < total:
                return i

    # Calcuate ps, the probabiltiy that each individual has for
    # being a parent

    # normalize the distribution of fitnesses to lie between 0 and 1
    f_normalized = np.array([x for x in fitnesses])
    if min(f_normalized) < max(f_normalized):
        f_normalized -= min(f_normalized)
    if max(f_normalized) > 0.0:
        f_normalized /= max(f_normalized)
    sum_f = max(0.01, sum(f_normalized))

    # ps: the probability of being selected as a parent of each
    # individual in the next generation
    ps = [f / sum_f for f in f_normalized]

    #### NOW ACTUALLY CREATE THE NEXT GENERATION

    # "elitism" : we seed the next generation of a copy of the best
    # individual from the previous generation
    best_index = np.argmax(fitnesses)
    best_individual = all_individuals[best_index]
    next_generation = [SethController(genome=best_individual.genome)]

    # ...and then populate the rest of the next generation by selecting
    # parents in a weighted manner such that higher fitness parents have
    # a higher chance of being selected.
    while len(next_generation) < POP_SIZE * PER_GROUP:
        a_i = weighted_choice(ps)
        b_i = weighted_choice(ps)
        mama = all_individuals[a_i]
        dada = all_individuals[b_i]
        baby = mama.procreate_with(dada)
        next_generation.append(baby)

    # replace the old generation with the new
    random.shuffle(next_generation)
    group_gen = []
    for item in range(math.ceil(len(next_generation)/PER_GROUP)):
        try:
            group_gen.extend([next_generation[item*PER_GROUP:item*PER_GROUP + PER_GROUP]])
        except IndexError:
            final_group = next_generation[item*PER_GROUP:]
            while len(final_group) < PER_GROUP:
                final_group.append(SethController())
            group_gen.extend([final_group])
    pop = group_gen.copy()

    print(f'GENERATION # {generation_index}.\t(mean/min/max)' +
          f'({np.mean(fitnesses):.4f}/{np.min(fitnesses):.4f}/{np.max(fitnesses):.4f})')
    generation_index += 1


def evolve():
    global fitnesses, generation_index

    print(f'Every generation is {POP_SIZE * N_TRIALS} fitness evaluations.')

    while True:
        fitnesses = generation()

        if generation_index % DRAW_EVERY_NTH_GENERATION == 0:
            fitness_plots(savepath, pop_fit_history)
            plot_population_genepool(savepath, pop)
        if generation_index >= 150:
            break


if __name__ == '__main__':
    evolve()
