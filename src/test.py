import numpy as np
import pandas as pd
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from datetime import datetime
from tqdm import tqdm


'''
type: <class 'numpy.ndarray'>
[x, y, z, x', y', z', s1, s2, s3, s4, s5]

[x, y, z] : Start coordinate
[x', y', z'] : Goal coordinate
(y = height. Therefore use (x, z) coordinate.)

s1: front right
s2: right
s3: front left
s4: left
s5: center
(Maximum value: 20.0)
'''

'''
type: <class 'numpy.ndarray'>
[steering angle, left torque, right torque]

steering angle range: [-1, 1] where 1 represents 45 degrees
torque range: [-150, 150]
'''


def get_steer_weight(sensors, front_weight, side_weight, gamma, threshold):
    front_min_val = np.min([sensors[0], sensors[2], sensors[4]])
    if front_min_val < threshold:
        steer_weight = gamma * np.max([front_weight, side_weight])
        if sensors[0] > sensors[2]:
            return steer_weight
        return -steer_weight
        
    steer_weight = (front_weight * (sensors[0] - sensors[2]) + side_weight * (sensors[1] - sensors[3])) / (sensors[:-1].sum() + 1e-6)
    return gamma * steer_weight


def get_action(memory_buffer, front_weight, side_weight, sensitivity, gamma, threshold):
    weights = [get_steer_weight(sensors, front_weight, side_weight, gamma ** i, threshold) for i, sensors in enumerate(memory_buffer)]
    steer_weight = np.sum(weights) * (1 - gamma) / ((1 - gamma ** len(weights)) * np.max([front_weight, side_weight]))  # value: distance, gamma: mass
    steer = np.tanh(sensitivity * steer_weight)  # tanh mapping

    speeds = [np.min([sensors[0], sensors[2], sensors[4]]) * (gamma ** i) for i, sensors in enumerate(memory_buffer)]
    speed_weight = np.sum(speeds) * (1 - gamma) / (1 - gamma ** len(speeds))  # value: distance, gamma: mass
    speed = -3 * ((speed_weight - 20) ** 2) / 8 + 150  # -x^2 mapping

    return [steer, speed, speed]


def print_log(init_loc, cur_loc, cur_obs):
    print(f'init loc [{init_loc[0]:7.3f}, {init_loc[2]:7.3f}]')
    print(f'cur  loc [{cur_loc[0]:7.3f}, {cur_loc[2]:7.3f}]')
    print(f'cur  obs [{cur_obs[3]:7.3f}, {cur_obs[2]:7.3f}, {cur_obs[4]:7.3f}, {cur_obs[0]:7.3f}, {cur_obs[1]:7.3f}]')


def main(ROAD, SPEED=8, MAX_FRAMES=10_000, FRONT_WEIGHT=1, SIDE_WEIGHT=8, SENSITIVITY=2,
         THRESHOLD=6, BUF_SIZE=16, GAMMA=0.97, GOAL_DIST=10, PRINT_LOG=False, LOG_FILE=None):
    '''
    SENSITIVITY:    steering sensitivity
    THRESHOLD:      if a minimum value of three front sensors is smaller than threshold, follow max
    GOAL_DIST:      if manhattan distance is smaller than goal dist, the truck moves forward
    '''

    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(width=1000, height=800, time_scale=SPEED)
    env = UnityEnvironment(file_name=f'../Road{ROAD}/Prototype 1', side_channels=[channel])
    env.reset()

    behavior_name = list(env.behavior_specs)[0]  # List['My Behavior?team=0']

    decision_steps, _ = env.get_steps(behavior_name)
    init_loc = decision_steps.obs[0][0][:3]
    dest_loc = decision_steps.obs[0][0][3:6]
    init_obs = decision_steps.obs[0][0][6:]
    memory_buffer = [init_obs] * BUF_SIZE

    goal = 0

    for i in range(MAX_FRAMES):
        decision_steps, _ = env.get_steps(behavior_name)
        cur_loc = decision_steps.obs[0][0][:3]
        cur_obs = decision_steps.obs[0][0][6:]
        del memory_buffer[-1]
        memory_buffer.insert(0, cur_obs)

        if abs(cur_loc[0] - dest_loc[0]) + abs(cur_loc[2] - dest_loc[2]) < GOAL_DIST:
            goal = 1
            break
        else:
            action = get_action(memory_buffer, FRONT_WEIGHT, SIDE_WEIGHT, SENSITIVITY, GAMMA, THRESHOLD)
        
        if PRINT_LOG:
            print(f'\nFrame [{i:4d}]')
            print_log(init_loc, cur_loc, cur_obs)
            print(f'steer {action[0]:6.3f} \t speed {action[1]:6.3f}')

        env.set_actions(behavior_name, np.array([action]))
        env.step()

        # If returned to initial point after some frames, break
        if i > 16 and round(cur_loc[0]) == round(init_loc[0]) and round(cur_loc[2]) == round(init_loc[2]):
            break
    
    LOG_FILE.write(','.join(map(str, [i, SIDE_WEIGHT, SENSITIVITY, THRESHOLD, BUF_SIZE, GAMMA, goal])) + '\n')
    env.close()


if __name__ == '__main__':
    ROAD = 2
    GOAL_DIST = 10  # It should be greater than 5
    SPEED = 16
    LOAD_COMBINATIONS = True

    # COMBINATIONS_DIR is only useful if LOAD_COMBINATIONS is True
    COMBINATIONS_DIR = '../Combinations/Comb(20211030_14-43-10).csv'
    RANGES = {
        'side_weight': np.arange(2, 11),            # 9
        'sensitivity': np.arange(0.5, 3.1, 0.5),    # 6
        'threshold': np.arange(5, 8),               # 3
        'bufsize': np.arange(8, 36, 4),             # 7
        'gamma': np.arange(0.95, 0.99, 0.01)        # 4
    }

    now = f'{datetime.now():%Y%m%d_%H-%M-%S}'
    log_file_name = f'Road{ROAD}_GoalDist{GOAL_DIST}({now})'

    if LOAD_COMBINATIONS:
        combinations = pd.read_csv(COMBINATIONS_DIR).values
    else:
        combinations = np.meshgrid(*RANGES.values())
        combinations = np.array([el.flatten() for el in combinations]).T

        comb_df = pd.DataFrame(combinations)
        comb_df.columns = RANGES.keys()
        comb_df.to_csv(f'../Combinations/Comb{now}.csv', index=False)

    log_file = open(f'../Logs/Road{ROAD}/Log_{log_file_name}.txt', 'w')
    log_file.write('frame,side weight,sensitivity,threshold,buf size,gamma,goal\n')

    for sw, st, th, bf, gm in tqdm(combinations):
        main(ROAD, SPEED=SPEED, SIDE_WEIGHT=sw, SENSITIVITY=st, THRESHOLD=th,
             BUF_SIZE=int(bf), GAMMA=gm, GOAL_DIST=GOAL_DIST, LOG_FILE=log_file)
    
    log_file.close()
