import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


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


def main():
    SPEED = 8
    FRAMES = 2000
    ROAD = 1
    FRONT_WEIGHT = 1
    SIDE_WEIGHT = 9
    SENSITIVITY = 1     # steering sensitivity
    THRESHOLD = 7       # if a minimum value of three front sensors is smaller than threshold, follow max
    BUF_SIZE = 24
    GAMMA = 0.95
    GOAL_DIST = 10      # if manhattan distance is smaller than goal dist, the truck moves forward
    PRINT_LOG = False

    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(width=1000, height=800, time_scale=SPEED)
    env = UnityEnvironment(file_name=f'../Maps/Road{ROAD}/Prototype 1', side_channels=[channel])
    env.reset()

    behavior_name = list(env.behavior_specs)[0]  # List['My Behavior?team=0']

    decision_steps, _ = env.get_steps(behavior_name)
    init_loc = decision_steps.obs[0][0][:3]
    dest_loc = decision_steps.obs[0][0][3:6]
    init_obs = decision_steps.obs[0][0][6:]
    memory_buffer = [init_obs] * BUF_SIZE

    for i in range(FRAMES):
        decision_steps, _ = env.get_steps(behavior_name)
        cur_loc = decision_steps.obs[0][0][:3]
        cur_obs = decision_steps.obs[0][0][6:]
        del memory_buffer[-1]
        memory_buffer.insert(0, cur_obs)

        if abs(cur_loc[0] - dest_loc[0]) + abs(cur_loc[2] - dest_loc[2]) < GOAL_DIST:
            action = [0, 150, 150]
        else:
            action = get_action(memory_buffer, FRONT_WEIGHT, SIDE_WEIGHT, SENSITIVITY, GAMMA, THRESHOLD)

        if PRINT_LOG:
            print(f'\nFrame [{i:4d}]')
            print_log(init_loc, cur_loc, cur_obs)
            print(f'steer {action[0]:6.3f} \t speed {action[1]:6.3f}')

        # Set the actions
        env.set_actions(behavior_name, np.array([action]))
        # Move the simulation forward
        env.step()

    env.close()


if __name__ == '__main__':
    main()
