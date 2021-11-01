import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


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
    steer_weight = np.sum(weights) * (1 - gamma) / ((1 - gamma ** len(weights)) * np.max([front_weight, side_weight]))
    steer = np.tanh(sensitivity * steer_weight)

    speeds = [np.min([sensors[0], sensors[2], sensors[4]]) * (gamma ** i) for i, sensors in enumerate(memory_buffer)]
    speed_weight = np.sum(speeds) * (1 - gamma) / (1 - gamma ** len(speeds))
    speed = -3 * ((speed_weight - 20) ** 2) / 8 + 150

    return [steer, speed, speed]


def main():
    ROAD_DIR = '../Maps/Road1/Prototype 1'

    SPEED = 8
    FRAMES = 1_000_000_000

    FRONT_WEIGHT = 1
    SIDE_WEIGHT = 7
    SENSITIVITY = 2.5
    THRESHOLD = 6
    BUF_SIZE = 16
    GAMMA = 0.95
    GOAL_DIST = 10

    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(width=1000, height=800, time_scale=SPEED)
    env = UnityEnvironment(file_name=ROAD_DIR, side_channels=[channel])
    env.reset()

    behavior_name = list(env.behavior_specs)[0]

    decision_steps, _ = env.get_steps(behavior_name)
    dest_loc = decision_steps.obs[0][0][3:6]
    init_obs = decision_steps.obs[0][0][6:]
    memory_buffer = [init_obs] * BUF_SIZE

    for _ in range(FRAMES):
        decision_steps, _ = env.get_steps(behavior_name)
        cur_loc = decision_steps.obs[0][0][:3]
        cur_obs = decision_steps.obs[0][0][6:]
        del memory_buffer[-1]
        memory_buffer.insert(0, cur_obs)

        if abs(cur_loc[0] - dest_loc[0]) + abs(cur_loc[2] - dest_loc[2]) < GOAL_DIST:
            action = [0, 150, 150]
        else:
            action = get_action(memory_buffer, FRONT_WEIGHT, SIDE_WEIGHT, SENSITIVITY, GAMMA, THRESHOLD)

        env.set_actions(behavior_name, np.array([action]))
        env.step()

    env.close()


if __name__ == '__main__':
    main()
