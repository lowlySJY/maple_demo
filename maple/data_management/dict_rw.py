import random

import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def save_paths(path, file_dir):
    filename = file_dir
    if not os.path.exists(filename):
        os.makedirs(filename)
    os.chdir(filename)
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M")
    file_str = os.listdir(filename)
    dt_string = '2022-08-17-24-53.npy'
    if (dt_string in file_str):
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    np.save(dt_string, path)

def load_paths(max_path_length, discard_incomplete_paths, file_dir):
    filename = file_dir
    os.chdir(filename)
    paths = []
    for file in os.listdir(filename):
        path = np.load(file, allow_pickle=True).item()
        if (len(path['rewards']) < max_path_length) and discard_incomplete_paths:
            continue
        else:
            path['actions'] = path['actions'][-max_path_length:]
            path['observations'] = path['observations'][-max_path_length:]
            path['next_observations'] = path['next_observations'][-max_path_length:]
            path['rewards'] = path['rewards'][-max_path_length:]
            path['env_infos'] = path['env_infos'][-max_path_length:]
            path['terminals'] = path['terminals'][-max_path_length:]
            path['skill_names'] = path['skill_names'][-max_path_length:]
            path['path_length'] = max_path_length
            path['max_path_length'] = max_path_length
            paths.append(path)
    return paths

def plot_paths(path1, path2):
    rewards_p1 = np.array(path1['rewards'])
    rewards_p2 = np.array(path2['rewards'])
    # plt.hold(True)
    index_p1 = np.array(range(0, len(rewards_p1)))
    index_p2 = np.array(range(0, len(rewards_p2)))
    figure(figsize=(10, 6))
    plt.title("Compare rewards under different env with same demo-actions")
    plt.xlabel("env step")
    plt.ylabel("reward")
    plt.plot(index_p1, rewards_p1, color='b', label="rewards by experts")
    plt.plot(index_p2, rewards_p2, color='r', label="rewards by env")
    plt.legend(loc="center right")
    plt.show()
    plt.close()



if __name__ == '__main__':
    o = np.ones([7, 21])
    a = np.ones([7, 13])
    r = np.ones([7, 1])
    path = dict(
            observations=o,
            actions=a,
            rewards=np.array(r),
        )
    o = np.zeros([8, 21])
    a = np.zeros([8, 13])
    r = np.zeros([8, 1])
    path = dict(
            observations=o,
            actions=a,
            rewards=np.array(r),
        )
    # save_paths(path, '/home/jinyi/文档/code/maple/data/lift/demo')
    discard = False
    max_length_path = 120
    path_expl = load_paths(max_length_path, discard_incomplete_paths=discard, file_dir='/home/jinyi/文档/code/maple/data/lift/demo/expl')
    path_expert = load_paths(max_length_path, discard_incomplete_paths=discard, file_dir='/home/jinyi/文档/code/maple/data/lift/demo/expert')
    size = min(len(path_expl), len(path_expert))
    indices = random.randint(0, size - 1)
    plot_paths(path_expert[indices], path_expl[indices])