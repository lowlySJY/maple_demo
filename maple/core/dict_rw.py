import numpy as np
import os
from datetime import datetime

def save_paths(path, file_dir):
    filename = file_dir
    if not os.path.exists(filename):
        os.makedirs(filename)
    os.chdir(filename)
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M")
    print(filename)
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
    paths = load_paths(120, discard_incomplete_paths=True, file_dir='/home/jinyi/文档/code/maple/data/lift/demo')
    for p in paths:
        print(p['observations'])
        print(p['rewards'])