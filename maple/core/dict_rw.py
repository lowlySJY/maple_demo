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

def load_paths():
    filename = '/home/jinyi/文档/code/maple/data/lift/demo'
    os.chdir(filename)
    paths = []
    for file in os.listdir(filename):
        path = np.load(file, allow_pickle=True).item()
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
    paths = load_paths()
    for p in paths:
        print(p['observations'])
        print(p['rewards'])