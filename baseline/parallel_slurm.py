import os
from copy import copy
from itertools import product
import click
from time import sleep
from random import shuffle
from templates import *

cur_dir = os.getcwd()
RESULTS_DIR = os.path.join(cur_dir, "param_search/")
print(RESULTS_DIR)

def set_results_dir(d):
    global RESULTS_DIR
    RESULTS_DIR = d

layer_sizes = {
        'CartPole-v0': [64],
        'CartPole-v1': [64],
        'Acrobot-v1': [64, 64],
        'LunarLander-v2': [256, 128]
}

episodes = {
        'CartPole-v0': 2000,
        'CartPole-v1': 2000,
        'Acrobot-v1': 10000,
        'LunarLander-v2': 10000 # episodes reduced because otherwise it takes forever
        # (later steps makes sure that at least 10,000 episodes are executed, but this mainly changes the number of steps)
}

hours = {
        'CartPole-v0': 1,
        'CartPole-v1': 1,
        'Acrobot-v1': 8,
        'LunarLander-v2': 12
}

mem = {
        'CartPole-v0': 500,
        'CartPole-v1': 500,
        'Acrobot-v1': 1000,
        'LunarLander-v2': 2000
}

search_space = {
        'tnuf': [1], # Baselines default
        'batch_size': [32, 128],
        'mem_len': [500, 2500, 50000],
        'exploration_frac': [0.01, 0.1, 0.5],
        'activation': ['tanh', 'relu'],
        'lr': [1e-7, 1e-8, '5e-2', '1e-4', '1e-5', '5e-3', '5e-5', 5e-6, 1e-6, 5e-4],
        # lrs also include default baselines learning rate (5e-4) copied from earlier runs
        'env': layer_sizes.keys()
     }

def submit(param_str, param_set, *args):
    slurm_file = slurm_template(param_str, os.path.join(RESULTS_DIR, param_str+'.py'), RESULTS_DIR, *args)
    script_file = script_template(param_set)
    file = open(RESULTS_DIR+param_str+'.py', 'w')
    file.write(script_file)
    file.close()
    file = open(RESULTS_DIR+param_str+".sbatch", 'w')
    file.write(slurm_file)
    file.close()
    cmd = "sbatch "+RESULTS_DIR+param_str+".sbatch"
    print("Running command: {}".format(cmd))
    # os.system(cmd)

def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in product(*iter(dicts.values())))

def parse_string(x):
    if isinstance(x, list):
        return "["+'_'.join(map(str, x))+"]"
    else:
        return x

@click.command()
@click.option('--env', default='all')
def submit_all(env):
    space = search_space
    names = list()
    combos = list(dict_product(space))
    shuffle(combos)
    for param_set_ in combos:
        if env != 'all' and env != param_set_['env']:
            continue
        param_set = copy(param_set_)
        param_set['layers'] = layer_sizes[param_set['env']]
        param_set['episodes'] = episodes[param_set['env']]
        print(param_set)
        param_str = param_set['env'] + '__'
        param_str += "__".join([str(x)+"_"+str(parse_string(y)) for x, y in param_set.items()])
        submit(param_str, param_set, hours[param_set['env']], mem[param_set['env']])
        names.append(param_str)
    return names

if __name__ == '__main__':
    submit_all()
