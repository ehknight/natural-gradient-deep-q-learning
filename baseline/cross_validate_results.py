import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set()
import numpy as np

import re
import click
import glob
from calc_avg import calc_avg
from calc_avg import avgs as get_avgs
from parallel_slurm import episodes
from warnings import warn
import matplotlib.pyplot as plt

@click.command()
def main():
    envs = ['CartPole-v0', 'CartPole-v1', 'Acrobot-v1', 'LunarLander-v2']
    all_avgs = {}
    for env in envs:
        out_fnames = glob.glob('cross_validate/out/{}_t*.out'.format(env))
        outs = [open(x).readlines() for x in out_fnames]

        if '.' in env: eps = episodes[env.split('.')[-1]]
        else: eps = episodes[env]

        try:
            avgs = [calc_avg(eval(x[-2]), eps)[0] for x in outs]
            all_avgs[env] = [get_avgs(eval(x[-2]), eps) for x in outs]
        except Exception as e:
            print(eps)
            print(env.split('.')[-1])
            print('^^')
            warn(str(e))
            continue

        print('{}: {}'.format(env, np.average(avgs)))
        print(avgs)

    envs = ['CartPole-v0', 'CartPole-v1', 'Acrobot-v1', 'LunarLander-v2']
    fig, axs = plt.subplots(1, len(envs))
    fig.set_figheight(4)
    fig.set_figwidth(15)
    print(all_avgs.keys())
    for env, ax in zip(envs, axs):
        [ax.plot(x, alpha=0.5, color='gray') for x in all_avgs[env]]
        ax.plot([np.average(x) for x in zip(*all_avgs[env])])
        ax.set_title(env.split('.')[-1])
    fig.savefig('./plot.png', bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()

