import re
import os
import glob
import click
import linecache
import subprocess
import numpy as np
from shutil import copyfile
from pprint import pprint
from calc_avg import calc_avg
from warnings import warn

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

@click.command()
@click.option('--write', default=False, help='Write best to cross_val/py')
@click.option('--assertsame', default=False, help='Verify that cross_val/py matches best')
@click.option('--folder', default='param_search')
@click.option('--crossvalidate', default=False)
def main(write, folder, crossvalidate, assertsame):
    envs = ['CartPole-v0', 'CartPole-v1', 'Acrobot-v1', 'LunarLander-v2']
    maxlens = [2000, 2000, 10000, 10000]
    for env, maxlen in zip(envs, maxlens):
        avgs = []
        print(env)
        activ = '*'
        selector = '{}/*{}*{}*.out'.format(folder, env, activ)
        file_list =glob.glob(selector)
        print(selector)
        print(len(file_list))
        for out_file in file_list:
            try:
                lines = open(out_file).readlines()
                line = lines[-2] # baselines has a '^^^^' on the last line
                avg = calc_avg(eval(line), maxlen)
            except:
                warn(out_file)
                continue
            py_file = '/'+re.findall(r"\/ng_({}.*)_t0\.out".format(env), out_file)[0]+".py"
            avgs.append(('/'.join(out_file.split('/')[:-1])+py_file, avg))
            # avgs.append((out_file, avg))

        if crossvalidate:
            avg = np.average(map(lambda x: x[1][0], avgs))
            pprint(avg)
        else:
            sorted_runs = sorted(avgs, key=lambda x: (x[1][0], -x[1][1]))
            best = sorted_runs[-1]
            pprint(best)

        if write:
            copyfile(best[0], os.path.join('cross_validate', 'py', env+'.py'))
        if assertsame:
            assert open(best[0]).read() == open(os.path.join('cross_validate', 'py', env+'.py')).read()


if __name__ == '__main__':
    main()
