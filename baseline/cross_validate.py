import click
import os
from os.path import join
from glob import glob
from parallel_slurm import hours, mem
from templates import slurm_template

@click.command()
@click.option('--d', default='cross_validate')
@click.option('--trials', default=10)
@click.option('--select_env', default='')
def main(d, trials, select_env):
    for py_file in glob(join(d, 'py', '*.py')):
        env = py_file.split('/')[-1].split('.')[0]
        if select_env not in py_file:
            continue
        print(env)
        for trial in range(trials):
            sbatch_fname = join(d, 'sbatch/', env+'_t{}.sbatch'.format(trial))
            sbatch_file = open(sbatch_fname, 'w')
            to_write = slurm_template(
                env, join(os.getcwd(), d, 'py', env+'.py'),
                join(os.getcwd(), d, 'out/'), hours[env], mem[env], trial)

            assert 'neon' not in to_write
            sbatch_file.write(to_write)
            sbatch_file.close()
            cmd = 'sbatch '+sbatch_fname
            print(cmd)
            os.system(cmd)

if __name__ == '__main__':
    main()
