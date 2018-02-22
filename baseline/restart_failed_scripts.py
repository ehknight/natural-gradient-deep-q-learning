import glob
from os import system
import click
import re

@click.command()
@click.option('--time_multiplier', default=0, type=int)
def main(time_multiplier):
    err_files = glob.glob("param_search/*.err")
    error_messages = ['CANCELLED', 'Exited with exit code 1'] #['Error', 'Timeout', 'TIMEOUT', 'CANCELLED']
    find_time = re.compile(r'--time=(\d+):')
    for err_file in err_files:
        f = open(err_file)
        contents = f.read()
        for err in error_messages:
            if err in contents:
                sbatch = re.findall(r'ng_(.*)_t0', err_file)[0].split('/')[-1]
                sbatch_name = 'param_search/'+sbatch+'.sbatch'
                if time_multiplier:
                    with open(sbatch_name, 'r+b') as sbatch_file:
                        sbatch_contents = str(sbatch_file.read())
                        new_time = int(find_time.findall(sbatch_contents)[0]) * time_multiplier
                        new_time = '--time='+str(new_time) + ':'
                        new_contents = find_time.sub(new_time, sbatch_contents)
                        print(new_contents.split('\\n'))
                        assert len(re.findall(r'bin', new_contents)) == 1
                    with open(sbatch_name, 'w') as sbatch_file:
                        sbatch_file.write(new_contents)
                print(sbatch_name)
                # system('sbatch {}'.format(sbatch_name))
                break
        f.close()

main()
