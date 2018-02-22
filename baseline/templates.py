script_template = lambda params: """
import gym
from baselines import deepq
import tensorflow as tf

episode_rewards = []

def callback(lcl, glb):
    global episode_rewards
    episode_rewards = lcl['episode_rewards']
    if len(episode_rewards) > {episodes}:
        return True
    else:
        return False

def main():
    global episode_rewards
    env = gym.make("{env}")
    max_timesteps_env = env.env._spec.__dict__['tags']['wrapper_config.TimeLimit.max_episode_steps']
    model = deepq.models.mlp({layers}, activation_fn=tf.nn.{activation})
    act = deepq.learn(
        env,
        lr={lr},
        q_func=model,
        target_network_update_freq={tnuf},
        batch_size={batch_size},
        max_timesteps=max_timesteps_env*{episodes},
        buffer_size={mem_len},
        exploration_fraction={exploration_frac},
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print('done')
    return episode_rewards
    # print("Saving model to cartpole_model.pkl")
    # act.save("cartpole_model.pkl")


if __name__ == '__main__':
    print(main())
    print('^^^^^^^^^^^^^')
""".format(**params)



slurm_template = lambda job_name, file, dir_path, run_time, mem, trials=0: """#!/bin/bash
#SBATCH --job-name=t{5}_{0}
#SBATCH --output={2}ng_{0}_t{5}.out
#SBATCH --error={2}ng_{0}_t{5}.err
#SBATCH --time={3}:00:00
#SBATCH --nodes=1
#SBATCH --mem={4}

module load python/2.7.5
# module load opencv/3.0.0
module load cuda/8.0
module load cuDNN/v5.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/sw/free/cuda/8.0/lib64/

source /scratch/PI/menon/scripts/python/misc/ehk/bin/activate
export PATH=$PATH:/scratch/PI/menon/scripts/python/misc/ffmpeg-3.3.1-64bit-static # required for gym
export PATH=/home/ehk/anaconda2/bin:$PATH
export PATH=/home/ehk/lib:$PATH
export PATH=/scratch/PI/menon/scripts/python/misc/lasagne:$PATH
alias cmake='/home/ehk/bin/cmake'
alias ale='/scratch/PI/menon/scripts/python/misc/natgrad/build/ale_0.4.4/ale_0_4/ale'

module load swig/3.0.8
source deactivate
module unload tensorflow.1/1.1.0
module load gcc/5.3.0
module load intelmpi
source activate tf

stdbuf -i0 -o0 -e0 srun python {1}
""".format(job_name, file, dir_path, run_time, mem, trials)
