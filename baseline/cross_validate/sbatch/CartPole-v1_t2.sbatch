#!/bin/bash
#SBATCH --job-name=t2_CartPole-v1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem=500

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

srun python /scratch/PI/menon/scripts/python/misc/natgrad/natural-learning/baseline_search_tanh_and_relu_stnuf_1_multiple_lrs/cross_validate/py/CartPole-v1.py
