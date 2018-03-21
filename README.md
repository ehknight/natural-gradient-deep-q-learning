# Natural Gradient Deep Q-Learning (NGDQN)

Please find our paper on arXiv [here](https://arxiv.org/abs/1803.07482).

This repository contains supplementary code for natural gradient Q-Learning. As of right now, the repository just contains the baseline code, which you can run yourself to replace the baseline results. 

## Baselines
The final scripts from the grid search used for crossvalidation located in
```baselines/cross_validate/py/```

Cross validation can be run with `python cross_validate.py`, and results can be viewed with `python cross_validate_results.py`.  To run the grid search, run `python parallel_slurm.py`.
