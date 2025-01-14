# Boosting Cooperative Decision-Making in Air Combat Environment Through Action Representation Learning and Mutual Support
We implemented a cooperative multi-agent reinforcement learning environment in the Air Combat Environment. This environment includes various scenarios such as 1v1, 2v2, and 4v4, and supports multiple armaments such as missiles, guns, and chaff.

## Install 

```shell
# create python env
conda create -n jsbsim python=3.8
# install dependency
pip install torch pymap3d jsbsim==1.1.6 geographiclib gym==0.20.0 wandb icecream setproctitle. 

- Download Shapely‑1.7.1‑cp38‑cp38‑win_amd64.whl from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely), and `pip install shaply` from local file.

- Initialize submodules(*JSBSim-Team/jsbsim*): `git submodule init; git submodule update`
```
## Envs
We provide all task configs in  `envs/JSBSim/configs`, each config corresponds to a task.

## Quick Start
### Training

```bash
cd scripts
bash train_*.sh
```
We have provide scripts for five tasks in `scripts/`.

- `train_heading.sh` is for SingleControl environment heading task.
- `train_vsbaseline.sh` is for SingleCombat vs-baseline tasks.
- `train_selfplay.sh` is for SingleCombat self-play tasks. 
- `train_selfplay_shoot.sh` is for SingleCombat self-play shoot missile tasks.
- `train_share_selfplay.sh` is for MultipleCombat self-play tasks.

It can be adapted to other tasks by modifying a few parameter settings. 

- `--env-name` includes options ['SingleControl', 'SingleCombat', 'MultipleCombat'].
- `--scenario` corresponds to yaml file in `envs/JBSim/configs` one by one.
- `--algorithm` includes options [ppo, mappo], ppo for SingleControl and SingleCombat, mappo for MultipleCombat

The description of parameter setting refers to `config.py`.
Note that we set parameters `--use-selfplay --selfplay-algorithm --n-choose-opponents --use-eval --n-eval-rollout-threads --eval-interval --eval-episodes` in selfplay-setting training. `--use-prior` is only set true for shoot missile tasks.
We use wandb to track the training process. If you set `--use-wandb`, please replace the `--wandb-name` with your name. 

### Evaluate and Render
```bash
cd renders
python render*.py
```
This will generate a `*.acmi` file. We can use [**TacView**](https://www.tacview.net/), a universal flight analysis tool, to open the file and watch the render videos.

## Citing
If you find this repo useful, pleased use the following citation:
````
@misc{liu2022light,
  author = {Qihan Liu and Yuhua Jiang and Xiaoteng Ma},
  title = {Light Aircraft Game: A lightweight, scalable, gym-wrapped aircraft competitive environment with baseline reinforcement learning algorithms},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/liuqh16/CloseAirCombat}},
}
