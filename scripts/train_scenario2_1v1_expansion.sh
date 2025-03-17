#!/bin/sh
env="MultipleCombat"
scenario="scenario2/scenario2"
algo="mappo"
<<<<<<< HEAD:scripts/train_MI_support_2v2.sh
exp="MI_experiment_move_vs_firearm_unlimited"
seed=0

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python train/train_jsbsim.py \
=======
exp="jhs_scenario2"
seed=0

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=3 python train/train_jsbsim.py \
>>>>>>> ac8504c6dfc6d247cf5d21e0cc8f487d01420e4a:scripts/train_scenario2_1v1_expansion.sh
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 32 --cuda --log-interval 1 --save-interval 1 \
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps 1e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 60 \
    --use-selfplay --selfplay-algorithm "fsp" --n-choose-opponents 1 \
    --use-eval --n-eval-rollout-threads 1 --eval-interval 1 --eval-episodes 1 \
    --use-prior --use-wandb