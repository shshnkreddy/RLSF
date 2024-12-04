CUDA_VISIBLE_DEVICES=0 python ../Trains/train_PPO.py \
--env_name='SafeDriverLaneChange' --max_episode_length=100 \
--seed=0 --num_training_step=10000000 \
--gamma=0.99 --cost_gamma=0.99 \
--number_layers=1 --hidden_units_actor=32 --hidden_units_critic=32 \
--coef_ent=0.0001 --reward_factor=1.0 --cost_limit=0.0 \
--lr_actor=0.0001 --lr_critic=0.0001 --lr_penalty=0.1 --clip_eps=0.2 \
--num_eval_episodes=100 --eval_num_envs=10 --max_grad_norm=1.0 --epoch_ppo=160 \
--buffer_size=50000 --eval_interval=250000 --num_envs=10  \
--weight_path='./weights/' \
--wandb_log=False --conv=False \