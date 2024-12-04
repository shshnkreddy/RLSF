python ../Trains/train_prefim.py \
--env_name='SafetyPointCircle1-v0' --max_episode_length=500 --segment_length=500 \
--seed=0 --num_training_step=100000 \
--gamma=0.99 --cost_gamma=0.99 \
--number_layers=3 --hidden_units_actor=256 --hidden_units_critic=256 \
--number_layers_clfs=2 --hidden_units_clfs=64 \
--coef_ent=0.0001 --reward_factor=1.0 --env_cost_limit=0.0 \
--lr_actor=0.0001 --lr_critic=0.0001 --lr_penalty=0.01 --clip_eps=0.2 \
--num_eval_episodes=100 --eval_num_envs=0 --max_grad_norm=1.0 --epoch_ppo=160 \
--buffer_size=50000 --eval_interval=50000 --num_envs=10  \
--batch_size=4096 --epoch_clfs=500 --lr_clfs=0.001 \
--wandb_log=False \
--alpha=0.1 --clip_dev=15.0 \
--n_ensemble=1 \
--aug_state=False --pos_weight=1.0 --strat_schedule='novel' \
--start_bad=0 --warm_start_steps=0 --encode_action=False \
--k=15 --over_sample=False --hinge_coeff=0.0 --total_queries=0 \
--feedback_buffer_size=2000000
    

