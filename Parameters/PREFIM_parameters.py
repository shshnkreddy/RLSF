import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()

training_group = parser.add_argument_group('PREFIM_training')

training_group.add_argument('--env_name',type=str,default='SafetyPointGoal1-v0')
training_group.add_argument('--gamma', type=float, default=0.99)
training_group.add_argument('--cost_gamma', type=float, default=0.99)
training_group.add_argument('--device_name', type=str, default='cuda')
training_group.add_argument('--seed', type=int, default=0)
training_group.add_argument('--buffer_size',type=int,default=20000)
training_group.add_argument('--feedback_buffer_size', type=int, default=2000000)
training_group.add_argument('--mix',type=int,default=1)
training_group.add_argument('--hidden_units_actor',type=int,default=256)
training_group.add_argument('--hidden_units_critic',type=int,default=256)
training_group.add_argument('--hidden_units_clfs',type=int,default=256)
training_group.add_argument('--number_layers',type=int,default=2)
training_group.add_argument('--number_layers_clfs',type=int,default=2)
training_group.add_argument('--conv',type=str,default='False')

training_group.add_argument('--lr_actor', type=float, default=0.0001)
training_group.add_argument('--lr_critic', type=float, default=0.0001)
training_group.add_argument('--lr_cost_critic', type=float, default=0.0001)
training_group.add_argument('--lr_penalty', type=float, default=0.0001)
training_group.add_argument('--lr_clfs', type=float, default=0.0001)

training_group.add_argument('--epoch_ppo',type=int,default=80)
training_group.add_argument('--epoch_clfs',type=int,default=80)
training_group.add_argument('--clip_eps', type=float, default=0.2)
training_group.add_argument('--lambd', type=float, default=0.97)
training_group.add_argument('--coef_ent', type=float, default=0.01)
training_group.add_argument('--max_grad_norm', type=float, default=1.0)
training_group.add_argument('--num_training_step',type=int,default=int(1e7))
training_group.add_argument('--eval_interval',type=int,default=int(60000))
training_group.add_argument('--num_eval_episodes',type=int,default=50)
training_group.add_argument('--max_episode_length',type=int,default=1000)
training_group.add_argument('--segment_length',type=int,default=1000)
training_group.add_argument('--mixup',type=str,default='False')
# confidence level for the classifier
training_group.add_argument('--class_prob',type=float,default=0.5)
training_group.add_argument('--n_ensemble',type=int,default=1)
training_group.add_argument('--reward_factor',type=float,default=3.0)
training_group.add_argument('--weight_path', type=str, default='./weights/')
training_group.add_argument('--pos_weight', type=float, default=1.0) 
# Querying Strategy. Options -> ['novel', '<random, entropy>_<uniform, decreasing>']
training_group.add_argument('--strat_schedule', type=str, default='novel')
# Size of embedding dim in Hash Map
training_group.add_argument('--k', type=int, default=12)
# NA for novelty sampling
training_group.add_argument('--total_queries', type=int, default=1000)

training_group.add_argument('--begin_cpu',type=int,default=None)
training_group.add_argument('--end_cpu',type=int,default=None)
# cost limit for the environment
training_group.add_argument('--env_cost_limit',type=float,default=25.0)
# cost limit for the inferred cost function
training_group.add_argument('--alpha', type=float, default=0.1)
training_group.add_argument('--clip_dev',type=float,default=100.0)
training_group.add_argument('--batch_size',type=int,default=None)
training_group.add_argument('--num_envs',type=int,default=20)
training_group.add_argument('--eval_num_envs',type=int,default=50)
training_group.add_argument('--risk_level',type=float,default=1.0)
training_group.add_argument('--start_bad',type=int,default=0)
training_group.add_argument('--warm_start_steps',type=int,default=0)
training_group.add_argument('--wandb_log',type=str,default='True')
# True for transfer experiments
training_group.add_argument('--aug_state',type=str,default='False')
training_group.add_argument('--debug',type=str,default='False')
# Whether to encode actions in the classifier
training_group.add_argument('--encode_action',type=str,default='True')
# Oversample minority class (bad states)
training_group.add_argument('--over_sample',type=str,default='False') 
# Apply a hinge loss to the classifier
training_group.add_argument('--hinge_coeff',type=float,default=0.0)


def get_bool(value):
    if (value == 'True'):
        return True
    elif (value == 'False'):
        return False
    else:
        raise 'value must in {True,False}'

#-------------------------------------------------------------------------------------------------#

# training
args = parser.parse_args()

gamma                                   = args.gamma
device                                  = args.device_name
seed                                    = args.seed
buffer_size                             = args.buffer_size
mix                                     = args.mix

hidden_units_actor                      = []
hidden_units_critic                     = []
hidden_units_clfs                       = []
for _ in range(args.number_layers):
    hidden_units_actor.append(args.hidden_units_actor)
    hidden_units_critic.append(args.hidden_units_critic)

for _ in range(args.number_layers_clfs):
    hidden_units_clfs.append(args.hidden_units_clfs)

max_eval_return                         = -np.inf

begin_cpu                               = args.begin_cpu
end_cpu                                 = args.end_cpu
lr_actor                                = args.lr_actor
lr_critic                               = args.lr_critic
lr_cost_critic                          = args.lr_cost_critic
lr_penalty                              = args.lr_penalty
lr_clfs                                 = args.lr_clfs
epoch_ppo                               = args.epoch_ppo
epoch_clfs                               = args.epoch_clfs
clip_eps                                = args.clip_eps
lambd                                   = args.lambd
coef_ent                                = args.coef_ent
max_grad_norm                           = args.max_grad_norm
num_training_step                       = args.num_training_step
eval_interval                           = args.eval_interval
num_eval_episodes                       = args.num_eval_episodes
env_name                                = args.env_name
reward_factor                           = args.reward_factor
max_episode_length                      = args.max_episode_length
segment_length                          = args.segment_length
env_cost_limit                          = args.env_cost_limit
alpha                                   = args.alpha
clip_dev                                = args.clip_dev
num_envs                                = args.num_envs
eval_num_envs                           = args.eval_num_envs
risk_level                              = args.risk_level
weight_path                             = args.weight_path
cost_gamma                              = args.cost_gamma
batch_size                              = args.batch_size
start_bad                               = args.start_bad
wandb_log                               = get_bool(args.wandb_log)
n_ensemble                              = args.n_ensemble
class_prob                              = args.class_prob
aug_state                               = get_bool(args.aug_state)
pos_weight                              = args.pos_weight
encode_action                           = get_bool(args.encode_action)
warm_start_steps                        = args.warm_start_steps
k                                       = args.k
over_sample                            = get_bool(args.over_sample)
hinge_coeff                             = args.hinge_coeff
conv                                    = get_bool(args.conv)
strat_schedule                          = args.strat_schedule
total_queries                           = args.total_queries
feedback_buffer_size                    = args.feedback_buffer_size

if not os.path.exists(weight_path):
    os.makedirs(weight_path)

log_path = f'{weight_path}/log_data'
if not os.path.exists(log_path):
    os.makedirs(log_path)

eval_return = open(f'{log_path}/return_{seed}.txt','w')
eval_cost = open(f'{log_path}/cost_{seed}.txt','w')
