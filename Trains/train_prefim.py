import sys
sys.path.append('..')
sys.path.append('./')
from Parameters.PREFIM_parameters import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from Sources.utils import save_frames_as_gif

#------------------------------------------#
def main():
    import wandb
    from Sources.utils import create_folder

    strat_schedule = args.strat_schedule
    seed = args.seed

    name = f'LCOST-{n_ensemble}-{env_name}-{strat_schedule}'
    if(wandb_log):
        wandb.init(name=name, group=f'{env_name}', project='PREF')

    global weight_path
    weight_path = './weights'
    weight_path = f'{weight_path}/{env_name}/LCOST-{strat_schedule}/{seed}'
    create_folder(weight_path)

    if(wandb_log):
        wandb.config.update(args)
        
    #------------------------------------------#
    import safety_gymnasium
    from Sources.wrapper import CostWrapper, NavClassifierWrapper, VelClassifierWrapper

    if('Safety' in env_name):
        sample_env = safety_gymnasium.make(env_name, render_mode='rgb_array', camera_name='fixedfar')
        wrappers = [CostWrapper]

        env = safety_gymnasium.vector.make(env_id=env_name, num_envs=num_envs, wrappers=wrappers)
        if (eval_num_envs):
            test_env = safety_gymnasium.vector.make(env_id=env_name, num_envs=eval_num_envs, wrappers=wrappers)
        else:
            test_env = None

    elif('Driver' in env_name):
        from Sources.envs.Driver.driver import get_driver
        viz = False
        if('viz' in env_name.lower()):
            viz = True
        if('blocking' in env_name.lower()):
            scenario = 'blocked'
        elif('two' in env_name.lower()):
            scenario = 'twolanes'
        elif('change' in env_name.lower()):
            scenario = 'changing_lane'
        elif('stopping' in env_name.lower()):
            scenario = 'stopping'

        sample_env = get_driver(scenario=scenario, viz_obs=viz)
        
        env = safety_gymnasium.vector.SafetyAsyncVectorEnv([lambda: get_driver(scenario=scenario, viz_obs=viz, constraint=True) for _ in range(num_envs)])
        if (eval_num_envs):
            test_env = safety_gymnasium.vector.SafetyAsyncVectorEnv([lambda: get_driver(scenario=scenario, viz_obs=viz, constraint=True) for _ in range(eval_num_envs)])
        else:
            test_env = None

    elif('BiasedPendulum' in env_name):
        import gymnasium as gym
        from Sources.wrapper import BiasedPendulumWrapper
        
        def BiasedPendulum():
            env = gym.make('InvertedPendulum-v4')
            env = BiasedPendulumWrapper(env)
            return env

        sample_env = BiasedPendulum()
        env = safety_gymnasium.vector.SafetyAsyncVectorEnv([BiasedPendulum for _ in range(num_envs)])
        if (eval_num_envs):
            test_env = safety_gymnasium.vector.SafetyAsyncVectorEnv([BiasedPendulum for _ in range(eval_num_envs)])
        else:
            test_env = None

    elif('BlockedSwimmer' in env_name):
        import gymnasium as gym
        from Sources.wrapper import BlockedSwimmerWrapper

        def BlockedSwimmer():
            env = gym.make('Swimmer-v4')
            env = BlockedSwimmerWrapper(env)
            return env
        
        sample_env = BlockedSwimmer()
        env = safety_gymnasium.vector.SafetyAsyncVectorEnv([BlockedSwimmer for _ in range(num_envs)])
        if (eval_num_envs):
            test_env = safety_gymnasium.vector.SafetyAsyncVectorEnv([BlockedSwimmer for _ in range(eval_num_envs)])
        else:
            test_env = None

    else:
        raise ValueError('Unknown environment')
    #------------------------------------------#
    from Sources.algo.prefim import PREFIM
    from Sources.buffer import Trajectory_Buffer_Continuous, Trajectory_Buffer_Query, Schedule
    from copy import deepcopy
    import threading
    import torch
    import setproctitle
    from torch import nn
    from Sources.density import SimHash

    #------------------------------------------#
    def evaluate(algo, env,max_episode_length, t):
        global max_eval_return
        mean_return = 0.0
        mean_cost = 0.0
        failed_case = []
        cost_sum = [0 for _ in range(eval_num_envs)]

        for step in range(num_eval_episodes//eval_num_envs):
            state,_ = env.reset()
            episode_return = 0.0
            episode_cost = 0.0
            for iter in range(max_episode_length):
                if (iter%100 == 0):
                    print(f'valid {step+1}/{num_eval_episodes//eval_num_envs}: {iter/max_episode_length*100:.2f}% {iter}/{max_episode_length}', end='\r')
                action = algo.exploit(state)
                state, reward, cost, done, _, _ = env.step(action)
                episode_return += np.sum(reward)
                episode_cost += np.sum(cost)
                for idx in range(eval_num_envs):
                    cost_sum[idx] += cost[idx]
            for idx in range(eval_num_envs):
                failed_case.append(cost_sum[idx])
                cost_sum[idx] = 0
            mean_return += episode_return 
            mean_cost += episode_cost 

        mean_return = mean_return/num_eval_episodes
        mean_cost = mean_cost/num_eval_episodes
        tmp_arr = np.asarray(failed_case)

        success_rate = np.sum(tmp_arr<=env_cost_limit)/num_eval_episodes
        value = (mean_return * success_rate)/10
        if (value>max_eval_return):
            max_eval_return = value
            algo.save_models(f'{weight_path}/({value:.3f})-({success_rate:.2f})-({mean_return:.2f})-({mean_cost:.2f})')
        else:
            max_eval_return*=0.999
        print(f'[Eval] R: {mean_return:.2f}, C: {mean_cost:.2f}, '+
            f'SR: {success_rate:.2f}, '
            f'V: {value:.2f}, maxV: {max_eval_return:.2f}')
        if(wandb_log):
            wandb.log({'eval/return':mean_return, 'eval/cost':mean_cost})

    def render(env, algo, t):
        state, _ = env.reset()
        done = False
        truncated = False
        rewards = []
        costs = []
        costs_clfs = []
        frames = []

        i = 0
        while not done and not truncated:
            i += 1
            action = algo.exploit([state])[0]
            
            pred_cost = torch.sigmoid(algo.clfs[0](torch.tensor(np.array([state]), device=device, dtype=torch.float32), torch.tensor(np.array([action]), device=device, dtype=torch.float32))).detach().cpu().numpy()[0]
            clfs_cost = 1.0 if pred_cost > 0.5 else 0.0
            
            state, reward, cost, done, truncated, _ = env.step(action)
           
            costs_clfs.append(clfs_cost)
            rewards.append(reward)
            costs.append(cost)
            frames.append(env.render())

        frames = np.array(frames)

        print(f'Episode length: {i}\n')

        gif_path = f'{weight_path}/step_{t}/'
        create_folder(gif_path)
        save_frames_as_gif(frames, path=gif_path, filename='episode.gif', costs=np.cumsum(costs), clfs_costs=np.cumsum(costs_clfs), rewards=np.cumsum(rewards))
        plt.close()


    def train(env,test_env,algo,eval_algo):
        t = [0 for _ in range(num_envs)]
        eval_thread = None
        state,_ = env.reset()

        print('start training')
        for step in range(1,num_training_step//num_envs+1):
            if (step%100==0):
                print(f'train: {step/(num_training_step//num_envs)*100:.2f}% {step}/{num_training_step//num_envs}', end='\r')
                if(wandb_log):
                    wandb.log({'train/step': step/(num_training_step//num_envs)*100})
            state, t = algo.step(env, state, t, step*num_envs)
            if algo.is_update(step*num_envs):
                    eval_return.write(f'{np.mean(algo.return_reward)}\n')
                    eval_return.flush()
                    eval_cost.write(f'{np.mean(algo.return_cost)}\n')
                    eval_cost.flush()
                    algo.update()
               
            if (step) % (eval_interval//num_envs) == 0 or step==1:
                algo.save_models(f'{weight_path}/step_{step}')
                if (test_env):
                    if eval_thread is not None:
                        eval_thread.join()
                    eval_algo.copyNetworksFrom(algo)
                    eval_algo.eval()
                    eval_thread = threading.Thread(target=evaluate, 
                    args=(eval_algo,test_env,max_episode_length, step))
                    eval_thread.start()
                # Render if applicable
                if(('Driver' in env_name or 'Carla' in env_name or 'Highway' in env_name) and eval_algo is not None):
                    print('Rendering')
                    render(sample_env, eval_algo, step)
        algo.save_models(f'{weight_path}/s{seed}-finish')
        if(eval_thread is not None):
            eval_thread.join()    

    state_shape=sample_env.observation_space.shape
    action_shape=sample_env.action_space.shape
    aug_state_shape = None
    if(aug_state):
        if('Circle' in env_name):
            aug_state_shape = (16,)
        elif('Goal' in env_name):
            aug_state_shape = (16*3,)
        else:
            raise ValueError('Unknown environment for Aug State')
    sample_env.close()

    exp_good_buffer = Trajectory_Buffer_Continuous(
        buffer_size=feedback_buffer_size,
        state_shape=state_shape,
        action_shape=action_shape,
        device='cpu',
        aug_state_shape=aug_state_shape, 
        priority=False
    )

    exp_bad_buffer = Trajectory_Buffer_Continuous(
        buffer_size=feedback_buffer_size,
        state_shape=state_shape,
        action_shape=action_shape,
        device='cpu',
        aug_state_shape=aug_state_shape, 
        priority=False
    )


    strat = strat_schedule.split('_')[0]
    if(len(strat_schedule.split('_'))>1):
        schedule = strat_schedule.split('_')[1]
        print(f'Strat: {strat} Schedule: {schedule}')
        scheduler = Schedule(n_samples_rollout=buffer_size, total_traj_queries=total_queries, max_episode_length=max_episode_length, schedule=schedule, total_timesteps=num_training_step)
    else:
        scheduler = None

    if(aug_state):
        tmp_query_buffer = Trajectory_Buffer_Query(
            segment_length=segment_length,
            env_cost_limit=env_cost_limit,
            state_shape=aug_state_shape,
            action_shape=action_shape,
            scheduler=scheduler
        )
        _state_shape = aug_state_shape

    else:
        tmp_query_buffer = Trajectory_Buffer_Query(
            segment_length=segment_length,
            env_cost_limit=env_cost_limit,
            state_shape=state_shape,
            action_shape=action_shape,
            scheduler=scheduler
        )
        _state_shape = state_shape

    
    if(strat=='novel'):
        hash_table = SimHash(k=k, state_shape=_state_shape, device=device, action_shape=action_shape, use_actions=False, feature_state_dims=None)
        
    else:
        hash_table = None
        
    setproctitle.setproctitle(f'{env_name}-PREFCRL-{seed}')
    algo = PREFIM(env_name=env_name,exp_good_buffer=exp_good_buffer,exp_bad_buffer=exp_bad_buffer, tmp_query_buffer=tmp_query_buffer,
            state_shape=state_shape, action_shape=action_shape,
            device=device, seed=seed, gamma=gamma,cost_gamma=cost_gamma,buffer_size=buffer_size,
            mix=mix, hidden_units_actor=hidden_units_actor,
            hidden_units_critic=hidden_units_critic,units_clfs=hidden_units_clfs,
            lr_actor=lr_actor,lr_critic=lr_critic,lr_cost_critic=lr_cost_critic,lr_penalty=lr_penalty, epoch_ppo=epoch_ppo,
            epoch_clfs=epoch_clfs,batch_size=batch_size,lr_clfs=lr_clfs,clip_eps=clip_eps, lambd=lambd, coef_ent=coef_ent,
            max_grad_norm=max_grad_norm,reward_factor=reward_factor,max_episode_length=max_episode_length,
            env_cost_limit=env_cost_limit,risk_level=risk_level,num_envs=num_envs,
            start_bad=start_bad, wandb_log=wandb_log, alpha=alpha, clip_dev=clip_dev, segment_length=segment_length, n_ensemble=n_ensemble,
            class_prob=class_prob, aug_state=aug_state, aug_state_shape=aug_state_shape, 
            pos_weight=pos_weight, strat=strat, encode_action=encode_action, warm_start_steps=warm_start_steps, 
            hash_map=hash_table, over_sample=over_sample, hinge_coeff=hinge_coeff, conv=conv)
    
    if(test_env):
        eval_algo = deepcopy(algo)
    else:
        eval_algo = None

    
    train(env=env,test_env=test_env,algo=algo,eval_algo=eval_algo)

    env.close()
    if (test_env):
        test_env.close()

    wandb.finish()

if __name__ == '__main__':
    main()