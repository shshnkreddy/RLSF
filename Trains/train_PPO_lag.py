import sys
sys.path.append('..')
sys.path.append('./')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from Sources.utils import save_frames_as_gif
from Parameters.PPO_parameters import *

#------------------------------------------#
def main():
    import wandb
    if(wandb_log):
        wandb.init(name=f'PPOLag-{env_name}', group=f'{env_name}', project='PREF')
        wandb.config.update(args)

    from Sources.utils import create_folder
    global weight_path
    weight_path = './weights'
    weight_path = f'{weight_path}/{env_name}/PPOLag'
    create_folder(weight_path)
    #------------------------------------------#
    import safety_gymnasium
    from Sources.wrapper import CostWrapper
    if('Safety' in env_name):
        sample_env = safety_gymnasium.make(env_name)
        env = safety_gymnasium.vector.make(env_id=env_name, num_envs=num_envs, wrappers=[CostWrapper])
        if (eval_num_envs):
            test_env = safety_gymnasium.vector.make(env_id=env_name, num_envs=eval_num_envs, wrappers=[CostWrapper])
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

    elif('BlockedWalker' in env_name):
        import gymnasium as gym
        from Sources.wrapper import BlockedWalkerWrapper

        def BlockedWalker():
            env = gym.make('Walker2d-v4')
            env = BlockedWalkerWrapper(env)
            return env
        
        sample_env = BlockedWalker()
        env = safety_gymnasium.vector.SafetyAsyncVectorEnv([BlockedWalker for _ in range(num_envs)])
        if (eval_num_envs):
            test_env = safety_gymnasium.vector.SafetyAsyncVectorEnv([BlockedWalker for _ in range(eval_num_envs)])
        else:
            test_env = None

    else:
        raise ValueError('Unknown environment')
    #------------------------------------------#
    from Sources.algo.ppo import PPO_lag

    from copy import deepcopy
    import threading
    import torch
    import setproctitle
    from torch import nn

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

        success_rate = np.sum(tmp_arr<=cost_limit)/num_eval_episodes
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
        frames = []
        states = [] 

        i = 0
        while not done and not truncated:
            i += 1
            action = algo.exploit([state])[0]
            state, reward, cost, done, truncated, _ = env.step(action)
            rewards.append(reward)
            costs.append(cost)
            frames.append(env.render())
            if(state.shape[0]==1):
                states.append(state[0])
            else:
                states.append(state)

        frames = np.array(frames)

        print(f'Episode length: {i}\n')

        gif_path = f'{weight_path}/Step_{t}/'
        create_folder(gif_path)
        save_frames_as_gif(frames, path=gif_path, filename='episode.gif', costs=np.cumsum(costs), clfs_costs=np.cumsum(costs), rewards=np.cumsum(rewards))
        # save_frames_as_gif(states, path=gif_path, filename='states.gif', costs=np.cumsum(costs), clfs_costs=np.cumsum(costs), rewards=np.cumsum(rewards))
        plt.close()


    def train(env,test_env,algo,eval_algo):
        t = [0 for _ in range(num_envs)]
        eval_thread = None
        state,_ = env.reset()

        print('start training')
        for step in range(1,num_training_step//num_envs+1):
            if (step%100 == 0):
                print(f'train: {step/(num_training_step//num_envs)*100:.2f}% {step}/{num_training_step//num_envs}', end='\r')
                if(wandb_log):
                    wandb.log({'train/step': step/(num_training_step//num_envs)*100})
            state, t = algo.step(env, state, t)
            if algo.is_update(step*num_envs):
                    eval_return.write(f'{np.mean(algo.return_reward)}\n')
                    eval_return.flush()
                    eval_cost.write(f'{np.mean(algo.return_cost)}\n')
                    eval_cost.flush()
                    algo.update()
                    
            if step % (eval_interval//num_envs) == 0 or step==1:
                algo.save_models(f'{weight_path}/s{seed}-latest')
                if (test_env):
                    if eval_thread is not None:
                        eval_thread.join()
                    eval_algo.copyNetworksFrom(algo)
                    eval_algo.eval()
                    eval_thread = threading.Thread(target=evaluate, 
                    args=(eval_algo,test_env,max_episode_length, step))
                    eval_thread.start()

                #Render if applicable
                if('Driver' in env_name or 'Carla' in env_name or 'Highway' in env_name):
                    render(sample_env, eval_algo, step)

        if(eval_thread is not None):
            eval_thread.join()
        algo.save_models(f'{weight_path}/s{seed}-finish')

    state_shape=sample_env.observation_space.shape
    action_shape=sample_env.action_space.shape
    sample_env.close()

    setproctitle.setproctitle(f'{env_name}-PPO-lag-{seed}')
    
    algo = PPO_lag(env_name=env_name,
            state_shape=state_shape, action_shape=action_shape,
            device=device, seed=seed, gamma=gamma,cost_gamma=cost_gamma,buffer_size=buffer_size,
            mix=mix, hidden_units_actor=hidden_units_actor,
            hidden_units_critic=hidden_units_critic,units_clfs=hidden_units_clfs,
            lr_actor=lr_actor,lr_critic=lr_critic,lr_cost_critic=lr_cost_critic,lr_penalty=lr_penalty, epoch_ppo=epoch_ppo,
            epoch_clfs=epoch_clfs,batch_size=batch_size,lr_clfs=lr_clfs,clip_eps=clip_eps, lambd=lambd, coef_ent=coef_ent,
            max_grad_norm=max_grad_norm,reward_factor=reward_factor,max_episode_length=max_episode_length,
            cost_limit=cost_limit,risk_level=risk_level,num_envs=num_envs,wandb_log=wandb_log, conv=conv)

    eval_algo = deepcopy(algo)
    
    train(env=env,test_env=test_env,algo=algo,eval_algo=eval_algo)

    env.close()
    if (test_env):
        test_env.close()

    wandb.finish()

if __name__ == '__main__':
    main()